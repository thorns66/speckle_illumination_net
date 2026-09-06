function report = run_test_rl5_baselines(repoRoot, outputRoot, sampleId, mode, scope, subsetIndices)
% Reuse the frozen MATLAB solver on retained-scale test sensor statistics.
% scope='full100' (default) or 'subsets10'. No source dataset writes.
if nargin < 5, scope = 'full100'; end
mode = validatestring(mode, {'mean','taylor'});
scope = validatestring(scope, {'full100','subsets10'});
assert(ismember(sampleId, {'P07','T01','T02'}), 'baseline:TestObject');
addpath(fullfile(repoRoot,'matlab_code','cell_dataset'), ...
    fullfile(repoRoot,'matlab_code','pilot_dataset'), ...
    fullfile(repoRoot,'matlab_code','Util'),fullfile(repoRoot,'matlab_code','Solver'));
dataRoot = fullfile(repoRoot,'data','matlab_cells_pilot_v2_r04');
cfg = cell_simulation_config(sampleId,dataRoot);
splits = jsondecode(fileread(fullfile(dataRoot,'dataset_splits.json')));
row = splits.samples(strcmp({splits.samples.sample_id},sampleId));
assert(isscalar(row) && strcmp(row.split,'test'), 'baseline:Split');
assert(~startsWith(outputRoot,[dataRoot filesep]) && ~strcmp(outputRoot,dataRoot), ...
    'baseline:OutputMustNotBeDataset');
device = gpuDevice(1); % launcher restricts CUDA_VISIBLE_DEVICES to one UUID
started = tic;
fprintf('%s %s %s | load resident H/Ht on %s\n',sampleId,mode,scope,device.Name);
psf = pilot_load_psf_gpu(cfg);
if strcmp(scope,'full100'), subsets = 0; else, subsets = 1:10; end
if nargin >= 6
    assert(all(ismember(subsetIndices,subsets)) && numel(unique(subsetIndices))==numel(subsetIndices));
    subsets = subsetIndices;
end
artifacts = cell(1,numel(subsets));
for position = 1:numel(subsets)
    subset = subsets(position);
    if subset == 0
        tag = 'full100'; indices = 1:100;
        stack = zeros(260,260,100,'double');
        for frame = indices
            source = fullfile(cfg.sample_dir,'sensor_frames',sprintf('frame_%03d.mat',frame));
            s = load(source,'sample_id','frame_index','sensor_pre_detector');
            assert(strcmp(s.sample_id,sampleId) && s.frame_index == frame, 'baseline:FrameOwner');
            validateattributes(s.sensor_pre_detector,{'single'}, ...
                {'finite','nonnegative','size',[260,260]});
            stack(:,:,frame) = double(s.sensor_pre_detector);
        end
        meanSensor = mean(stack,3); varSensor = var(stack,0,3);
        clear stack;
        statisticsSource = fullfile(cfg.sample_dir,'sensor_frames');
    else
        tag = sprintf('subset_%02d',subset);
        statisticsSource = fullfile(cfg.sample_dir,'subsets',sprintf('subset_%02d.mat',subset));
        s = load(statisticsSource,'sample_id','subset_index','input_indices', ...
            'input_physics_mean_float','input_physics_variance_nminus1_float');
        assert(strcmp(s.sample_id,sampleId) && s.subset_index == subset, 'baseline:SubsetOwner');
        indices = double(s.input_indices);
        assert(numel(indices)==10 && numel(unique(indices))==10, 'baseline:InputFrames');
        meanSensor = s.input_physics_mean_float;
        varSensor = s.input_physics_variance_nminus1_float;
    end
    destination = fullfile(outputRoot,sampleId,tag,mode);
    if ~isfolder(destination), mkdir(destination); end
    matPath = fullfile(destination,'reconstruction.mat');
    tifPath = fullfile(destination,'reconstruction.tif');
    jsonPath = fullfile(destination,'complete.json');
    assert(~isfile(matPath) && ~isfile(tifPath) && ~isfile(jsonPath), ...
        'baseline:ExistingOutput','Refusing to overwrite existing baseline: %s',destination);
    if strcmp(mode,'mean'), sensor = single(meanSensor); else, sensor = single(varSensor); end
    fprintf('%s %s %s | start exactly 5 iterations\n',sampleId,tag,mode);
    [raw,diagnostic] = pilot_reconstruct_volume(psf,sensor,5,mode);
    if strcmp(mode,'taylor'), reconstruction = sqrt(raw); else, reconstruction = raw; end
    record = struct('sample_id',sampleId,'method',mode,'frame_count',numel(indices), ...
        'subset_index',subset,'input_indices',indices,'iterations',5,'z_um',psf.z_um, ...
        'sensor_mean',meanSensor,'sensor_variance_nminus1',varSensor, ...
        'raw_solver_output',raw,'reconstruction',reconstruction,'diagnostic',diagnostic, ...
        'stream','physics sensor_pre_detector; double statistics; single solver', ...
        'statistics_source',statisticsSource,'psf_source',psf.source_path, ...
        'selected_psf_indices_one_based',psf.selected_indices_one_based, ...
        'array_axis_order','YXZ','taylor_policy','H.^2/Ht.^2; sqrt of raw output once', ...
        'gain_policy','retained scale; no max normalization; no GT-derived gain in reconstruction');
    pilot_atomic_save(matPath,record);
    pilot_write_tiff(tifPath,reconstruction);
    record = rmfield(record,{'sensor_mean','sensor_variance_nminus1','raw_solver_output','reconstruction'});
    record.complete = true; record.output_mat = matPath; record.output_tif = tifPath;
    record.elapsed_seconds_from_worker_start = toc(started);
    cell_write_json(jsonPath,record);
    artifacts{position} = jsonPath;
    fprintf('%s %s %s | COMPLETE %.1f seconds\n',sampleId,tag,mode,toc(started));
end
report = struct('complete',true,'sample_id',sampleId,'method',mode, ...
    'scope',scope,'artifacts',{artifacts},'elapsed_seconds',toc(started));
end
