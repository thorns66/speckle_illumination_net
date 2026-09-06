function report = run_matlab_pilot(sampleIds, outputRoot, workerCount)
%RUN_MATLAB_PILOT Execute the frozen five-object MATLAB pilot protocol.
% By default this runs P01 only on six process workers, with worker k bound
% to GPU k. Call run_matlab_pilot({'P01','P02'}, root, 6) to extend it.
% Completed, schema-matching artifacts are reused; stages never overlap.
if nargin < 1 || isempty(sampleIds), sampleIds = {'P01'}; end
if nargin < 2, outputRoot = ''; end
if nargin < 3 || isempty(workerCount), workerCount = 6; end
sampleIds = normalize_sample_ids(sampleIds);
validateattributes(workerCount, {'numeric'}, ...
    {'scalar','integer','positive','finite'}, mfilename, 'workerCount');

pilotRoot = fileparts(mfilename('fullpath'));
codeRoot = fileparts(pilotRoot);
addpath(pilotRoot, fullfile(codeRoot, 'Util'), fullfile(codeRoot, 'Solver'));
firstCfg = pilot_dataset_config(sampleIds{1}, outputRoot);
if isempty(outputRoot), outputRoot = firstCfg.output_root; end
if ~isfolder(outputRoot), mkdir(outputRoot); end
jobRoot = fullfile(outputRoot, '_parallel_jobs');
if ~isfolder(jobRoot), mkdir(jobRoot); end

availableGpuCount = gpuDeviceCount('available');
assert(availableGpuCount >= workerCount, 'pilot:GPUCount', ...
    'Requested %d process workers but only %d GPUs are available.', ...
    workerCount, availableGpuCount);
pool = gcp('nocreate');
if isempty(pool)
    cluster = parcluster('Processes');
    cluster.JobStorageLocation = jobRoot;
    pool = parpool(cluster, workerCount);
else
    assert(isa(pool, 'parallel.ProcessPool') && pool.NumWorkers == workerCount, ...
        'pilot:ExistingPool', ['An existing pool is not the requested %d-worker ' ...
        'process pool. Close it explicitly before running the pilot.'], workerCount);
end

% Bind once. The subsequent parfor loops use the same process workers, so
% no two workers silently select the default GPU 1.
spmd
    workerGpuCount = gpuDeviceCount('available');
    assert(workerGpuCount >= numlabs, 'pilot:WorkerGPUCount', ...
        'Worker sees only %d GPUs for %d labs.', workerGpuCount, numlabs);
    selectedGpu = gpuDevice(labindex);
    workerGpu = struct('worker_index', labindex, ...
        'gpu_index', selectedGpu.Index, 'gpu_name', selectedGpu.Name);
end
gpuMap = repmat(struct('worker_index', 0, 'gpu_index', 0, 'gpu_name', ''), ...
    1, workerCount);
for worker = 1:workerCount, gpuMap(worker) = workerGpu{worker}; end
assert(isequal([gpuMap.gpu_index], 1:workerCount), 'pilot:GPUMap', ...
    'Workers were not mapped one-to-one to GPUs 1:%d.', workerCount);

reportTemplate = struct('sample_id', '', 'truth_depth_um', 0, ...
    'worker_gpu_map', [], 'elapsed_seconds', 0, 'validation', struct);
report = repmat(reportTemplate, 1, numel(sampleIds));
for sampleNumber = 1:numel(sampleIds)
    cfg = pilot_dataset_config(sampleIds{sampleNumber}, outputRoot);
    sampleStart = tic;
    fprintf('\n=== MATLAB pilot %s (%g um): prepare ===\n', ...
        cfg.sample_id, cfg.truth_depth_um);
    prepared = pilot_prepare_sample(cfg.sample_id, outputRoot);

    forwardComplete = stage_complete(cfg, 'sensor');
    if ~forwardComplete
        fprintf('=== %s: simulate 100 frames with truth-only PSF ===\n', cfg.sample_id);
        truthPsf = parallel.pool.Constant(@() pilot_load_truth_psf(cfg));
        parfor frame = 1:cfg.frame_count
            pilot_forward_frame(cfg, prepared, truthPsf.Value, frame);
        end
        delete(truthPsf);
        assert(stage_complete(cfg, 'sensor'), 'pilot:ForwardIncomplete', ...
            'Sensor generation did not produce 100 owned frames.');
    else
        fprintf('=== %s: sensor stage already complete; reusing it ===\n', cfg.sample_id);
    end

    reconComplete = stage_complete(cfg, 'recon');
    subsetComplete = stage_complete(cfg, 'subset');
    fullPsf = [];
    if ~reconComplete || ~subsetComplete
        % Constructed independently on each process: the ~10 GB H/Ht value
        % is never materialized in or broadcast from the client process.
        fullPsf = parallel.pool.Constant(@() pilot_load_psf_gpu(cfg));
    end
    if ~reconComplete
        fprintf('=== %s: reconstruct 100 frames at all 10 depths ===\n', cfg.sample_id);
        parfor frame = 1:cfg.frame_count
            pilot_reconstruct_frame(cfg, fullPsf.Value, frame);
        end
        assert(stage_complete(cfg, 'recon'), 'pilot:ReconIncomplete', ...
            'Frame reconstruction did not produce 100 owned volumes.');
    else
        fprintf('=== %s: frame reconstruction already complete; reusing it ===\n', ...
            cfg.sample_id);
    end

    if ~stage_complete(cfg, 'algorim')
        fprintf('=== %s: export depth-wise Windows AlgoRIM stacks ===\n', cfg.sample_id);
        pilot_export_algorim(cfg.sample_dir, cfg.frame_count, cfg.z_um);
    else
        fprintf('=== %s: AlgoRIM export already complete; reusing it ===\n', cfg.sample_id);
    end

    if ~subsetComplete
        fprintf('=== %s: build ten disjoint 10/90 subsets ===\n', cfg.sample_id);
        parfor subsetIndex = 1:cfg.subset_count
            pilot_build_subset(cfg, fullPsf.Value, subsetIndex);
        end
        assert(stage_complete(cfg, 'subset'), 'pilot:SubsetIncomplete', ...
            'Subset stage did not produce ten owned reconstructions.');
    else
        fprintf('=== %s: subset stage already complete; reusing it ===\n', cfg.sample_id);
    end
    if ~isempty(fullPsf), delete(fullPsf); end

    fprintf('=== %s: validate, hash, and publish manifest ===\n', cfg.sample_id);
    validation = pilot_validate_sample(cfg);
    report(sampleNumber) = struct('sample_id', cfg.sample_id, ...
        'truth_depth_um', cfg.truth_depth_um, 'worker_gpu_map', gpuMap, ...
        'elapsed_seconds', toc(sampleStart), 'validation', validation);
end
end

function ids = normalize_sample_ids(value)
if ischar(value) || (isstring(value) && isscalar(value)), value = cellstr(value); end
if isstring(value), value = cellstr(value(:)); end
assert(iscell(value) && ~isempty(value), 'pilot:SampleIDs', ...
    'sampleIds must be a nonempty character vector, string array, or cellstr.');
ids = cellfun(@(x) upper(char(x)), value(:)', 'UniformOutput', false);
assert(numel(unique(ids)) == numel(ids), 'pilot:SampleIDs', ...
    'Duplicate sample IDs are not allowed in one run.');
for index = 1:numel(ids), pilot_dataset_config(ids{index}); end
end

function complete = stage_complete(cfg, stage)
% Cheap but schema-aware resume check. Deep numerical checks and hashes are
% performed by pilot_validate_sample after every stage is present.
switch stage
    case 'sensor'
        complete = check_owned_files(fullfile(cfg.sample_dir, 'sensor_frames'), ...
            'frame_%03d.mat', cfg.frame_count, cfg, ...
            {'sensor_pre_detector','sensor_legacy_float','sensor_legacy_uint8'}, ...
            'frame_index');
    case 'recon'
        complete = check_owned_files(fullfile(cfg.sample_dir, 'recon_frames'), ...
            'frame_%03d.mat', cfg.frame_count, cfg, ...
            {'w_legacy_raw','w_physics_raw'}, 'frame_index');
    case 'subset'
        complete = check_owned_files(fullfile(cfg.sample_dir, 'subsets'), ...
            'subset_%02d.mat', cfg.subset_count, cfg, ...
            {'input_indices','holdout_indices','legacy_mean_raw', ...
             'legacy_taylor_raw','physics_mean_raw','physics_taylor_raw'}, ...
            'subset_index');
    case 'algorim'
        complete = false;
        manifestPath = fullfile(cfg.sample_dir, 'algorim', 'manifest.mat');
        if ~isfile(manifestPath), return; end
        try
            old = load(manifestPath, 'manifest');
            manifest = old.manifest;
            complete = strcmp(manifest.producer, 'pilot_export_algorim/1') && ...
                manifest.frame_count == cfg.frame_count && ...
                isequal(double(manifest.z_um), double(cfg.z_um));
            if ~complete, return; end
            for depth = 1:numel(cfg.z_um)
                file = sprintf('depth_%03dum.tif', cfg.z_um(depth));
                for stream = {'legacy_per_frame_normalized','physics_common_scale'}
                    path = fullfile(cfg.sample_dir, 'algorim', stream{1}, file);
                    if ~isfile(path) || numel(imfinfo(path)) ~= cfg.frame_count
                        complete = false; return;
                    end
                end
            end
        catch
            complete = false;
        end
    otherwise
        error('pilot:Stage', 'Unknown pipeline stage: %s', stage);
end
end

function complete = check_owned_files(folder, pattern, count, cfg, fields, indexField)
complete = isfolder(folder);
if ~complete, return; end
for index = 1:count
    path = fullfile(folder, sprintf(pattern, index));
    if ~isfile(path), complete = false; return; end
    try
        names = {whos('-file', path).name};
        required = [{'schema_version','dataset_id','sample_id',indexField}, fields];
        if ~all(ismember(required, names)), complete = false; return; end
        header = load(path, 'schema_version', 'dataset_id', 'sample_id', indexField);
        if header.schema_version ~= cfg.schema_version || ...
                ~strcmp(header.dataset_id, cfg.dataset_id) || ...
                ~strcmp(header.sample_id, cfg.sample_id) || ...
                header.(indexField) ~= index
            complete = false; return;
        end
    catch
        complete = false; return;
    end
end
end
