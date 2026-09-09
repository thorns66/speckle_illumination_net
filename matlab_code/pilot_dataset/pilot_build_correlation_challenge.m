function pilot_build_correlation_challenge(manifestPath)
% Explicit additive test challenge; original sensor files are read-only.
manifest = jsondecode(fileread(manifestPath));
assert(strcmp(getenv('CUDA_VISIBLE_DEVICES'), manifest.gpu_uuid));
gpuDevice(1);
psf = [];
for object = 1:numel(manifest.samples)
    sample = manifest.samples(object);
    source = load(fullfile(sample.source_dir, 'prepared.mat'));
    cfg = source.cfg;
    cfg.sample_dir = sample.challenge_dir;
    cfg.subset_count = 2;
    assert(cfg.iterations == 3 && cfg.frame_count == 100);
    source.cfg = cfg;
    source.input_indices = [sample.low.input_indices(:)'; sample.high.input_indices(:)'];
    source.holdout_indices = [sample.low.holdout_indices(:)'; sample.high.holdout_indices(:)'];
    preparedPath = fullfile(cfg.sample_dir, 'prepared.mat');
    if ~isfile(preparedPath), save(preparedPath, '-struct', 'source', '-v7.3'); end
    if isempty(psf), psf = pilot_load_psf_gpu(cfg); end
    % Include original subset 01 as a solver/adapter numerical regression.
    original = load(fullfile(sample.source_dir, 'subsets', 'subset_01.mat'));
    for subset = 1:3
        if subset <= 2
            inputIndices = source.input_indices(subset,:);
            holdoutIndices = source.holdout_indices(subset,:);
            outputPath = fullfile(cfg.sample_dir, 'subsets', sprintf('subset_%02d.mat', subset));
        else
            inputIndices = double(original.input_indices(:)');
            holdoutIndices = double(original.holdout_indices(:)');
            outputPath = fullfile(cfg.sample_dir, 'regression.mat');
        end
        if isfile(outputPath), continue; end
        assert(numel(unique(inputIndices)) == 10 && numel(holdoutIndices) == 90);
        assert(isequal(sort([inputIndices,holdoutIndices]),1:100));
        [mu, variance] = statistics(cfg, inputIndices);
        [targetMu, targetVariance] = statistics(cfg, holdoutIndices);
        [meanRaw, meanDiagnostic] = pilot_reconstruct_volume(psf, single(mu), 3, 'mean');
        [taylorRaw, taylorDiagnostic] = pilot_reconstruct_volume(psf, single(variance), 3, 'taylor');
        record = struct('schema_version',cfg.schema_version,'dataset_id',cfg.dataset_id, ...
            'sample_id',cfg.sample_id,'subset_index',double(subset),'iterations',3, ...
            'input_indices',inputIndices,'holdout_indices',holdoutIndices,'z_um',double(cfg.z_um), ...
            'input_physics_mean_float',mu,'input_physics_variance_nminus1_float',variance, ...
            'holdout_physics_mean_float',targetMu,'holdout_physics_variance_nminus1_float',targetVariance, ...
            'physics_mean_raw',meanRaw,'physics_taylor_raw',taylorRaw, ...
            'physics_taylor_sqrt_float',sqrt(taylorRaw),'physics_mean_diagnostic',meanDiagnostic, ...
            'physics_taylor_diagnostic',taylorDiagnostic);
        if subset == 3
            assert(isequal(mu,original.input_physics_mean_float));
            assert(norm(variance(:)-original.input_physics_variance_nminus1_float(:)) / ...
                max(norm(original.input_physics_variance_nminus1_float(:)),1e-30) < 1e-12);
            record.mean_relative_l2 = double(norm(meanRaw(:)-original.physics_mean_raw(:))) / ...
                max(double(norm(original.physics_mean_raw(:))),1e-30);
            record.taylor_relative_l2 = double(norm(taylorRaw(:)-original.physics_taylor_raw(:))) / ...
                max(double(norm(original.physics_taylor_raw(:))),1e-30);
            assert(record.mean_relative_l2 < 1e-5 && record.taylor_relative_l2 < 1e-5);
        end
        pilot_atomic_save(outputPath,record);
        fprintf('CHALLENGE_COMPLETE %s subset=%d %s\n', cfg.sample_id,subset,datestr(now,31));
    end
end
end

function [mu, variance] = statistics(cfg, indices)
frames = zeros([cfg.image_size,numel(indices)],'double');
for k = 1:numel(indices)
    frame = load(fullfile(cfg.sample_dir,'sensor_frames',sprintf('frame_%03d.mat',indices(k))), ...
        'sensor_pre_detector','sample_id','frame_index');
    assert(strcmp(frame.sample_id,cfg.sample_id) && frame.frame_index == indices(k));
    frames(:,:,k) = double(frame.sensor_pre_detector);
end
mu = mean(frames,3);
variance = var(frames,0,3);
end
