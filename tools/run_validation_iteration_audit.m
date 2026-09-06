function run_validation_iteration_audit(repoRoot, outputRoot, sampleId, modeName, smokeOnly)
% Validation reconstruction scan for historical update and standard RL.
% Each worker owns one (sample, statistic) pair and one visible GPU.

arguments
    repoRoot (1,:) char
    outputRoot (1,:) char
    sampleId (1,:) char
    modeName (1,:) char
    smokeOnly (1,1) logical = false
end

validSamples = {'P09', 'V01', 'V02'};
validModes = {'mean', 'taylor'};
assert(any(strcmp(sampleId, validSamples)), 'Unexpected validation sample: %s', sampleId);
assert(any(strcmp(modeName, validModes)), 'Unexpected mode: %s', modeName);

addpath(fullfile(repoRoot, 'matlab_code', 'Solver'));
addpath(fullfile(repoRoot, 'matlab_code', 'Util'));
addpath(fullfile(repoRoot, 'matlab_code', 'pilot_dataset'));
addpath(fullfile(repoRoot, 'matlab_code', 'cell_dataset'));
addpath(fullfile(repoRoot, 'tools'));

gpu = gpuDevice(1);
fprintf('worker sample=%s mode=%s gpu=%s free_GiB=%.2f\n', ...
    sampleId, modeName, gpu.Name, gpu.AvailableMemory / 2^30);

dataRoot = fullfile(repoRoot, 'data', 'matlab_cells_pilot_v2_r04');
cfg = cell_simulation_config(sampleId, dataRoot);
assert(strcmp(cfg.split, 'validation'), 'Sample is not in the validation split.');
sampleRoot = cfg.sample_dir;
preparedPath = fullfile(sampleRoot, 'prepared.mat');
geometryPath = fullfile(sampleRoot, 'geometry.json');
sensorRoot = fullfile(sampleRoot, 'sensor_frames');
subsetRoot = fullfile(sampleRoot, 'subsets');

assert(isfile(preparedPath), 'Missing %s', preparedPath);
assert(isfile(geometryPath), 'Missing %s', geometryPath);

pixelSizeUm = double(cfg.object_pixel_pitch_um);
zUm = double(cfg.z_um(:)');
psf = pilot_load_psf_gpu(cfg);

if strcmp(modeName, 'taylor')
    Hselected = psf.H .^ 2;
    HtSelected = psf.Ht .^ 2;
else
    Hselected = psf.H;
    HtSelected = psf.Ht;
end

configure_operators(Hselected, cfg.image_size, zUm);

if ~smokeOnly && strcmp(sampleId, 'P09') && strcmp(modeName, 'mean')
    auditDir = fullfile(outputRoot, 'operator_audit');
    if ~isfolder(auditDir)
        mkdir(auditDir);
    end
    run_operator_audit(psf.H, psf.Ht, pixelSizeUm, cfg.image_size, zUm, auditDir);
end
clear psf

% Historical update: every fixed 10-frame subset and the full 100 frames.
if smokeOnly
    historicalJobs = {'subset_01'};
    maximumIterations = 3;
else
    historicalJobs = [arrayfun(@(k) sprintf('subset_%02d', k), 1:10, ...
        'UniformOutput', false), {'full_100'}];
    maximumIterations = 50;
end
for jobIndex = 1:numel(historicalJobs)
    tag = historicalJobs{jobIndex};
    [sensorStatistic, sourceInfo] = load_statistic( ...
        modeName, tag, subsetRoot, sensorRoot);
    trajectoryDir = fullfile(outputRoot, 'historical_isra', sampleId, tag, modeName);
    run_trajectory(Hselected, HtSelected, sensorStatistic, trajectoryDir, ...
        sourceInfo, 'historical_isra', modeName, sampleId, tag, true, maximumIterations);
    clear sensorStatistic
end

% Standard Poisson-form RL is an algorithm-contract audit. It is run on one
% fixed 10-frame subset and the full 100 frames, not treated as a tuned Taylor
% likelihood (variance data are not Poisson measurements).
if smokeOnly
    standardJobs = {};
else
    standardJobs = {'subset_01', 'full_100'};
end
for jobIndex = 1:numel(standardJobs)
    tag = standardJobs{jobIndex};
    [sensorStatistic, sourceInfo] = load_statistic( ...
        modeName, tag, subsetRoot, sensorRoot);
    trajectoryDir = fullfile(outputRoot, 'standard_rl', sampleId, tag, modeName);
    run_trajectory(Hselected, HtSelected, sensorStatistic, trajectoryDir, ...
        sourceInfo, 'standard_rl', modeName, sampleId, tag, false, maximumIterations);
    clear sensorStatistic
end

fprintf('worker_complete sample=%s mode=%s\n', sampleId, modeName);
end


function configure_operators(H, imageSize, zUm)
global volumeResolution zeroImageEx exsize
assert(size(H, 5) == numel(zUm), 'PSF depth count does not match z_um.');
assert(isequal(imageSize, [260, 260]), 'Unexpected image size.');
volumeResolution = [imageSize, numel(zUm)];
msize = [size(H, 1), size(H, 2)];
mmid = floor(msize / 2);
candidate = imageSize + mmid;
exsize = [min(2^ceil(log2(candidate(1))), 128 * ceil(candidate(1) / 128)), ...
    min(2^ceil(log2(candidate(2))), 128 * ceil(candidate(2) / 128))];
zeroImageEx = gpuArray.zeros(exsize, 'single');
end


function [statistic, sourceInfo] = load_statistic(modeName, tag, subsetRoot, sensorRoot)
if startsWith(tag, 'subset_')
    subsetNumber = sscanf(tag, 'subset_%d');
    subsetPath = fullfile(subsetRoot, sprintf('subset_%02d.mat', subsetNumber));
    assert(isfile(subsetPath), 'Missing subset: %s', subsetPath);
    if strcmp(modeName, 'mean')
        fieldName = 'input_physics_mean_float';
    else
        fieldName = 'input_physics_variance_nminus1_float';
    end
    subset = load(subsetPath, fieldName, 'input_indices');
    statistic = single(subset.(fieldName));
    sourceInfo = struct( ...
        'kind', 'fixed_10_frame_subset', ...
        'path', subsetPath, ...
        'field', fieldName, ...
        'frame_indices_one_based', double(subset.input_indices(:)'));
else
    sensorFiles = dir(fullfile(sensorRoot, 'frame_*.mat'));
    assert(numel(sensorFiles) == 100, ...
        'Expected 100 sensor frames in %s, found %d.', sensorRoot, numel(sensorFiles));
    [~, order] = sort({sensorFiles.name});
    sensorFiles = sensorFiles(order);
    first = load(fullfile(sensorFiles(1).folder, sensorFiles(1).name), 'sensor_pre_detector');
    stack = zeros([size(first.sensor_pre_detector), numel(sensorFiles)], 'double');
    stack(:, :, 1) = double(first.sensor_pre_detector);
    for frameIndex = 2:numel(sensorFiles)
        frame = load(fullfile(sensorFiles(frameIndex).folder, sensorFiles(frameIndex).name), ...
            'sensor_pre_detector');
        stack(:, :, frameIndex) = double(frame.sensor_pre_detector);
    end
    if strcmp(modeName, 'mean')
        statistic = mean(stack, 3);
        fieldName = 'mean(double(sensor_pre_detector),3)';
    else
        statistic = var(stack, 0, 3);
        fieldName = 'var(double(sensor_pre_detector),0,3)';
    end
    clear stack
    sourceInfo = struct( ...
        'kind', 'full_100_frames', ...
        'path', sensorRoot, ...
        'field', fieldName, ...
        'frame_indices_one_based', 1:100);
end

assert(all(isfinite(statistic), 'all'), 'Non-finite sensor statistic.');
assert(all(statistic >= 0, 'all'), 'Negative sensor statistic.');
end


function run_trajectory(H, Ht, sensorStatistic, trajectoryDir, sourceInfo, ...
        algorithmName, modeName, sampleId, tag, compareIteration3, maximumIterations)
hasJson = isfile(fullfile(trajectoryDir, 'complete.json'));
hasMat = isfile(fullfile(trajectoryDir, 'trajectory.mat'));
if hasJson && hasMat
    fprintf('skip_complete trajectory=%s\n', trajectoryDir);
    return
elseif hasJson || hasMat
    error('Partial trajectory requires manual audit: %s', trajectoryDir);
end
if ~isfolder(trajectoryDir)
    mkdir(trajectoryDir);
end

checkpointIterations = [5, 10, 20, 50];
checkpointIterations = checkpointIterations(checkpointIterations <= maximumIterations);
if compareIteration3 && startsWith(tag, 'subset_')
    saveIterations = [3, checkpointIterations];
else
    saveIterations = checkpointIterations;
end

sensorGpu = single(gpuArray(sensorStatistic));
Hty = backwardProjectGPU(Ht, sensorGpu);
X = Hty;

if strcmp(algorithmName, 'standard_rl')
    sensitivity = backwardProjectGPU(Ht, ones(size(sensorGpu), 'single', 'gpuArray'));
    sensitivity = max(sensitivity, 0);
else
    sensitivity = [];
end

result = struct();
result.sample_id = sampleId;
result.mode = modeName;
result.algorithm = algorithmName;
result.tag = tag;
result.source_info = sourceInfo;
result.iteration_times_seconds = zeros(1, maximumIterations);

totalTimer = tic;
for iteration = 1:maximumIterations
    iterationTimer = tic;
    if strcmp(algorithmName, 'historical_isra')
        projected = forwardProjectGPU(H, X);
        denominator = backwardProjectGPU(Ht, projected);
        X = X .* (Hty ./ denominator);
        X(isnan(X)) = 0;  % exact historical behavior
    else
        X = max(X, 0);
        projected = max(forwardProjectGPU(H, X), 0);
        ratio = zeros(size(projected), 'single', 'gpuArray');
        validProjected = projected > 0;
        ratio(validProjected) = sensorGpu(validProjected) ./ projected(validProjected);
        correction = max(backwardProjectGPU(Ht, ratio), 0);
        nextX = zeros(size(X), 'single', 'gpuArray');
        validSensitivity = sensitivity > 0;
        nextX(validSensitivity) = X(validSensitivity) .* ...
            correction(validSensitivity) ./ sensitivity(validSensitivity);
        X = max(nextX, 0);
        X(~isfinite(X)) = 0;
    end
    wait(gpuDevice);
    result.iteration_times_seconds(iteration) = toc(iterationTimer);

    if any(saveIterations == iteration)
        rawUnsanitized = single(gather(X));
        assert(all(isfinite(rawUnsanitized), 'all'), ...
            'Non-finite reconstruction at iteration %d.', iteration);
        % Preserve the in-GPU trajectory exactly; sanitize only its gathered
        % checkpoint with the same two-gate FFT-roundoff policy used by the
        % frozen pilot reconstruction wrapper.
        [raw, roundoff] = pilot_sanitize_roundoff( ...
            rawUnsanitized, sprintf('%s %s iter %d', algorithmName, modeName, iteration));
        if strcmp(modeName, 'taylor')
            reconstruction = sqrt(raw);
        else
            reconstruction = raw;
        end
        fieldName = sprintf('reconstruction_iter_%03d', iteration);
        result.(fieldName) = reconstruction;
        result.(sprintf('roundoff_iter_%03d', iteration)) = roundoff;
        if iteration == 3
            result.raw_iter_003 = raw;
        end
        fprintf('%s %s %s %s iter=%d elapsed=%.1fs\n', ...
            sampleId, modeName, algorithmName, tag, iteration, toc(totalTimer));
    end
end

result.total_seconds = toc(totalTimer);
device = gpuDevice();
result.gpu_name = device.Name;
result.created_at = char(datetime('now', 'TimeZone', 'local', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));

if compareIteration3 && startsWith(tag, 'subset_')
    subsetPath = sourceInfo.path;
    if strcmp(modeName, 'mean')
        oldField = 'physics_mean_raw';
    else
        oldField = 'physics_taylor_raw';
    end
    old = load(subsetPath, oldField);
    oldRaw = single(old.(oldField));
    delta = result.raw_iter_003 - oldRaw;
    result.iteration3_regression = struct( ...
        'old_field', oldField, ...
        'relative_l2', double(norm(delta(:)) / max(norm(oldRaw(:)), eps('single'))), ...
        'max_abs', double(max(abs(delta(:)))));
end

matPath = fullfile(trajectoryDir, 'trajectory.mat');
tmpMatPath = [matPath, '.tmp'];
save(tmpMatPath, '-struct', 'result', '-v7.3');
movefile(tmpMatPath, matPath, 'f');

completion = rmfield(result, fields_with_large_arrays(result));
completion.mat_path = matPath;
completion.saved_iterations = saveIterations;
completionPath = fullfile(trajectoryDir, 'complete.json');
write_json_atomic(completionPath, completion);
end


function names = fields_with_large_arrays(record)
allNames = fieldnames(record);
keep = false(size(allNames));
for k = 1:numel(allNames)
    keep(k) = startsWith(allNames{k}, 'reconstruction_iter_') || ...
        startsWith(allNames{k}, 'raw_iter_');
end
names = allNames(keep);
end


function run_operator_audit(Hbase, HtBase, pixelSizeUm, imageSize, zUm, auditDir)
auditPath = fullfile(auditDir, 'operator_audit.mat');
jsonPath = fullfile(auditDir, 'operator_audit.json');
if isfile(auditPath) && isfile(jsonPath)
    fprintf('skip_complete operator_audit=%s\n', jsonPath);
    return
elseif isfile(auditPath) || isfile(jsonPath)
    error('Partial operator audit requires manual review in %s', auditDir);
end

rng(20260906, 'twister');
audit = struct();
audit.created_at = char(datetime('now', 'TimeZone', 'local', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
audit.pixel_size_um = pixelSizeUm;
audit.z_um = zUm;

for modeCell = {'mean', 'taylor'}
    currentMode = modeCell{1};
    if strcmp(currentMode, 'taylor')
        currentH = Hbase .^ 2;
        currentHt = HtBase .^ 2;
    else
        currentH = Hbase;
        currentHt = HtBase;
    end
    configure_operators(currentH, imageSize, zUm);
    x = single(gpuArray.rand(imageSize(1), imageSize(2), numel(zUm)));
    y = single(gpuArray.rand(imageSize(1), imageSize(2)));
    Ax = forwardProjectGPU(currentH, x);
    Aty = backwardProjectGPU(currentHt, y);
    lhs = double(gather(sum(Ax .* y, 'all')));
    rhs = double(gather(sum(x .* Aty, 'all')));
    sensitivity = gather(backwardProjectGPU(currentHt, ...
        ones(size(y), 'single', 'gpuArray')));
    positiveSensitivity = sensitivity(sensitivity > 0 & isfinite(sensitivity));
    modeAudit = struct();
    modeAudit.inner_product_lhs = lhs;
    modeAudit.inner_product_rhs = rhs;
    modeAudit.adjoint_relative_error = abs(lhs - rhs) / max([abs(lhs), abs(rhs), eps]);
    modeAudit.sensitivity_min_positive = double(min(positiveSensitivity));
    modeAudit.sensitivity_max = double(max(positiveSensitivity));
    modeAudit.sensitivity_mean = double(mean(positiveSensitivity));
    modeAudit.sensitivity_cv = double(std(positiveSensitivity) / mean(positiveSensitivity));
    modeAudit.sensitivity_zero_fraction = double(mean(sensitivity(:) <= 0));

    kernelMassH = gather(squeeze(sum(currentH, [1, 2])));
    kernelMassHt = gather(squeeze(sum(currentHt, [1, 2])));
    modeAudit.H_kernel_mass_min = double(min(kernelMassH, [], 'all'));
    modeAudit.H_kernel_mass_max = double(max(kernelMassH, [], 'all'));
    modeAudit.H_kernel_mass_mean = double(mean(kernelMassH, 'all'));
    modeAudit.H_kernel_mass_cv = double(std(kernelMassH(:)) / mean(kernelMassH(:)));
    modeAudit.Ht_kernel_mass_min = double(min(kernelMassHt, [], 'all'));
    modeAudit.Ht_kernel_mass_max = double(max(kernelMassHt, [], 'all'));
    modeAudit.Ht_kernel_mass_mean = double(mean(kernelMassHt, 'all'));
    modeAudit.Ht_kernel_mass_cv = double(std(kernelMassHt(:)) / mean(kernelMassHt(:)));
    audit.(currentMode) = modeAudit;
    clear currentH currentHt x y Ax Aty sensitivity positiveSensitivity
end

save(auditPath, '-struct', 'audit', '-v7.3');
write_json_atomic(jsonPath, audit);
fprintf('operator_audit_complete path=%s\n', jsonPath);
end


function write_json_atomic(path, value)
tmpPath = [path, '.tmp'];
fid = fopen(tmpPath, 'w');
assert(fid >= 0, 'Could not open %s', tmpPath);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(value, PrettyPrint=true));
clear cleanup
movefile(tmpPath, path, 'f');
end
