function manifest = pilot_validate_sample(configOrSample, outputRoot)
%PILOT_VALIDATE_SAMPLE Deep validation and SHA-256 manifest for one sample.
% This is deliberately stricter than the resume checks in run_matlab_pilot:
% it recomputes every 10/90 statistic from the authoritative float frames.
if nargin < 1 || isempty(configOrSample), configOrSample = 'P01'; end
if isstruct(configOrSample)
    cfg = configOrSample;
else
    if nargin < 2, outputRoot = ''; end
    cfg = pilot_dataset_config(configOrSample, outputRoot);
end
assert(isequal(double(cfg.z_um), 10:10:100), 'pilot:ValidationDepths', ...
    'The pilot reconstruction grid must be exactly 10:10:100 um.');
preparedPath = fullfile(cfg.sample_dir, 'prepared.mat');
assert(isfile(preparedPath), 'pilot:ValidationMissing', 'Missing prepared.mat.');
prepared = load(preparedPath);
isContinuousVolume=isfield(cfg,'ground_truth_mode')&& ...
    strcmp(cfg.ground_truth_mode,'continuous_multidepth_or_independent_control');
if isContinuousVolume
    requiredPrepared={'cfg','ground_truth','ground_truth_fine','truth_source_sha256', ...
        'occupied_truth_indices_one_based','input_indices','holdout_indices'};
else
    requiredPrepared={'cfg','object','ground_truth','truth_index_one_based', ...
        'input_indices','holdout_indices'};
end
assert(all(isfield(prepared, requiredPrepared)), 'pilot:ValidationSchema', ...
    'prepared.mat is incomplete.');
assert(strcmp(prepared.cfg.dataset_id, cfg.dataset_id) && ...
    strcmp(prepared.cfg.sample_id, cfg.sample_id) && ...
    prepared.cfg.schema_version == cfg.schema_version && ...
    isequal(double(prepared.cfg.z_um), double(cfg.z_um)), ...
    'pilot:ValidationSchema', 'prepared.mat does not match cfg.');
validate_volume(prepared.ground_truth, cfg, 'ground_truth');
if isContinuousVolume
    truth=load(cfg.truth_path,'ground_truth','ground_truth_fine','source_sha256');
    assert(strcmp(prepared.truth_source_sha256,cfg.truth_source_sha256)&& ...
        strcmp(truth.source_sha256,cfg.truth_source_sha256)&& ...
        isequal(prepared.ground_truth,truth.ground_truth)&& ...
        isequal(prepared.ground_truth_fine,truth.ground_truth_fine), ...
        'pilot:ValidationObject','Prepared volume differs from approved truth.');
    truthIndex=double(prepared.occupied_truth_indices_one_based);
    expected=find(squeeze(sum(sum(prepared.ground_truth,1),2))>0)';
    assert(isequal(truthIndex,expected)&&~isempty(truthIndex), ...
        'pilot:ValidationObject','Occupied depth metadata is incorrect.');
else
    assert(isa(prepared.object,'double')&&isequal(size(prepared.object),cfg.image_size)&& ...
        all(isfinite(prepared.object(:)))&&all(prepared.object(:)>=0), ...
        'pilot:ValidationObject','Object must be finite nonnegative 260-by-260 double.');
    truthIndex=find(cfg.z_um==cfg.truth_depth_um);
    assert(isscalar(truthIndex)&&prepared.truth_index_one_based==truthIndex&& ...
        isequal(prepared.ground_truth(:,:,truthIndex),single(prepared.object))&& ...
        nnz(prepared.ground_truth(:,:,[1:truthIndex-1,truthIndex+1:end]))==0, ...
        'pilot:ValidationObject','Ground truth is not confined to its declared layer.');
end
validate_all_splits(prepared, cfg);

legacyFrames = zeros(cfg.image_size(1), cfg.image_size(2), cfg.frame_count, 'double');
physicsFrames = zeros(cfg.image_size(1), cfg.image_size(2), cfg.frame_count, 'double');
for frame = 1:cfg.frame_count
    path = fullfile(cfg.sample_dir, 'sensor_frames', sprintf('frame_%03d.mat', frame));
    assert(isfile(path), 'pilot:ValidationMissing', 'Missing %s.', path);
    data = load(path);
    require_owner(data, cfg, 'frame_index', frame, path);
    required={'sensor_pre_detector','sensor_legacy_before_max', ...
        'sensor_legacy_float','sensor_legacy_uint8'};
    if isContinuousVolume
        required=[required,{'occupied_truth_indices_one_based','z_um'}];
    else
        required=[required,{'truth_index_one_based'}];
    end
    assert(all(isfield(data, required)), 'pilot:ValidationSchema', ...
        'Sensor frame schema is incomplete: %s', path);
    validate_image(data.sensor_pre_detector, 'single', cfg, path);
    validate_image(data.sensor_legacy_before_max, 'single', cfg, path);
    validate_image(data.sensor_legacy_float, 'double', cfg, path);
    validate_image(data.sensor_legacy_uint8, 'uint8', cfg, path);
    if isContinuousVolume
        correctTruth=isequal(double(data.occupied_truth_indices_one_based),truthIndex)&& ...
            isequal(double(data.z_um),double(cfg.z_um));
    else
        correctTruth=data.truth_index_one_based==truthIndex;
    end
    assert(correctTruth&&max(data.sensor_legacy_float(:)) == 1 && ...
        isequal(im2uint8(data.sensor_legacy_float), data.sensor_legacy_uint8), ...
        'pilot:ValidationSensor', 'Invalid normalization/encoding in %s.', path);
    tiffPath = fullfile(cfg.sample_dir, 'sensor_frames', ...
        sprintf('frame_%03d_legacy.tif', frame));
    assert(isfile(tiffPath) && isequal(imread(tiffPath), data.sensor_legacy_uint8), ...
        'pilot:ValidationSensor', 'Legacy TIFF round-trip mismatch for frame %d.', frame);
    legacyFrames(:,:,frame) = data.sensor_legacy_float;
    physicsFrames(:,:,frame) = double(data.sensor_pre_detector);
end

for frame = 1:cfg.frame_count
    path = fullfile(cfg.sample_dir, 'recon_frames', sprintf('frame_%03d.mat', frame));
    assert(isfile(path), 'pilot:ValidationMissing', 'Missing %s.', path);
    data = load(path);
    require_owner(data, cfg, 'frame_index', frame, path);
    assert(isfield(data, 'z_um') && isequal(double(data.z_um), double(cfg.z_um)), ...
        'pilot:ValidationDepths', 'Frame %d has the wrong depth grid.', frame);
    rawNames = {'w_legacy_raw','w_physics_raw'};
    previewNames = {'w_legacy_preview_uint16','w_physics_preview_uint16'};
    for stream = 1:2
        validate_volume(data.(rawNames{stream}), cfg, rawNames{stream});
        expected = pilot_legacy_preview(data.(rawNames{stream}), 'mean');
        assert(isa(data.(previewNames{stream}), 'uint16') && ...
            isequal(data.(previewNames{stream}), expected), ...
            'pilot:ValidationPreview', 'Frame %d preview mismatch.', frame);
    end
end

for subsetIndex = 1:cfg.subset_count
    path = fullfile(cfg.sample_dir, 'subsets', sprintf('subset_%02d.mat', subsetIndex));
    assert(isfile(path), 'pilot:ValidationMissing', 'Missing %s.', path);
    data = load(path);
    require_owner(data, cfg, 'subset_index', subsetIndex, path);
    assert(isfield(data, 'z_um') && isequal(double(data.z_um), double(cfg.z_um)) && ...
        isequal(double(data.input_indices), double(prepared.input_indices(subsetIndex,:))) && ...
        isequal(double(data.holdout_indices), double(prepared.holdout_indices(subsetIndex,:))), ...
        'pilot:ValidationSubset', 'Subset %d indices/depths changed.', subsetIndex);
    input = prepared.input_indices(subsetIndex,:);
    holdout = prepared.holdout_indices(subsetIndex,:);
    expectedStats = struct( ...
        'input_legacy_mean_float', mean(legacyFrames(:,:,input), 3), ...
        'input_legacy_variance_nminus1_float', var(legacyFrames(:,:,input), 0, 3), ...
        'holdout_legacy_mean_float', mean(legacyFrames(:,:,holdout), 3), ...
        'holdout_legacy_variance_nminus1_float', var(legacyFrames(:,:,holdout), 0, 3), ...
        'input_physics_mean_float', mean(physicsFrames(:,:,input), 3), ...
        'input_physics_variance_nminus1_float', var(physicsFrames(:,:,input), 0, 3), ...
        'holdout_physics_mean_float', mean(physicsFrames(:,:,holdout), 3), ...
        'holdout_physics_variance_nminus1_float', var(physicsFrames(:,:,holdout), 0, 3));
    statNames = fieldnames(expectedStats);
    for index = 1:numel(statNames)
        name = statNames{index};
        assert(isfield(data, name) && isequal(data.(name), expectedStats.(name)), ...
            'pilot:ValidationStatistics', ...
            'Subset %d does not exactly reproduce %s.', subsetIndex, name);
    end
    expectedMeanNormalized = expectedStats.input_legacy_mean_float / ...
        max(expectedStats.input_legacy_mean_float(:));
    expectedVarianceNormalized = expectedStats.input_legacy_variance_nminus1_float / ...
        max(expectedStats.input_legacy_variance_nminus1_float(:));
    assert(isequal(data.legacy_mean_normalized_float, expectedMeanNormalized) && ...
        isequal(data.legacy_variance_normalized_float, expectedVarianceNormalized) && ...
        isequal(data.legacy_mean_uint8, im2uint8(expectedMeanNormalized)) && ...
        isequal(data.legacy_variance_uint8, im2uint8(expectedVarianceNormalized)) && ...
        isequal(data.legacy_mean_input_single, single(data.legacy_mean_uint8)/single(255)) && ...
        isequal(data.legacy_taylor_input_single, single(data.legacy_variance_uint8)/single(255)) && ...
        isequal(data.physics_mean_input_single, single(expectedStats.input_physics_mean_float)) && ...
        isequal(data.physics_taylor_input_single, single(expectedStats.input_physics_variance_nminus1_float)), ...
        'pilot:ValidationStatistics', 'Subset %d input encoding changed.', subsetIndex);
    rawNames = {'legacy_mean_raw','legacy_taylor_raw', ...
        'physics_mean_raw','physics_taylor_raw'};
    previewNames = {'legacy_mean_preview_uint16','legacy_taylor_preview_uint16', ...
        'physics_mean_preview_uint16','physics_taylor_preview_uint16'};
    previewModes = {'mean','taylor','mean','taylor'};
    previewPaths = { ...
        fullfile(cfg.sample_dir, 'previews', sprintf('subset_%02d_legacy_mean.tif', subsetIndex)), ...
        fullfile(cfg.sample_dir, 'previews', sprintf('subset_%02d_legacy_taylor_sqrt.tif', subsetIndex)), ...
        fullfile(cfg.sample_dir, 'previews', sprintf('subset_%02d_physics_mean.tif', subsetIndex)), ...
        fullfile(cfg.sample_dir, 'previews', sprintf('subset_%02d_physics_taylor_sqrt.tif', subsetIndex))};
    for stream = 1:4
        validate_volume(data.(rawNames{stream}), cfg, rawNames{stream});
        expectedPreview = pilot_legacy_preview(data.(rawNames{stream}), previewModes{stream});
        assert(isa(data.(previewNames{stream}), 'uint16') && ...
            isequal(data.(previewNames{stream}), expectedPreview), ...
            'pilot:ValidationPreview', 'Subset %d preview mismatch.', subsetIndex);
        assert(isfile(previewPaths{stream}) && numel(imfinfo(previewPaths{stream})) == 10, ...
            'pilot:ValidationPages', 'Subset preview must contain exactly 10 pages: %s', ...
            previewPaths{stream});
        assert(isequal(read_tiff_stack(previewPaths{stream}), expectedPreview), ...
            'pilot:ValidationPreview', 'Published subset TIFF differs from its MAT array.');
    end
end
clear legacyFrames physicsFrames;

validate_algorim(cfg);
groundTruthTiff = fullfile(cfg.sample_dir, 'previews', 'ground_truth_float.tif');
assert(isfile(groundTruthTiff) && numel(imfinfo(groundTruthTiff)) == 10 && ...
    isequal(read_tiff_stack(groundTruthTiff), prepared.ground_truth), ...
    'pilot:ValidationPages', 'Ground-truth TIFF must be the exact 10-page float volume.');

hashes = hash_artifacts(cfg.sample_dir);
manifestTruthDepth=[];
if ~isContinuousVolume, manifestTruthDepth=cfg.truth_depth_um; end
manifest = struct('schema_version', cfg.schema_version, ...
    'producer', 'pilot_validate_sample/1', 'complete', true, ...
    'dataset_id', cfg.dataset_id, 'sample_id', cfg.sample_id, ...
    'truth_depth_um', manifestTruthDepth, 'z_um', double(cfg.z_um), ...
    'ground_truth_mode', ternary(isContinuousVolume, ...
        'continuous_multidepth_or_independent_control','single_plane'), ...
    'frame_count', cfg.frame_count, 'subset_count', cfg.subset_count, ...
    'input_frames_per_subset', cfg.input_frames, ...
    'holdout_frames_per_subset', cfg.holdout_frames, ...
    'variance_normalization', 'N-1 (MATLAB var(...,0,3))', ...
    'artifact_count', numel(hashes), 'artifacts', hashes, ...
    'validated_utc', char(datetime('now','TimeZone','UTC', ...
        'Format','yyyy-MM-dd''T''HH:mm:ss.SSSXXX')));
pilot_atomic_save(fullfile(cfg.sample_dir, 'validation_manifest.mat'), ...
    struct('manifest', manifest));
write_json_atomic(fullfile(cfg.sample_dir, 'validation_manifest.json'), manifest);
end

function value=ternary(condition,yes,no)
if condition, value=yes; else, value=no; end
end

function validate_all_splits(prepared, cfg)
assert(isequal(size(prepared.input_indices), [cfg.subset_count,cfg.input_frames]) && ...
    isequal(size(prepared.holdout_indices), [cfg.subset_count,cfg.holdout_frames]), ...
    'pilot:ValidationSplit', 'Split matrices have the wrong shape.');
assert(isequal(sort(prepared.input_indices(:))', 1:cfg.frame_count), ...
    'pilot:ValidationSplit', 'The ten input subsets must partition frames 1:100.');
for subsetIndex = 1:cfg.subset_count
    input = double(prepared.input_indices(subsetIndex,:));
    holdout = double(prepared.holdout_indices(subsetIndex,:));
    assert(numel(unique(input)) == cfg.input_frames && ...
        numel(unique(holdout)) == cfg.holdout_frames && ...
        isempty(intersect(input, holdout)) && ...
        isequal(sort([input,holdout]), 1:cfg.frame_count), ...
        'pilot:ValidationSplit', 'Subset %d is not an exact 10/90 split.', subsetIndex);
end
end

function require_owner(data, cfg, indexField, expectedIndex, path)
required = {'schema_version','dataset_id','sample_id',indexField};
assert(all(isfield(data, required)) && data.schema_version == cfg.schema_version && ...
    strcmp(data.dataset_id, cfg.dataset_id) && strcmp(data.sample_id, cfg.sample_id) && ...
    data.(indexField) == expectedIndex, 'pilot:ValidationSchema', ...
    'Artifact does not belong to this protocol: %s', path);
end

function validate_image(image, className, cfg, path)
assert(isa(image, className) && isequal(size(image), cfg.image_size) && ...
    isreal(image) && all(isfinite(double(image(:)))) && all(image(:) >= 0), ...
    'pilot:ValidationImage', 'Invalid image field in %s.', path);
end

function validate_volume(volume, cfg, name)
assert(isa(volume, 'single') && ...
    isequal(size(volume), [cfg.image_size,numel(cfg.z_um)]) && ...
    isreal(volume) && all(isfinite(volume(:))) && all(volume(:) >= 0), ...
    'pilot:ValidationVolume', '%s must be finite nonnegative 260-by-260-by-10 single.', name);
end

function validate_algorim(cfg)
manifestPath = fullfile(cfg.sample_dir, 'algorim', 'manifest.mat');
assert(isfile(manifestPath), 'pilot:ValidationMissing', 'Missing AlgoRIM manifest.');
data = load(manifestPath, 'manifest');
manifest = data.manifest;
assert(strcmp(manifest.producer, 'pilot_export_algorim/1') && ...
    manifest.frame_count == cfg.frame_count && ...
    isequal(double(manifest.z_um), double(cfg.z_um)) && ...
    isequal(double(manifest.frame_indices), 1:cfg.frame_count), ...
    'pilot:ValidationAlgoRIM', 'AlgoRIM manifest does not match the sample.');
for depthIndex = 1:numel(cfg.z_um)
    name = sprintf('depth_%03dum.tif', cfg.z_um(depthIndex));
    for stream = {'legacy_per_frame_normalized','physics_common_scale'}
        path = fullfile(cfg.sample_dir, 'algorim', stream{1}, name);
        info = imfinfo(path);
        assert(numel(info) == cfg.frame_count && all([info.BitDepth] == 16), ...
            'pilot:ValidationAlgoRIM', ...
            'AlgoRIM stack must have 100 uint16 pages: %s', path);
    end
end
end

function volume = read_tiff_stack(path)
info = imfinfo(path);
t = Tiff(path, 'r');
first = t.read();
volume = zeros(size(first,1), size(first,2), numel(info), 'like', first);
volume(:,:,1) = first;
try
    for page = 2:numel(info)
        t.setDirectory(page);
        volume(:,:,page) = t.read();
    end
    t.close();
catch exception
    try, t.close(); catch, end
    rethrow(exception);
end
end

function records = hash_artifacts(sampleDir)
listing = dir(fullfile(sampleDir, '**', '*'));
listing = listing(~[listing.isdir]);
relative = cell(1, numel(listing));
keep = true(1, numel(listing));
for index = 1:numel(listing)
    absolute = fullfile(listing(index).folder, listing(index).name);
    relative{index} = absolute(numel(sampleDir)+2:end);
    portableRelative = strrep(relative{index}, '\', '/');
    keep(index) = ~startsWith(portableRelative, 'logs/') && ~ismember(relative{index}, ...
        {'validation_manifest.mat','validation_manifest.json', ...
        'full_run_report.mat','full_run_report.json'});
end
listing = listing(keep);
relative = relative(keep);
[relative, order] = sort(relative);
listing = listing(order);
records = repmat(struct('relative_path','','bytes',0,'sha256',''), 1, numel(listing));
for index = 1:numel(listing)
    absolute = fullfile(listing(index).folder, listing(index).name);
    records(index) = struct('relative_path', relative{index}, ...
        'bytes', double(listing(index).bytes), 'sha256', sha256_file(absolute));
end
end

function hash = sha256_file(path)
digest = javaMethod('getInstance', 'java.security.MessageDigest', 'SHA-256');
fid = fopen(path, 'rb');
assert(fid ~= -1, 'pilot:ValidationHash', 'Cannot open artifact for hashing: %s', path);
try
    while true
        bytes = fread(fid, 1024*1024, '*uint8');
        if isempty(bytes), break; end
        digest.update(typecast(bytes(:), 'int8'));
    end
    fclose(fid);
catch exception
    fclose(fid);
    rethrow(exception);
end
raw = typecast(digest.digest(), 'uint8');
hash = lower(reshape(dec2hex(raw, 2).', 1, []));
end

function write_json_atomic(path, payload)
parent = fileparts(path);
temporary = [tempname(parent) '.json'];
fid = fopen(temporary, 'w', 'n', 'UTF-8');
assert(fid ~= -1, 'pilot:ValidationManifest', 'Cannot create temporary JSON manifest.');
try
    count = fprintf(fid, '%s\n', jsonencode(payload, 'PrettyPrint', true));
    assert(count > 0, 'pilot:ValidationManifest', 'Could not write JSON manifest.');
    fclose(fid);
    [ok, message] = movefile(temporary, path, 'f');
    assert(ok, 'pilot:ValidationManifest', 'Could not publish JSON: %s', message);
catch exception
    try, fclose(fid); catch, end
    if isfile(temporary), delete(temporary); end
    rethrow(exception);
end
end
