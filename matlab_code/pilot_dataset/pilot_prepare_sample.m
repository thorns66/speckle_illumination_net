function summary = pilot_prepare_sample(sampleId, outputRoot)
%PILOT_PREPARE_SAMPLE Freeze object, GT, illumination and ten 10/90 splits.
if nargin < 2, outputRoot = ''; end
cfg = pilot_dataset_config(sampleId, outputRoot);
assert(isfile(cfg.psf_path), 'pilot:MissingPSF', 'Missing configured PSF.');
if isfolder(cfg.sample_dir)
    existing = fullfile(cfg.sample_dir, 'prepared.mat');
    assert(isfile(existing), 'pilot:Ownership', ...
        'Sample directory exists without prepared.mat; refusing to mix outputs.');
    old = load(existing, 'cfg');
    assert(isfield(old, 'cfg') && strcmp(old.cfg.dataset_id, cfg.dataset_id) && ...
        strcmp(old.cfg.sample_id, cfg.sample_id) && old.cfg.schema_version == cfg.schema_version, ...
        'pilot:Ownership', 'Existing sample belongs to another protocol.');
    summary = load(existing);
    return;
end
mkdir(cfg.sample_dir);
subdirs = {'products','sensor_frames','recon_frames','subsets','previews'};
for k = 1:numel(subdirs), mkdir(fullfile(cfg.sample_dir, subdirs{k})); end
[object, object_meta] = pilot_make_object(cfg.sample_id, cfg.code_root);
assert(object_meta.depth_um == cfg.truth_depth_um, 'pilot:ObjectDepth', ...
    'Object metadata and frozen config disagree.');
gt = zeros(cfg.image_size(1), cfg.image_size(2), numel(cfg.z_um), 'single');
truthIndex = find(cfg.z_um == cfg.truth_depth_um);
assert(isscalar(truthIndex), 'pilot:ObjectDepth', 'Truth depth must be in z_um.');
gt(:,:,truthIndex) = single(object);

oldRng = rng;
restoreRng = onCleanup(@() rng(oldRng)); %#ok<NASGU>
rng(cfg.subset_seed, 'twister');
permutation = randperm(cfg.frame_count);
input_indices = reshape(permutation, cfg.input_frames, cfg.subset_count).';
holdout_indices = zeros(cfg.subset_count, cfg.holdout_frames);
for s = 1:cfg.subset_count
    holdout_indices(s,:) = setdiff(1:cfg.frame_count, input_indices(s,:), 'stable');
    assert(isempty(intersect(input_indices(s,:), holdout_indices(s,:))) && ...
        isequal(sort([input_indices(s,:), holdout_indices(s,:)]), 1:cfg.frame_count), ...
        'pilot:Split', 'Invalid 10/90 split.');
end

illumination_meta = struct('source', '', 'absolute_scale_recoverable', false);
if strcmp(cfg.sample_id, 'P01')
    for frame = 1:cfg.frame_count
        source = fullfile(cfg.product_path, sprintf('img_%d.png', frame));
        assert(isfile(source), 'pilot:MissingProduct', 'Missing supplied product %s', source);
    end
    illumination_meta.source = 'supplied 8-bit object-times-speckle products';
    illumination_meta.absolute_scale_recoverable = false;
else
    [rawIntensity, normalizedIntensity, illumination_meta] = ...
        pilot_generate_illumination(cfg.illumination_seed, cfg.frame_count);
    illumination_meta.absolute_scale_recoverable = true;
    illumination_meta.raw_storage_class = 'single';
    illumination_meta.normalized_storage_class = 'uint8 after original per-frame max';
    illumination = single(rawIntensity); %#ok<NASGU>
    illumination_legacy_uint8 = im2uint8(normalizedIntensity); %#ok<NASGU>
    save(fullfile(cfg.sample_dir, 'illumination.mat'), 'illumination', ...
        'illumination_legacy_uint8', 'illumination_meta', '-v7.3');
end

prepared = struct('cfg', cfg, 'object', object, 'object_meta', object_meta, ...
    'ground_truth', gt, 'truth_index_one_based', truthIndex, ...
    'input_indices', input_indices, 'holdout_indices', holdout_indices, ...
    'illumination_meta', illumination_meta, ...
    'protocol_changes_from_legacy_main', {{ ...
    'same ten frames provide mean; no extra uniform exposure', ...
    'unknown-depth reconstruction uses every 10--100 um layer', ...
    'ten deterministic disjoint 10-frame subsets with 90-frame holdouts', ...
    'both pre-detector-normalization and legacy per-frame-normalized sensors are retained'}});
pilot_atomic_save(fullfile(cfg.sample_dir, 'prepared.mat'), prepared);
pilot_write_tiff(fullfile(cfg.sample_dir, 'previews', 'ground_truth_float.tif'), gt);
imwrite(im2uint8(object), fullfile(cfg.sample_dir, 'previews', 'object_uint8.tif'), ...
    'tif', 'Compression', 'none');
summary = prepared;
end
