function record = pilot_forward_frame(cfg, prepared, psf, frame)
%PILOT_FORWARD_FRAME Exact ACC simulation with retained and legacy scales.
validateattributes(frame, {'numeric'}, {'scalar','integer','>=',1,'<=',cfg.frame_count});
outputPath = fullfile(cfg.sample_dir, 'sensor_frames', sprintf('frame_%03d.mat', frame));
if isfile(outputPath)
    record = load(outputPath);
    assert(strcmp(record.sample_id, cfg.sample_id) && record.frame_index == frame && ...
        record.schema_version == cfg.schema_version, 'pilot:StaleFrame', ...
        'Existing frame does not match the frozen protocol.');
    return;
end
object = double(prepared.object);
if strcmp(cfg.sample_id, 'P01')
    sourcePath = fullfile(cfg.product_path, sprintf('img_%d.png', frame));
    productUint8 = imread(sourcePath);
    assert(isa(productUint8, 'uint8') && isequal(size(productUint8), cfg.image_size), ...
        'pilot:ProductEncoding', 'P01 source must be 260-by-260 uint8.');
    productLegacy = double(productUint8);
    productLegacy = productLegacy / max(productLegacy(:));
    productPreDetector = productLegacy;
    illuminationPeak = NaN;
    productPeak = double(max(productUint8(:)));
else
    m = matfile(fullfile(cfg.sample_dir, 'illumination.mat'));
    illuminationRaw = double(m.illumination(:,:,frame));
    illuminationLegacy = double(m.illumination_legacy_uint8(:,:,frame)) / 255;
    productPreDetector = object .* illuminationRaw;
    productLegacy = object .* illuminationLegacy;
    % Mirror main.m: read an encoded object-times-speckle frame, cast to
    % double, then divide this input by its own maximum.
    productUint8 = im2uint8(productLegacy);
    imwrite(productUint8, fullfile(cfg.sample_dir, 'products', ...
        sprintf('object_speckle_%03d.png', frame)));
    productLegacy = double(productUint8);
    productPeak = max(productLegacy(:));
    assert(productPeak > 0, 'pilot:EmptyProduct', 'Encoded product is empty.');
    productLegacy = productLegacy / productPeak;
    illuminationPeak = max(illuminationRaw(:));
end
truthIndex = prepared.truth_index_one_based;
if isfield(psf, 'truth_only') && psf.truth_only
    assert(psf.truth_index_one_based == truthIndex, 'pilot:PSFDepth', ...
        'Truth-only PSF does not match the prepared sample depth.');
    Htruth = psf.H;
    CAtruth = psf.CAindex;
else
    Htruth = psf.H(:,:,:,:,truthIndex);
    CAtruth = psf.CAindex(truthIndex,:);
end
t0 = tic;
sensorPreDetector = single(forwardProjectACC(Htruth, productPreDetector, CAtruth));
preSeconds = toc(t0);
if isequal(productPreDetector, productLegacy)
    sensorLegacyBeforeMax = sensorPreDetector;
    legacySeconds = 0;
else
    t1 = tic;
    sensorLegacyBeforeMax = single(forwardProjectACC(Htruth, productLegacy, CAtruth));
    legacySeconds = toc(t1);
end
sensorMax = max(sensorLegacyBeforeMax(:));
assert(isfinite(sensorMax) && sensorMax > 0, 'pilot:EmptySensor', 'Sensor frame is empty.');
% main.m uses double(single ACC output) / its maximum and computes stats
% before TIFF encoding. Keep double here for exact subset mean/variance.
sensorLegacyFloat = double(sensorLegacyBeforeMax) / double(sensorMax);
sensorLegacyUint8 = im2uint8(sensorLegacyFloat);
imwrite(sensorLegacyFloat, fullfile(cfg.sample_dir, 'sensor_frames', ...
    sprintf('frame_%03d_legacy.tif', frame)), 'tif', 'Compression', 'none');
record = struct('schema_version', cfg.schema_version, 'dataset_id', cfg.dataset_id, ...
    'sample_id', cfg.sample_id, 'frame_index', double(frame), ...
    'truth_depth_um', cfg.truth_depth_um, 'truth_index_one_based', truthIndex, ...
    'sensor_pre_detector', sensorPreDetector, ...
    'sensor_legacy_before_max', sensorLegacyBeforeMax, ...
    'sensor_legacy_float', sensorLegacyFloat, ...
    'sensor_legacy_uint8', sensorLegacyUint8, ...
    'illumination_peak_raw', double(illuminationPeak), ...
    'object_product_peak_before_main_normalization', double(productPeak), ...
    'sensor_legacy_max', double(sensorMax), ...
    'pre_detector_forward_seconds', preSeconds, 'legacy_forward_seconds', legacySeconds, ...
    'p01_absolute_scale_recoverable', ~strcmp(cfg.sample_id, 'P01'));
pilot_atomic_save(outputPath, record);
end
