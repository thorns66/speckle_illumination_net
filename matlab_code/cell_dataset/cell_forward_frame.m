function record = cell_forward_frame(cfg,prepared,psf,frame)
%CELL_FORWARD_FRAME Simulate one continuous-volume sensor realization.
validateattributes(frame,{'numeric'},{'scalar','integer','>=',1,'<=',cfg.frame_count});
outputPath=fullfile(cfg.sample_dir,'sensor_frames',sprintf('frame_%03d.mat',frame));
if isfile(outputPath)
    record=load(outputPath);
    assert(record.schema_version==cfg.schema_version&&strcmp(record.dataset_id,cfg.dataset_id)&& ...
        strcmp(record.sample_id,cfg.sample_id)&&record.frame_index==frame,'cells:StaleFrame');
    return;
end
m=matfile(fullfile(cfg.sample_dir,'illumination_3d.mat'));
illuminationRaw=single(m.illumination_raw(:,:,:,frame));
illuminationPeak=max(illuminationRaw,[],'all');
assert(isfinite(illuminationPeak)&&illuminationPeak>0,'cells:Illumination','Invalid frame.');
groundTruth=single(prepared.ground_truth);
productPreDetector=groundTruth.*illuminationRaw;
legacyNormalized=groundTruth.*(illuminationRaw/illuminationPeak);
legacyUint8=uint8(round(255*min(max(legacyNormalized,0),1)));
productPeak=max(legacyUint8,[],'all');
assert(productPeak>0,'cells:EmptyProduct','Quantized 3D product is empty.');
productLegacy=single(legacyUint8)/single(productPeak);
t0=tic; sensorPreDetector=cell_forward_project_acc(psf.H,productPreDetector,psf.CAindex);
preSeconds=toc(t0);
t1=tic; sensorLegacyBeforeMax=cell_forward_project_acc(psf.H,productLegacy,psf.CAindex);
legacySeconds=toc(t1);
sensorMax=max(sensorLegacyBeforeMax,[],'all');
assert(isfinite(sensorMax)&&sensorMax>0,'cells:EmptySensor','Forward result is empty.');
sensorLegacyFloat=double(sensorLegacyBeforeMax)/double(sensorMax);
sensorLegacyUint8=im2uint8(sensorLegacyFloat);
imwrite(sensorLegacyFloat,fullfile(cfg.sample_dir,'sensor_frames', ...
    sprintf('frame_%03d_legacy.tif',frame)),'tif','Compression','none');
record=struct('schema_version',cfg.schema_version,'dataset_id',cfg.dataset_id, ...
    'sample_id',cfg.sample_id,'frame_index',double(frame),'z_um',double(cfg.z_um), ...
    'occupied_truth_indices_one_based',prepared.occupied_truth_indices_one_based, ...
    'sensor_pre_detector',sensorPreDetector,'sensor_legacy_before_max',sensorLegacyBeforeMax, ...
    'sensor_legacy_float',sensorLegacyFloat,'sensor_legacy_uint8',sensorLegacyUint8, ...
    'illumination_peak_raw_yxz',double(illuminationPeak), ...
    'object_product_peak_legacy_uint8',double(productPeak), ...
    'sensor_legacy_max',double(sensorMax),'pre_detector_forward_seconds',preSeconds, ...
    'legacy_forward_seconds',legacySeconds,'forward_operator','sum_z forwardProjectACC(H_z,g_z .* I_z)', ...
    'illumination_depth_coupling','one propagated complex field shared by all z', ...
    'legacy_product_policy','one global YXZ illumination scale, uint8, then divide by product maximum');
pilot_atomic_save(outputPath,record);
end
