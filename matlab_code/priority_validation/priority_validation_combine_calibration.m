function report = priority_validation_combine_calibration(outputRoot)
%PRIORITY_VALIDATION_COMBINE_CALIBRATION Combine 16 immutable 64-frame batches.
addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))),'pilot_dataset'), ...
    fullfile(fileparts(fileparts(mfilename('fullpath'))),'cell_dataset'));
folder=fullfile(outputRoot,'calibration'); outputPath=fullfile(folder,'illumination_moments_1024.mat');
jsonPath=fullfile(folder,'illumination_moments_1024.json');
if isfile(outputPath)&&isfile(jsonPath)
    report=jsondecode(fileread(jsonPath)); assert(report.complete&&report.frame_count==1024); return;
end
assert(~isfile(outputPath)&&~isfile(jsonPath),'priority:PartialCalibrationCombination');
total=zeros(260,260,10,'double'); totalSquares=total; frameCount=0; seeds=zeros(1,16);
for batch=1:16
    path=fullfile(folder,'batches',sprintf('batch_%02d.mat',batch));
    assert(isfile(path),'priority:MissingCalibrationBatch','Missing %s',path);
    value=load(path,'seed','frame_count','z_um','batch_sum','batch_sum_squares');
    assert(value.seed==2026090790+batch-1&&value.frame_count==64&&isequal(double(value.z_um),10:10:100));
    total=total+value.batch_sum; totalSquares=totalSquares+value.batch_sum_squares;
    frameCount=frameCount+value.frame_count; seeds(batch)=value.seed;
end
illumination_mean=total/frameCount;
illumination_variance_nminus1=(totalSquares-frameCount*illumination_mean.^2)/(frameCount-1);
illumination_variance_nminus1=max(illumination_variance_nminus1,0);
payload=struct('schema_version',1,'frame_count',frameCount,'batch_count',16, ...
    'frames_per_batch',64,'seeds',seeds,'z_um',10:10:100, ...
    'illumination_mean',single(illumination_mean), ...
    'illumination_variance_nminus1',single(illumination_variance_nminus1), ...
    'array_axis_order','YXZ','normalization','raw generator intensity; no rescaling');
pilot_atomic_save(outputPath,payload);
report=struct('complete',true,'frame_count',frameCount,'batch_count',16, ...
    'seeds',seeds,'output_mat',outputPath, ...
    'mean_min',min(illumination_mean,[],'all'),'mean_max',max(illumination_mean,[],'all'), ...
    'variance_min',min(illumination_variance_nminus1,[],'all'), ...
    'variance_max',max(illumination_variance_nminus1,[],'all'));
cell_write_json(jsonPath,report);
end
