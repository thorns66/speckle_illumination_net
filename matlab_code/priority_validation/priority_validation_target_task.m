function report = priority_validation_target_task(repoRoot, outputRoot, sampleId, repeatIndex)
%PRIORITY_VALIDATION_TARGET_TASK Independent 90-frame targets for loss scoring.
addpath(fullfile(repoRoot,'matlab_code','cell_dataset'), ...
    fullfile(repoRoot,'matlab_code','pilot_dataset'));
assert(ismember(sampleId,{'P08','P10','P09'}),'priority:TargetSample');
validateattributes(repeatIndex,{'numeric'},{'scalar','integer','>=',1,'<=',3});
folder=fullfile(outputRoot,'loss_targets',sampleId,sprintf('repeat_%02d',repeatIndex));
if ~isfolder(folder), mkdir(folder); end
outputPath=fullfile(folder,'holdout_statistics.mat'); jsonPath=fullfile(folder,'complete.json');
if isfile(outputPath)&&isfile(jsonPath)
    report=jsondecode(fileread(jsonPath)); assert(report.complete); return;
end
assert(~isfile(outputPath)&&~isfile(jsonPath),'priority:PartialTarget');
dataRoot=fullfile(repoRoot,'data','matlab_cells_pilot_v2_r04');
cfg=cell_simulation_config(sampleId,dataRoot); prepared=load(fullfile(dataRoot,sampleId,'prepared.mat'),'ground_truth');
illuminationPath=fullfile(outputRoot,'shared_illumination',sprintf('repeat_%02d.mat',repeatIndex));
illumination=matfile(illuminationPath); psf=cell_load_forward_psf(cfg);
input=zeros(260,260,10,'double'); holdout=zeros(260,260,90,'double');
for frame=1:100
    raw=single(illumination.illumination_raw(:,:,:,frame));
    sensor=cell_forward_project_acc(psf.H,single(prepared.ground_truth).*raw,psf.CAindex);
    if frame<=10, input(:,:,frame)=double(sensor); else, holdout(:,:,frame-10)=double(sensor); end
end
payload=struct('schema_version',1,'sample_id',sampleId,'repeat_index',repeatIndex, ...
    'input_indices',1:10,'holdout_indices',11:100,'z_um',cfg.z_um, ...
    'input_mean',single(mean(input,3)),'input_variance_nminus1',single(var(input,0,3)), ...
    'holdout_mean',single(mean(holdout,3)), ...
    'holdout_variance_nminus1',single(var(holdout,0,3)), ...
    'illumination_source',illuminationPath,'noise_model','none');
pilot_atomic_save(outputPath,payload);
report=struct('complete',true,'sample_id',sampleId,'repeat_index',repeatIndex, ...
    'holdout_frames',90,'output_mat',outputPath); cell_write_json(jsonPath,report);
end
