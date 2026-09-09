function report = priority_validation_scene_task(repoRoot, outputRoot, family, depthUm, repeatIndex)
%PRIORITY_VALIDATION_SCENE_TASK Simulate and reconstruct one immutable scene/repeat.
addpath(fullfile(repoRoot,'matlab_code','priority_validation'), ...
    fullfile(repoRoot,'matlab_code','cell_dataset'), ...
    fullfile(repoRoot,'matlab_code','pilot_dataset'), ...
    fullfile(repoRoot,'matlab_code','Util'),fullfile(repoRoot,'matlab_code','Solver'));
family=validatestring(family,{'points','lines','axial_pairs'});
validateattributes(repeatIndex,{'numeric'},{'scalar','integer','>=',1,'<=',3});
if strcmp(family,'axial_pairs'), sceneId='axial_pairs';
else, sceneId=sprintf('%s_z%03d',family,depthUm); end
sampleId=sprintf('%s_r%02d',sceneId,repeatIndex);
sampleDir=fullfile(outputRoot,'generated',sampleId);
completePath=fullfile(sampleDir,'complete.json');
if isfile(completePath)
    report=jsondecode(fileread(completePath));
    assert(report.complete&&strcmp(report.sample_id,sampleId),'priority:StaleScene'); return;
end
for folder={sampleDir,fullfile(sampleDir,'sensor_frames'),fullfile(sampleDir,'subsets'), ...
        fullfile(sampleDir,'baselines'),fullfile(sampleDir,'previews')}
    if ~isfolder(folder{1}), mkdir(folder{1}); end
end

truth=priority_validation_make_truth(family,depthUm);
preparedPath=fullfile(sampleDir,'prepared.mat'); geometryPath=fullfile(sampleDir,'geometry.json');
if ~isfile(preparedPath)
    payload=truth; payload.sample_id=sampleId; payload.scene_id=sceneId; payload.repeat_index=repeatIndex;
    pilot_atomic_save(preparedPath,payload);
    pilot_write_tiff(fullfile(sampleDir,'previews','ground_truth.tif'),truth.ground_truth);
    geometry=rmfield(payload,{'ground_truth','ground_truth_fine','x_um','y_um','fine_z_um'});
    geometry.complete=true; cell_write_json(geometryPath,geometry);
else
    existing=load(preparedPath,'sample_id','scene_id','repeat_index','ground_truth');
    assert(strcmp(existing.sample_id,sampleId)&&strcmp(existing.scene_id,sceneId)&& ...
        existing.repeat_index==repeatIndex&&isequal(existing.ground_truth,truth.ground_truth), ...
        'priority:StaleTruth');
end

illuminationPath=fullfile(outputRoot,'shared_illumination',sprintf('repeat_%02d.mat',repeatIndex));
assert(isfile(illuminationPath),'priority:MissingIllumination'); illumination=matfile(illuminationPath);
cfg=cell_simulation_config('P08',fullfile(repoRoot,'data','matlab_cells_pilot_v2_r04'));
cfg.sample_id=sampleId; cfg.sample_dir=sampleDir; cfg.frame_count=100;
psf=cell_load_forward_psf(cfg);
stack=zeros(260,260,100,'single');
if strcmp(family,'axial_pairs')
    holdoutLayerSum=zeros(260,260,10,'double'); holdoutLayerSumSquares=holdoutLayerSum;
end
for frame=1:100
    framePath=fullfile(sampleDir,'sensor_frames',sprintf('frame_%03d.mat',frame));
    if isfile(framePath)
        saved=load(framePath,'sample_id','frame_index','sensor_pre_detector');
        assert(strcmp(saved.sample_id,sampleId)&&saved.frame_index==frame,'priority:StaleFrame');
        sensor=saved.sensor_pre_detector;
        % Axial layer moments must be rebuilt deterministically even on resume.
        needLayerMoments=strcmp(family,'axial_pairs')&&frame>=11;
    else
        needLayerMoments=strcmp(family,'axial_pairs')&&frame>=11;
        raw=single(illumination.illumination_raw(:,:,:,frame));
        [sensor,layerSensors]=project_layers(psf,truth.ground_truth.*raw,needLayerMoments);
        payload=struct('schema_version',1,'sample_id',sampleId,'scene_id',sceneId, ...
            'repeat_index',repeatIndex,'frame_index',frame,'z_um',truth.z_um, ...
            'sensor_pre_detector',sensor,'illumination_source',illuminationPath, ...
            'forward_operator','sum_z forwardProjectACC(H_z,truth_z*illumination_z)');
        pilot_atomic_save(framePath,payload);
    end
    validateattributes(sensor,{'single'},{'size',[260,260],'finite','nonnegative'});
    stack(:,:,frame)=sensor;
    if needLayerMoments
        if ~exist('layerSensors','var')
            raw=single(illumination.illumination_raw(:,:,:,frame));
            [~,layerSensors]=project_layers(psf,truth.ground_truth.*raw,true);
        end
        holdoutLayerSum=holdoutLayerSum+double(layerSensors);
        holdoutLayerSumSquares=holdoutLayerSumSquares+double(layerSensors).^2;
        clear layerSensors raw;
    end
end
clear psf illumination;

inputMean=mean(double(stack(:,:,1:10)),3); inputVariance=var(double(stack(:,:,1:10)),0,3);
holdoutMean=mean(double(stack(:,:,11:100)),3); holdoutVariance=var(double(stack(:,:,11:100)),0,3);
cfg=cell_simulation_config('P08',fullfile(repoRoot,'data','matlab_cells_pilot_v2_r04'));
device=gpuDevice(1); fprintf('%s | reconstruct on %s\n',sampleId,device.Name);
psf=pilot_load_psf_gpu(cfg);
[mean3,diagMean3]=pilot_reconstruct_volume(psf,single(inputMean),3,'mean');
[taylor3,diagTaylor3]=pilot_reconstruct_volume(psf,single(inputVariance),3,'taylor');
[mean5,diagMean5]=pilot_reconstruct_volume(psf,single(inputMean),5,'mean');
[taylor5,diagTaylor5]=pilot_reconstruct_volume(psf,single(inputVariance),5,'taylor');
taylorSqrt3=sqrt(taylor3); taylorSqrt5=sqrt(taylor5);
subsetPath=fullfile(sampleDir,'subsets','subset_01.mat');
if ~isfile(subsetPath)
    payload=struct('schema_version',1,'sample_id',sampleId,'subset_index',1,'iterations',3, ...
        'input_indices',1:10,'holdout_indices',11:100,'z_um',truth.z_um, ...
        'input_physics_mean_float',single(inputMean), ...
        'input_physics_variance_nminus1_float',single(inputVariance), ...
        'holdout_physics_mean_float',single(holdoutMean), ...
        'holdout_physics_variance_nminus1_float',single(holdoutVariance), ...
        'physics_mean_raw',mean3,'physics_taylor_raw',taylor3, ...
        'physics_taylor_sqrt_float',taylorSqrt3,'frame_policy','first 10 input; last 90 holdout');
    if strcmp(family,'axial_pairs')
        layerMean=holdoutLayerSum/90;
        layerVariance=(holdoutLayerSumSquares-90*layerMean.^2)/89;
        payload.holdout_layer_sensor_mean=single(layerMean);
        payload.holdout_layer_sensor_variance_nminus1=single(max(layerVariance,0));
    end
    pilot_atomic_save(subsetPath,payload);
end
save_baseline('mean_rl3',mean3,diagMean3); save_baseline('taylor_rl3',taylorSqrt3,diagTaylor3);
save_baseline('mean_rl5',mean5,diagMean5); save_baseline('taylor_rl5',taylorSqrt5,diagTaylor5);
report=struct('complete',true,'sample_id',sampleId,'scene_id',sceneId,'family',family, ...
    'depth_um',depthUm,'repeat_index',repeatIndex,'input_indices',1:10, ...
    'holdout_indices',11:100,'subset_mat',subsetPath,'prepared_mat',preparedPath, ...
    'checkpoint_input_iterations',3,'baseline_iterations',[3,5]);
cell_write_json(completePath,report);

    function [sensor,layers]=project_layers(localPsf,product,keepLayers)
        sensor=zeros(260,260,'single');
        if keepLayers, layers=zeros(260,260,10,'single'); else, layers=[]; end
        for layer=1:10
            if any(product(:,:,layer)>0,'all')
                contribution=single(forwardProjectACC(localPsf.H(:,:,:,:,layer), ...
                    product(:,:,layer),localPsf.CAindex(layer,:)));
                sensor=sensor+contribution;
                if keepLayers, layers(:,:,layer)=contribution; end
            end
        end
    end

    function save_baseline(name,volume,diagnostic)
        folder=fullfile(sampleDir,'baselines',name); if ~isfolder(folder), mkdir(folder); end
        matPath=fullfile(folder,'reconstruction.mat'); tifPath=fullfile(folder,'reconstruction.tif');
        if ~isfile(matPath)
            payload=struct('sample_id',sampleId,'method',name,'reconstruction',volume, ...
                'diagnostic',diagnostic,'input_indices',1:10,'z_um',truth.z_um, ...
                'gain_policy','retained scale; no normalization');
            pilot_atomic_save(matPath,payload); pilot_write_tiff(tifPath,volume);
        end
    end
end
