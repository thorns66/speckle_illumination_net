function report = priority_validation_generate_illumination(outputRoot, mode, index)
%PRIORITY_VALIDATION_GENERATE_ILLUMINATION Generate shared repeat or calibration moments.
addpath(fileparts(mfilename('fullpath')), ...
    fullfile(fileparts(fileparts(mfilename('fullpath'))),'cell_dataset'), ...
    fullfile(fileparts(fileparts(mfilename('fullpath'))),'pilot_dataset'));
mode=validatestring(mode,{'repeat','calibration'});
zUm=10:10:100;

if strcmp(mode,'repeat')
    validateattributes(index,{'numeric'},{'scalar','integer','>=',1,'<=',3});
    seeds=[2026090701,2026090702,2026090703]; seed=seeds(index);
    folder=fullfile(outputRoot,'shared_illumination');
    if ~isfolder(folder), mkdir(folder); end
    path=fullfile(folder,sprintf('repeat_%02d.mat',index));
    completePath=fullfile(folder,sprintf('repeat_%02d_complete.json',index));
    if isfile(path)&&isfile(completePath)
        prior=jsondecode(fileread(completePath));
        assert(prior.complete&&prior.seed==seed&&prior.frame_count==100,'priority:StaleIllumination');
        report=prior; return;
    end
    assert(~isfile(path)&&~isfile(completePath),'priority:PartialIllumination');
    [illumination_raw,~,illumination_meta]=cell_generate_illumination_3d(seed,100,zUm);
    payload=struct('schema_version',1,'purpose','priority validation shared illumination', ...
        'repeat_index',index,'seed',seed,'illumination_raw',illumination_raw, ...
        'illumination_meta',illumination_meta);
    pilot_atomic_save(path,payload); clear illumination_raw payload;
    report=struct('complete',true,'mode','repeat','repeat_index',index,'seed',seed, ...
        'frame_count',100,'z_um',zUm,'output_mat',path);
    cell_write_json(completePath,report);
else
    validateattributes(index,{'numeric'},{'scalar','integer','>=',1,'<=',16});
    seed=2026090790+index-1; folder=fullfile(outputRoot,'calibration','batches');
    if ~isfolder(folder), mkdir(folder); end
    path=fullfile(folder,sprintf('batch_%02d.mat',index));
    completePath=fullfile(folder,sprintf('batch_%02d_complete.json',index));
    if isfile(path)&&isfile(completePath)
        prior=jsondecode(fileread(completePath));
        assert(prior.complete&&prior.seed==seed&&prior.frame_count==64,'priority:StaleCalibration');
        report=prior; return;
    end
    assert(~isfile(path)&&~isfile(completePath),'priority:PartialCalibration');
    [raw,~,meta]=cell_generate_illumination_3d(seed,64,zUm);
    batch_sum=sum(double(raw),4); batch_sum_squares=sum(double(raw).^2,4);
    payload=struct('schema_version',1,'seed',seed,'frame_count',64,'z_um',zUm, ...
        'batch_sum',batch_sum,'batch_sum_squares',batch_sum_squares,'generator_meta',meta);
    pilot_atomic_save(path,payload); clear raw batch_sum batch_sum_squares payload;
    report=struct('complete',true,'mode','calibration','batch_index',index, ...
        'seed',seed,'frame_count',64,'output_mat',path);
    cell_write_json(completePath,report);
end
end
