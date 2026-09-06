function complete = cell_stage_complete(cfg,stage)
%CELL_STAGE_COMPLETE Cheap schema-aware resume gate; deep checks run last.
switch stage
    case 'sensor'
        complete=check_files(fullfile(cfg.sample_dir,'sensor_frames'), ...
            'frame_%03d.mat',cfg.frame_count,cfg,{'sensor_pre_detector','sensor_legacy_float'},'frame_index');
        if complete
            for k=1:cfg.frame_count
                if ~isfile(fullfile(cfg.sample_dir,'sensor_frames',sprintf('frame_%03d_legacy.tif',k)))
                    complete=false; return;
                end
            end
        end
    case 'recon'
        complete=check_files(fullfile(cfg.sample_dir,'recon_frames'), ...
            'frame_%03d.mat',cfg.frame_count,cfg,{'w_legacy_raw','w_physics_raw'},'frame_index');
    case 'subset'
        complete=check_files(fullfile(cfg.sample_dir,'subsets'),'subset_%02d.mat', ...
            cfg.subset_count,cfg,{'input_indices','holdout_indices','legacy_mean_raw', ...
            'legacy_taylor_raw','physics_mean_raw','physics_taylor_raw'},'subset_index');
    case 'algorim'
        complete=false; path=fullfile(cfg.sample_dir,'algorim','manifest.mat');
        if ~isfile(path), return; end
        try
            old=load(path,'manifest'); manifest=old.manifest;
            complete=strcmp(manifest.producer,'pilot_export_algorim/1')&& ...
                manifest.frame_count==cfg.frame_count&&isequal(double(manifest.z_um),double(cfg.z_um));
            if ~complete, return; end
            for depth=cfg.z_um
                name=sprintf('depth_%03dum.tif',depth);
                for stream={'legacy_per_frame_normalized','physics_common_scale'}
                    file=fullfile(cfg.sample_dir,'algorim',stream{1},name);
                    if ~isfile(file)||numel(imfinfo(file))~=cfg.frame_count, complete=false; return; end
                end
            end
        catch
            complete=false;
        end
    case 'validated'
        complete=false; path=fullfile(cfg.sample_dir,'validation_manifest.mat');
        if ~isfile(path), return; end
        try
            old=load(path,'manifest'); complete=old.manifest.complete&& ...
                strcmp(old.manifest.dataset_id,cfg.dataset_id)&&strcmp(old.manifest.sample_id,cfg.sample_id)&& ...
                old.manifest.frame_count==cfg.frame_count;
        catch
            complete=false;
        end
    otherwise
        error('cells:Stage','Unknown stage %s.',stage);
end
end

function complete=check_files(folder,pattern,count,cfg,fields,indexField)
complete=isfolder(folder); if ~complete, return; end
for index=1:count
    path=fullfile(folder,sprintf(pattern,index));
    if ~isfile(path), complete=false; return; end
    try
        names={whos('-file',path).name};
        required=[{'schema_version','dataset_id','sample_id',indexField},fields];
        if ~all(ismember(required,names)), complete=false; return; end
        h=load(path,'schema_version','dataset_id','sample_id',indexField);
        if h.schema_version~=cfg.schema_version||~strcmp(h.dataset_id,cfg.dataset_id)|| ...
                ~strcmp(h.sample_id,cfg.sample_id)||h.(indexField)~=index
            complete=false; return;
        end
    catch
        complete=false; return;
    end
end
end
