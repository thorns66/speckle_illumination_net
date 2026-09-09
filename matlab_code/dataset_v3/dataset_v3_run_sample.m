function dataset_v3_run_sample(sampleId,outputRoot,stage)
approval=jsondecode(fileread(fullfile(outputRoot,'SIMULATION_APPROVED.json')));
assert(approval.approved&&approval.allowed_physical_gpu_indices==0&&approval.allow_sharing&& ...
    ~approval.allow_fallback_gpu,'v3:Approval');
assert(any(strcmp(cellstr(string(approval.approved_new_sample_ids)),sampleId)),'v3:Approval');
cfg=dataset_v3_simulation_config(sampleId,outputRoot);
truthRecord=approval.truth_sources.(sampleId);
assert(strcmp(cfg.truth_source_sha256,truthRecord.geometry_source_sha256),'v3:ApprovalHash');
progress(cfg,stage,0,0);
switch stage
    case 'sensor'
        assert(isempty(getenv('CUDA_VISIBLE_DEVICES')),'v3:CPUOnly');
        prepared=dataset_v3_prepare_sample(cfg);
        psf=cell_load_forward_psf(cfg);
        for frame=1:cfg.frame_count
            cell_forward_frame(cfg,prepared,psf,frame);
            progress(cfg,stage,frame,cfg.frame_count);
        end
        assert(cell_stage_complete(cfg,'sensor'),'v3:SensorIncomplete');
    case 'reconstruct'
        assert(strcmp(getenv('CUDA_VISIBLE_DEVICES'),approval.gpu_uuid),'v3:GPUUUID');
        assert(gpuDeviceCount('available')==1,'v3:GPUVisibility');
        selected=gpuDevice(1);
        assert(contains(selected.Name,'A40')&&selected.AvailableMemory>=20*1024^3,'v3:GPUMemory');
        hardware=struct('physical_index',0,'logical_index',selected.Index,'uuid',approval.gpu_uuid, ...
            'name',selected.Name,'available_bytes_before',selected.AvailableMemory, ...
            'total_bytes',selected.TotalMemory,'sharing_authorized',true);
        cell_write_json(fullfile(outputRoot,'progress',[sampleId '_gpu.json']),hardware);
        assert(cell_stage_complete(cfg,'sensor'),'v3:SensorIncomplete');
        psf=pilot_load_psf_gpu(cfg);
        for frame=1:cfg.frame_count
            pilot_reconstruct_frame(cfg,psf,frame);
            progress(cfg,'per_frame_rl3',frame,cfg.frame_count);
        end
        for subset=1:cfg.subset_count
            pilot_build_subset(cfg,psf,subset);
            progress(cfg,'subset_rl3',subset,cfg.subset_count);
        end
        clear psf selected;
        gpuDevice([]); % Release this MATLAB context only; never touch another process.
    case 'validate'
        assert(isempty(getenv('CUDA_VISIBLE_DEVICES')),'v3:CPUOnly');
        assert(cell_stage_complete(cfg,'recon')&&cell_stage_complete(cfg,'subset'),'v3:ReconIncomplete');
        pilot_export_algorim(cfg.sample_dir,cfg.frame_count,cfg.z_um);
        validation=pilot_validate_sample(cfg);
        assert(validation.complete,'v3:Validation');
    otherwise
        error('v3:Stage','Unknown stage %s.',stage);
end
progress(cfg,[stage '_complete'],1,1);
end

function progress(cfg,stage,completed,total)
record=struct('sample_id',cfg.sample_id,'stage',stage,'completed',completed,'total',total, ...
    'updated_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ssXXX')));
cell_write_json(fullfile(cfg.output_root,'progress',[cfg.sample_id '.json']),record);
fprintf('%s | %s | %d/%d | %s\n',cfg.sample_id,stage,completed,total,datestr(now,30));
end
