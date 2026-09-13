function dataset_v3_reconstruct_p12_tail(outputRoot)
%DATASET_V3_RECONSTRUCT_P12_TAIL Build disjoint tail outputs on physical GPU 0.
approval=jsondecode(fileread(fullfile(outputRoot,'SIMULATION_APPROVED.json')));
assert(approval.approved&&any(strcmp(cellstr(string(approval.approved_new_sample_ids)),'P12')), ...
    'v3p12:Approval');
assert(strcmp(getenv('CUDA_VISIBLE_DEVICES'),approval.gpu_uuid), ...
    'v3p12:GPUUUID');
assert(gpuDeviceCount('available')==1,'v3p12:GPUVisibility');
selected=gpuDevice(1);
assert(contains(selected.Name,'A40')&&selected.AvailableMemory>=20*1024^3, ...
    'v3p12:GPUMemory');
cfg=dataset_v3_simulation_config('P12',outputRoot);
assert(strcmp(cfg.geometry_version,'root_native_pixel_v4')&&cfg.cell_count==18, ...
    'v3p12:TruthRevision','Only the approved 18-cell native-pixel P12 may run.');
assert(cell_stage_complete(cfg,'sensor'),'v3p12:SensorIncomplete');
psf=pilot_load_psf_gpu(cfg);
progressPath=fullfile(outputRoot,'progress','P12_reconstruct_tail.json');
frames=100:-1:55;
for k=1:numel(frames)
    frame=frames(k);
    path=fullfile(cfg.sample_dir,'recon_frames',sprintf('frame_%03d.mat',frame));
    if ~isfile(path), pilot_reconstruct_frame(cfg,psf,frame); end
    report(progressPath,'frame',k,numel(frames),frame);
end
subsets=10:-1:6;
for k=1:numel(subsets)
    subset=subsets(k);
    path=fullfile(cfg.sample_dir,'subsets',sprintf('subset_%02d.mat',subset));
    if ~isfile(path), pilot_build_subset(cfg,psf,subset); end
    report(progressPath,'subset',k,numel(subsets),subset);
end
clear psf selected;
gpuDevice([]);
cell_write_json(fullfile(outputRoot,'progress','P12_reconstruct_tail_complete.json'), ...
    struct('complete',true,'sample_id','P12','physical_gpu_index',0, ...
    'frame_indices',frames,'subset_indices',subsets));
end

function report(path,kind,completed,total,index)
record=struct('sample_id','P12','stage','reconstruct_tail','kind',kind, ...
    'completed',completed,'total',total,'last_index',index, ...
    'updated_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ssXXX')));
cell_write_json(path,record);
fprintf('P12 | reconstruct_tail | %s %d | %d/%d | %s\n', ...
    kind,index,completed,total,datestr(now,30));
end
