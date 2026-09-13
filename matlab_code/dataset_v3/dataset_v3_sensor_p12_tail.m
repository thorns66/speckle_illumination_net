function dataset_v3_sensor_p12_tail(outputRoot)
%DATASET_V3_SENSOR_P12_TAIL Build disjoint tail sensor frames on CPU.
assert(isempty(getenv('CUDA_VISIBLE_DEVICES')),'v3p12:CPUOnly');
approval=jsondecode(fileread(fullfile(outputRoot,'SIMULATION_APPROVED.json')));
assert(approval.approved&&any(strcmp(cellstr(string(approval.approved_new_sample_ids)),'P12')), ...
    'v3p12:Approval');
cfg=dataset_v3_simulation_config('P12',outputRoot);
assert(strcmp(cfg.geometry_version,'root_native_pixel_v4')&&cfg.cell_count==18, ...
    'v3p12:TruthRevision','Only the approved 18-cell native-pixel P12 may run.');
prepared=load(fullfile(cfg.sample_dir,'prepared.mat'));
assert(isequaln(prepared.cfg,cfg),'v3p12:PreparedStale');
assert(isfile(fullfile(cfg.sample_dir,'illumination_3d.mat')),'v3p12:IlluminationMissing');
psf=cell_load_forward_psf(cfg);
progressPath=fullfile(outputRoot,'progress','P12_sensor_tail.json');
frames=100:-1:56;
for k=1:numel(frames)
    frame=frames(k);
    path=fullfile(cfg.sample_dir,'sensor_frames',sprintf('frame_%03d.mat',frame));
    if ~isfile(path), cell_forward_frame(cfg,prepared,psf,frame); end
    report(progressPath,k,numel(frames),frame);
end
cell_write_json(fullfile(outputRoot,'progress','P12_sensor_tail_complete.json'), ...
    struct('complete',true,'sample_id','P12','device','CPU', ...
    'frame_indices',frames));
end

function report(path,completed,total,index)
record=struct('sample_id','P12','stage','sensor_tail','device','CPU', ...
    'completed',completed,'total',total,'last_index',index, ...
    'updated_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ssXXX')));
cell_write_json(path,record);
fprintf('P12 | sensor_tail | frame %d | %d/%d | %s\n', ...
    index,completed,total,datestr(now,30));
end
