function dataset_v3_run_shard(planPath,slot)
% Independent complete-frame/subset jobs: no cross-worker reductions/writes.
plan=jsondecode(fileread(planPath));
assert(plan.version==1&&plan.authorized&&isequal(double(plan.allowed_physical_gpu_indices(:)'),0:5), ...
    'v3parallel:Authorization');
assert(ismember(slot,0:5),'v3parallel:Slot');
worker=plan.workers(slot+1); assert(worker.physical_index==slot,'v3parallel:Slot');
root=plan.dataset_root; control=plan.control_root;
assert(ismember(plan.sample_id,{'T04','V03'}),'v3parallel:Sample');
approval=jsondecode(fileread(fullfile(root,'SIMULATION_APPROVED.json')));
cfg=dataset_v3_simulation_config(plan.sample_id,root);
assert(approval.approved&&strcmp(cfg.truth_source_sha256, ...
    approval.truth_sources.(plan.sample_id).geometry_source_sha256),'v3parallel:Truth');
jobs=worker.jobs;
if isempty(jobs), return; end
if strcmp(plan.stage,'sensor')
    assert(isempty(getenv('CUDA_VISIBLE_DEVICES')),'v3parallel:CPU');
    prepared=load(fullfile(cfg.sample_dir,'prepared.mat'));
    assert(isequaln(prepared.cfg,cfg),'v3parallel:Prepared');
    psf=cell_load_forward_psf(cfg);
elseif strcmp(plan.stage,'reconstruct')
    assert(strcmp(getenv('CUDA_VISIBLE_DEVICES'),worker.gpu_uuid),'v3parallel:UUID');
    assert(gpuDeviceCount('available')==1,'v3parallel:GPUCount');
    selected=gpuDevice(1);
    assert(contains(selected.Name,'A40')&&selected.AvailableMemory>=20*1024^3,'v3parallel:Memory');
    hardware=struct('physical_index',slot,'uuid',worker.gpu_uuid,'logical_index',selected.Index, ...
        'name',selected.Name,'available_bytes_before',selected.AvailableMemory);
    cell_write_json(fullfile(control,'progress',sprintf('%s_gpu%d_hardware.json',plan.sample_id,slot)),hardware);
    psf=pilot_load_psf_gpu(cfg);
    % One retained-scale reference reconstruction checks each device without
    % publishing or overwriting a dataset frame; sample/output semantics fixed.
    parityPath=fullfile(control,'progress',sprintf('gpu%d_parity.json',slot));
    if ~isfile(parityPath)
        ref=load(fullfile(root,'T04','recon_frames','frame_001.mat'),'physics_input_single','w_physics_raw');
        actual=pilot_reconstruct_volume(psf,ref.physics_input_single,cfg.iterations,'mean');
        relative=norm(double(actual(:))-double(ref.w_physics_raw(:)))/max(norm(double(ref.w_physics_raw(:))),eps);
        assert(relative<=1e-5,'v3parallel:Parity','GPU %d reference mismatch: %.9g',slot,relative);
        cell_write_json(parityPath,struct('passed',true,'physical_index',slot,'relative_l2',relative, ...
            'reference','T04/recon_frames/frame_001.mat: w_physics_raw','iterations',3));
    end
else
    error('v3parallel:Stage','Unsupported stage.');
end
progressPath=fullfile(control,'progress',sprintf('%s_%s_slot%d.json',plan.sample_id,plan.stage,slot));
for k=1:numel(jobs)
    job=jobs(k); assert(job.index>=1&&job.index==floor(job.index),'v3parallel:Index');
    switch job.kind
        case 'sensor'
            assert(strcmp(plan.stage,'sensor')&&job.index<=100,'v3parallel:Job');
            target=fullfile(cfg.sample_dir,'sensor_frames',sprintf('frame_%03d.mat',job.index));
        case 'frame'
            assert(strcmp(plan.stage,'reconstruct')&&job.index<=100,'v3parallel:Job');
            target=fullfile(cfg.sample_dir,'recon_frames',sprintf('frame_%03d.mat',job.index));
        case 'subset'
            assert(strcmp(plan.stage,'reconstruct')&&job.index<=10,'v3parallel:Job');
            target=fullfile(cfg.sample_dir,'subsets',sprintf('subset_%02d.mat',job.index));
        otherwise
            error('v3parallel:Job','Unknown job kind.');
    end
    assert(~isfile(target),'v3parallel:Overwrite','Job output already exists: %s',target);
    switch job.kind
        case 'sensor', cell_forward_frame(cfg,prepared,psf,job.index);
        case 'frame', pilot_reconstruct_frame(cfg,psf,job.index);
        case 'subset', pilot_build_subset(cfg,psf,job.index);
    end
    cell_write_json(progressPath,struct('sample_id',plan.sample_id,'stage',plan.stage, ...
        'slot',slot,'completed',k,'total',numel(jobs),'last_kind',job.kind,'last_index',job.index, ...
        'updated_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ssXXX'))));
    fprintf('%s | %s | slot %d | %d/%d | %s %d\n',plan.sample_id,plan.stage,slot,k,numel(jobs),job.kind,job.index);
end
if strcmp(plan.stage,'reconstruct'), clear psf selected; gpuDevice([]); end
end
