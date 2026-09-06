function report = cell_run_full_sample(sampleId,outputRoot,gpuIndex)
%CELL_RUN_FULL_SAMPLE Resume-safe end-to-end pipeline for one approved GT.
cfg=cell_simulation_config(sampleId,outputRoot);
logPath=fullfile(cfg.sample_dir,'logs',sprintf('full_run_gpu_%d.log',gpuIndex));
if ~isfolder(fullfile(cfg.sample_dir,'logs')), mkdir(fullfile(cfg.sample_dir,'logs')); end
diary(logPath); cleanup=onCleanup(@() diary('off')); %#ok<NASGU>
started=tic; fprintf('\n%s | GPU %d | START %s\n',cfg.sample_id,gpuIndex,datestr(now,30));
if cell_stage_complete(cfg,'validated')&&isfile(fullfile(cfg.sample_dir,'full_run_report.mat'))
    old=load(fullfile(cfg.sample_dir,'full_run_report.mat'),'report'); report=old.report;
    officialSplit=cell_dataset_split(cfg.sample_id);
    if ~strcmp(report.split,officialSplit)
        report.split=officialSplit;
        pilot_atomic_save(fullfile(cfg.sample_dir,'full_run_report.mat'),struct('report',report));
        cell_write_json(fullfile(cfg.sample_dir,'full_run_report.json'),report);
    end
    fprintf('%s | already deeply validated; reuse complete sample\n',cfg.sample_id);
    return;
end
prepared=cell_prepare_simulation_sample(cfg);

if ~cell_stage_complete(cfg,'sensor')
    fprintf('%s | load ten-depth CPU H and simulate missing sensor frames\n',cfg.sample_id);
    forwardPsf=cell_load_forward_psf(cfg);
    for frame=1:cfg.frame_count
        cell_forward_frame(cfg,prepared,forwardPsf,frame);
        if mod(frame,10)==0, fprintf('%s | sensor %d/%d\n',cfg.sample_id,frame,cfg.frame_count); end
    end
    clear forwardPsf;
end
assert(cell_stage_complete(cfg,'sensor'),'cells:SensorIncomplete');

needsGpu=~cell_stage_complete(cfg,'recon')||~cell_stage_complete(cfg,'subset');
if needsGpu
    fprintf('%s | load resident ten-depth H/Ht on GPU %d\n',cfg.sample_id,gpuIndex);
    psf=pilot_load_psf_gpu(cfg);
else
    psf=[];
end
if ~cell_stage_complete(cfg,'recon')
    for frame=1:cfg.frame_count
        pilot_reconstruct_frame(cfg,psf,frame);
        if mod(frame,10)==0, fprintf('%s | per-frame RL %d/%d\n',cfg.sample_id,frame,cfg.frame_count); end
    end
end
assert(cell_stage_complete(cfg,'recon'),'cells:ReconIncomplete');
if ~cell_stage_complete(cfg,'algorim')
    fprintf('%s | export twenty 100-page Windows AlgoRIM input stacks\n',cfg.sample_id);
    pilot_export_algorim(cfg.sample_dir,cfg.frame_count,cfg.z_um);
end
if ~cell_stage_complete(cfg,'subset')
    for subset=1:cfg.subset_count
        pilot_build_subset(cfg,psf,subset);
        fprintf('%s | subset %d/%d\n',cfg.sample_id,subset,cfg.subset_count);
    end
end
if ~isempty(psf), clear psf; end
assert(cell_stage_complete(cfg,'subset')&&cell_stage_complete(cfg,'algorim'),'cells:FinalStageIncomplete');
fprintf('%s | deep numerical validation and SHA-256 manifest\n',cfg.sample_id);
validation=pilot_validate_sample(cfg);
report=struct('sample_id',cfg.sample_id,'split',cell_dataset_split(cfg.sample_id),'gpu_index',gpuIndex, ...
    'elapsed_seconds',toc(started),'complete',true,'artifact_count',validation.artifact_count, ...
    'validation_manifest',fullfile(cfg.sample_dir,'validation_manifest.json'), ...
    'finished_at',datestr(now,30));
pilot_atomic_save(fullfile(cfg.sample_dir,'full_run_report.mat'),struct('report',report));
cell_write_json(fullfile(cfg.sample_dir,'full_run_report.json'),report);
fprintf('%s | COMPLETE in %.2f h\n',cfg.sample_id,report.elapsed_seconds/3600);
end
