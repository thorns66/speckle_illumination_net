function report = run_dataset_v3_preview(outputRoot)
%RUN_DATASET_V3_PREVIEW Strictly CPU truth-only; no automatic next stage.
assert(isempty(getenv('CUDA_VISIBLE_DEVICES')),'v3:GPU','Preview requires CUDA_VISIBLE_DEVICES empty.');
assert(isfolder(outputRoot) && isfile(fullfile(outputRoot,'preview_contract.json')), ...
    'v3:Ownership','Use the dated Python launcher to reserve an owned output directory.');
contract = jsondecode(fileread(fullfile(outputRoot,'preview_contract.json')));
assert(strcmp(contract.stage,'truth_preview_only') && ~contract.morphology_approved && ...
    isequal(contract.allowed_physical_gpu_indices,0) && strcmp(contract.required_gpu_model,'A40'), ...
    'v3:Contract','Invalid preview contract.');
ids = {'T03','T04','V03'};
for k=1:3
    assert(~isfolder(fullfile(outputRoot,ids{k})),'v3:Overwrite','Refusing to overwrite existing sample.');
end
report = struct('stage','truth_preview_only','preview_complete',false, ...
    'dataset_complete',false,'morphology_approved',false,'forward_started',false, ...
    'rl_started',false,'training_started',false,'gpu_used',false,'samples',{{}});
for k=1:3
    cfg = dataset_v3_config(ids{k},outputRoot);
    fprintf('CPU TRUTH %s | seed %.0f\n',cfg.sample_id,cfg.geometry_seed);
    sample = dataset_v3_make_truth(cfg);
    sample.metrics = dataset_v3_validate_truth(sample);
    sample.source_sha256 = contract.geometry_source_sha256;
    sample.approval_status = 'pending_user_morphology_review';
    before = rng; restore = onCleanup(@() rng(before));
    rng(cfg.subset_seed,'twister');
    sample.input_indices = reshape(randperm(100),10,10)';
    sample.holdout_indices = zeros(10,90);
    for s=1:10, sample.holdout_indices(s,:)=setdiff(1:100,sample.input_indices(s,:),'stable'); end
    clear restore;
    mkdir(cfg.sample_dir); previewDir=fullfile(cfg.sample_dir,'previews'); mkdir(previewDir);
    pilot_atomic_save(fullfile(cfg.sample_dir,'truth.mat'),sample);
    pilot_write_tiff(fullfile(cfg.sample_dir,'ground_truth_float.tif'),sample.ground_truth);
    pilot_write_tiff(fullfile(cfg.sample_dir,'ground_truth_fine_float.tif'),sample.ground_truth_fine);
    cell_write_json(fullfile(cfg.sample_dir,'geometry.json'),sample.meta);
    cell_write_json(fullfile(cfg.sample_dir,'config.json'),cfg);
    cell_write_json(fullfile(cfg.sample_dir,'truth_metrics.json'),sample.metrics);
    cell_plot_truth(sample,previewDir);
    report.samples{end+1}=struct('sample_id',cfg.sample_id,'split',cfg.split, ...
        'truth_path',fullfile(cfg.sample_dir,'truth.mat'),'preview_dir',previewDir, ...
        'metrics',sample.metrics);
    fprintf('CPU TRUTH COMPLETE %s | z mass %s\n',cfg.sample_id,mat2str(sample.metrics.mass_fraction,3));
end
report.preview_complete=true;
cell_write_json(fullfile(outputRoot,'preview_report.json'),report);
fprintf('STOP: human morphology review required. No acquisition, RL or training.\n');
end
