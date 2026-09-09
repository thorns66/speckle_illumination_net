function report = run_t04_axial_preview(outputRoot)
%RUN_T04_AXIAL_PREVIEW Writes only a new, owned T04 truth directory on CPU.
assert(isempty(getenv('CUDA_VISIBLE_DEVICES')),'t04:GPU','All CUDA devices must be hidden.');
path=fullfile(outputRoot,'preview_contract.json');
assert(isfile(path),'t04:Ownership','Use the dated Python launcher.');
contract=jsondecode(fileread(path));
assert(strcmp(contract.stage,'truth_preview_only') && ~contract.morphology_approved && ...
    isequal(contract.allowed_physical_gpu_indices,0) && strcmp(contract.required_gpu_model,'A40') && ...
    strcmp(contract.geometry_revision,'t04_axial_line_pairs_v2'), ...
    't04:Contract','Wrong preview contract.');
cfg=t04_axial_config(outputRoot);
assert(~isfolder(cfg.sample_dir),'t04:Overwrite','Refusing to overwrite an existing T04.');
sample=t04_axial_make_truth(cfg); sample.metrics=t04_axial_validate_truth(sample);
sample.source_sha256=contract.geometry_source_sha256;
sample.approval_status='pending_user_morphology_review';
old=rng; cleanup=onCleanup(@() rng(old)); %#ok<NASGU>
rng(cfg.subset_seed,'twister'); sample.input_indices=reshape(randperm(100),10,10)';
sample.holdout_indices=zeros(10,90);
for k=1:10, sample.holdout_indices(k,:)=setdiff(1:100,sample.input_indices(k,:),'stable'); end
mkdir(cfg.sample_dir); previewDir=fullfile(cfg.sample_dir,'previews'); mkdir(previewDir);
pilot_atomic_save(fullfile(cfg.sample_dir,'truth.mat'),sample);
pilot_write_tiff(fullfile(cfg.sample_dir,'ground_truth_float.tif'),sample.ground_truth);
pilot_write_tiff(fullfile(cfg.sample_dir,'ground_truth_fine_float.tif'),sample.ground_truth_fine);
cell_write_json(fullfile(cfg.sample_dir,'config.json'),cfg);
cell_write_json(fullfile(cfg.sample_dir,'geometry.json'),sample.meta);
cell_write_json(fullfile(cfg.sample_dir,'truth_metrics.json'),sample.metrics);
cell_plot_truth(sample,previewDir);
report=struct('stage','truth_preview_only','sample_id','T04', ...
    'geometry_revision',cfg.geometry_revision,'preview_complete',true, ...
    'dataset_complete',false,'morphology_approved',false,'gpu_used',false, ...
    'forward_started',false,'rl_started',false,'training_started',false,'metrics',sample.metrics);
cell_write_json(fullfile(outputRoot,'preview_report.json'),report);
fprintf('T04 CPU TRUTH COMPLETE: %d regions, %d pairs, %d controls, %d lines.\n', ...
    sample.metrics.region_count,sample.metrics.pair_count, ...
    sample.metrics.single_control_count,sample.metrics.line_count);
fprintf('STOP for human morphology review. No acquisition, RL, migration or training.\n');
end
