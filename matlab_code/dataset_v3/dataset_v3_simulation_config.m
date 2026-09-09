function cfg = dataset_v3_simulation_config(sampleId, outputRoot)
% Approved truth is loaded, not regenerated from a mutable default generator.
path=fullfile(outputRoot,sampleId,'truth.mat');
truth=load(path,'cfg','source_sha256');
assert(strcmp(truth.cfg.sample_id,sampleId),'v3:Owner','Wrong truth owner.');
cfg=truth.cfg;
protocol=cell_simulation_config('P06',outputRoot);
names={'psf_path','psf_depth_policy','reconstruction_depth_policy', ...
    'illumination_model','illumination_reference_z_um','illumination_na', ...
    'illumination_wavelength_nm','illumination_pixel_pitch_um', ...
    'illumination_padded_sampling','illumination_crop_policy', ...
    'illumination_normalization','noise_model','mean_policy','variance_policy'};
for k=1:numel(names), cfg.(names{k})=protocol.(names{k}); end
cfg.geometry_schema_version=cfg.schema_version; cfg.schema_version=1;
cfg.dataset_id='speckle_dataset_v3_simulation_v1';
cfg.output_root=outputRoot; cfg.sample_dir=fullfile(outputRoot,sampleId);
cfg.truth_path=path; cfg.ground_truth_mode='continuous_multidepth_or_independent_control';
cfg.truth_depth_um=[]; cfg.truth_index_one_based=[];
cfg.truth_source_sha256=truth.source_sha256;
cfg.stage='approved_full_simulation'; cfg.morphology_approved=true;
cfg.allowed_physical_gpu_indices=0; cfg.required_gpu_model='A40';
cfg.allow_sharing=true; cfg.allow_fallback_gpu=false;
cfg.future_gpu_policy='Physical 0 A40 only; sharing explicitly authorized; never terminate existing processes';
cfg.preview_compute='CPU_ONLY'; % Historical truth preview, not a simulation device selector.
assert(isequal(cfg.z_um,10:10:100)&&cfg.iterations==3&&cfg.frame_count==100&& ...
    cfg.input_frames==10&&cfg.holdout_frames==90&&cfg.subset_count==10,'v3:Protocol');
end
