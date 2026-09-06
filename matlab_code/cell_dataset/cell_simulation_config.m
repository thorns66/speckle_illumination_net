function cfg = cell_simulation_config(sampleId, outputRoot)
%CELL_SIMULATION_CONFIG Frozen full-simulation protocol for approved truth.
if nargin<2, outputRoot=''; end
truthCfg=cell_dataset_config(sampleId,outputRoot);
if isempty(outputRoot), outputRoot=truthCfg.output_root; end
repoRoot=truthCfg.repo_root;
cfg=truthCfg;
cfg.schema_version=1;
cfg.geometry_schema_version=truthCfg.schema_version;
cfg.dataset_id='matlab_cells_pilot_v2_r04_simulation_v1';
cfg.output_root=char(outputRoot);
cfg.sample_dir=fullfile(outputRoot,cfg.sample_id);
cfg.truth_path=fullfile(cfg.sample_dir,'truth.mat');
cfg.ground_truth_mode='continuous_multidepth_or_independent_control';
cfg.truth_depth_um=[];
cfg.truth_index_one_based=[];
cfg.psf_path=fullfile(repoRoot,'psf', ...
    'NEW_modifyfobj_PSFmatrix_M4NA0.15MLPitch220fml4000OSR3chunk05from10to130zspacing14.6154Nnum49lambda532n1a0_-11b0_2.9333.mat');
cfg.psf_depth_policy='match_x3objspace_not_filename_or_zspacing';
cfg.reconstruction_depth_policy='all_ten_layers_without_truth_support_metadata';
cfg.illumination_model='shared_complex_field_angular_spectrum';
cfg.illumination_reference_z_um=0;
cfg.illumination_na=0.05;
cfg.illumination_wavelength_nm=488;
cfg.illumination_pixel_pitch_um=1.125;
cfg.illumination_padded_sampling=520;
cfg.illumination_crop_policy='central_260_from_520_matching_original_generator';
cfg.illumination_normalization='physics raw; legacy one scale over complete YXZ frame';
cfg.noise_model='none';
cfg.mean_policy='same ten input frames only; no extra uniform exposure';
cfg.variance_policy='sample variance N-1 using MATLAB var(...,0,3)';
cfg.truth_source_sha256=cell_geometry_hash();
cfg.stage='approved_full_simulation';
end
