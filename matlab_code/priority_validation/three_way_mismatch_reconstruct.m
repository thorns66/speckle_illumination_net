function three_way_mismatch_reconstruct(repoRoot,outputRoot,sampleId)
addpath(fullfile(repoRoot,'matlab_code','cell_dataset'),fullfile(repoRoot,'matlab_code','pilot_dataset'), ...
 fullfile(repoRoot,'matlab_code','Util'),fullfile(repoRoot,'matlab_code','Solver'));
folder=fullfile(outputRoot,'mismatch_generated',sampleId);
path=fullfile(folder,'subsets','subset_01.mat');if isfile(path),return;end
cfg=cell_simulation_config('P08',fullfile(repoRoot,'data','speckle_data_now'));
psf=pilot_load_psf_gpu(cfg);data=load(fullfile(folder,'input_statistics.mat'));
[mean3,dm]=pilot_reconstruct_volume(psf,data.input_physics_mean_float,3,'mean');
[taylor3,dt]=pilot_reconstruct_volume(psf,data.input_physics_variance_nminus1_float,3,'taylor');
data.sample_id=sampleId;data.subset_index=1;data.iterations=3;data.schema_version=1;
data.physics_mean_raw=mean3;data.physics_taylor_raw=taylor3;data.physics_taylor_sqrt_float=sqrt(taylor3);
data.mean_diagnostic=dm;data.taylor_diagnostic=dt;
if ~isfolder(fullfile(folder,'subsets')),mkdir(fullfile(folder,'subsets'));end
pilot_atomic_save(path,data);
fprintf('NA mismatch RL3 complete: %s\n',sampleId);
end
