function three_way_brightness_inputs(repoRoot,outputRoot,selectedSamples)
% Independent brightness sweep: scale raw ten frames, then rerun MATLAB RL3.
addpath(fullfile(repoRoot,'matlab_code','cell_dataset'), ...
 fullfile(repoRoot,'matlab_code','pilot_dataset'),fullfile(repoRoot,'matlab_code','Util'), ...
 fullfile(repoRoot,'matlab_code','Solver'));
sources={fullfile(repoRoot,'outputs','priority_validation_20260907','generated','points_z060_r01'), ...
 fullfile(repoRoot,'outputs','priority_validation_20260907','generated','lines_z060_r01'), ...
 fullfile(repoRoot,'data','speckle_data_now','P09')};
cfg=cell_simulation_config('P08',fullfile(repoRoot,'data','speckle_data_now'));
psf=pilot_load_psf_gpu(cfg);
if nargin<3,selectedSamples=1:numel(sources);end
for j=selectedSamples
 source=sources{j};[~,sampleId]=fileparts(source);
 subset=load(fullfile(source,'subsets','subset_01.mat'),'input_indices','z_um');
 frames=zeros(260,260,10,'single');
 for k=1:10
  frame=load(fullfile(source,'sensor_frames',sprintf('frame_%03d.mat',subset.input_indices(k))), ...
   'sensor_pre_detector');
  frames(:,:,k)=frame.sensor_pre_detector;
 end
 destination=fullfile(outputRoot,'brightness_inputs',sampleId);
 if ~isfolder(destination),mkdir(destination);end
 for gain=[0.1,0.5,1,2]
  path=fullfile(destination,sprintf('gain_%g.mat',gain));
  if isfile(path),continue;end
  scaled=single(double(frames).*gain);
  mu=single(mean(double(scaled),3));variance=single(var(double(scaled),0,3));
  [gmean,diagMean]=pilot_reconstruct_volume(psf,mu,3,'mean');
  [taylor,diagTaylor]=pilot_reconstruct_volume(psf,variance,3,'taylor');
  payload=struct('sample_id',sampleId,'gain',gain,'input_indices',subset.input_indices, ...
   'z_um',subset.z_um,'input_frames',scaled,'input_mean',mu,'input_variance',variance, ...
   'g_mean',gmean,'f_var',sqrt(taylor),'taylor_raw',taylor,'mean_diagnostic',diagMean, ...
   'taylor_diagnostic',diagTaylor,'source',source,'target_or_gt_used',false);
  pilot_atomic_save(path,payload);
  fprintf('BRIGHTNESS %s gain=%g\n',sampleId,gain);
 end
end
end
