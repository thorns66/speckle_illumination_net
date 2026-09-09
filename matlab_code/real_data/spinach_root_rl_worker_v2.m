function spinach_root_rl_worker_v2(manifestPath, mode)
% Full-field exploratory RL3 and common forward diagnostics.
mode=validatestring(mode,{'mean','taylor'},mfilename,'mode');
cfg=jsondecode(fileread(manifestPath));
assert(strcmp(getenv('CUDA_VISIBLE_DEVICES'),cfg.gpu_uuid.(mode)));
gpuDevice(1); psfCfg=struct('psf_path',cfg.psf_path,'z_um',double(cfg.z_um));
psf=pilot_load_psf_gpu(psfCfg);
sensorPath=cfg.mean_tiff;if strcmp(mode,'taylor'),sensorPath=cfg.variance_tiff;end
sensor=single(imread(sensorPath));
assert(isequal(size(sensor),double(cfg.image_shape_yx(:)')));
[raw,diagnostic]=reconstruct_dynamic(psf,sensor,3,mode);
comparison=raw;
if strcmp(mode,'taylor'),comparison=sqrt(raw);end
setup_fft(size(sensor),size(psf.H,1));
predictedMean=single(gather(forwardProjectGPU(psf.H,gpuArray(comparison))));
predictedVariance=single(gather(forwardProjectGPU(psf.H.^2,gpuArray(comparison.^2))));
record=struct('schema_version',2,'mode',mode,'iterations',3,'z_um',double(cfg.z_um), ...
 'input_indices',double(cfg.input_indices),'image_shape_yx',double(size(sensor)), ...
 'reconstruction_raw',raw,'predicted_mean',predictedMean, ...
 'predicted_variance',predictedVariance,'diagnostic',diagnostic,'input_policy', ...
 'rectified uint8 divided by 255 once; no additional frame normalization');
if strcmp(mode,'taylor')
 record.reconstruction_sqrt=comparison;record.projection_of_sqrt=predictedMean;
end
outputPath=cfg.output_mat.(mode);temporary=[outputPath '.tmp.mat'];
save(temporary,'-struct','record','-v7.3');movefile(temporary,outputPath,'f');
fprintf('SPINACH_RL_COMPLETE %s %s\n',mode,datestr(now,31));
end

function [Xraw,diagnostic]=reconstruct_dynamic(psf,sensor,iterations,mode)
if strcmp(mode,'taylor'),H=psf.H.^2;Ht=psf.Ht.^2;else,H=psf.H;Ht=psf.Ht;end
assert(size(H,5)==10&&isequal(size(H),size(Ht)));setup_fft(size(sensor),size(H,1));
global volumeResolution;volumeResolution=[size(sensor,1),size(sensor,2),size(H,5)];
forwardFUN=@(x)forwardProjectGPU(H,x);backwardFUN=@(y)backwardProjectGPU(Ht,y);
t0=tic;Htf=backwardFUN(single(sensor));backwardSeconds=toc(t0);Xguess=Htf;t1=tic;
Xguess=deconvRL(forwardFUN,backwardFUN,Htf,double(iterations),Xguess);iterationSeconds=toc(t1);
Xraw=single(gather(Xguess));assert(all(isfinite(Xraw(:))));fallback=false;
try
 [Xraw,roundoff]=pilot_sanitize_roundoff(Xraw,['real ' mode]);
catch exception
 if ~strcmp(exception.identifier,'pilot:MaterialNegativeReconstruction'),rethrow(exception);end
 fallback=true;positiveForward=@(x)max(forwardProjectGPU(H,max(x,0)),0);
 positiveBackward=@(y)max(backwardProjectGPU(Ht,max(y,0)),0);
 Htf=positiveBackward(single(sensor));Xguess=pilot_deconv_rl_nonnegative(positiveForward,positiveBackward,Htf,double(iterations));
 Xraw=single(gather(Xguess));[Xraw,roundoff]=pilot_sanitize_roundoff(Xraw,['real ' mode ' fallback']);
end
diagnostic=struct('mode',mode,'iterations',iterations,'backward_seconds',backwardSeconds, ...
 'iteration_seconds',iterationSeconds,'minimum',double(min(Xraw(:))), ...
 'maximum',double(max(Xraw(:))),'roundoff',roundoff,'positivity_fallback',fallback, ...
 'solver','simulation multiplicative RL update generalized only to full field');
end

function setup_fft(sensorSize,kernelSize)
global zeroImageEx exsize;candidate=double(sensorSize(1:2))+floor(double(kernelSize)/2);
exsize=[min(2^ceil(log2(candidate(1))),128*ceil(candidate(1)/128)), ...
 min(2^ceil(log2(candidate(2))),128*ceil(candidate(2)/128))];
zeroImageEx=gpuArray.zeros(exsize,'single');
end
