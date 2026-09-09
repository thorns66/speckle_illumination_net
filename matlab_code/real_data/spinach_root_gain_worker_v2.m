function spinach_root_gain_worker_v2(manifestPath)
% Defining E3 global q/gain step and common forward diagnostics.
cfg=jsondecode(fileread(manifestPath));assert(strcmp(getenv('CUDA_VISIBLE_DEVICES'),cfg.gpu_uuid.gain));
gpuDevice(1);psfCfg=struct('psf_path',cfg.psf_path,'z_um',double(cfg.z_um));psf=pilot_load_psf_gpu(psfCfg);
loaded=load(cfg.network_base_mat,'base_reconstruction');base=single(loaded.base_reconstruction);
assert(isequal(size(base),[double(cfg.image_shape_yx(:)'),numel(cfg.z_um)]));
q=base/max(sum(base,'all','double'),1e-30);setup_fft(size(base(:,:,1)),size(psf.H,1));
hq=single(gather(forwardProjectGPU(psf.H,gpuArray(q))));mu=single(imread(cfg.mean_tiff));
a0=sum(double(hq(:)).*double(mu(:)))/max(sum(double(hq(:)).^2),1e-30);a0=max(a0,0);
gain=a0*double(cfg.e3_gain_factor);reconstruction=single(q*gain);
predictedMean=hq*single(gain);
predictedVariance=single(gather(forwardProjectGPU(psf.H.^2,gpuArray(reconstruction.^2))));
record=struct('schema_version',2,'reconstruction',reconstruction,'q',q,'projection_of_q',hq, ...
 'analytic_a0',a0,'gain_factor',double(cfg.e3_gain_factor),'gain',gain, ...
 'predicted_mean',predictedMean,'predicted_variance',predictedVariance, ...
 'policy','full-field E3 global q normalization and analytic mean gain');
temporary=[cfg.network_final_mat '.tmp.mat'];save(temporary,'-struct','record','-v7.3');
movefile(temporary,cfg.network_final_mat,'f');fprintf('SPINACH_GAIN_COMPLETE %s\n',datestr(now,31));
end

function setup_fft(sensorSize,kernelSize)
global zeroImageEx exsize;candidate=double(sensorSize(1:2))+floor(double(kernelSize)/2);
exsize=[min(2^ceil(log2(candidate(1))),128*ceil(candidate(1)/128)), ...
 min(2^ceil(log2(candidate(2))),128*ceil(candidate(2)/128))];
zeroImageEx=gpuArray.zeros(exsize,'single');
end
