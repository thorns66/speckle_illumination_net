function psf = pilot_load_psf_gpu(cfg)
%PILOT_LOAD_PSF_GPU Load the ten-layer PSF once and retain it on this GPU.
% forwardProjectGPU/backwardProjectGPU already convert every kernel slice
% with gpuArray. Supplying gpuArray H/Ht only removes repeated host-device
% transfers; it does not alter the original loop order or arithmetic.
assert(gpuDeviceCount('available') >= 1, 'pilot:GPUCount', ...
    'A worker GPU must be selected before loading the resident PSF.');
psf = pilot_load_psf(cfg);
psf.H = gpuArray(psf.H);
psf.Ht = gpuArray(psf.Ht);
psf.storage = 'worker-local GPU-resident single';
psf.numerical_policy = ...
    'transfer-only optimization; original forward/backward/deconvRL unchanged';
end
