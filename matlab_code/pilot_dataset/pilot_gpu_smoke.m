function result = pilot_gpu_smoke(deviceIndex)
%PILOT_GPU_SMOKE One-layer CPU/GPU parity and historical-frame regression.
% This preflight does not create dataset output and does not load the full
% ten-depth PSF. It checks the exact operators used before a long pilot run.
if nargin < 1, deviceIndex = 1; end
validateattributes(deviceIndex, {'numeric'}, ...
    {'scalar','integer','positive','finite'}, mfilename, 'deviceIndex');
pilotRoot = fileparts(mfilename('fullpath'));
codeRoot = fileparts(pilotRoot);
addpath(pilotRoot, fullfile(codeRoot, 'Util'), fullfile(codeRoot, 'Solver'));
cfg = pilot_dataset_config('P01');
assert(gpuDeviceCount('available') >= deviceIndex, 'pilot:GPUCount', ...
    'Requested GPU %d is unavailable.', deviceIndex);
device = gpuDevice(deviceIndex);

psf = pilot_load_truth_psf(cfg);
m = matfile(cfg.psf_path);
Ht = single(m.Ht(:,:,:,:,psf.source_index_one_based));
H = psf.H;
CA = psf.CAindex;
product = double(imread(fullfile(cfg.product_path, 'img_1.png')));
product = product / max(product(:));

global volumeResolution zeroImageEx exsize;
volumeResolution = [cfg.image_size,1];
msize = [size(H,1),size(H,2)];
candidate = cfg.image_size + floor(msize/2);
exsize = [min(2^ceil(log2(candidate(1))),128*ceil(candidate(1)/128)), ...
    min(2^ceil(log2(candidate(2))),128*ceil(candidate(2)/128))];
zeroImageEx = gpuArray.zeros(exsize, 'single');

cpuForward = single(forwardProjectACC(H, product, CA));
gpuForward = single(gather(forwardProjectGPU(H, single(product))));
[forwardRelL2, forwardMaxAbs] = errors(cpuForward, gpuForward);
assert(forwardRelL2 <= 5e-5 && ...
    forwardMaxAbs <= 1e-4*double(max(abs(cpuForward(:)))), ...
    'pilot:ForwardParity', 'CPU/GPU forward mismatch: relL2=%g, maxAbs=%g.', ...
    forwardRelL2, forwardMaxAbs);

cpuBackward = single(backwardProjectACC(Ht, cpuForward, CA));
gpuBackward = single(gather(backwardProjectGPU(Ht, cpuForward)));
[backwardRelL2, backwardMaxAbs] = errors(cpuBackward, gpuBackward);
assert(backwardRelL2 <= 5e-5 && ...
    backwardMaxAbs <= 1e-4*double(max(abs(cpuBackward(:)))), ...
    'pilot:BackwardParity', 'CPU/GPU backward mismatch: relL2=%g, maxAbs=%g.', ...
    backwardRelL2, backwardMaxAbs);

cpuForwardFcn = @(x) forwardProjectACC(H, x, CA);
cpuBackwardFcn = @(y) backwardProjectACC(Ht, y, CA);
cpuRl = deconvRL(cpuForwardFcn, cpuBackwardFcn, double(cpuBackward), ...
    cfg.iterations, double(cpuBackward));
gpuForwardFcn = @(x) forwardProjectGPU(H, x);
gpuBackwardFcn = @(y) backwardProjectGPU(Ht, y);
gpuStart = backwardProjectGPU(Ht, cpuForward);
gpuRl = single(gather(deconvRL(gpuForwardFcn, gpuBackwardFcn, gpuStart, ...
    cfg.iterations, gpuStart)));
[rlRelL2, rlMaxAbs] = errors(single(cpuRl), gpuRl);
assert(rlRelL2 <= 1e-3, 'pilot:RLParity', ...
    'Three-step CPU/GPU RL mismatch: relL2=%g, maxAbs=%g.', rlRelL2, rlMaxAbs);

normalized = double(cpuForward) / double(max(cpuForward(:)));
regenerated = im2uint8(normalized);
% Keep the non-ASCII path legible and portable across MATLAB source encodings.
historicalPath = fullfile(cfg.repo_root, 'data', ...
    native2unicode(uint8([229,136,134,232,190,168,231,142,135,229,155,190, ...
    228,187,191,231,156,159]), 'UTF-8'), 'img_detph50_1.tif');
assert(isfile(historicalPath), 'pilot:GoldenMissing', ...
    'Missing historical detector frame: %s', historicalPath);
historical = imread(historicalPath);
adu = abs(double(regenerated)-double(historical));
maxAdu = max(adu(:));
identicalFraction = nnz(adu == 0) / numel(adu);
assert(maxAdu <= 1 && identicalFraction >= 0.9999, 'pilot:GoldenMismatch', ...
    'Historical detector frame mismatch: max ADU=%g, identical fraction=%g.', ...
    maxAdu, identicalFraction);

result = struct('passed', true, 'gpu_index', device.Index, ...
    'gpu_name', device.Name, 'truth_depth_um', cfg.truth_depth_um, ...
    'psf_source_index_one_based', psf.source_index_one_based, ...
    'caindex', CA, 'forward_rel_l2', forwardRelL2, ...
    'forward_max_abs', forwardMaxAbs, 'backward_rel_l2', backwardRelL2, ...
    'backward_max_abs', backwardMaxAbs, 'rl_3step_rel_l2', rlRelL2, ...
    'rl_3step_max_abs', rlMaxAbs, 'historical_max_adu', maxAdu, ...
    'historical_identical_fraction', identicalFraction);
end

function [relativeL2, maximumAbsolute] = errors(reference, candidate)
difference = double(candidate)-double(reference);
relativeL2 = norm(difference(:)) / max(norm(double(reference(:))), eps);
maximumAbsolute = max(abs(difference(:)));
end
