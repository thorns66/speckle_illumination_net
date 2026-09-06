function [rawIntensity, normalizedIntensity, meta] = pilot_generate_illumination(seed, frameCount)
%PILOT_GENERATE_ILLUMINATION Pure illumination from generate_speckle_NA05.m.
%   Returns unquantized double Y-X-frame arrays. No sample multiplication,
%   photon noise, PNG encoding, or propagation along z takes place here.
%   The original random-phase/pad/FFT/pupil/IFFT/crop operations are preserved.
%   Seeding is explicit and the caller's random-number state is restored.

validateattributes(seed, {'numeric'}, {'real', 'finite', 'scalar', 'integer', ...
    '>=', 0, '<=', 2^32 - 1}, mfilename, 'seed');
validateattributes(frameCount, {'numeric'}, {'real', 'finite', 'scalar', ...
    'integer', 'positive'}, mfilename, 'frameCount');
originalRng = rng;
restoreRng = onCleanup(@() rng(originalRng));
rng(double(seed), 'twister');

NA = 0.05;
lambda = 488e-9;
pixel_size = 4.5e-6 / 4;
sampling = 260;
fov = sampling * pixel_size;

phase = rand(sampling, sampling, frameCount) .* 2 * pi;
SLM = exp(1j * phase);
SLM = padarray(SLM, [sampling / 2, sampling / 2], 0);
phase_fft = fftshift(fft2(SLM));

f0 = NA / lambda;
L = 2 * fov;
du = pixel_size;
fu = -1 / (2 * du):1 / L:1 / (2 * du) - 1 / L;
fv = fu;
[Fu, Fv] = meshgrid(fu, fv);
r = sqrt(Fu.^2 + Fv.^2);
H = zeros(size(r, 1), size(r, 2));
H((r / f0) <= 1) = 1;

speckle = phase_fft .* H;
speckle = ifft2(ifftshift(speckle));
speckle = abs(speckle).^2;
crop = sampling / 2 + 1:sampling / 2 + sampling;
rawIntensity = speckle(crop, crop, :);
normalizedIntensity = zeros(sampling, sampling, frameCount);
framePeaks = zeros(1, frameCount);
for a = 1:frameCount
    framePeaks(a) = max(max(rawIntensity(:, :, a)));
    assert(isfinite(framePeaks(a)) && framePeaks(a) > 0, ...
        'pilot:InvalidIllumination', 'Generated frame %d has no finite positive peak.', a);
    normalizedIntensity(:, :, a) = rawIntensity(:, :, a) ./ framePeaks(a);
end

meta = struct('generator_version', 'original_NA05_seeded_v1', ...
    'source_algorithm', 'generate_speckle_NA05.m, random phase through central crop', ...
    'seed', double(seed), 'rng_algorithm', 'twister', 'frame_count', double(frameCount), ...
    'NA', NA, 'lambda_m', lambda, 'pixel_size_m', pixel_size, ...
    'sampling', sampling, 'padded_sampling', 2 * sampling, ...
    'frame_peaks', framePeaks, 'pupil_nonzero_count', nnz(H), ...
    'normalization', 'each central-cropped intensity frame divided by its own maximum', ...
    'contains_object', false, 'axis_order', 'YXT', 'quantized', false);
end
