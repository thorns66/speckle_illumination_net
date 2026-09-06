function tests = test_pilot_objects_speckle
%TEST_PILOT_OBJECTS_SPECKLE Native-grid geometry and original-algorithm parity.
tests = functiontests(localfunctions);
end

function test_objects_are_valid_reproducible_and_have_expected_depths(testCase)
ids = {'P01', 'P02', 'P03', 'P04', 'P05'};
depths = [50, 20, 80, 30, 100];
for k = 1:numel(ids)
    [object, meta] = pilot_make_object(ids{k});
    repeated = pilot_make_object(ids{k});
    verifyClass(testCase, object, 'double');
    verifySize(testCase, object, [260, 260]);
    verifyTrue(testCase, all(isfinite(object(:))));
    verifyGreaterThanOrEqual(testCase, min(object(:)), 0);
    verifyEqual(testCase, max(object(:)), 1);
    verifyEqual(testCase, object, repeated);
    verifyEqual(testCase, meta.depth_um, depths(k));
    verifyEqual(testCase, meta.pixel_size_m, 4.5e-6 / 4);
    if k > 1
        verifyGreaterThanOrEqual(testCase, meta.minimum_border_margin_px, 16);
    end
end
end

function test_original_target_is_not_resized_or_modified(testCase)
[actual, meta] = pilot_make_object('P01');
expected = double(imread(meta.source_path));
expected = expected / max(expected(:));
verifyEqual(testCase, actual, expected);
end

function test_unknown_sample_is_rejected(testCase)
verifyError(testCase, @() pilot_make_object('P06'), 'pilot:UnknownSample');
end

function test_speckle_matches_original_expressions_exactly(testCase)
% Test odd and even batch sizes: original fftshift also shifts the frame axis.
for frameCount = [1, 3, 4]
    seed = 1729;
    [raw, normalized, meta] = pilot_generate_illumination(seed, frameCount);
    [expectedRaw, expectedNormalized] = original_expressions(seed, frameCount);
    verifyEqual(testCase, raw, expectedRaw);
    verifyEqual(testCase, normalized, expectedNormalized);
    verifyClass(testCase, raw, 'double');
    verifyEqual(testCase, [size(raw, 1), size(raw, 2), size(raw, 3)], [260, 260, frameCount]);
    verifyTrue(testCase, all(isfinite(raw(:))));
    verifyGreaterThanOrEqual(testCase, min(raw(:)), 0);
    verifyEqual(testCase, squeeze(max(max(normalized, [], 1), [], 2)), ones(frameCount, 1));
    verifyEqual(testCase, meta.NA, 0.05);
    verifyEqual(testCase, meta.lambda_m, 488e-9);
    verifyFalse(testCase, meta.contains_object);
    verifyFalse(testCase, meta.quantized);
end
end

function test_seed_is_reproducible_and_caller_rng_is_unchanged(testCase)
originalRng = rng;
restoreRng = onCleanup(@() rng(originalRng));
rng(999, 'twister');
before = rng;
[a, an] = pilot_generate_illumination(12, 2);
verifyEqual(testCase, rng, before);
[b, bn] = pilot_generate_illumination(12, 2);
verifyEqual(testCase, a, b);
verifyEqual(testCase, an, bn);
c = pilot_generate_illumination(13, 2);
verifyNotEqual(testCase, a, c);
end

function [raw, normalized] = original_expressions(seed, frameCount)
% Direct mathematical transcription of the original active speckle section.
originalRng = rng;
restoreRng = onCleanup(@() rng(originalRng));
rng(seed, 'twister');
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
raw = zeros(sampling, sampling, frameCount);
normalized = zeros(sampling, sampling, frameCount);
for a = 1:frameCount
    raw(:, :, a) = speckle(sampling / 2 + 1:sampling / 2 + sampling, ...
        sampling / 2 + 1:sampling / 2 + sampling, a);
    normalized(:, :, a) = raw(:, :, a) ./ max(max(raw(:, :, a)));
end
end
