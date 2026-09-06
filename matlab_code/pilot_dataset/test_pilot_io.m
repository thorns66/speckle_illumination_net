function tests = test_pilot_io
%TEST_PILOT_IO CPU-only tests: runtests('test_pilot_io.m').
tests = functiontests(localfunctions);
end

function setup(testCase)
testCase.TestData.directory = tempname;
mkdir(testCase.TestData.directory);
end

function teardown(testCase)
if isfolder(testCase.TestData.directory)
    rmdir(testCase.TestData.directory, 's');
end
end

function testFloatTiffRoundTrip(testCase)
volume = reshape(single([-2.5, 0, 0.001, pi, 1e6, 0.3, 8, 9, 11, 12, 13, 14]), 2, 3, 2);
path = fullfile(testCase.TestData.directory, 'float.tif');
pilot_write_tiff(path, volume);
actual = read_stack(path);
verifyClass(testCase, actual, 'single');
verifyEqual(testCase, actual, volume);
t = Tiff(path, 'r');
cleanup = onCleanup(@() t.close()); %#ok<NASGU>
verifyEqual(testCase, t.getTag('SampleFormat'), Tiff.SampleFormat.IEEEFP);
verifyEqual(testCase, t.getTag('BitsPerSample'), 32);
end

function testUint16TiffAndSafeReplacement(testCase)
path = fullfile(testCase.TestData.directory, 'integer.tif');
pilot_write_tiff(path, uint16(reshape(1:24, 3, 4, 2)));
replacement = uint16([0, 65535; 44, 19]);
pilot_write_tiff(path, replacement);
verifyEqual(testCase, read_stack(path), replacement);
verifyNumElements(testCase, imfinfo(path), 1);
failed = false;
try, pilot_write_tiff(path, single([1, NaN])); catch, failed = true; end
verifyTrue(testCase, failed);
verifyEqual(testCase, read_stack(path), replacement);
end

function testLegacyFormula(testCase)
X = reshape(single([0, 1e-8, 0.17, 0.5, 3, 7, 12, 0.009, 0.26, 11, 0.004, 1]), 2, 3, 2);
Q = uint16(round(65535 * (0.66 / max(X(:))) * X));
expectedMean = uint16(round(65535 * double(Q) / max(double(Q(:)))));
root = sqrt(double(Q));
expectedTaylor = uint16(round(65535 * root / max(root(:))));
verifyEqual(testCase, pilot_legacy_preview(X, 'mean'), expectedMean);
verifyEqual(testCase, pilot_legacy_preview(X, 'taylor'), expectedTaylor);
verifyEqual(testCase, pilot_legacy_preview(zeros(2, 3, 2, 'single'), 'taylor'), ...
    zeros(2, 3, 2, 'uint16'));
end

function testCommonGainPreservesFrameAndDepthScale(testCase)
sampleDir = testCase.TestData.directory;
mkdir(fullfile(sampleDir, 'recon_frames'));
base = reshape(single([0, 1, 3, 2, 4, 6, 0, 2, 4, 4, 8, 12]), 2, 3, 2);
for frame = 1:3
    w_legacy_raw = frame * base; %#ok<NASGU>
    w_physics_raw = frame * base; %#ok<NASGU>
    save(fullfile(sampleDir, 'recon_frames', sprintf('frame_%03d.mat', frame)), ...
        'w_legacy_raw', 'w_physics_raw');
end
manifest = pilot_export_algorim(sampleDir, 3, [10, 20]);
verifyEqual(testCase, manifest.physics.common_uint16_gain, 65535 / 36);
verifyEqual(testCase, manifest.output_axes, 'YXF');
for depth = 1:2
    name = sprintf('depth_%03dum.tif', depth * 10);
    physics = read_stack(fullfile(sampleDir, 'algorim', 'physics_common_scale', name));
    legacy = read_stack(fullfile(sampleDir, 'algorim', 'legacy_per_frame_normalized', name));
    for frame = 1:3
        expected = uint16(round(double(frame * base(:, :, depth)) * 65535 / 36));
        verifyEqual(testCase, physics(:, :, frame), expected);
        preview = pilot_legacy_preview(frame * base, 'mean');
        verifyEqual(testCase, legacy(:, :, frame), preview(:, :, depth));
    end
    verifyEqual(testCase, legacy(:, :, 1), legacy(:, :, 3));
    verifyGreaterThan(testCase, max(physics(:, :, 3), [], 'all'), ...
        max(physics(:, :, 1), [], 'all'));
end
% Repeating an owned export is safe and does not append duplicate pages.
pilot_export_algorim(sampleDir, 3, [10, 20]);
path = fullfile(sampleDir, 'algorim', 'physics_common_scale', 'depth_010um.tif');
verifyNumElements(testCase, imfinfo(path), 3);
% Missing input fails before replacing an already complete export.
delete(fullfile(sampleDir, 'recon_frames', 'frame_003.mat'));
verifyError(testCase, @() pilot_export_algorim(sampleDir, 3, [10, 20]), ...
    'pilot:exportMissingFrame');
verifyNumElements(testCase, imfinfo(path), 3);
end

function testIncompleteAndForeignExportsFail(testCase)
sampleDir = testCase.TestData.directory;
mkdir(fullfile(sampleDir, 'recon_frames'));
w_legacy_raw = ones(2, 3, 2, 'single'); %#ok<NASGU>
path = fullfile(sampleDir, 'recon_frames', 'frame_001.mat');
save(path, 'w_legacy_raw');
verifyError(testCase, @() pilot_export_algorim(sampleDir, 1, [10, 20]), ...
    'pilot:exportSchema');
verifyFalse(testCase, isfolder(fullfile(sampleDir, 'algorim')));
w_physics_raw = ones(2, 3, 2, 'single'); %#ok<NASGU>
save(path, 'w_legacy_raw', 'w_physics_raw');
mkdir(fullfile(sampleDir, 'algorim'));
verifyError(testCase, @() pilot_export_algorim(sampleDir, 1, [10, 20]), ...
    'pilot:exportOwnership');
end

function volume = read_stack(path)
info = imfinfo(path);
t = Tiff(path, 'r');
cleanup = onCleanup(@() t.close()); %#ok<NASGU>
first = t.read();
volume = zeros(size(first, 1), size(first, 2), numel(info), 'like', first);
volume(:, :, 1) = first;
for page = 2:numel(info)
    t.setDirectory(page);
    volume(:, :, page) = t.read();
end
end
