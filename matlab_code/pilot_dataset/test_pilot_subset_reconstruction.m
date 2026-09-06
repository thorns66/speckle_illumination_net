function tests = test_pilot_subset_reconstruction
%TEST_PILOT_SUBSET_RECONSTRUCTION Focused CPU tests with an injected solver.
tests = functiontests(localfunctions);
end

function setup(testCase)
global PILOT_TEST_RECON_CALLS;
PILOT_TEST_RECON_CALLS = {};
testCase.TestData.directory = tempname;
mkdir(testCase.TestData.directory);
[cfg, psf] = make_fixture(testCase.TestData.directory);
testCase.TestData.cfg = cfg;
testCase.TestData.psf = psf;
end

function teardown(testCase)
global PILOT_TEST_RECON_CALLS;
PILOT_TEST_RECON_CALLS = {};
if isfolder(testCase.TestData.directory)
    rmdir(testCase.TestData.directory, 's');
end
end

function testFrameUsesExactTiffEquivalentInput(testCase)
global PILOT_TEST_RECON_CALLS;
cfg = testCase.TestData.cfg;
psf = testCase.TestData.psf;
record = pilot_reconstruct_frame(cfg, psf, 1, @fake_reconstruct);
sensor = load(fullfile(cfg.sample_dir, 'sensor_frames', 'frame_001.mat'));
expectedLegacy = single(im2uint8(sensor.sensor_legacy_float)) / single(255);
verifyEqual(testCase, PILOT_TEST_RECON_CALLS{1}.sensor, expectedLegacy);
verifyEqual(testCase, PILOT_TEST_RECON_CALLS{2}.sensor, sensor.sensor_pre_detector);
verifyEqual(testCase, {PILOT_TEST_RECON_CALLS{1}.mode, ...
    PILOT_TEST_RECON_CALLS{2}.mode}, {'mean','mean'});
verifyClass(testCase, record.w_legacy_raw, 'single');
verifySize(testCase, record.w_legacy_raw, [260,260,10]);
verifyEqual(testCase, record.w_legacy_preview_uint16, ...
    pilot_legacy_preview(record.w_legacy_raw, 'mean'));
% Completed records are resume-safe and do not invoke the solver again.
PILOT_TEST_RECON_CALLS = {};
resumed = pilot_reconstruct_frame(cfg, psf, 1, @unexpected_reconstruct);
verifyEqual(testCase, resumed.w_legacy_raw, record.w_legacy_raw);
verifyEmpty(testCase, PILOT_TEST_RECON_CALLS);
end

function testSubsetUsesFloatStatsNminusOneAndSeparateProducts(testCase)
global PILOT_TEST_RECON_CALLS;
cfg = testCase.TestData.cfg;
psf = testCase.TestData.psf;
record = pilot_build_subset(cfg, psf, 1, @fake_reconstruct);
legacy = zeros(260,260,2);
physics = zeros(260,260,2);
for position = 1:2
    frame = [1,3];
    data = load(fullfile(cfg.sample_dir, 'sensor_frames', ...
        sprintf('frame_%03d.mat', frame(position))));
    legacy(:,:,position) = data.sensor_legacy_float;
    physics(:,:,position) = double(data.sensor_pre_detector);
end
expectedLegacyMean = mean(legacy,3);
expectedLegacyVar = var(legacy,0,3);
expectedPhysicsMean = mean(physics,3);
expectedPhysicsVar = var(physics,0,3);
verifyEqual(testCase, record.input_legacy_mean_float, expectedLegacyMean);
verifyEqual(testCase, record.input_legacy_variance_nminus1_float, expectedLegacyVar);
verifyEqual(testCase, record.input_physics_mean_float, expectedPhysicsMean);
verifyEqual(testCase, record.input_physics_variance_nminus1_float, expectedPhysicsVar);
verifyNotEqual(testCase, expectedLegacyVar, var(legacy,1,3));
expectedMeanInput = single(im2uint8(expectedLegacyMean / ...
    max(expectedLegacyMean(:)))) / single(255);
expectedVarInput = single(im2uint8(expectedLegacyVar / ...
    max(expectedLegacyVar(:)))) / single(255);
verifyEqual(testCase, PILOT_TEST_RECON_CALLS{1}.sensor, expectedMeanInput);
verifyEqual(testCase, PILOT_TEST_RECON_CALLS{2}.sensor, expectedVarInput);
verifyEqual(testCase, PILOT_TEST_RECON_CALLS{3}.sensor, single(expectedPhysicsMean));
verifyEqual(testCase, PILOT_TEST_RECON_CALLS{4}.sensor, single(expectedPhysicsVar));
verifyEqual(testCase, {PILOT_TEST_RECON_CALLS{1}.mode, ...
    PILOT_TEST_RECON_CALLS{2}.mode, PILOT_TEST_RECON_CALLS{3}.mode, ...
    PILOT_TEST_RECON_CALLS{4}.mode}, {'mean','taylor','mean','taylor'});
requiredRaw = {'legacy_mean_raw','legacy_taylor_raw', ...
    'physics_mean_raw','physics_taylor_raw'};
for index = 1:numel(requiredRaw)
    verifyClass(testCase, record.(requiredRaw{index}), 'single');
    verifySize(testCase, record.(requiredRaw{index}), [260,260,10]);
end
verifyEqual(testCase, record.legacy_taylor_sqrt_float, ...
    sqrt(record.legacy_taylor_raw));
verifyNotEqual(testCase, single(record.legacy_taylor_preview_uint16), ...
    record.legacy_taylor_sqrt_float);
verifyEqual(testCase, record.legacy_taylor_preview_uint16, ...
    pilot_legacy_preview(record.legacy_taylor_raw, 'taylor'));
previewPaths = struct2cell(record.preview_paths);
for index = 1:numel(previewPaths)
    verifyTrue(testCase, isfile(previewPaths{index}));
    verifyNumElements(testCase, imfinfo(previewPaths{index}), 10);
end
% Holdout frames are disjoint and use the same exact estimators.
verifyEmpty(testCase, intersect(record.input_indices, record.holdout_indices));
verifyEqual(testCase, sort([record.input_indices, record.holdout_indices]), 1:4);
end

function [cfg, psf] = make_fixture(root)
sampleDir = fullfile(root, 'PTEST');
mkdir(sampleDir);
mkdir(fullfile(sampleDir, 'sensor_frames'));
mkdir(fullfile(sampleDir, 'recon_frames'));
mkdir(fullfile(sampleDir, 'subsets'));
mkdir(fullfile(sampleDir, 'previews'));
cfg = struct('schema_version',1, 'dataset_id','pilot_test', ...
    'sample_id','PTEST', 'sample_dir',sampleDir, 'frame_count',4, ...
    'subset_count',1, 'input_frames',2, 'holdout_frames',2, ...
    'image_size',[260,260], 'z_um',10:10:100, 'iterations',3);
input_indices = [1,3]; %#ok<NASGU>
holdout_indices = [2,4]; %#ok<NASGU>
save(fullfile(sampleDir, 'prepared.mat'), 'cfg', 'input_indices', ...
    'holdout_indices', '-v7');
[yy, xx] = ndgrid(0:259, 0:259);
base = 0.02 + 0.30 * mod(3*xx + 5*yy, 31) / 30;
for frame = 1:4
    sensor_legacy_float = base + 0.07 * frame + ...
        0.01 * mod(xx + frame*yy, 7); %#ok<NASGU>
    sensor_legacy_float = sensor_legacy_float / ...
        max(sensor_legacy_float(:)); %#ok<NASGU>
    sensor_legacy_uint8 = im2uint8(sensor_legacy_float); %#ok<NASGU>
    sensor_pre_detector = single((1 + 0.5*frame) * base + ...
        0.03 * mod(2*xx + frame*yy, 11)); %#ok<NASGU>
    schema_version = cfg.schema_version; %#ok<NASGU>
    dataset_id = cfg.dataset_id; %#ok<NASGU>
    sample_id = cfg.sample_id; %#ok<NASGU>
    frame_index = frame; %#ok<NASGU>
    save(fullfile(sampleDir, 'sensor_frames', sprintf('frame_%03d.mat', frame)), ...
        'schema_version','dataset_id','sample_id','frame_index', ...
        'sensor_legacy_float','sensor_legacy_uint8','sensor_pre_detector','-v7');
end
psf = struct('z_um', cfg.z_um);
end

function [X, diagnostic] = fake_reconstruct(psf, sensor, iterations, mode)
global PILOT_TEST_RECON_CALLS;
PILOT_TEST_RECON_CALLS{end+1} = struct('sensor',sensor, ...
    'iterations',iterations,'mode',mode); %#ok<AGROW>
gain = single(1);
if strcmp(mode, 'taylor'), gain = single(2); end
X = repmat(gain * single(sensor), 1, 1, numel(psf.z_um));
for depth = 1:numel(psf.z_um)
    X(:,:,depth) = X(:,:,depth) * single(depth / numel(psf.z_um));
end
diagnostic = struct('mock',true,'mode',mode,'iterations',iterations);
end

function [X, diagnostic] = unexpected_reconstruct(varargin) %#ok<STOUT,INUSD>
error('pilot:testUnexpectedSolver', 'Resume path unexpectedly invoked the solver.');
end
