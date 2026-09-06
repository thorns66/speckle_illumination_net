function tests = test_pilot_orchestration
%TEST_PILOT_ORCHESTRATION Protocol/split and CPU-only helper smoke tests.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
pilotRoot = fileparts(mfilename('fullpath'));
codeRoot = fileparts(pilotRoot);
addpath(pilotRoot, fullfile(codeRoot, 'Util'), fullfile(codeRoot, 'Solver'));
end

function testFrozenSampleTableAndPSFDepths(testCase)
ids = {'P01','P02','P03','P04','P05'};
depths = [50,20,80,30,100];
for index = 1:numel(ids)
    cfg = pilot_dataset_config(ids{index});
    verifyEqual(testCase, cfg.truth_depth_um, depths(index));
    verifyEqual(testCase, cfg.z_um, 10:10:100);
    verifyEqual(testCase, cfg.frame_count, 100);
    verifyEqual(testCase, cfg.input_frames, 10);
    verifyEqual(testCase, cfg.holdout_frames, 90);
    verifyEqual(testCase, cfg.iterations, 3);
end
cfg = pilot_dataset_config('P01');
metadata = load(cfg.psf_path, 'x3objspace', 'CAindex');
zUm = double(metadata.x3objspace(:)) * 1e6;
selected = zeros(1,10);
for depth = 1:10
    hit = find(abs(zUm - cfg.z_um(depth)) < 1e-4);
    verifyNumElements(testCase, hit, 1);
    selected(depth) = hit;
end
m = matfile(cfg.psf_path);
hSize = size(m, 'H');
verifyEqual(testCase, double(metadata.CAindex(selected,:)), ...
    repmat([1,hSize(1)], 10, 1));
end

function testPrepareIsDeterministicAndSplitsAreDisjoint(testCase)
root = new_test_root();
cleanup = onCleanup(@() cleanup_root(root)); %#ok<NASGU>
first = pilot_prepare_sample('P01', root);
second = pilot_prepare_sample('P01', root);
verifyEqual(testCase, second.input_indices, first.input_indices);
verifyEqual(testCase, second.holdout_indices, first.holdout_indices);
verifyEqual(testCase, sort(first.input_indices(:))', 1:100);
for subset = 1:10
    input = first.input_indices(subset,:);
    holdout = first.holdout_indices(subset,:);
    verifyEmpty(testCase, intersect(input, holdout));
    verifyEqual(testCase, sort([input,holdout]), 1:100);
end
verifyNumElements(testCase, imfinfo(fullfile(first.cfg.sample_dir, ...
    'previews', 'ground_truth_float.tif')), 10);
end

function testHoldoutCannotChangeInputStatistics(testCase)
rng(20260904, 'twister');
frames = rand(13, 11, 100);
input = [2,5,9,13,27,40,54,71,83,99];
holdout = setdiff(1:100, input, 'stable');
meanBefore = mean(frames(:,:,input), 3);
varianceBefore = var(frames(:,:,input), 0, 3);
frames(:,:,holdout) = frames(:,:,holdout) + 1000 * rand(size(frames(:,:,holdout)));
verifyEqual(testCase, mean(frames(:,:,input), 3), meanBefore);
verifyEqual(testCase, var(frames(:,:,input), 0, 3), varianceBefore);
end

function testFrameAndSubsetHelpersWithStubSolver(testCase)
root = new_test_root();
cleanup = onCleanup(@() cleanup_root(root)); %#ok<NASGU>
cfg = pilot_dataset_config('P01', root);
mkdir(cfg.sample_dir);
mkdir(fullfile(cfg.sample_dir, 'sensor_frames'));
mkdir(fullfile(cfg.sample_dir, 'recon_frames'));
mkdir(fullfile(cfg.sample_dir, 'subsets'));
mkdir(fullfile(cfg.sample_dir, 'previews'));
input_indices = reshape(1:100, 10, 10).'; %#ok<NASGU>
holdout_indices = zeros(10,90); %#ok<NASGU>
for subset = 1:10
    holdout_indices(subset,:) = setdiff(1:100, input_indices(subset,:), 'stable');
end
save(fullfile(cfg.sample_dir, 'prepared.mat'), 'cfg', ...
    'input_indices', 'holdout_indices');
[yy,xx] = ndgrid(1:260,1:260);
for frame = 1:100
    sensor_legacy_float = mod(xx + 2*yy + frame, 29) / 28; %#ok<NASGU>
    sensor_legacy_float(1,1) = 1;
    sensor_legacy_uint8 = im2uint8(sensor_legacy_float); %#ok<NASGU>
    sensor_pre_detector = single(0.03*frame + sensor_legacy_float); %#ok<NASGU>
    schema_version = cfg.schema_version; %#ok<NASGU>
    dataset_id = cfg.dataset_id; %#ok<NASGU>
    sample_id = cfg.sample_id; %#ok<NASGU>
    frame_index = frame; %#ok<NASGU>
    save(fullfile(cfg.sample_dir, 'sensor_frames', ...
        sprintf('frame_%03d.mat', frame)), 'schema_version', 'dataset_id', ...
        'sample_id', 'frame_index', 'sensor_legacy_float', ...
        'sensor_legacy_uint8', 'sensor_pre_detector');
end
psf = struct('z_um', cfg.z_um);
frame = pilot_reconstruct_frame(cfg, psf, 1, @stub_reconstruct);
verifySize(testCase, frame.w_legacy_raw, [260,260,10]);
verifyClass(testCase, frame.w_legacy_raw, 'single');
subset = pilot_build_subset(cfg, psf, 1, @stub_reconstruct);
verifyEqual(testCase, subset.input_indices, 1:10);
verifyEqual(testCase, subset.legacy_variance_normalization, ...
    'N-1: var(sensor_legacy_float,0,3)');
verifyEqual(testCase, subset.input_legacy_variance_nminus1_float, ...
    var(cat_sensor(root, 1:10, 'sensor_legacy_float'), 0, 3));
paths = struct2cell(subset.preview_paths);
for index = 1:numel(paths)
    verifyNumElements(testCase, imfinfo(paths{index}), 10);
end
end

function [X, diagnostic] = stub_reconstruct(psf, sensor, iterations, mode)
X = zeros(size(sensor,1), size(sensor,2), numel(psf.z_um), 'single');
base = single(sensor);
for depth = 1:numel(psf.z_um), X(:,:,depth) = base * depth + single(1e-7); end
diagnostic = struct('mode', mode, 'iterations', iterations, ...
    'z_um', psf.z_um, 'nonfinite_count', 0);
end

function stack = cat_sensor(root, indices, field)
cfg = pilot_dataset_config('P01', root);
stack = zeros(260,260,numel(indices));
for position = 1:numel(indices)
    data = load(fullfile(cfg.sample_dir, 'sensor_frames', ...
        sprintf('frame_%03d.mat', indices(position))), field);
    stack(:,:,position) = data.(field);
end
end

function root = new_test_root()
base = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
    'data', '_pilot_test_tmp');
if ~isfolder(base), mkdir(base); end
root = tempname(base);
mkdir(root);
end

function cleanup_root(root)
if isfolder(root), rmdir(root, 's'); end
end
