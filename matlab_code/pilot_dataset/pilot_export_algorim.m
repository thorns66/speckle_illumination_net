function manifest = pilot_export_algorim(sampleDir, frameCount, zUm)
%PILOT_EXPORT_ALGORIM Export a frame stack per depth for Windows AlgoRIM.
% Inputs: recon_frames/frame_NNN.mat, with finite nonnegative single YXZ
% w_legacy_raw and w_physics_raw. All frames are validated before exporting.
% Output TIFF axes are YXF. The physics gain is shared by all frames/depths.
% A complete staged directory is published with rollback on rename failure.
validateattributes(frameCount, {'numeric'}, ...
    {'scalar', 'integer', 'positive', 'finite'}, mfilename, 'frameCount');
validateattributes(zUm, {'numeric'}, ...
    {'vector', 'integer', 'positive', 'finite', 'nonempty'}, mfilename, 'zUm');
assert(numel(unique(zUm)) == numel(zUm), 'pilot:exportDepths', ...
    'Depth coordinates must be unique to avoid overwriting TIFF files.');
assert(isfolder(sampleDir), 'pilot:exportSource', 'Sample directory does not exist.');
[ok, attributes] = fileattrib(sampleDir);
assert(ok, 'pilot:exportSource', 'Cannot resolve sample directory.');
sampleDir = attributes.Name;
zUm = double(zUm(:)');
depthCount = numel(zUm);
sourceDir = fullfile(sampleDir, 'recon_frames');
assert(isfolder(sourceDir), 'pilot:exportSource', 'Missing recon_frames directory.');
sourceFiles = arrayfun(@(f) sprintf('frame_%03d.mat', f), 1:frameCount, ...
    'UniformOutput', false);
depthFiles = arrayfun(@(z) sprintf('depth_%03dum.tif', z), zUm, 'UniformOutput', false);
volumeSize = [];
physicsMax = 0;
legacyMax = zeros(1, frameCount);
legacyFirstGain = zeros(1, frameCount);
legacySecondGain = zeros(1, frameCount);
sourceBytes = zeros(1, frameCount);
sourceModified = zeros(1, frameCount);
for frame = 1:frameCount
    path = fullfile(sourceDir, sourceFiles{frame});
    [data, volumeSize] = read_frame(path, volumeSize, depthCount);
    metadata = dir(path);
    sourceBytes(frame) = metadata.bytes;
    sourceModified(frame) = metadata.datenum;
    physicsMax = max(physicsMax, double(max(data.w_physics_raw(:))));
    legacyMaximum = max(data.w_legacy_raw(:));
    legacyMax(frame) = double(legacyMaximum);
    if legacyMaximum > 0
        firstGain = 0.66 / legacyMaximum;
        quantizedMaximum = uint16(round(65535 * firstGain * legacyMaximum));
        legacyFirstGain(frame) = double(firstGain);
        legacySecondGain(frame) = 65535 / double(quantizedMaximum);
    end
end
clear data;
if physicsMax > 0, commonGain = 65535 / physicsMax; else, commonGain = 1; end
exportDir = fullfile(sampleDir, 'algorim');
previousFiles = {};
if isfolder(exportDir)
    previousFiles = validate_owned_export(exportDir, sampleDir);
elseif isfile(exportDir)
    error('pilot:exportOwnership', 'algorim is an existing file; refusing to overwrite it.');
end
stageDir = [tempname(sampleDir) '_algorim_staging'];
backupDir = [tempname(sampleDir) '_algorim_previous'];
legacyDir = fullfile(stageDir, 'legacy_per_frame_normalized');
physicsDir = fullfile(stageDir, 'physics_common_scale');
stacks = cell(2, depthCount);
try
    mkdir(legacyDir);
    mkdir(physicsDir);
    mode = 'w';
if prod(volumeSize(1:2)) * frameCount * 2 > 3.5 * 1024^3, mode = 'w8'; end
tags = struct('ImageLength', volumeSize(1), 'ImageWidth', volumeSize(2), ...
    'Photometric', Tiff.Photometric.MinIsBlack, 'BitsPerSample', 16, ...
    'SamplesPerPixel', 1, 'SampleFormat', Tiff.SampleFormat.UInt, ...
    'PlanarConfiguration', Tiff.PlanarConfiguration.Chunky, ...
    'Compression', Tiff.Compression.None, 'RowsPerStrip', min(volumeSize(1), 32), ...
    'Software', 'MATLAB pilot_export_algorim; axes Y,X,frame');
for depth = 1:depthCount
    stacks{1, depth} = Tiff(fullfile(legacyDir, depthFiles{depth}), mode);
    stacks{2, depth} = Tiff(fullfile(physicsDir, depthFiles{depth}), mode);
end
for frame = 1:frameCount
    path = fullfile(sourceDir, sourceFiles{frame});
    metadata = dir(path);
    assert(isscalar(metadata) && metadata.bytes == sourceBytes(frame) && ...
        metadata.datenum == sourceModified(frame), 'pilot:exportSourceChanged', ...
        'Source changed while exporting: %s', path);
    data = read_frame(path, volumeSize, depthCount);
    assert(double(max(data.w_physics_raw(:))) <= physicsMax, ...
        'pilot:exportSourceChanged', 'A source exceeds the validated global maximum.');
    legacy = pilot_legacy_preview(data.w_legacy_raw, 'mean');
    physics = uint16(round(double(data.w_physics_raw) .* commonGain));
    for depth = 1:depthCount
        write_page(stacks{1, depth}, legacy(:, :, depth), tags, frame < frameCount);
        write_page(stacks{2, depth}, physics(:, :, depth), tags, frame < frameCount);
    end
end
clear data legacy physics;
close_stacks();
manifest = struct;
manifest.schema_version = 1;
manifest.producer = 'pilot_export_algorim/1';
manifest.sample_dir = sampleDir;
manifest.input_schema = struct('folder', 'recon_frames', ...
    'files', {sourceFiles}, 'fields', {{'w_legacy_raw', 'w_physics_raw'}}, ...
    'class', 'single', 'axes', 'YXZ', 'size_yxz', volumeSize, ...
    'finite_nonnegative_required', true);
manifest.frame_count = frameCount;
manifest.frame_indices = 1:frameCount;
manifest.z_um = zUm;
manifest.output_axes = 'YXF';
manifest.output_page_semantics = 'Page f is frame f; one TIFF per physical depth.';
manifest.output_class = 'uint16';
manifest.depth_files = depthFiles;
manifest.legacy = struct('folder', 'legacy_per_frame_normalized', ...
    'source_field', 'w_legacy_raw', 'normalization_domain', 'all YXZ voxels, per frame', ...
    'frame_maxima', legacyMax, 'first_stage_gain', legacyFirstGain, ...
    'second_stage_gain', legacySecondGain, ...
    'zero_frame_policy', 'Both gains and every output voxel are zero.', ...
    'formula', ['Q = uint16(round(65535*(0.66/max(X(:)))*X)); ' ...
    'output = uint16(round(65535*double(Q)/max(double(Q(:)))))']);
manifest.physics = struct('folder', 'physics_common_scale', ...
    'source_field', 'w_physics_raw', 'global_max_yxzf', physicsMax, ...
    'common_uint16_gain', commonGain, 'normalization_domain', 'all YXZF voxels', ...
    'formula', 'uint16(round(double(w_physics_raw)*common_uint16_gain))', ...
    'inverse_scale', 'double(TIFF)/common_uint16_gain', ...
    'max_quantization_error_raw_units', 0.5 / commonGain);
manifest.source_file_bytes = sourceBytes;
manifest.source_file_modified_datenum = sourceModified;
manifest.warnings = { ...
    'Legacy per-frame normalization changes temporal means/variances and is a regression display export, not a statistics-preserving dataset.', ...
    'Physics TIFF uses one gain for every frame and depth; uint16 quantization remains. Original single arrays are authoritative.', ...
    'AlgoRIM output is an auxiliary reference, not ground truth. Run AlgoRIM separately on each depth stack.'};
save(fullfile(stageDir, 'manifest.mat'), 'manifest', '-v7');
write_json(fullfile(stageDir, 'manifest.json'), manifest);
% Rename on the same filesystem. Keep the previous completed export until
% the new directory is published, and restore it if publication fails.
if isfolder(exportDir)
    [ok, message] = movefile(exportDir, backupDir);
    assert(ok, 'pilot:exportPublish', 'Cannot stage previous export: %s', message);
end
[ok, message] = movefile(stageDir, exportDir);
if ~ok
    if isfolder(backupDir), movefile(backupDir, exportDir); end
    error('pilot:exportPublish', 'Cannot publish complete export: %s', message);
end
if isfolder(backupDir)
    % Only remove files enumerated by the validated previous manifest. Do
    % not recursively delete a pre-existing user directory.
    remove_owned_export(backupDir, previousFiles);
end
catch exception
    close_stacks();
    cleanup_staging();
    rethrow(exception);
end

    function close_stacks()
        for index = 1:numel(stacks)
            if ~isempty(stacks{index})
                try, stacks{index}.close(); catch, end
                stacks{index} = [];
            end
        end
    end

    function cleanup_staging()
        for index = 1:numel(stacks)
            if ~isempty(stacks{index})
                try, stacks{index}.close(); catch, end
            end
        end
        if isfolder(stageDir), rmdir(stageDir, 's'); end
        if isfolder(backupDir) && ~isfolder(exportDir)
            movefile(backupDir, exportDir);
        end
    end
end

function [data, volumeSize] = read_frame(path, volumeSize, depthCount)
assert(isfile(path), 'pilot:exportMissingFrame', 'Missing source frame: %s', path);
variables = whos('-file', path);
variableNames = {variables.name};
assert(all(ismember({'w_legacy_raw', 'w_physics_raw'}, variableNames)), ...
    'pilot:exportSchema', 'Missing required raw reconstruction field in %s', path);
data = load(path, 'w_legacy_raw', 'w_physics_raw');
fields = {'w_legacy_raw', 'w_physics_raw'};
for index = 1:numel(fields)
    name = fields{index};
    assert(isfield(data, name), 'pilot:exportSchema', 'Missing %s in %s', name, path);
    X = data.(name);
    validateattributes(X, {'single'}, ...
        {'real', 'finite', 'nonnegative', 'nonempty', 'nonsparse'}, mfilename, name);
    shape = [size(X, 1), size(X, 2), size(X, 3)];
    assert(ndims(X) <= 3 && shape(3) == depthCount, 'pilot:exportShape', ...
        'Expected YXZ data with %d depths in %s.', depthCount, path);
    if isempty(volumeSize), volumeSize = shape; end
    assert(isequal(shape, volumeSize), 'pilot:exportShape', ...
        'Source volume shape mismatch in %s.', path);
end
end

function write_page(t, page, tags, hasNext)
t.setTag(tags);
t.write(page);
if hasNext, t.writeDirectory(); end
end

function write_json(path, manifest)
fid = fopen(path, 'w', 'n', 'UTF-8');
assert(fid ~= -1, 'pilot:exportManifest', 'Cannot create manifest: %s', path);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
payload = jsonencode(manifest, 'PrettyPrint', true);
count = fprintf(fid, '%s\n', payload);
assert(count > 0, 'pilot:exportManifest', 'Could not write manifest.');
end

function files = validate_owned_export(exportDir, sampleDir)
manifestPath = fullfile(exportDir, 'manifest.mat');
assert(isfile(manifestPath), 'pilot:exportOwnership', ...
    'Existing algorim lacks this exporter''s ownership manifest.');
old = load(manifestPath, 'manifest');
assert(isfield(old, 'manifest') && isfield(old.manifest, 'producer') && ...
    strcmp(old.manifest.producer, 'pilot_export_algorim/1') && ...
    isfield(old.manifest, 'sample_dir') && strcmp(old.manifest.sample_dir, sampleDir) && ...
    isfield(old.manifest, 'depth_files'), 'pilot:exportOwnership', ...
    'Existing algorim was not produced by this exporter for this sample.');
depthFiles = old.manifest.depth_files;
assert(iscellstr(depthFiles) && all(cellfun(@(x) ...
    ~isempty(regexp(x, '^depth_[0-9]+um\.tif$', 'once')), depthFiles)), ...
    'pilot:exportOwnership', 'Invalid filenames in the existing export manifest.');
files = {'manifest.mat', 'manifest.json'};
folders = {'legacy_per_frame_normalized', 'physics_common_scale'};
for folder = 1:numel(folders)
    files = [files, cellfun(@(f) fullfile(folders{folder}, f), ...
        depthFiles(:)', 'UniformOutput', false)]; %#ok<AGROW>
end
allowed = [files, folders];
existing = dir(fullfile(exportDir, '**', '*'));
for index = 1:numel(existing)
    if ismember(existing(index).name, {'.', '..'}), continue; end
    absolute = fullfile(existing(index).folder, existing(index).name);
    relative = absolute(numel(exportDir) + 2:end);
    assert(ismember(relative, allowed), 'pilot:exportOwnership', ...
        'Refusing to replace an export containing an unrecognized path: %s', relative);
end
end

function remove_owned_export(path, files)
for index = 1:numel(files)
    owned = fullfile(path, files{index});
    if isfile(owned), delete(owned); end
end
rmdir(fullfile(path, 'legacy_per_frame_normalized'));
rmdir(fullfile(path, 'physics_common_scale'));
rmdir(path);
end
