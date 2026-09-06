function pilot_write_tiff(path, volume)
%PILOT_WRITE_TIFF Atomically write Y-by-X-by-page single or uint16 data.
% No clipping, normalization, or change of floating-point sample values occurs.
validateattributes(volume, {'single', 'uint16'}, ...
    {'real', 'nonempty', 'nonsparse', 'finite'}, mfilename, 'volume');
assert(ndims(volume) <= 3, 'pilot:tiffShape', ...
    'TIFF input must have axes Y, X, page.');
assert((ischar(path) && isrow(path)) || ...
    (isstring(path) && isscalar(path)), 'pilot:tiffPath', ...
    'path must be a character row or string scalar.');
path = char(path);
[parentDir, ~, ~] = fileparts(path);
if isempty(parentDir), parentDir = pwd; end
assert(isfolder(parentDir), 'pilot:tiffParent', ...
    'Output directory does not exist: %s', parentDir);
assert(~isfolder(path), 'pilot:tiffPath', 'Output path is a directory: %s', path);
temporaryPath = [tempname(parentDir) '.tif'];
t = [];
if isa(volume, 'single')
    bits = 32;
    sampleFormat = Tiff.SampleFormat.IEEEFP;
else
    bits = 16;
    sampleFormat = Tiff.SampleFormat.UInt;
end
mode = 'w';
if numel(volume) * bits / 8 > 3.5 * 1024^3, mode = 'w8'; end
try
    t = Tiff(temporaryPath, mode);
    tags = struct('ImageLength', size(volume, 1), ...
        'ImageWidth', size(volume, 2), 'Photometric', Tiff.Photometric.MinIsBlack, ...
        'BitsPerSample', bits, 'SamplesPerPixel', 1, ...
        'SampleFormat', sampleFormat, 'PlanarConfiguration', Tiff.PlanarConfiguration.Chunky, ...
        'Compression', Tiff.Compression.None, 'RowsPerStrip', min(size(volume, 1), 32), ...
        'Software', 'MATLAB pilot_write_tiff; axes Y,X,page; no normalization');
    for page = 1:size(volume, 3)
        t.setTag(tags);
        t.write(volume(:, :, page));
        if page < size(volume, 3), t.writeDirectory(); end
    end
    t.close();
    t = [];
    [ok, message] = movefile(temporaryPath, path, 'f');
    assert(ok, 'pilot:tiffPublish', 'Could not publish TIFF: %s', message);
catch exception
    if ~isempty(t)
        try, t.close(); catch, end
    end
    if isfile(temporaryPath), delete(temporaryPath); end
    rethrow(exception);
end
end
