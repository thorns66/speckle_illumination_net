function preview = pilot_legacy_preview(X, mode)
%PILOT_LEGACY_PREVIEW Reproduce the original two-stage uint16 display pipeline.
% mean: Reconstruction3D.m; taylor: Reconstruction3D_speckle.m sqrt output.
% Both gains use the whole volume, never individual depth planes. A caller
% handling separate frames calls this function once for each complete volume.
if isa(X, 'gpuArray'), X = gather(X); end
validateattributes(X, {'single', 'double'}, ...
    {'real', 'nonempty', 'nonsparse', 'finite', 'nonnegative'}, mfilename, 'X');
assert(ndims(X) <= 3, 'pilot:previewShape', 'X must have axes Y, X, Z.');
mode = validatestring(mode, {'mean', 'taylor'}, mfilename, 'mode');
maximum = max(X(:));
if maximum == 0
    % The old zero/zero path ultimately casts NaNs to zero. Make that case
    % explicit without introducing nonfinite intermediate data.
    preview = zeros(size(X), 'uint16');
    return;
end
Xvolume = uint16(round(65535 * (0.66 / maximum) * X));
XguessFinal = double(Xvolume);
if strcmp(mode, 'taylor'), XguessFinal = sqrt(XguessFinal); end
preview = uint16(round(65535 * XguessFinal / max(XguessFinal(:))));
end
