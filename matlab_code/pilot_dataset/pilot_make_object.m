function [object, meta] = pilot_make_object(sampleId, matlabRoot)
%PILOT_MAKE_OBJECT Deterministic, single-plane pilot objects on the native grid.
%   OBJECT is a 260-by-260 nonnegative double image with peak intensity one.
%   Depth is ground-truth metadata only; it must not select reconstruction PSFs.
%   P01 is the unchanged source target (apart from conversion/normalization).
%   P02--P05 are analytic phantoms, with no random draws or image resizing.

if nargin < 2 || isempty(matlabRoot)
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
end
sampleId = upper(char(sampleId));
sampling = 260;
[x, y] = meshgrid(1:sampling, 1:sampling);
object = zeros(sampling, sampling);
meta = struct('sample_id', sampleId, 'description', '', 'depth_um', [], ...
    'pixel_size_m', 4.5e-6 / 4, 'sampling', sampling, ...
    'generator_version', 'pilot_objects_v1', 'source_path', '', ...
    'normalization', 'divide by object maximum', ...
    'geometry_units', 'native object-grid pixels');

switch sampleId
    case 'P01'
        meta.description = 'Original USAF/star resolution target';
        meta.depth_um = 50;
        meta.source_path = fullfile(matlabRoot, 'usaf_260.bmp');
        assert(isfile(meta.source_path), 'pilot:MissingObject', ...
            'Missing original target: %s', meta.source_path);
        object = double(imread(meta.source_path));
        assert(ismatrix(object) && isequal(size(object), [sampling, sampling]), ...
            'pilot:ObjectShape', 'P01 must be the native 260-by-260 grayscale target.');

    case 'P02'
        meta.description = 'Line pairs with three orientations and three periods';
        meta.depth_um = 20;
        centers = [58, 130, 202];
        angles = [0, 45, 90];
        periods = [6, 10, 16];
        for row = 1:3
            theta = angles(row) * pi / 180;
            for col = 1:3
                dx = x - centers(col);
                dy = y - centers(row);
                u = dx * cos(theta) + dy * sin(theta);
                v = -dx * sin(theta) + dy * cos(theta);
                patch = abs(u) <= 24 & abs(v) <= 24;
                bars = mod(u + periods(col) / 4, periods(col)) < periods(col) / 2;
                object = max(object, double(patch & bars));
            end
        end
        meta.periods_px = periods;
        meta.normal_angles_deg = angles;
        meta.patch_centers_px = centers;

    case 'P03'
        meta.description = 'Sparse compact points and isolated short filaments';
        meta.depth_um = 80;
        points = [40,45; 85,38; 130,56; 188,39; 221,79; 48,114; ...
            107,128; 164,112; 207,144; 45,214; 112,206; 181,213];
        sigmas = [0.75, 1.1, 1.6];
        for k = 1:size(points, 1)
            sigma = sigmas(mod(k - 1, numel(sigmas)) + 1);
            d = hypot(x - points(k, 1), y - points(k, 2));
            object = max(object, compact_gaussian(d, sigma));
        end
        segments = [63,73,84,83; 113,87,137,87; 171,70,187,87; ...
            67,156,87,174; 126,163,127,186; 171,167,195,164; ...
            217,185,225,205];
        for k = 1:size(segments, 1)
            vertices = reshape(segments(k, :), 2, 2).';
            sigma = 0.8 + 0.25 * mod(k, 3);
            object = max(object, 0.8 * filament(x, y, vertices, sigma));
        end
        meta.point_centers_px = points;
        meta.point_sigmas_px = sigmas;
        meta.segment_endpoints_px = segments;

    case 'P04'
        meta.description = 'Curved crossing filaments with explicit local gaps';
        meta.depth_um = 30;
        t = (32:2:228).';
        curve1 = [t, 130 + 39 * sin((t - 32) / 196 * 2 * pi)];
        keep1 = ~(t > 72 & t < 86) & ~(t > 180 & t < 194);
        object = max(object, gapped_filament(x, y, curve1, keep1, 1.15));
        curve2 = [130 + 40 * sin((t - 32) / 196 * 2 * pi), t];
        keep2 = ~(t > 192 & t < 206);
        object = max(object, 0.8 * gapped_filament(x, y, curve2, keep2, 1.45));
        curve3 = [t, 45 + 0.78 * (t - 32) + 12 * sin((t - 32) / 196 * pi)];
        object = max(object, 0.65 * filament(x, y, curve3, 0.9));
        branch1 = [43,196; 62,191; 81,184; 96,174; 111,159];
        branch2 = [163,71; 183,58; 202,57; 218,65];
        object = max(object, 0.5 * filament(x, y, branch1, 1.0));
        object = max(object, 0.7 * filament(x, y, branch2, 1.2));
        meta.gap_intervals_px = {[72,86; 180,194], [192,206]};
        meta.filament_sigmas_px = [1.15, 1.45, 0.9, 1.0, 1.2];

    case 'P05'
        meta.description = 'Irregular membrane and rings with weak fine structure';
        meta.depth_um = 100;
        dx = x - 137;
        dy = y - 130;
        theta = atan2(dy, dx);
        radius = 57 + 7 * cos(3 * theta) + 4 * sin(5 * theta);
        object = compact_gaussian(abs(hypot(dx, dy) - radius), 1.6);
        ring1 = compact_gaussian(abs(hypot(x - 60, y - 61) - 22), 1.0);
        ring2 = compact_gaussian(abs(hypot((x - 210) / 1.1, y - 203) - 20), 0.9);
        object = max(object, 0.3 * ring1);
        object = max(object, 0.12 * ring2);
        t = linspace(-0.2 * pi, 1.3 * pi, 100).';
        arc = [135 + 31 * cos(t), 132 + 24 * sin(t)];
        object = max(object, 0.10 * filament(x, y, arc, 0.7));
        weakSpot = compact_gaussian(hypot(x - 139, y - 126), 1.0);
        object = max(object, 0.08 * weakSpot);
        meta.relative_feature_levels = [1, 0.3, 0.12, 0.10, 0.08];
        meta.membrane_center_px = [137, 130];

    otherwise
        error('pilot:UnknownSample', 'Unknown sample ID: %s. Expected P01--P05.', sampleId);
end

assert(all(isfinite(object(:))) && all(object(:) >= 0), ...
    'pilot:InvalidObject', 'Object values must be finite and nonnegative.');
peak = max(object(:));
assert(peak > 0, 'pilot:EmptyObject', 'Object must contain a nonzero structure.');
object = object / peak;
[rows, cols] = find(object > 0);
meta.support_bbox_yx = [min(rows), min(cols), max(rows), max(cols)];
meta.minimum_border_margin_px = min([min(rows) - 1, min(cols) - 1, ...
    sampling - max(rows), sampling - max(cols)]);
meta.nonzero_fraction = nnz(object) / numel(object);
meta.pre_normalization_peak = peak;
if ~strcmp(sampleId, 'P01')
    assert(meta.minimum_border_margin_px >= 16, ...
        'pilot:ClippedObject', 'Synthetic targets must have a zero-valued border margin.');
end
end

function result = compact_gaussian(distance, sigma)
% An explicit three-sigma support keeps the background exactly zero.
result = exp(-0.5 * (distance / sigma).^2);
result(distance > 3 * sigma) = 0;
end

function result = filament(x, y, vertices, sigma)
% Distance to a finite polyline gives a reproducible, rounded filament profile.
minDistance = inf(size(x));
for k = 1:size(vertices, 1) - 1
    p = vertices(k, :);
    v = vertices(k + 1, :) - p;
    s = ((x - p(1)) * v(1) + (y - p(2)) * v(2)) / sum(v.^2);
    s = min(max(s, 0), 1);
    minDistance = min(minDistance, hypot(x - p(1) - s * v(1), y - p(2) - s * v(2)));
end
result = compact_gaussian(minDistance, sigma);
end

function result = gapped_filament(x, y, vertices, keep, sigma)
% Draw disconnected runs independently: never bridge over a removed interval.
result = zeros(size(x));
edges = diff([false; keep(:); false]);
starts = find(edges == 1);
stops = find(edges == -1) - 1;
for k = 1:numel(starts)
    result = max(result, filament(x, y, vertices(starts(k):stops(k), :), sigma));
end
end
