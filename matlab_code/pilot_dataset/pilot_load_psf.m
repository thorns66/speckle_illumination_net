function psf = pilot_load_psf(cfg)
%PILOT_LOAD_PSF Load exactly the ten protocol depths from the V7.3 PSF.
assert(isfile(cfg.psf_path), 'pilot:MissingPSF', 'Missing PSF: %s', cfg.psf_path);
meta = load(cfg.psf_path, 'x3objspace', 'zspacing', 'CAindex');
assert(all(isfield(meta, {'x3objspace','CAindex'})), 'pilot:PSFSchema', ...
    'PSF must contain x3objspace and CAindex.');
zAllUm = double(meta.x3objspace(:)) * 1e6;
indices = zeros(1, numel(cfg.z_um));
for k = 1:numel(cfg.z_um)
    hit = find(abs(zAllUm - cfg.z_um(k)) < 1e-4);
    assert(isscalar(hit), 'pilot:PSFDepth', ...
        'Depth %.6g um must have exactly one x3objspace match.', cfg.z_um(k));
    indices(k) = hit;
end
m = matfile(cfg.psf_path);
hSize = size(m, 'H');
htSize = size(m, 'Ht');
assert(numel(hSize) == 5 && isequal(hSize, htSize), 'pilot:PSFShape', ...
    'H/Ht must have identical five-dimensional shapes.');
assert(max(indices) <= hSize(5), 'pilot:PSFShape', 'Selected depth exceeds H/Ht.');
ca = double(meta.CAindex(indices, :));
assert(isequal(size(ca), [numel(indices), 2]), 'pilot:CAIndex', 'Invalid CAindex.');
% GPU routines use the full stored kernel and ignore CAindex. They match ACC
% only for full support. Refuse a silent operator change.
assert(all(ca(:,1) == 1) && all(ca(:,2) == hSize(1)) && hSize(1) == hSize(2), ...
    'pilot:CAIndex', 'Selected CAindex is cropped; GPU and ACC would differ.');
psf = struct;
psf.H = single(m.H(:,:,:,:,indices));
psf.Ht = single(m.Ht(:,:,:,:,indices));
psf.CAindex = ca;
psf.z_um = cfg.z_um;
psf.selected_indices_one_based = indices;
psf.source_path = cfg.psf_path;
psf.source_size = hSize;
psf.selected_size = size(psf.H);
psf.x3objspace_um = zAllUm;
psf.stored_zspacing_um = NaN;
if isfield(meta, 'zspacing'), psf.stored_zspacing_um = double(meta.zspacing) * 1e6; end
psf.zspacing_conflicts_with_x3objspace = numel(zAllUm) > 1 && ...
    abs(psf.stored_zspacing_um - median(diff(zAllUm))) > 1e-4;
end
