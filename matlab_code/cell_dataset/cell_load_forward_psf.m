function psf = cell_load_forward_psf(cfg)
%CELL_LOAD_FORWARD_PSF Load H only, avoiding an unnecessary host Ht copy.
meta=load(cfg.psf_path,'x3objspace','CAindex');
zAllUm=double(meta.x3objspace(:))*1e6; indices=zeros(1,numel(cfg.z_um));
for depth=1:numel(cfg.z_um)
    hit=find(abs(zAllUm-cfg.z_um(depth))<1e-4);
    assert(isscalar(hit),'cells:PSFDepth','Depth %g um is not unique.',cfg.z_um(depth));
    indices(depth)=hit;
end
m=matfile(cfg.psf_path); hSize=size(m,'H'); ca=double(meta.CAindex(indices,:));
assert(all(ca(:,1)==1)&&all(ca(:,2)==hSize(1)),'cells:PSFSupport','Cropped CAindex is unsupported.');
psf=struct('H',single(m.H(:,:,:,:,indices)),'CAindex',ca,'z_um',cfg.z_um, ...
    'selected_indices_one_based',indices,'source_path',cfg.psf_path);
end
