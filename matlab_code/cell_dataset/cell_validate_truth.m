function metrics = cell_validate_truth(sample)
%CELL_VALIDATE_TRUTH Check values, physical mass and native/fine consistency.
g=sample.ground_truth; f=sample.ground_truth_fine; cfg=sample.cfg;
assert(isa(g,'single')&&isequal(size(g),[260,260,10]),'cells:Shape','Invalid coarse GT.');
assert(isa(f,'single')&&isequal(size(f),[260,260,100]),'cells:Shape','Invalid fine GT.');
assert(all(isfinite(g(:)))&&all(isfinite(f(:)))&&min(f(:))>=0&&min(g(:))>=0, ...
    'cells:Values','Truth must be nonnegative and finite.');
assert(abs(double(max(f(:)))-1)<1e-6,'cells:Scale','Expected one global normalization.');
fineMass=sum(double(f(:)))*cfg.fine_dz_um;
coarseMass=sum(double(g(:)))*10;
relativeError=abs(fineMass-coarseMass)/fineMass;
assert(relativeError<2e-6,'cells:Mass','Z binning failed mass conservation.');
for k=1:10
    selected=cfg.fine_z_um>=cfg.z_um(k)-5 & cfg.fine_z_um<cfg.z_um(k)+5;
    expected=mean(f(:,:,selected),3);
    assert(isequal(expected,g(:,:,k)),'cells:Binning','Incorrect coarse layer.');
end
mass=squeeze(sum(sum(double(g),1),2))'; p=mass/sum(mass);
occupied=find(p>1e-8);
is3d=ismember(cfg.sample_id,{'P06','P07','P08','P09','P10'});
if is3d
    assert(numel(occupied)>=4&&all(diff(occupied)==1), ...
        'cells:Support','3D sample must occupy consecutive depth bins.');
else
    assert(numel(occupied)==1,'cells:Support','Legacy control must have one occupied plane.');
end
border=max([max(g(1,:,:),[],'all'),max(g(end,:,:),[],'all'), ...
    max(g(:,1,:),[],'all'),max(g(:,end,:),[],'all')]);
assert(border==0,'cells:Border','Lateral support is clipped by FOV.');
assert(~isequal(g(:,:,occupied(1)),g(:,:,occupied(end)))||~is3d, ...
    'cells:Extrusion','3D target must not be a repeated 2D image.');
metrics=struct('mass_fraction',p,'occupied_z_um',cfg.z_um(occupied), ...
    'centroid_z_um',sum(p.*cfg.z_um),'peak_z_um',cfg.z_um(find(p==max(p),1)), ...
    'fine_to_coarse_mass_relative_error',relativeError, ...
    'max_coarse_voxel_fraction',double(max(g(:)))/sum(double(g(:))), ...
    'coarse_peak',double(max(g(:))),'fine_peak',double(max(f(:))), ...
    'physical_integrated_mass',coarseMass*cfg.object_pixel_pitch_um^2, ...
    'nonzero_coarse_voxels',nnz(g),'lateral_border_max',double(border), ...
    'status','numeric_checks_passed_morphology_not_yet_approved');
end
