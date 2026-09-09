function metrics = t04_axial_validate_truth(sample)
%T04_AXIAL_VALIDATE_TRUTH Check each local pair, not mixed global projections.
cfg=sample.cfg; g=sample.ground_truth; f=sample.ground_truth_fine;
assert(strcmp(cfg.sample_id,'T04') && strcmp(cfg.geometry_revision,'t04_axial_line_pairs_v2'), ...
    't04:Owner','Wrong sample/revision.');
assert(isa(g,'single') && isequal(size(g),[260,260,10]) && ...
    isa(f,'single') && isequal(size(f),[260,260,100]),'t04:Shape','Incorrect truth shapes.');
assert(all(isfinite(g(:))) && all(isfinite(f(:))) && min(g(:))>=0 && min(f(:))>=0, ...
    't04:Values','Invalid truth values.');
assert(abs(double(max(f(:)))-1)<1e-6,'t04:Scale','Expected one global normalization.');
massFine=sum(double(f(:)))*cfg.fine_dz_um; massCoarse=sum(double(g(:)))*10;
relative=abs(massFine-massCoarse)/massFine;
assert(relative<2e-6,'t04:Mass','Depth mass not conserved.');
for k=1:10
    selected=cfg.fine_z_um>=cfg.z_um(k)-5 & cfg.fine_z_um<cfg.z_um(k)+5;
    assert(isequal(g(:,:,k),mean(f(:,:,selected),3)),'t04:Binning','Incorrect slab averaging.');
end
border=max([max(f([1,end],:,:),[],'all'),max(f(:,[1,end],:),[],'all'),max(f(:,:,[1,end]),[],'all')]);
assert(border==0,'t04:Border','Support clipped by field boundaries.');
regions=t04_axial_regions(cfg);
assert(isequaln(regions,sample.meta.geometry),'t04:Metadata','Region metadata changed.');
ncols=numel(cfg.axial_board.separations_um);
covered=false(260,260); local=cell(size(regions)); totalLines=0; unitMass=[];
adjacentPairs=0;
for k=1:numel(regions)
    r=regions{k}; bounds=r.roi_bounds_xy_um;
    ix=find(cfg.x_um>=bounds(1) & cfg.x_um<=bounds(3));
    iy=find(cfg.y_um>=bounds(2) & cfg.y_um<=bounds(4));
    assert(~any(covered(iy,ix),'all'),'t04:ROI','Overlapping scoring ROIs.');
    covered(iy,ix)=true;
    roi=g(iy,ix,:); profile=reshape(sum(sum(double(roi),1),2),1,[]);
    occupied=find(profile>0); expected=find(ismember(cfg.z_um,r.z_um));
    assert(isequal(occupied,expected),'t04:Support','Region has a missing/extra depth plane.');
    totalLines=totalLines+numel(r.z_um);
    reference=roi(:,:,expected(1))/r.amplitudes(1);
    for j=1:numel(expected)
        actual=roi(:,:,expected(j))/r.amplitudes(j);
        assert(max(abs(actual-reference),[],'all')<2e-7,'t04:XYCoincidence','Paired XY masks differ.');
    end
    ratios=profile(expected)/profile(expected(1));
    assert(max(abs(ratios-r.amplitudes/r.amplitudes(1)))<2e-6,'t04:Ratio','Brightness ratio changed.');
    interior=[]; samplingCase='single_layer_control';
    fineROI=f(iy,ix,:); fineCC=bwconncomp(fineROI>0,6); nativeCC=bwconncomp(roi>0,6);
    assert(fineCC.NumObjects==numel(r.z_um),'t04:FineCount','Fine-geometry layers must be separate.');
    if numel(r.z_um)==2
        interior=cfg.z_um(cfg.z_um>r.z_um(1) & cfg.z_um<r.z_um(2));
        if isempty(interior)
            adjacentPairs=adjacentPairs+1;
            samplingCase='adjacent_slabs_no_valley_sample';
            assert(r.separation_um==10 && nativeCC.NumObjects==1, ...
                't04:Adjacency','A 10 um pair must occupy two touching native slabs.');
        else
            samplingCase='separated_with_interior_samples';
            assert(nativeCC.NumObjects==2 && all(profile(ismember(cfg.z_um,interior))==0), ...
                't04:Valley','Separated native layers must retain their empty interior samples.');
        end
    else
        assert(nativeCC.NumObjects==1,'t04:SingleCount','Single-layer control is not connected.');
    end
    unitMass(end+1)=sum(profile)/sum(r.amplitudes); %#ok<AGROW>
    local{k}=struct('region_id',r.region_id,'family',r.family,'z_um',r.z_um, ...
        'separation_um',r.separation_um,'amplitudes',r.amplitudes, ...
        'observed_mass_ratios',ratios,'axial_mass',profile, ...
        'roi_bounds_xy_um',bounds,'occupied_layer_count',numel(occupied), ...
        'native_interior_z_um',interior,'native_sampling_case',samplingCase, ...
        'native_component_count',nativeCC.NumObjects,'fine_component_count',fineCC.NumObjects);
end
outside=g; outside(repmat(covered,1,1,10))=0;
assert(~any(outside(:)),'t04:Outside','Unexpected fluorescence outside test regions.');
cc=bwconncomp(g>0,6); fineCC=bwconncomp(f>0,6);
assert(totalLines==7*ncols && fineCC.NumObjects==totalLines && ...
    cc.NumObjects==totalLines-adjacentPairs,'t04:Count','Unexpected fine/native component counts.');
assert((max(unitMass)-min(unitMass))/mean(unitMass)<2e-6, ...
    't04:Area','Pixel phase changed the integrated line area.');
mass=reshape(sum(sum(double(g),1),2),1,[]); p=mass/sum(mass);
metrics=struct('region_count',numel(regions),'pair_count',3*ncols, ...
    'single_control_count',ncols,'line_count',totalLines, ...
    'weak_pair_count',ncols,'separations_um',cfg.axial_board.separations_um, ...
    'adjacent_native_pair_count',adjacentPairs,'native_component_count',cc.NumObjects, ...
    'fine_component_count',fineCC.NumObjects, ...
    'mass_fraction',p,'occupied_z_um',cfg.z_um(p>0), ...
    'centroid_z_um',sum(p.*cfg.z_um),'fine_to_coarse_mass_relative_error',relative, ...
    'fine_peak',double(max(f(:))),'coarse_peak',double(max(g(:))), ...
    'theoretical_geometry_axial_fwhm_um',2*sqrt(2*log(2))*cfg.axial_board.axial_sigma_um, ...
    'unit_line_mass_relative_spread',(max(unitMass)-min(unitMass))/mean(unitMass), ...
    'local_regions',{local},'status','numeric_checks_passed_morphology_not_yet_approved');
end
