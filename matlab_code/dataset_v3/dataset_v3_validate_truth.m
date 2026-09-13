function metrics = dataset_v3_validate_truth(sample)
%DATASET_V3_VALIDATE_TRUTH Keep legacy validation and approved hashes intact.
cfg = sample.cfg; g = sample.ground_truth; f = sample.ground_truth_fine;
assert(isa(g,'single') && isequal(size(g),[260,260,10]),'v3:Shape','Invalid native GT.');
assert(isa(f,'single') && isequal(size(f),[260,260,100]),'v3:Shape','Invalid fine GT.');
assert(all(isfinite(g(:))) && all(isfinite(f(:))) && min(g(:))>=0 && min(f(:))>=0, ...
    'v3:Values','Nonfinite or negative truth.');
assert(abs(double(max(f(:)))-1)<1e-6,'v3:Scale','Expected one global normalization.');
relative = abs(sum(double(f(:)))*cfg.fine_dz_um-sum(double(g(:)))*10)/sum(double(f(:)));
assert(relative<2e-6,'v3:Mass','Mass is not conserved.');
for k = 1:10
    selected = cfg.fine_z_um>=cfg.z_um(k)-5 & cfg.fine_z_um<cfg.z_um(k)+5;
    binned=mean(f(:,:,selected),3);
    if strcmp(cfg.geometry_kind,'nonoverlapping_deformed_cell_section')
        assert(max(abs(double(g(:,:,k))-double(binned)),[],'all')<2e-7, ...
            'v3:Binning','P12 slab averaging exceeds the cross-language FP32 tolerance.');
    else
        assert(isequal(g(:,:,k),binned),'v3:Binning','Incorrect slab averaging.');
    end
end
mass = squeeze(sum(sum(double(g),1),2))'; p = mass/sum(mass); occupied = find(p>1e-8);
border = max([max(f([1,end],:,:),[],'all'),max(f(:,[1,end],:),[],'all'), ...
    max(f(:,:,[1,end]),[],'all')]);
assert(border==0,'v3:Border','Support is clipped by the field of view.');
geometry = sample.meta.geometry;
specific = struct();
switch cfg.geometry_kind
    case 'single_plane_chart'
        assert(isequal(cfg.z_um(occupied),50),'v3:Depth','Chart must occupy only 50 um.');
        assert(numel(geometry)==12,'v3:Chart','Expected twelve three-bar groups.');
        mask = max(g,[],3)>0; expectedMask = false(260,260);
        for k = 1:numel(geometry)
            group = geometry{k}; b = group.bars_xy_one_based; w = group.width_px;
            if strcmp(group.orientation,'horizontal')
                assert(all(b(:,3)-b(:,1)+1==5*w) && all(b(:,4)-b(:,2)+1==w) && ...
                    all(diff(b(:,2))==2*w),'v3:Chart','Invalid horizontal bars.');
            else
                assert(all(b(:,4)-b(:,2)+1==5*w) && all(b(:,3)-b(:,1)+1==w) && ...
                    all(diff(b(:,1))==2*w),'v3:Chart','Invalid vertical bars.');
            end
            for j=1:3, expectedMask(b(j,2):b(j,4),b(j,1):b(j,3))=true; end
        end
        assert(isequal(mask,expectedMask),'v3:Chart','Extra labels, fiducials or missing bars in truth.');
        cc = bwconncomp(mask,4);
        assert(cc.NumObjects==36,'v3:Chart','Bars touch or are missing.');
        specific = struct('bar_count',36,'group_count',12,'widths_px',[2,3,4,6,8,10], ...
            'widths_um',[2,3,4,6,8,10]*cfg.object_pixel_pitch_um);
    case 'spatial_network'
        net = geometry{1}; nodes = net.nodes_xyz_um; edges = net.edge_indices_one_based;
        G = graph(edges(:,1),edges(:,2),[],size(nodes,1));
        assert(size(nodes,1)==15 && size(edges,1)==22,'v3:Network','Incorrect topology.');
        assert(numel(unique(conncomp(G)))==1,'v3:Network','Disconnected network.');
        assert(rank(nodes-mean(nodes,1))==3,'v3:Network','Network is planar.');
        for rail=1:3
            assert(all(diff(nodes((rail-1)*5+(1:5),3))>0),'v3:Network','Depth must increase along each rail.');
        end
        plane = [ones(size(nodes,1),1),nodes(:,1:2)];
        residual = nodes(:,3)-plane*(plane\nodes(:,3));
        specific = struct('node_count',15,'edge_count',22,'cycle_rank',8, ...
            'centerline_z_range_um',[min(nodes(:,3)),max(nodes(:,3))], ...
            'nonplanar_z_residual_rms_um',sqrt(mean(residual.^2)), ...
            'tube_fwhm_um',2*sqrt(2*log(2))*cfg.network.tube_sigma_um);
    case 'scattered_spheres'
        spheres = geometry(cellfun(@(s) strcmp(s.kind,'solid_sphere'),geometry));
        assert(numel(spheres)==60,'v3:Beads','Expected exactly sixty solid spheres.');
        centers = cell2mat(cellfun(@(s) s.center_xyz_um(:)',spheres,'UniformOutput',false)');
        radii = cellfun(@(s) s.radius_um,spheres)';
        settings = cfg.beads; near = 0; minimum = inf;
        for k=2:60
            delta = centers(1:k-1,:)-centers(k,:);
            support = radii(1:k-1)+radii(k)+2*settings.edge_width_um;
            gaps = sqrt(sum(delta.^2,2))-support;
            minimum = min(minimum,min(gaps)); xy = sqrt(sum(delta(:,1:2).^2,2));
            assert(all(gaps>=settings.minimum_surface_gap_um),'v3:Beads','Spheres touch.');
            assert(all(xy>=settings.minimum_projected_radius_sum_fraction*support), ...
                'v3:Beads','Excessive projected overlap.');
            near = near+nnz(xy<support+settings.xy_clear_gap_um);
        end
        assert(near<=6,'v3:Beads','Too many near projected pairs.');
        assert(all(radii>=5.8 & radii<=7.8),'v3:Beads','Unexpected radii.');
        specific = struct('bead_count',60,'near_projected_pairs',near, ...
            'minimum_3d_surface_gap_um',minimum,'center_z_range_um',[min(centers(:,3)),max(centers(:,3))]);
    case 'nonoverlapping_deformed_cell_section'
        assert(strcmp(cfg.sample_id,'P12') && strcmp(cfg.split,'train'), ...
            'v3:RootCells','The reviewed root-cell section must be P12 in train.');
        assert(strcmp(cfg.geometry_version,'root_native_pixel_v4') && ...
            cfg.cell_count==18 && numel(geometry)==cfg.cell_count, ...
            'v3:RootCells','Expected exactly 18 reviewed native-pixel-calibrated cells.');
        assert(isequal(double(cfg.allowed_truth_z_um),[40,50,60]) && ...
            isequal(double(cfg.z_um(occupied)),[40,50,60]), ...
            'v3:RootCells','P12 must occupy only 40, 50 and 60 um.');
        assert(cfg.lateral_quadrature_samples_per_pixel==3 && ...
            cfg.local_slice_full_support_thickness_um==14, ...
            'v3:RootCells','P12 sampling or thickness differs from the reviewed truth.');
        m=sample.metrics;
        assert(m.cell_count==18 && m.filled_cell_overlap_subpixels==0 && ...
            m.projected_cell_overlap_subpixels==0 && m.maximum_cell_ownership==1 && ...
            logical(m.all_cells_and_lumens_connected) && ...
            logical(m.all_individual_lumens_dark) && ...
            logical(m.contacts_are_complementary_boundaries) && ...
            logical(m.cell_region_fixed_in_xy_across_depth) && ...
            m.out_of_allowed_layers_nonzero_voxels==0, ...
            'v3:RootCells','P12 non-overlap, lumen or depth audit failed.');
        c=m.calibration;
        assert(~logical(c.real_resizing_applied) && logical(c.no_forward_or_network_run) && ...
            c.accepted_real_object_ring_count==25 && ...
            abs(c.real_object_diameter_px.median-44)<1e-9 && ...
            abs(c.synthetic_wall_peak_diameter_px.median-44)<1e-9, ...
            'v3:RootCells','P12 native-pixel size calibration changed.');
        assert(all(diff(occupied)==1) && ~isequal(g(:,:,occupied(1)),g(:,:,occupied(end))), ...
            'v3:RootCells','P12 must retain its reviewed thin three-layer profile.');
        specific = struct('cell_count',double(m.cell_count), ...
            'touching_pairs',double(m.touching_pairs), ...
            'geometry_sampling_pitch_um',double(m.geometry_sampling_pitch_um), ...
            'occupied_z_um',double(cfg.z_um(occupied)));
    otherwise
        error('v3:Kind','Unknown geometry kind.');
end
if ~ismember(cfg.geometry_kind,{'single_plane_chart','nonoverlapping_deformed_cell_section'})
    assert(numel(occupied)>=4 && all(diff(occupied)==1),'v3:Depth','Expected a continuous multi-depth object.');
    assert(~isequal(g(:,:,occupied(1)),g(:,:,occupied(end))),'v3:Extrusion','Repeated 2D truth.');
end
metrics = struct('mass_fraction',p,'occupied_z_um',cfg.z_um(occupied), ...
    'centroid_z_um',sum(p.*double(cfg.z_um)),'fine_to_coarse_mass_relative_error',relative, ...
    'fine_peak',double(max(f(:))),'coarse_peak',double(max(g(:))), ...
    'border_max',double(border),'specific',specific, ...
    'status','numeric_checks_passed_morphology_not_yet_approved');
end
