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
    assert(isequal(g(:,:,k),mean(f(:,:,selected),3)),'v3:Binning','Incorrect slab averaging.');
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
    otherwise
        error('v3:Kind','Unknown geometry kind.');
end
if ~strcmp(cfg.geometry_kind,'single_plane_chart')
    assert(numel(occupied)>=4 && all(diff(occupied)==1),'v3:Depth','Expected a continuous multi-depth object.');
    assert(~isequal(g(:,:,occupied(1)),g(:,:,occupied(end))),'v3:Extrusion','Repeated 2D truth.');
end
metrics = struct('mass_fraction',p,'occupied_z_um',cfg.z_um(occupied), ...
    'centroid_z_um',sum(p.*cfg.z_um),'fine_to_coarse_mass_relative_error',relative, ...
    'fine_peak',double(max(f(:))),'coarse_peak',double(max(g(:))), ...
    'border_max',double(border),'specific',specific, ...
    'status','numeric_checks_passed_morphology_not_yet_approved');
end
