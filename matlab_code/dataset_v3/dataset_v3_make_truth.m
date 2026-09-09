function sample = dataset_v3_make_truth(cfg)
%DATASET_V3_MAKE_TRUTH Exact native-grid chart and continuous XYZ segments.
old = rng; cleanup = onCleanup(@() rng(old)); %#ok<NASGU>
rng(cfg.geometry_seed,'twister');
if strcmp(cfg.sample_id,'V03')
    % Reuse the frozen P08 algorithm, NOT its existing geometry or positions.
    % Only this local generator adapter uses the template ID; the published
    % config and every output artifact are owned by V03.
    template = cfg; template.sample_id = 'P08';
    sample = cell_make_truth(template);
    sample.cfg = cfg;
    sample.meta.generator_template = 'cell_make_truth/P08; independent seed and count';
    return;
end
v = zeros(260,260,100,'single'); geometry = {};
switch cfg.sample_id
    case 'T03'
        mask = zeros(260,260,'single'); c = cfg.chart;
        for row = 1:3
            for col = 1:2
                w = c.widths_px(row,col); side = 5*w;
                totalWidth = 2*side+c.pair_gap_px;
                x0 = round(c.pair_center_x_px(col)-(totalWidth-1)/2);
                y0 = round(c.pair_center_y_px(row)-(side-1)/2);
                for orientation = 1:2
                    xStart = x0+(orientation-1)*(side+c.pair_gap_px);
                    bars = zeros(3,4);
                    for j = 0:2
                        if orientation == 1
                            bounds = [xStart,y0+2*j*w,xStart+side-1,y0+2*j*w+w-1];
                        else
                            bounds = [xStart+2*j*w,y0,xStart+2*j*w+w-1,y0+side-1];
                        end
                        mask(bounds(2):bounds(4),bounds(1):bounds(3)) = 1;
                        bars(j+1,:) = bounds;
                    end
                    orientations = {'horizontal','vertical'};
                    geometry{end+1} = struct('kind','three_bar_group', ...
                        'orientation',orientations{orientation},'width_px',w, ...
                        'width_um',w*cfg.object_pixel_pitch_um,'gap_px',w, ...
                        'length_px',side,'row',row,'column',col, ...
                        'bbox_xy_one_based',[xStart,y0,xStart+side-1,y0+side-1], ...
                        'bars_xy_one_based',bars,'depth_um',c.depth_um); %#ok<AGROW>
                end
            end
        end
        dz = cfg.fine_z_um-c.depth_um;
        profile = exp(-dz.^2/(2*c.axial_sigma_um^2));
        profile(abs(dz)>3*c.axial_sigma_um) = 0;
        v = mask.*reshape(single(profile),1,1,[]);
    case 'T04'
        s = cfg.network; nodes = zeros(15,3); edges = zeros(22,2); e = 0;
        for rail = 1:3
            for station = 1:5
                index = (rail-1)*5+station;
                nodes(index,:) = [s.station_xy_um(station)+s.rail_offsets_xy_um(rail,:), ...
                    s.rail_depths_um(rail,station)];
                if station<5, e=e+1; edges(e,:)=[index,index+1]; end
                if rail<3, e=e+1; edges(e,:)=[index,index+5]; end
            end
        end
        for index = 1:size(edges,1)
            first = nodes(edges(index,1),:); last = nodes(edges(index,2),:);
            sigma = s.tube_sigma_um;
            lo = min(first,last)-3*sigma; hi = max(first,last)+3*sigma;
            ix = find(cfg.x_um>=lo(1) & cfg.x_um<=hi(1));
            iy = find(cfg.y_um>=lo(2) & cfg.y_um<=hi(2));
            iz = find(cfg.fine_z_um>=lo(3) & cfg.fine_z_um<=hi(3));
            [x,y,z] = meshgrid(cfg.x_um(ix),cfg.y_um(iy),cfg.fine_z_um(iz));
            delta = last-first;
            t = ((x-first(1))*delta(1)+(y-first(2))*delta(2)+(z-first(3))*delta(3))/sum(delta.^2);
            t = min(max(t,0),1);
            distance2 = (x-first(1)-t*delta(1)).^2+(y-first(2)-t*delta(2)).^2+(z-first(3)-t*delta(3)).^2;
            density = s.amplitude*exp(-distance2/(2*sigma^2));
            density(distance2>9*sigma^2) = 0;
            v(iy,ix,iz) = max(v(iy,ix,iz),single(density));
        end
        geometry = {struct('kind','network_topology','nodes_xyz_um',nodes, ...
            'edge_indices_one_based',edges,'tube_sigma_um',s.tube_sigma_um, ...
            'amplitude',s.amplitude,'cycle_rank',8, ...
            'construction','three noncoplanar rails with ten sparse cross-depth connections')};
    otherwise
        error('v3:Sample','Unsupported sample ID.');
end
peak = double(max(v(:))); assert(peak>0,'v3:Empty','Empty object.');
fine = v/peak; coarse = zeros(260,260,10,'single');
for layer = 1:10
    selected = cfg.fine_z_um>=cfg.z_um(layer)-5 & cfg.fine_z_um<cfg.z_um(layer)+5;
    coarse(:,:,layer) = mean(fine(:,:,selected),3);
end
meta = struct('normalization','one global fine-volume peak; no per-layer normalization', ...
    'raw_global_peak',peak,'geometry_units','um','geometry_axis_order','XYZ', ...
    'array_axis_order','YXZ','coarse_voxel_semantics','average density in centered 10 um slabs', ...
    'fine_grid_semantics','1 um geometry quadrature; not optical resolution', ...
    'compositing','maximum union; no intersection brightening','geometry',{geometry});
sample = struct('cfg',cfg,'ground_truth',coarse,'ground_truth_fine',fine,'meta',meta);
end
