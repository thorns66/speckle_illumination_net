function sample = cell_make_truth(cfg)
%CELL_MAKE_TRUTH Continuous physical-XYZ geometry, voxelized in MATLAB.
% Fine geometry has native XY pitch and 1 um Z quadrature. Ten-plane GT is
% the mean density in each 10 um slab. Fine Z is NOT a finer optical model.
% All geometry is sampled from one fixed seed, with the caller RNG restored.
oldRng = rng; cleanup = onCleanup(@() rng(oldRng)); %#ok<NASGU>
rng(cfg.geometry_seed, 'twister');
x = cfg.x_um; y = cfg.y_um; z = cfg.fine_z_um;
v = zeros(numel(y),numel(x),numel(z),'single');
geometry = {};
switch cfg.sample_id
    case 'P06'
        trunks = { ...
            [34,65,24;75,90,33;119,123,47;160,137,63;205,172,77;251,191,85], ...
            [42,224,78;80,193,69;119,153,63;161,119,59;208,89,47;248,53,36], ...
            [58,38,61;100,51,63;136,78,64;171,86,63;220,105,66]};
        for j=1:3
            addtube(trunks{j},1.45+0.2*j,1-0.18*(j-1));
        end
        branches = { ...
            [75,90,33;72,120,40;45,147,50], ...
            [119,123,47;121,93,43;143,57,31], ...
            [160,137,63;183,126,73;210,135,84], ...
            [205,172,77;216,206,71;247,224,62], ...
            [80,193,69;64,166,63;36,159,58], ...
            [119,153,63;148,176,54;160,211,38], ...
            [208,89,47;232,110,34;252,142,25], ...
            [136,78,64;142,105,77;134,129,84]};
        for j=1:numel(branches), addtube(branches{j},0.85+0.35*rand,0.35+0.4*rand); end
    case 'P07'
        centers = [79,86,40;198,100,54;136,207,65];
        orientations = [0.55,2.2,-0.25];
        for c=1:3
            for j=1:13
                t = linspace(-1,1,13)';
                a = orientations(c)+0.5*(rand-0.5);
                lateral = (j-7)*3.1;
                u = t*(31+15*rand);
                w = lateral+5*sin(pi*t+0.3*j);
                p = [centers(c,1)+u*cos(a)-w*sin(a), ...
                    centers(c,2)+u*sin(a)+w*cos(a), ...
                    centers(c,3)+9*t+4*sin(pi*t+0.7*j)];
                if mod(j,5)==0
                    addtube(p(1:5,:),0.95,0.5); addtube(p(8:end,:),0.95,0.5);
                else
                    addtube(p,0.85+0.65*rand,0.28+0.65*rand);
                end
            end
        end
    case 'P08'
        % Random positions throughout XYZ, not samples of a curve or grid.
        % Physical supports cannot touch. Most XY silhouettes are separated;
        % only a small bounded number of mild projected overlaps is allowed.
        [beadCenters,beadRadii,beadLayout]=build_scattered_beads(cfg.beads);
        for beadIndex=1:cfg.beads.count
            addsolidball(beadCenters(beadIndex,:),beadRadii(beadIndex), ...
                0.62+0.34*rand,cfg.beads.edge_width_um,beadIndex);
        end
        geometry{end+1}=beadLayout;
    case 'P09'
        centers = [78,85,42;200,92,54;134,205,65];
        for c=1:3
            for j=1:17
                a=2*pi*rand; radius=8+25*sqrt(rand);
                origin=centers(c,:)+[radius*cos(a),radius*sin(a),8*(rand-0.5)];
                theta=2*pi*rand; lengthUm=8+17*rand;
                t=linspace(-1,1,11)';
                u=lengthUm*t/2; w=(2+4*rand)*sin(pi*(t+1)/2);
                p=[origin(1)+u*cos(theta)-w*sin(theta), ...
                    origin(2)+u*sin(theta)+w*cos(theta), ...
                    origin(3)+5*t+2*sin(pi*t)];
                sig=0.65+0.55*rand; amp=0.25+0.65*rand;
                addtube(p,sig,amp);
                if mod(j,4)==0
                    node=p(6,:);
                    addtube([node;node+[5,-5,4];node+[9,-7,8]],sig*0.8,amp*0.8);
                end
            end
        end
    case 'P10'
        % Broad continuous polygonal reticulum, visually inspired by the
        % user reference. A volumetric Voronoi edge graph has no prescribed
        % depth sheets; this is neither four islands nor an extruded 2D image.
        [reticulumNodes,reticulumEdges,nodeLayers,reticulumAudit]=build_reticulum_graph(cfg.reticulum);
        for edgeIndex=1:size(reticulumEdges,1)
            firstNode=reticulumNodes(reticulumEdges(edgeIndex,1),:);
            secondNode=reticulumNodes(reticulumEdges(edgeIndex,2),:);
            bendScale=min(1,norm(secondNode-firstNode)/8);
            edgeMid=(firstNode+secondNode)/2+bendScale*cfg.reticulum.midpoint_jitter_um*(rand(1,3)-0.5);
            tubeSigma=cfg.reticulum.tube_sigma_range_um(1)+diff(cfg.reticulum.tube_sigma_range_um)*rand;
            addtube([firstNode;edgeMid;secondNode],tubeSigma,0.45+0.45*rand);
        end
        geometry{end+1}=struct('kind','network_topology','nodes_xyz_um',reticulumNodes, ...
            'edge_indices_one_based',reticulumEdges,'node_depth_bin',nodeLayers, ...
            'reference','user-requested simpler continuous reticular morphology', ...
            'simplification',reticulumAudit, ...
            'construction','volumetric 3D Voronoi edges; fewer more-separated sites, short-edge contraction, smaller bends; no prescribed depth surfaces');
    case 'V01'
        for row=1:3
            for col=1:3
                center=[57+75*(col-1),59+74*(row-1),40];
                period=[6,10,16]; a=[0.2,0.8,1.4]; angle=a(row);
                for j=-2:2
                    t=linspace(-23,23,17)'; u=j*period(col)+1.8*sin(t/15);
                    p=[center(1)+u*cos(angle)-t*sin(angle), ...
                        center(2)+u*sin(angle)+t*cos(angle),40+0*t];
                    addtube(p,min(1.2,period(col)/5),0.3+0.25*row);
                end
            end
        end
    case 'V02'
        centers=[64,71,70;128,86,70;208,68,70;89,191,70;185,190,70];
        for j=1:5
            t=linspace(0,2*pi,80)'; r=21+3*sin(3*t+0.4*j);
            p=[centers(j,1)+r.*cos(t),centers(j,2)+0.8*r.*sin(t),70+0*t];
            addtube(p,0.85+0.1*j,0.25+0.12*j);
        end
        addtube([130,176,70;136,187,70;131,205,70],0.8,0.22);
    case 'T01'
        for j=1:22
            p=[35+215*rand,35+215*rand,60];
            addtube([p;p+[0.01,0,0]],0.7+0.6*rand,0.25+0.7*rand);
            if mod(j,3)==0, addtube([p+[6,1,0];p+[6.01,1,0]],0.9,0.5); end
        end
        for j=1:8
            p=[45+190*rand,45+190*rand,60]; a=2*pi*rand;
            addtube([p;p+[14*cos(a),14*sin(a),0]],1,0.6);
        end
    case 'T02'
        for j=1:6
            t=linspace(35,253,60)';
            p=[t,48+30*j+15*sin(t/40+0.7*j),90+0*t];
            addtube(p(1:23,:),0.85+0.08*j,0.25+0.1*j);
            addtube(p(27:end,:),0.85+0.08*j,0.25+0.1*j);
        end
        addtube([73,55,90;91,93,90;113,141,90;141,193,90;179,244,90],1.05,0.8);
end
globalPeak=double(max(v(:))); assert(globalPeak>0,'cells:Empty','Empty truth.');
fine=v/globalPeak;
coarse=zeros(numel(y),numel(x),numel(cfg.z_um),'single');
for k=1:numel(cfg.z_um)
    selected=z>=cfg.z_um(k)-5 & z<cfg.z_um(k)+5;
    coarse(:,:,k)=mean(fine(:,:,selected),3);
end
meta=struct('normalization','one global fine-volume maximum; no per-layer rescaling', ...
    'raw_global_peak',globalPeak,'geometry_units','um', ...
    'geometry_axis_order','XYZ','array_axis_order','YXZ', ...
    'coarse_voxel_semantics','average fluorophore density in centered 10 um slabs', ...
    'fine_grid_semantics','geometry quadrature only; not a finer PSF or optical reconstruction', ...
    'compositing','maximum union of fluorescent geometric supports', ...
    'biological_scope','scale-adapted morphology phantoms, not ultrastructural cell truth', ...
    'geometry', {geometry});
sample=struct('cfg',cfg,'ground_truth',coarse,'ground_truth_fine',fine,'meta',meta);

    function addtube(control,sigma,amplitude)
        dist=[0;cumsum(sqrt(sum(diff(control,1,1).^2,2)))];
        assert(all(diff(dist)>0),'cells:Curve','Duplicate control vertices.');
        query=linspace(0,dist(end),max(2,ceil(dist(end)/0.7)+1));
        points=interp1(dist,control,query,'pchip');
        for ii=1:size(points,1)-1
            segStart=points(ii,:); segDelta=points(ii+1,:)-segStart;
            low=min(segStart,segStart+segDelta)-3*sigma;
            high=max(segStart,segStart+segDelta)+3*sigma;
            ix=find(x>=low(1)&x<=high(1)); iy=find(y>=low(2)&y<=high(2)); iz=find(z>=low(3)&z<=high(3));
            if isempty(ix)||isempty(iy)||isempty(iz), continue; end
            [xx,yy,zz]=meshgrid(x(ix),y(iy),z(iz));
            projectionT=((xx-segStart(1))*segDelta(1)+(yy-segStart(2))*segDelta(2)+ ...
                (zz-segStart(3))*segDelta(3))/sum(segDelta.^2);
            projectionT=min(max(projectionT,0),1);
            r2=(xx-segStart(1)-projectionT*segDelta(1)).^2+ ...
                (yy-segStart(2)-projectionT*segDelta(2)).^2+ ...
                (zz-segStart(3)-projectionT*segDelta(3)).^2;
            density=amplitude*exp(-r2/(2*sigma^2)); density(r2>9*sigma^2)=0;
            v(iy,ix,iz)=max(v(iy,ix,iz),single(density));
        end
        geometry{end+1}=struct('kind','tube','control_xyz_um',control, ...
            'sigma_um',sigma,'amplitude',amplitude); %#ok<AGROW>
    end

    function addshell(center,radii,sigma,amplitude,phase)
        ix=find(abs(x-center(1))<radii(1)*1.12+3*sigma);
        iy=find(abs(y-center(2))<radii(2)*1.12+3*sigma);
        iz=find(abs(z-center(3))<radii(3)*1.12+3*sigma);
        [xx,yy,zz]=meshgrid(x(ix)-center(1),y(iy)-center(2),z(iz)-center(3));
        q=sqrt((xx/radii(1)).^2+(yy/radii(2)).^2+(zz/radii(3)).^2);
        grad=sqrt((xx/radii(1)^2).^2+(yy/radii(2)^2).^2+(zz/radii(3)^2).^2)./max(q,1e-9);
        perturb=0.045*sin(3*atan2(yy,xx)+phase).*sin(pi*zz/radii(3));
        distance=abs(q-1-perturb)./max(grad,1/max(radii));
        density=amplitude*exp(-distance.^2/(2*sigma^2)); density(distance>3*sigma)=0;
        v(iy,ix,iz)=max(v(iy,ix,iz),single(density));
        geometry{end+1}=struct('kind','shell','center_xyz_um',center, ...
            'radii_xyz_um',radii,'sigma_um',sigma,'amplitude',amplitude,'phase',phase);
    end

    function addsolidball(ballCenter,ballRadius,ballAmplitude,edgeWidth,beadId)
        ix=find(abs(x-ballCenter(1))<=ballRadius+edgeWidth);
        iy=find(abs(y-ballCenter(2))<=ballRadius+edgeWidth);
        iz=find(abs(z-ballCenter(3))<=ballRadius+edgeWidth);
        [xx,yy,zz]=meshgrid(x(ix)-ballCenter(1),y(iy)-ballCenter(2),z(iz)-ballCenter(3));
        radialDistance=sqrt(xx.^2+yy.^2+zz.^2);
        edgePosition=min(max((radialDistance-(ballRadius-edgeWidth))/(2*edgeWidth),0),1);
        density=ballAmplitude*0.5*(1+cos(pi*edgePosition));
        density(radialDistance>=ballRadius+edgeWidth)=0;
        v(iy,ix,iz)=max(v(iy,ix,iz),single(density));
        geometry{end+1}=struct('kind','solid_sphere','center_xyz_um',ballCenter, ...
            'radius_um',ballRadius,'amplitude',ballAmplitude,'edge_width_um',edgeWidth, ...
            'bead_index',beadId,'interior','uniform_filled');
    end

    function addsheet(center,radii,sigma,amplitude,phase)
        ix=find(abs(x-center(1))<radii(1)); iy=find(abs(y-center(2))<radii(2));
        iz=find(abs(z-center(3))<12);
        [xx,yy,zz]=meshgrid(x(ix)-center(1),y(iy)-center(2),z(iz)-center(3));
        surface=0.12*xx+0.08*yy+1.8*sin(xx/8+phase).*cos(yy/7);
        distance=abs(zz-surface);
        inside=(xx/radii(1)).^2+(yy/radii(2)).^2<1;
        holes=(xx+6).^2+(yy-2).^2<3.5^2 | (xx-6).^2+(yy+3).^2<2.5^2;
        density=amplitude*exp(-distance.^2/(2*sigma^2));
        density(~inside|holes|distance>3*sigma)=0;
        v(iy,ix,iz)=max(v(iy,ix,iz),single(density));
        geometry{end+1}=struct('kind','perforated_sheet','center_xyz_um',center, ...
            'radii_xy_um',radii,'sigma_um',sigma,'amplitude',amplitude,'phase',phase);
    end
end

function [centers,radii,audit]=build_scattered_beads(settings)
centers=zeros(settings.count,3); radii=zeros(settings.count,1);
overlapPairs=0; attempts=0;
for bead=1:settings.count
    radius=settings.radius_range_um(1)+diff(settings.radius_range_um)*rand;
    accepted=false;
    for trial=1:50000
        attempts=attempts+1;
        candidate=[settings.center_xy_range_um(1)+diff(settings.center_xy_range_um)*rand(1,2), ...
            settings.center_z_range_um(1)+diff(settings.center_z_range_um)*rand];
        offsets=centers(1:bead-1,:)-candidate;
        distance3=sqrt(sum(offsets.^2,2)); distanceXY=sqrt(sum(offsets(:,1:2).^2,2));
        supportSum=radii(1:bead-1)+radius+2*settings.edge_width_um;
        if any(distance3<supportSum+settings.minimum_surface_gap_um), continue; end
        projected=distanceXY<supportSum+settings.xy_clear_gap_um;
        % Only the final few placements may overlap in projection, never in
        % 3D. At most one neighbour can overlap each such sphere.
        if any(projected)
            if bead<=settings.count-settings.maximum_projected_overlap_pairs || ...
                    nnz(projected)>1 || overlapPairs>=settings.maximum_projected_overlap_pairs || ...
                    any(distanceXY<settings.minimum_projected_radius_sum_fraction*supportSum)
                continue;
            end
        end
        centers(bead,:)=candidate; radii(bead)=radius;
        overlapPairs=overlapPairs+nnz(projected); accepted=true; break;
    end
    assert(accepted,'cells:BeadPacking','Cannot place all separated spheres; do not silently reduce count or gap.');
end
audit=struct('kind','scattered_bead_layout','bead_count',settings.count, ...
    'sampling','uniform random XYZ proposals with hard minimum separation; no paths, rings or grid', ...
    'settings',settings,'near_projected_pairs',overlapPairs,'proposal_count',attempts);
end

function [allNodes,allEdges,layerLabels,audit]=build_reticulum_graph(settings)
% Neighbouring Delaunay tetrahedron circumcentres are true 3D Voronoi edges.
% Uniform 3D sites avoid the laminar bias of stacking 2D Voronoi diagrams.
sites=zeros(settings.site_count,3); accepted=0; attempts=0;
while accepted<size(sites,1) && attempts<20000
    attempts=attempts+1;
    candidate=[13+264*rand,13+264*rand,5+100*rand];
    if all(sum((sites(1:accepted,:)-candidate).^2,2)>=settings.site_minimum_spacing_um^2)
        accepted=accepted+1; sites(accepted,:)=candidate;
    end
end
assert(accepted==settings.site_count,'cells:ReticulumSites','Failed to sample separated 3D sites.');
[gx,gy,gz]=ndgrid([-40,330],[-40,330],[-40,150]);
sites=[sites;gx(:),gy(:),gz(:)];
dt=delaunayTriangulation(sites);
vertices=circumcenter(dt); adjacency=neighbors(dt);
source=repmat((1:size(vertices,1))',1,size(adjacency,2));
validPairs=isfinite(adjacency);
candidateEdges=unique(sort([source(validPairs),adjacency(validPairs)],2),'rows');
inside=all(isfinite(vertices),2) & all(vertices>=settings.centerline_bounds_um(1,:) & ...
    vertices<=settings.centerline_bounds_um(2,:),2);
candidateEdges=candidateEdges(all(inside(candidateEdges),2),:);
valid=find(inside); remap=zeros(size(vertices,1),1); remap(valid)=1:numel(valid);
allNodes=vertices(valid,:); allEdges=remap(candidateEdges);
G=graph(allEdges(:,1),allEdges(:,2),[],size(allNodes,1));
component=conncomp(G); counts=accumarray(component(:),1); [~,largest]=max(counts);
keep=component==largest;
allEdges=allEdges(all(keep(allEdges),2),:);
remap=zeros(size(allNodes,1),1); remap(keep)=1:nnz(keep);
allEdges=unique(sort(remap(allEdges),2),'rows'); allNodes=allNodes(keep,:);
edgesBefore=size(allEdges,1); nodesBefore=size(allNodes,1);
% Contract short edges rather than cutting random links. This preserves
% connectivity and replaces tiny crowded corners by shared junctions.
nodeWeights=ones(size(allNodes,1),1); contractions=0;
while true
    lengths=sqrt(sum((allNodes(allEdges(:,1),:)-allNodes(allEdges(:,2),:)).^2,2));
    [shortest,index]=min(lengths);
    if shortest>=settings.collapse_edges_below_um, break; end
    a=allEdges(index,1); b=allEdges(index,2);
    allNodes(a,:)=(nodeWeights(a)*allNodes(a,:)+nodeWeights(b)*allNodes(b,:))/(nodeWeights(a)+nodeWeights(b));
    nodeWeights(a)=nodeWeights(a)+nodeWeights(b);
    allEdges(allEdges==b)=a;
    allEdges=unique(sort(allEdges(allEdges(:,1)~=allEdges(:,2),:),2),'rows');
    contractions=contractions+1;
end
used=unique(allEdges(:)); remap=zeros(size(allNodes,1),1); remap(used)=1:numel(used);
allEdges=remap(allEdges); allNodes=allNodes(used,:);
audit=struct('settings',settings,'nodes_before_contraction',nodesBefore, ...
    'edges_before_contraction',edgesBefore,'short_edge_contractions',contractions, ...
    'node_count',size(allNodes,1),'edge_count',size(allEdges,1), ...
    'cycle_rank',size(allEdges,1)-size(allNodes,1)+1,'minimum_edge_length_um',min(lengths));
layerLabels=floor((allNodes(:,3)-20)/20)+1; % audit bins only, NOT generation surfaces
G=graph(allEdges(:,1),allEdges(:,2),[],size(allNodes,1));
assert(numel(unique(conncomp(G)))==1,'cells:ReticulumConnectivity','3D network must be connected.');
end
