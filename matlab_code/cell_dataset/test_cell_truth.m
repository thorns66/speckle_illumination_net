function tests = test_cell_truth
tests=functiontests(localfunctions);
end

function testAllFiveContinuousVolumes(testCase)
ids={'P06','P07','P08','P09','P10'};
for k=1:numel(ids)
    cfg=cell_dataset_config(ids{k}); sample=cell_make_truth(cfg);
    metrics=cell_validate_truth(sample);
    verifyGreaterThanOrEqual(testCase,numel(metrics.occupied_z_um),4);
    verifyLessThan(testCase,metrics.fine_to_coarse_mass_relative_error,2e-6);
    verifyEqual(testCase,size(sample.ground_truth),[260,260,10]);
    verifyEqual(testCase,size(sample.ground_truth_fine),[260,260,100]);
end
end

function testSinglePlaneIndependentControls(testCase)
ids={'V01','V02','T01','T02'}; depths=[40,70,60,90];
for k=1:numel(ids)
    sample=cell_make_truth(cell_dataset_config(ids{k}));
    metrics=cell_validate_truth(sample);
    verifyEqual(testCase,metrics.occupied_z_um,depths(k));
end
end

function testReproducibilityAndRngIsolation(testCase)
cfg=cell_dataset_config('P09'); before=rng;
a=cell_make_truth(cfg); after=rng;
verifyEqual(testCase,after,before);
b=cell_make_truth(cfg);
verifyEqual(testCase,a.ground_truth,b.ground_truth);
verifyEqual(testCase,a.ground_truth_fine,b.ground_truth_fine);
cfg.geometry_seed=cfg.geometry_seed+100;
c=cell_make_truth(cfg);
verifyFalse(testCase,isequal(a.ground_truth,c.ground_truth));
end

function testOutputDoesNotPretendToBeAcquisition(testCase)
cfg=cell_dataset_config('P06');
verifyEqual(testCase,cfg.stage,'truth_preview_only');
verifyEqual(testCase,cfg.iterations,3);
verifyEqual(testCase,cfg.z_um,10:10:100);
verifyEqual(testCase,cfg.fine_z_um,5.5:104.5);
verifyEqual(testCase,cfg.object_pixel_pitch_um,220/49/4,'AbsTol',1e-12);
end

function testWholeObjectSplits(testCase)
ids={'P06','P07','P08','P09','P10','V01','V02','T01','T02'};
expected={'train','test','train','validation','train', ...
    'validation','validation','test','test'};
seeds=zeros(1,numel(ids));
for k=1:numel(ids)
    cfg=cell_dataset_config(ids{k});
    verifyEqual(testCase,cell_dataset_split(ids{k}),expected{k}); seeds(k)=cfg.geometry_seed;
end
verifyEqual(testCase,numel(unique(seeds)),numel(ids));
end

function testValidationRejectsLayerRenormalization(testCase)
sample=cell_make_truth(cell_dataset_config('P06'));
g=sample.ground_truth;
for k=1:10
    peak=max(g(:,:,k),[],'all');
    if peak>0, g(:,:,k)=g(:,:,k)/peak; end
end
sample.ground_truth=g;
verifyError(testCase,@() cell_validate_truth(sample),'cells:Mass');
end

function testP08FilledScatteredNonTouchingBeads(testCase)
sample=cell_make_truth(cell_dataset_config('P08'));
geometry=sample.meta.geometry;
isSphere=cellfun(@(s) strcmp(s.kind,'solid_sphere'),geometry);
spheres=geometry(isSphere);
verifyEqual(testCase,numel(spheres),120);
verifyFalse(testCase,any(cellfun(@(s) ismember(s.kind,{'shell','tube','bead_chain'}),geometry)));
centers=zeros(numel(spheres),3); radii=zeros(numel(spheres),1);
for k=1:numel(spheres)
    ball=spheres{k};
    centers(k,:)=ball.center_xyz_um; radii(k)=ball.radius_um;
    verifyFalse(testCase,isfield(ball,'strand_id'));
    for fraction=[0,0.45]
        location=ball.center_xyz_um+[fraction*ball.radius_um,0,0];
        [~,ix]=min(abs(sample.cfg.x_um-location(1)));
        [~,iy]=min(abs(sample.cfg.y_um-location(2)));
        [~,iz]=min(abs(sample.cfg.fine_z_um-location(3)));
        expected=ball.amplitude/sample.meta.raw_global_peak;
        verifyGreaterThanOrEqual(testCase,double(sample.ground_truth_fine(iy,ix,iz)),0.99*expected);
    end
end
settings=sample.cfg.beads; nearPairs=0;
for k=2:numel(spheres)
    delta=centers(1:k-1,:)-centers(k,:);
    supportSum=radii(1:k-1)+radii(k)+2*settings.edge_width_um;
    gap=sqrt(sum(delta.^2,2))-supportSum;
    xyDistance=sqrt(sum(delta(:,1:2).^2,2));
    verifyGreaterThanOrEqual(testCase,min(gap),settings.minimum_surface_gap_um);
    verifyGreaterThanOrEqual(testCase,min(xyDistance./supportSum),settings.minimum_projected_radius_sum_fraction);
    nearPairs=nearPairs+nnz(xyDistance<supportSum+settings.xy_clear_gap_um);
end
verifyLessThanOrEqual(testCase,nearPairs,settings.maximum_projected_overlap_pairs);
verifyGreaterThan(testCase,max(centers(:,1))-min(centers(:,1)),220);
verifyGreaterThan(testCase,max(centers(:,2))-min(centers(:,2)),220);
verifyGreaterThan(testCase,max(centers(:,3))-min(centers(:,3)),45);
verifyGreaterThanOrEqual(testCase,min(radii),settings.radius_range_um(1));
verifyLessThanOrEqual(testCase,max(radii),settings.radius_range_um(2));
for quadrant=0:3
    groups=(centers(:,1)>145)+2*(centers(:,2)>145);
    verifyGreaterThanOrEqual(testCase,nnz(groups==quadrant),15);
end
end

function testP10SimplifiedConnectedCrossDepthNetwork(testCase)
sample=cell_make_truth(cell_dataset_config('P10'));
geometry=sample.meta.geometry;
networks=geometry(cellfun(@(s) strcmp(s.kind,'network_topology'),geometry));
verifyEqual(testCase,numel(networks),1);
net=networks{1}; nodes=net.nodes_xyz_um; edges=net.edge_indices_one_based;
G=graph(edges(:,1),edges(:,2),[],size(nodes,1));
verifyEqual(testCase,numel(unique(conncomp(G))),1);
verifyGreaterThan(testCase,size(nodes,1),300);
verifyGreaterThan(testCase,size(edges,1)-size(nodes,1)+1,100);
verifyGreaterThanOrEqual(testCase,size(edges,1),1200);
verifyLessThanOrEqual(testCase,size(edges,1),1600);
lengths=sqrt(sum((nodes(edges(:,1),:)-nodes(edges(:,2),:)).^2,2));
verifyGreaterThanOrEqual(testCase,min(lengths),sample.cfg.reticulum.collapse_edges_below_um);
verifyGreaterThan(testCase,net.simplification.short_edge_contractions,0);
verifyGreaterThan(testCase,max(nodes(:,1))-min(nodes(:,1)),200);
verifyGreaterThan(testCase,max(nodes(:,2))-min(nodes(:,2)),200);
verifyGreaterThan(testCase,max(nodes(:,3))-min(nodes(:,3)),60);
crossDepth=net.node_depth_bin(edges(:,1))~=net.node_depth_bin(edges(:,2));
verifyGreaterThan(testCase,nnz(crossDepth),60);
verifyFalse(testCase,any(cellfun(@(s) strcmp(s.kind,'perforated_sheet'),geometry)));
end

function testNewShapesReproduceWithoutChangingCallerRng(testCase)
for id={'P08','P10'}
    cfg=cell_dataset_config(id{1}); before=rng;
    first=cell_make_truth(cfg); after=rng;
    second=cell_make_truth(cfg);
    verifyEqual(testCase,after,before);
    verifyEqual(testCase,first.ground_truth,second.ground_truth);
    verifyEqual(testCase,first.ground_truth_fine,second.ground_truth_fine);
    verifyEqual(testCase,first.meta.geometry,second.meta.geometry);
end
end

function testOtherReviewedShapesRemainUnchanged(testCase)
cfg=cell_dataset_config('P06');
oldRoot=fullfile(cfg.repo_root,'data','matlab_cells_pilot_v2');
assumeTrue(testCase,isfile(fullfile(oldRoot,'P06','truth.mat')));
for id={'P06','P07','P09'}
    old=load(fullfile(oldRoot,id{1},'truth.mat'),'ground_truth','ground_truth_fine');
    fresh=cell_make_truth(cell_dataset_config(id{1}));
    verifyEqual(testCase,fresh.ground_truth,old.ground_truth);
    verifyEqual(testCase,fresh.ground_truth_fine,old.ground_truth_fine);
end
end
