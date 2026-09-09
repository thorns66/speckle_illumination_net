function tests = test_dataset_v3_truth
tests=functiontests(localfunctions);
end

function setupOnce(testCase)
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo,'matlab_code','cell_dataset'),fullfile(repo,'matlab_code','pilot_dataset'));
testCase.TestData.repo=repo;
end

function testAllShapesAndReproducibility(testCase)
for id={'T03','T04','V03'}
    cfg=dataset_v3_config(id{1}); before=rng;
    sample=dataset_v3_make_truth(cfg); after=rng;
    verifyEqual(testCase,after,before);
    again=dataset_v3_make_truth(cfg);
    verifyEqual(testCase,sample.ground_truth,again.ground_truth);
    verifyEqual(testCase,sample.ground_truth_fine,again.ground_truth_fine);
    metrics=dataset_v3_validate_truth(sample);
    verifyLessThan(testCase,metrics.fine_to_coarse_mass_relative_error,2e-6);
    verifyEqual(testCase,sample.cfg.sample_id,id{1});
    verifyFalse(testCase,sample.cfg.morphology_approved);
end
end

function testChartGeometry(testCase)
sample=dataset_v3_make_truth(dataset_v3_config('T03'));
metrics=dataset_v3_validate_truth(sample);
verifyEqual(testCase,metrics.occupied_z_um,50);
verifyEqual(testCase,metrics.specific.bar_count,36);
verifyEqual(testCase,metrics.specific.widths_px,[2,3,4,6,8,10]);
verifyLessThan(testCase,max(sample.ground_truth(:)),single(1));
end

function testSpatialNetwork(testCase)
sample=dataset_v3_make_truth(dataset_v3_config('T04'));
metrics=dataset_v3_validate_truth(sample);
verifyEqual(testCase,metrics.specific.cycle_rank,8);
verifyGreaterThan(testCase,metrics.specific.nonplanar_z_residual_rms_um,4);
verifyEqual(testCase,metrics.specific.centerline_z_range_um,[24,86]);
end

function testIndependentSolidBeads(testCase)
cfg=dataset_v3_config('V03'); sample=dataset_v3_make_truth(cfg);
metrics=dataset_v3_validate_truth(sample);
verifyEqual(testCase,metrics.specific.bead_count,60);
changed=cfg; changed.geometry_seed=cfg.geometry_seed+1;
other=dataset_v3_make_truth(changed);
verifyNotEqual(testCase,sample.ground_truth,other.ground_truth);
oldcfg=cell_dataset_config('P08'); old=cell_make_truth(oldcfg);
extract=@(g) cell2mat(cellfun(@(s) s.center_xyz_um(:)', ...
    g(cellfun(@(s) strcmp(s.kind,'solid_sphere'),g)),'UniformOutput',false)');
centers=extract(sample.meta.geometry); oldcenters=extract(old.meta.geometry);
verifyFalse(testCase,any(ismember(centers,oldcenters,'rows')));
verifyEqual(testCase,oldcfg.beads.count,120);
end

function testLayerRenormalizationRejected(testCase)
sample=dataset_v3_make_truth(dataset_v3_config('T04'));
for k=1:10
    p=max(sample.ground_truth(:,:,k),[],'all');
    if p>0, sample.ground_truth(:,:,k)=sample.ground_truth(:,:,k)/p; end
end
verifyError(testCase,@() dataset_v3_validate_truth(sample),'v3:Mass');
end

function testPolicyAndIds(testCase)
verifyError(testCase,@() dataset_v3_config('T01'),'v3:Sample');
verifyError(testCase,@() dataset_v3_config('P11'),'v3:Sample');
seeds=[];
for id={'T03','T04','V03'}
    cfg=dataset_v3_config(id{1}); seeds(end+1)=cfg.geometry_seed; %#ok<AGROW>
    verifyEqual(testCase,cfg.allowed_physical_gpu_indices,0);
    verifyEqual(testCase,cfg.required_gpu_model,'A40');
    verifyEqual(testCase,cfg.iterations,3);
    verifyEqual(testCase,cfg.preview_compute,'CPU_ONLY');
end
verifyEqual(testCase,numel(unique(seeds)),3);
verifyEqual(testCase,cell_dataset_split('T01'),'test');
verifyEqual(testCase,cell_dataset_split('P07'),'test');
verifyEqual(testCase,cell_dataset_split('P09'),'validation');
end
