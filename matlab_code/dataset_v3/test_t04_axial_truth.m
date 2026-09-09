function tests = test_t04_axial_truth
tests=functiontests(localfunctions);
end

function setupOnce(~)
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo,'matlab_code','cell_dataset'),fullfile(repo,'matlab_code','pilot_dataset'));
end

function testExactDepthMatrix(testCase)
cfg=t04_axial_config(); regions=t04_axial_regions(cfg);
expected={[20,30],[20,40],[20,50],[20,60], ...
    [50,60],[50,70],[50,80],[50,90], ...
    [30,40],[30,50],[30,60],[30,70],30,50,70,90};
verifyEqual(testCase,numel(regions),16);
verifyEqual(testCase,cfg.axial_board.separations_um,[10,20,30,40]);
for k=1:16
    verifyEqual(testCase,regions{k}.z_um,expected{k});
    if k>=9 && k<=12, verifyEqual(testCase,regions{k}.amplitudes,[1,0.5]); end
end
verifyEqual(testCase,cfg.axial_board.line_length_um,40);
verifyEqual(testCase,cfg.axial_board.line_width_um,6);
end

function testReproducibilityAndNoRngChange(testCase)
cfg=t04_axial_config(); before=rng;
a=t04_axial_make_truth(cfg); verifyEqual(testCase,rng,before);
b=t04_axial_make_truth(cfg);
verifyEqual(testCase,a.ground_truth,b.ground_truth);
verifyEqual(testCase,a.ground_truth_fine,b.ground_truth_fine);
end

function testLocalSupportsAndWeakRatios(testCase)
sample=t04_axial_make_truth(t04_axial_config()); m=t04_axial_validate_truth(sample);
verifyEqual(testCase,[m.region_count,m.pair_count,m.single_control_count,m.line_count],[16,12,4,28]);
for k=9:12, verifyEqual(testCase,m.local_regions{k}.observed_mass_ratios,[1,0.5],'AbsTol',1e-7); end
verifyLessThan(testCase,m.unit_line_mass_relative_spread,2e-6);
verifyEqual(testCase,m.occupied_z_um,20:10:90);
end

function testTenMicronPairsHaveNoInventedNativeValley(testCase)
s=t04_axial_make_truth(t04_axial_config()); m=t04_axial_validate_truth(s);
verifyEqual(testCase,m.adjacent_native_pair_count,3);
verifyEqual(testCase,[m.native_component_count,m.fine_component_count],[25,28]);
for k=[1,5,9]
    r=m.local_regions{k};
    verifyEqual(testCase,r.separation_um,10);
    verifyEqual(testCase,r.occupied_layer_count,2);
    verifyEmpty(testCase,r.native_interior_z_um);
    verifyEqual(testCase,r.native_sampling_case,'adjacent_slabs_no_valley_sample');
    verifyEqual(testCase,[r.native_component_count,r.fine_component_count],[1,2]);
    verifyEqual(testCase,diff(find(r.axial_mass>0)),1);
end
for k=[2:4,6:8,10:12]
    r=m.local_regions{k};
    verifyNotEmpty(testCase,r.native_interior_z_um);
    verifyEqual(testCase,r.native_sampling_case,'separated_with_interior_samples');
end
end

function testMassAndSlabConsistency(testCase)
s=t04_axial_make_truth(t04_axial_config()); m=t04_axial_validate_truth(s);
verifyLessThan(testCase,m.fine_to_coarse_mass_relative_error,2e-6);
verifyEqual(testCase,size(s.ground_truth),[260,260,10]);
verifyEqual(testCase,size(s.ground_truth_fine),[260,260,100]);
end

function testRejectsLayerNormalization(testCase)
s=t04_axial_make_truth(t04_axial_config());
for k=1:10
    p=max(s.ground_truth(:,:,k),[],'all');
    if p>0, s.ground_truth(:,:,k)=s.ground_truth(:,:,k)/p; end
end
verifyError(testCase,@() t04_axial_validate_truth(s),'t04:Mass');
end

function testRejectsLateralShortcut(testCase)
s=t04_axial_make_truth(t04_axial_config()); cfg=s.cfg;
selected=cfg.fine_z_um>=35 & cfg.fine_z_um<45;
s.ground_truth_fine(:,:,selected)=circshift(s.ground_truth_fine(:,:,selected),[0,2,0]);
s.ground_truth(:,:,4)=mean(s.ground_truth_fine(:,:,selected),3);
verifyError(testCase,@() t04_axial_validate_truth(s),'t04:XYCoincidence');
end

function testOldMeshAndOtherSamplesRemainAvailable(testCase)
old=dataset_v3_config('T04'); chart=dataset_v3_config('T03'); beads=dataset_v3_config('V03');
verifyEqual(testCase,old.geometry_kind,'spatial_network');
verifyEqual(testCase,chart.geometry_kind,'single_plane_chart');
verifyEqual(testCase,beads.beads.count,60);
cfg=t04_axial_config();
verifyEqual(testCase,cfg.sample_id,'T04'); verifyEqual(testCase,cfg.split,'test');
verifyEqual(testCase,cfg.allowed_physical_gpu_indices,0);
verifyEqual(testCase,cfg.required_gpu_model,'A40');
verifyEqual(testCase,cfg.preview_compute,'CPU_ONLY');
verifyFalse(testCase,cfg.morphology_approved);
end
