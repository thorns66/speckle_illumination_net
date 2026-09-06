function tests=test_cell_simulation
tests=functiontests(localfunctions);
end

function testPropagationDeterministicAndOriginalAtReference(testCase)
before=rng;
[raw,normalized,meta]=cell_generate_illumination_3d(1234,2,[0,10,100]);
after=rng;
verifyEqual(testCase,after,before);
[repeat,repeatNormalized]=cell_generate_illumination_3d(1234,2,[0,10,100]);
verifyEqual(testCase,repeat,raw); verifyEqual(testCase,repeatNormalized,normalized);
verifySize(testCase,raw,[260,260,3,2]); verifyClass(testCase,raw,'single');
verifyTrue(testCase,all(isfinite(raw),'all')); verifyGreaterThan(testCase,min(raw,[],'all'),-eps('single'));
for frame=1:2, verifyEqual(testCase,max(normalized(:,:,:,frame),[],'all'),single(1)); end
reference=pilot_generate_illumination(1234,1);
relative=norm(double(raw(:,:,1,1)-single(reference(:,:,1))),'fro')/norm(reference(:,:,1),'fro');
verifyLessThan(testCase,relative,2e-6);
verifyEqual(testCase,meta.depth_coupling,'same complex pupil field at every z; not independent layers');
end

function testSimulationConfigAndApproval(testCase)
cfg=cell_simulation_config('P08');
verifyEqual(testCase,cfg.z_um,10:10:100);
verifyEqual(testCase,cfg.frame_count,100); verifyEqual(testCase,cfg.input_frames,10);
verifyEqual(testCase,cfg.holdout_frames,90); verifyEqual(testCase,cfg.iterations,3);
verifyEqual(testCase,cfg.truth_source_sha256,cell_geometry_hash());
approval=jsondecode(fileread(fullfile(cfg.output_root,'SIMULATION_APPROVED.json')));
verifyTrue(testCase,approval.approved); verifyEqual(testCase,approval.truth_source_sha256,cell_geometry_hash());
end

function testForwardDepthSumAndSingleLayerReduction(testCase)
% Tiny synthetic H checks depth summation without the 257 MB production PSF.
H=zeros(3,3,1,1,2,'single'); H(:,:,1,1,1)=single([0,1,0;1,2,1;0,1,0]);
H(:,:,1,1,2)=single([1,0,1;0,1,0;1,0,1]); CA=[1,3;1,3];
v=zeros(7,7,2,'single'); v(4,4,1)=2; v(3,5,2)=3;
actual=cell_forward_project_acc(H,v,CA);
expected=single(forwardProjectACC(H(:,:,:,:,1),v(:,:,1),CA(1,:)))+ ...
    single(forwardProjectACC(H(:,:,:,:,2),v(:,:,2),CA(2,:)));
verifyEqual(testCase,actual,expected);
v(:,:,2)=0;
verifyEqual(testCase,cell_forward_project_acc(H,v,CA), ...
    single(forwardProjectACC(H(:,:,:,:,1),v(:,:,1),CA(1,:))));
end
