function tests=test_dataset_v3_migration
tests=functiontests(localfunctions);
end

function testMetadataChangesButAllNumericsStayIdentical(testCase)
a=struct('sample_id','T01','cfg',struct('sample_id','T01','split','test', ...
    'sample_dir','/old/T01','output_root','/old','seed',2028090508), ...
    'path','/old/T01/previews/a.tif','untouched_path','/oldish/T01', ...
    'arrays',{{single(reshape(1:24,2,3,4)),uint16(1:9),[1,3,9]}});
[b,changed]=dataset_v3_rewrite_metadata(a,'T01','P11','train',{'/old'},'/new');
verifyTrue(testCase,changed); verifyEqual(testCase,b.sample_id,'P11');
verifyEqual(testCase,b.cfg.split,'train'); verifyEqual(testCase,b.cfg.sample_dir,'/new/P11');
verifyEqual(testCase,b.path,'/new/P11/previews/a.tif');
verifyEqual(testCase,b.untouched_path,'/oldish/T01');
verifyTrue(testCase,dataset_v3_numeric_equal(a,b));
b.arrays{1}(1)=0; verifyFalse(testCase,dataset_v3_numeric_equal(a,b));
end

function testIdentityAndIdempotence(testCase)
a=struct('sample_id','P09','split','validation','cfg',struct('output_root','/old'));
[b,~]=dataset_v3_rewrite_metadata(a,'P09','P09','train',{'/old'},'/new');
[c,changed]=dataset_v3_rewrite_metadata(b,'P09','P09','train',{'/old'},'/new');
verifyFalse(testCase,changed); verifyEqual(testCase,b,c); verifyEqual(testCase,b.split,'train');
end

function testNumericComparisonRejectsSeedsIndicesAndTypeChanges(testCase)
a=struct('seed',1,'indices',[1,2,3],'truth',single(ones(2,2,2)));
b=a; b.seed=2; verifyFalse(testCase,dataset_v3_numeric_equal(a,b));
b=a; b.indices=[1,3,2]; verifyFalse(testCase,dataset_v3_numeric_equal(a,b));
b=a; b.truth=double(b.truth); verifyFalse(testCase,dataset_v3_numeric_equal(a,b));
end
