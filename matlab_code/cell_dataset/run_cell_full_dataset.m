function report = run_cell_full_dataset(gpuIndices,outputRoot)
%RUN_CELL_FULL_DATASET Multi-sample, one-process-per-GPU dataset build.
if nargin<1 || isempty(gpuIndices), gpuIndices=1:6; end
if nargin<2, outputRoot=''; end
ids={'P06','P07','P08','P09','P10','V01','V02','T01','T02'};
base=cell_dataset_config('P06',outputRoot); outputRoot=base.output_root;
validateattributes(gpuIndices,{'numeric'},{'vector','integer','positive','finite'});
gpuIndices=double(gpuIndices(:)');
assert(numel(unique(gpuIndices))==numel(gpuIndices),'cells:GPU','GPU indices must be unique.');
available=gpuDeviceCount('available');
assert(max(gpuIndices)<=available,'cells:GPU','Requested GPU %d but only %d are available.',max(gpuIndices),available);

% Controls were designed earlier but intentionally not generated before the
% five morphology review. Freeze them now under the same source fingerprint.
missing={};
for id={'V01','V02','T01','T02'}
    if ~isfile(fullfile(outputRoot,id{1},'truth.mat')), missing{end+1}=id{1}; end %#ok<AGROW>
end
if ~isempty(missing), run_cell_truth_preview(outputRoot,missing); end

approvalPath=fullfile(outputRoot,'SIMULATION_APPROVED.json');
assert(isfile(approvalPath),'cells:Approval','Missing frozen simulation approval.');
approval=jsondecode(fileread(approvalPath)); currentHash=cell_geometry_hash();
approvedIds=cellstr(string(approval.approved_new_sample_ids));
assert(approval.approved&&strcmp(approval.truth_source_sha256,currentHash)&& ...
    all(ismember(ids,approvedIds)),'cells:Approval', ...
    'Approval is absent, incomplete, or belongs to different geometry source.');
for index=1:numel(ids)
    truth=load(fullfile(outputRoot,ids{index},'truth.mat'),'source_sha256');
    assert(strcmp(truth.source_sha256,currentHash),'cells:Approval','%s truth fingerprint changed.',ids{index});
end
cell_write_json(fullfile(outputRoot,'MORPHOLOGY_REVIEW_REQUIRED.json'), ...
    struct('approved',true,'stage','full_simulation_authorized', ...
    'truth_source_sha256',currentHash,'message','User approved r04; full MATLAB dataset build authorized.'));

workerCount=min(numel(gpuIndices),numel(ids));
jobRoot=fullfile(outputRoot,'_parallel_jobs'); if ~isfolder(jobRoot), mkdir(jobRoot); end
pool=gcp('nocreate'); createdPool=false;
if isempty(pool)
    cluster=parcluster('Processes'); cluster.JobStorageLocation=jobRoot;
    pool=parpool(cluster,workerCount); createdPool=true;
else
    assert(isa(pool,'parallel.ProcessPool')&&pool.NumWorkers==workerCount, ...
        'cells:ExistingPool','Close the existing pool; expected %d process workers.',workerCount);
end
poolCleanup=onCleanup(@() close_created_pool(createdPool)); %#ok<NASGU>
fprintf('FULL DATASET: %d samples on GPU indices %s\n',numel(ids),mat2str(gpuIndices(1:workerCount)));
spmd
    selected=gpuDevice(gpuIndices(labindex));
    assigned=ids(labindex:numlabs:end);
    workerReports=cell(1,numel(assigned));
    fprintf('Worker %d -> GPU %d (%s), samples %s\n',labindex,selected.Index,selected.Name,strjoin(assigned,','));
    for sample=1:numel(assigned)
        workerReports{sample}=cell_run_full_sample(assigned{sample},outputRoot,selected.Index);
    end
end
reports={};
for worker=1:workerCount
    reports=[reports,workerReports{worker}]; %#ok<AGROW>
end
report=publish_final_manifest(outputRoot,reports,currentHash);
fprintf('FINAL DATASET COMPLETE: %s\n',report.manifest_path);
end

function report=publish_final_manifest(outputRoot,newReports,sourceHash)
legacySourceRoot=fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
    'data','matlab_pilot5_v1');
ids={'P01','P02','P03','P04','P05','P06','P07','P08','P09','P10','V01','V02','T01','T02'};
splits=cellfun(@cell_dataset_split,ids,'UniformOutput',false);
samples=cell(1,numel(ids));
for index=1:numel(ids)
    folder=fullfile(outputRoot,ids{index});
    if index<=5 && ~isfolder(folder)
        source=fullfile(legacySourceRoot,ids{index});
        assert(isfolder(source),'cells:FinalManifest', ...
            'Missing legacy source sample %s.',source);
        fprintf('Consolidate %s into unified dataset root\n',ids{index});
        [copied,message]=copyfile(source,folder);
        assert(copied,'cells:FinalManifest','Could not consolidate %s: %s',ids{index},message);
    end
    path=fullfile(folder,'validation_manifest.json');
    assert(isfile(path),'cells:FinalManifest','Missing validated sample %s.',ids{index});
    validation=jsondecode(fileread(path));
    assert(validation.complete&&validation.frame_count==100&&validation.subset_count==10, ...
        'cells:FinalManifest','Incomplete validation for %s.',ids{index});
    for stream={'legacy_per_frame_normalized','physics_common_scale'}
        files=dir(fullfile(folder,'algorim',stream{1},'depth_*um.tif'));
        assert(numel(files)==10,'cells:FinalManifest','%s lacks ten AlgoRIM depth stacks.',ids{index});
        for file=1:numel(files)
            assert(numel(imfinfo(fullfile(files(file).folder,files(file).name)))==100, ...
                'cells:FinalManifest','%s AlgoRIM stack has wrong page count.',ids{index});
        end
    end
    samples{index}=struct('sample_id',ids{index},'split',splits{index}, ...
        'sample_dir',folder,'validation_manifest',path,'artifact_count',validation.artifact_count, ...
        'legacy_reused',index<=5);
end
report=struct('producer','run_cell_full_dataset/1','complete',true, ...
    'dataset_name','matlab_complete_14_objects_v1','dataset_root',outputRoot, ...
    'all_samples_under_dataset_root',true,'truth_source_sha256',sourceHash, ...
    'counts',struct('objects',14,'train',8,'validation',3,'test',3, ...
        'frames_per_object',100,'input_subsets_per_object',10,'input_frames',10,'holdout_frames',90), ...
    'split_unit','whole object including every frame, subset and derivative', ...
    'split_revision','P07=test; P10=train', ...
    'samples',{samples},'new_sample_run_reports',{newReports}, ...
    'algorim_status','twenty 100-page uint16 input stacks per object exported; Windows application outputs not run', ...
    'completed_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss.SSSXXX')));
path=fullfile(outputRoot,'FINAL_DATASET_MANIFEST.json'); cell_write_json(path,report);
pilot_atomic_save(fullfile(outputRoot,'FINAL_DATASET_MANIFEST.mat'),struct('report',report));
write_complete_splits(outputRoot,samples,path);
report.manifest_path=path;
end

function write_complete_splits(outputRoot,samples,manifestPath)
% Replace the preview-stage split metadata only after every sample passed
% the final manifest checks. Keeping whole objects together prevents leakage
% between 10-frame inputs and their 90-frame complement constraints.
items=cell(1,numel(samples));
for index=1:numel(samples)
    sample=samples{index};
    if sample.legacy_reused
        stage='legacy_acquisition_complete_validated';
    else
        stage='full_simulation_complete_validated';
    end
    items{index}=struct('sample_id',sample.sample_id, ...
        'object_group_id',sample.sample_id,'split',sample.split, ...
        'sample_dir',sample.sample_dir,'stage',stage, ...
        'validation_manifest',sample.validation_manifest);
end
document=struct('version',2,'dataset_complete',true, ...
    'final_manifest',manifestPath, ...
    'split_unit','entire_object_including_all_subsets_and_derivatives', ...
    'counts',struct('train',8,'validation',3,'test',3), ...
    'samples',{items}, ...
    'holdout_note',['90 complement frames are training constraints for each ' ...
        '10-frame input; they are not an independent object-level test set'], ...
    'morphology_review_note',['All new morphologies were approved and frozen ' ...
        'before full simulation; test objects were not used to tune reconstruction'], ...
    'algorim_status',['twenty 100-page uint16 input stacks per object exported; ' ...
        'Windows AlgoRIM application outputs not run']);
cell_write_json(fullfile(outputRoot,'dataset_splits.json'),document);
end

function close_created_pool(created)
if created
    pool=gcp('nocreate'); if ~isempty(pool), delete(pool); end
end
end
