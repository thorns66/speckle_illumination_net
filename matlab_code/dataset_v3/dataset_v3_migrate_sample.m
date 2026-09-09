function dataset_v3_migrate_sample(outputRoot,sampleId)
assert(isempty(getenv('CUDA_VISIBLE_DEVICES')),'v3:CPUOnly');
jobs=jsondecode(fileread(fullfile(outputRoot,'migration_jobs.json')));
rows=jobs.samples; index=find(strcmp({rows.sample_id},sampleId)); assert(isscalar(index),'v3:MigrationOwner');
job=rows(index); folder=fullfile(outputRoot,sampleId);
assert(strcmp(folder,job.target_dir)&&~strcmp(folder,job.source_dir),'v3:MigrationTarget');
oldRoots=cellstr(string(job.recorded_roots));
listing=dir(fullfile(folder,'**','*.mat')); changedFiles={};
for k=1:numel(listing)
    path=fullfile(listing(k).folder,listing(k).name); relative=path(numel(folder)+2:end);
    if ismember(relative,{'validation_manifest.mat','full_run_report.mat'}), continue; end
    selected=~strcmp(job.source_sample_id,sampleId)|| ...
        ismember(relative,{'prepared.mat','truth.mat',fullfile('algorim','manifest.mat')})|| ...
        startsWith(relative,['subsets' filesep]);
    if ~selected, continue; end
    original=load(path);
    [updated,changed]=dataset_v3_rewrite_metadata(original,job.source_sample_id, ...
        sampleId,job.split,oldRoots,outputRoot);
    if changed
        pilot_atomic_save(path,updated);
        reread=load(path);
        assert(dataset_v3_numeric_equal(original,reread),'v3:NumericMutation','Numeric change: %s',path);
        assert(isequaln(updated,reread),'v3:Serialization','Metadata round trip failed.');
        changedFiles{end+1}=relative; %#ok<AGROW>
    end
end
prepared=load(fullfile(folder,'prepared.mat'),'cfg'); cfg=prepared.cfg;
assert(strcmp(cfg.sample_id,sampleId)&&strcmp(cfg.sample_dir,folder)&&strcmp(cfg.output_root,outputRoot), ...
    'v3:MigrationConfig');
if isfield(cfg,'split'), assert(strcmp(cfg.split,job.split),'v3:MigrationSplit'); end
validation=pilot_validate_sample(cfg);
report=struct('sample_id',sampleId,'source_sample_id',job.source_sample_id, ...
    'split',job.split,'complete',validation.complete,'numerical_payloads_unchanged',true, ...
    'changed_mat_files',{changedFiles},'source_directory',job.source_dir,'target_directory',folder);
cell_write_json(fullfile(outputRoot,'progress',['migration_' sampleId '.json']),report);
fprintf('MIGRATED AND VALIDATED %s -> %s (%d MAT metadata rewrites)\n', ...
    job.source_sample_id,sampleId,numel(changedFiles));
end
