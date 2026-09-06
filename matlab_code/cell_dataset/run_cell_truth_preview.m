function report = run_cell_truth_preview(outputRoot, sampleIds)
%RUN_CELL_TRUTH_PREVIEW One-command geometry simulation. NEVER runs imaging.
%   run_cell_truth_preview() creates P06--P10 and stops for human review.
%   run_cell_truth_preview('',{'V01','V02','T01','T02'}) also prepares controls.
% There is deliberately no automatic preview-to-forward continuation.
if nargin<1, outputRoot=''; end
if nargin<2, sampleIds={'P06','P07','P08','P09','P10'}; end
if ischar(sampleIds), sampleIds={sampleIds}; end
thisDir=fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(thisDir),'pilot_dataset'));
base=cell_dataset_config(sampleIds{1},outputRoot); outputRoot=base.output_root;
if ~isfolder(outputRoot), mkdir(outputRoot); end
report=struct('stage','truth_preview_only','morphology_approved',false, ...
    'forward_started',false,'rl_started',false,'algorim_started',false, ...
    'samples',{{}},'created_at',datestr(now,30));
meshes=cell(1,numel(sampleIds)); summaries=cell(1,numel(sampleIds));
for index=1:numel(sampleIds)
    cfg=cell_dataset_config(sampleIds{index},outputRoot);
    fprintf('PREVIEW ONLY %s: construct geometry, no speckle or PSF loading\n',cfg.sample_id);
    path=fullfile(cfg.sample_dir,'truth.mat');
    if isfolder(cfg.sample_dir)
        assert(isfile(path),'cells:Ownership','Existing directory has no owned truth.mat: %s',cfg.sample_dir);
        sample=load(path);
        assert(isfield(sample,'cfg') && isequaln(sample.cfg,cfg), ...
            'cells:Stale','Existing truth configuration differs. Use a NEW output directory.');
        assert(strcmp(sample.source_sha256,geometry_hash(thisDir)), ...
            'cells:Stale','Geometry source changed. Preserve this output and choose a NEW directory.');
    else
        sample=cell_make_truth(cfg);
        sample.metrics=cell_validate_truth(sample);
        sample.source_sha256=geometry_hash(thisDir);
        sample.approval_status='pending_user_morphology_review';
        old=rng; cleanup=onCleanup(@() rng(old)); %#ok<NASGU>
        rng(cfg.subset_seed,'twister');
        sample.input_indices=reshape(randperm(100),10,10)';
        sample.holdout_indices=zeros(10,90);
        for s=1:10, sample.holdout_indices(s,:)=setdiff(1:100,sample.input_indices(s,:),'stable'); end
        clear cleanup;
        mkdir(cfg.sample_dir);
        pilot_atomic_save(path,sample);
    end
    sample.metrics=cell_validate_truth(sample);
    previewDir=fullfile(cfg.sample_dir,'previews');
    if ~isfolder(previewDir), mkdir(previewDir); end
    pilot_write_tiff(fullfile(cfg.sample_dir,'ground_truth_float.tif'),sample.ground_truth);
    pilot_write_tiff(fullfile(cfg.sample_dir,'ground_truth_fine_float.tif'),sample.ground_truth_fine);
    cell_write_json(fullfile(cfg.sample_dir,'geometry.json'),sample.meta);
    cell_write_json(fullfile(cfg.sample_dir,'truth_metrics.json'),sample.metrics);
    meshes{index}=cell_plot_truth(sample,previewDir);
    summaries{index}=struct('cfg',cfg,'xy',max(sample.ground_truth,[],3), ...
        'xz',squeeze(max(sample.ground_truth,[],1))', ...
        'yz',squeeze(max(sample.ground_truth,[],2))');
    report.samples{end+1}=struct('sample_id',cfg.sample_id,'split',cfg.split, ...
        'source_sha256',sample.source_sha256,'truth_metrics',sample.metrics, ...
        'truth_path',path,'preview_dir',previewDir);
    fprintf('%s validated: z mass %s; waiting for visual approval\n', ...
        cfg.sample_id,mat2str(sample.metrics.mass_fraction,3));
end
fig=figure('Visible','off','Color','w','Position',[20,20,1800,330*numel(sampleIds)]);
cleanup=onCleanup(@() close(fig)); %#ok<NASGU>
layout=tiledlayout(fig,numel(sampleIds),4,'TileSpacing','compact','Padding','compact');
for k=1:numel(sampleIds)
    item=summaries{k}; cfg=item.cfg; limit=double(max(item.xy(:)));
    ax=nexttile(layout); imagesc(ax,cfg.x_um,cfg.y_um,item.xy,[0,limit]);
    axis(ax,'image'); colormap(ax,gray(256)); xlabel(ax,'X (um)'); ylabel(ax,'Y (um)');
    title(ax,sprintf('%s | %s\nXY MIP (native GT)',cfg.sample_id,cfg.description),'Interpreter','none');
    ax=nexttile(layout); imagesc(ax,cfg.x_um,cfg.z_um,item.xz,[0,limit]);
    axis(ax,'image'); colormap(ax,gray(256)); xlabel(ax,'X (um)'); ylabel(ax,'Depth (um)'); title(ax,'XZ MIP (native GT)');
    ax=nexttile(layout); imagesc(ax,cfg.y_um,cfg.z_um,item.yz,[0,limit]);
    axis(ax,'image'); colormap(ax,gray(256)); xlabel(ax,'Y (um)'); ylabel(ax,'Depth (um)'); title(ax,'YZ MIP (native GT)');
    ax=nexttile(layout); cell_draw_surface(ax,meshes{k},cfg); title(ax,'3D geometry | physical aspect 1:1:1');
end
title(layout,'GROUND TRUTH ONLY | left: native 10 um slabs; right: 1 um geometry | no forward / RL / training');
overviewName=['overview_' strjoin(sampleIds,'_') '.png'];
exportgraphics(fig,fullfile(outputRoot,overviewName),'Resolution',125);
report.overview=fullfile(outputRoot,overviewName);
cell_write_json(fullfile(outputRoot,['preview_report_' strjoin(sampleIds,'_') '.json']),report);
write_splits(outputRoot,base.repo_root);
cell_write_json(fullfile(outputRoot,'MORPHOLOGY_REVIEW_REQUIRED.json'), ...
    struct('approved',false,'stage','truth_preview_only', ...
    'message','STOP. No sensor simulation, RL or AlgoRIM until explicit user morphology confirmation.'));
fprintf('\nSTOP: previews ready. Await explicit morphology confirmation. No forward or reconstruction was run.\n');
end

function hash=geometry_hash(folder)
md=java.security.MessageDigest.getInstance('SHA-256');
for file={'cell_dataset_config.m','cell_make_truth.m'}
    md.update(uint8(unicode2native(fileread(fullfile(folder,file{1})),'UTF-8')));
end
hash=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end

function write_splits(root,repo)
ids={'P01','P02','P03','P04','P05','P06','P07','P08','P09','P10','V01','V02','T01','T02'};
splits=cellfun(@cell_dataset_split,ids,'UniformOutput',false);
items=cell(1,numel(ids));
for k=1:numel(ids)
    if k<=5
        samplePath=fullfile(repo,'data','matlab_pilot5_v1',ids{k}); stage='legacy_acquisition_ready';
    else
        samplePath=fullfile(root,ids{k});
        if isfile(fullfile(samplePath,'truth.mat')), stage='truth_preview_only'; else, stage='planned_not_generated'; end
    end
    items{k}=struct('sample_id',ids{k},'object_group_id',ids{k}, ...
        'split',splits{k},'sample_dir',samplePath,'stage',stage);
end
cell_write_json(fullfile(root,'dataset_splits.json'), ...
    struct('version',1,'split_unit','entire_object_including_all_subsets_and_derivatives', ...
    'counts',struct('train',8,'validation',3,'test',3),'samples',{items}, ...
    'holdout_note','90 frames used for gradients are training targets, never independent tests', ...
    'morphology_review_note','Test morphology previews may be inspected for validity, not used to tune reconstruction'));
end
