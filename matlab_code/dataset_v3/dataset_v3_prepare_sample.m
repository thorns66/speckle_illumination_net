function prepared = dataset_v3_prepare_sample(cfg)
truth=load(cfg.truth_path);
assert(strcmp(truth.source_sha256,cfg.truth_source_sha256),'v3:TruthHash');
if strcmp(cfg.sample_id,'T04')
    t04_axial_validate_truth(truth);
else
    dataset_v3_validate_truth(truth);
end
path=fullfile(cfg.sample_dir,'prepared.mat');
if isfile(path)
    prepared=load(path);
    assert(isequaln(prepared.cfg,cfg),'v3:PreparedStale');
else
    for name={'products','sensor_frames','recon_frames','subsets','previews','logs'}
        folder=fullfile(cfg.sample_dir,name{1}); if ~isfolder(folder), mkdir(folder); end
    end
    prepared=struct('cfg',cfg,'ground_truth',truth.ground_truth, ...
        'ground_truth_fine',truth.ground_truth_fine,'truth_meta',truth.meta, ...
        'truth_metrics',truth.metrics,'truth_source_sha256',truth.source_sha256, ...
        'input_indices',truth.input_indices,'holdout_indices',truth.holdout_indices, ...
        'occupied_truth_indices_one_based',find(squeeze(sum(sum(truth.ground_truth,1),2))>0)');
    pilot_atomic_save(path,prepared);
    pilot_write_tiff(fullfile(cfg.sample_dir,'previews','ground_truth_float.tif'),truth.ground_truth);
    cell_write_json(fullfile(cfg.sample_dir,'simulation_config.json'),cfg);
end
path=fullfile(cfg.sample_dir,'illumination_3d.mat');
if ~isfile(path)
    [illumination_raw,~,illumination_meta]=cell_generate_illumination_3d( ...
        cfg.illumination_seed,cfg.frame_count,cfg.z_um);
    payload=struct('schema_version',cfg.schema_version,'dataset_id',cfg.dataset_id, ...
        'sample_id',cfg.sample_id,'truth_source_sha256',cfg.truth_source_sha256, ...
        'illumination_raw',illumination_raw,'illumination_meta',illumination_meta);
    pilot_atomic_save(path,payload);
else
    old=load(path,'schema_version','dataset_id','sample_id','truth_source_sha256','illumination_meta');
    assert(old.schema_version==cfg.schema_version&&strcmp(old.dataset_id,cfg.dataset_id)&& ...
        strcmp(old.sample_id,cfg.sample_id)&&strcmp(old.truth_source_sha256,cfg.truth_source_sha256)&& ...
        isequal(double(old.illumination_meta.z_um),double(cfg.z_um))&& ...
        old.illumination_meta.frame_count==cfg.frame_count,'v3:IlluminationStale');
end
end
