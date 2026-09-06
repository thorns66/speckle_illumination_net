function report = REFRESH_DATASET_MANIFESTS(workerCount)
%REFRESH_DATASET_MANIFESTS Revalidate immutable artifacts and republish manifests.
% This never regenerates sensor/reconstruction data. Append-only logs are
% deliberately excluded from immutable artifact hashes by pilot_validate_sample.
if nargin<1 || isempty(workerCount), workerCount=6; end
validateattributes(workerCount,{'numeric'},{'scalar','integer','>=',1,'<=',6});
repoRoot=fileparts(mfilename('fullpath'));
outputRoot=fullfile(repoRoot,'data','matlab_cells_pilot_v2_r04');
addpath(fullfile(repoRoot,'matlab_code','cell_dataset'), ...
    fullfile(repoRoot,'matlab_code','pilot_dataset'), ...
    fullfile(repoRoot,'matlab_code','Util'),fullfile(repoRoot,'matlab_code','Solver'));
ids={'P01','P02','P03','P04','P05','P06','P07','P08','P09','P10','V01','V02','T01','T02'};

pool=gcp('nocreate');
if ~isempty(pool), delete(pool); end
cluster=parcluster('Processes');
cluster.JobStorageLocation=fullfile(outputRoot,'_manifest_refresh_jobs');
pool=parpool(cluster,min(workerCount,numel(ids)));
cleanup=onCleanup(@() close_pool()); %#ok<NASGU>
parfor index=1:numel(ids)
    sampleId=ids{index};
    if index<=5
        cfg=pilot_dataset_config(sampleId,outputRoot);
    else
        cfg=cell_simulation_config(sampleId,outputRoot);
    end
    fprintf('Revalidate immutable artifacts: %s\n',sampleId);
    pilot_validate_sample(cfg);
end
delete(pool);
clear cleanup;

% The full runner is resume-safe: completed samples are reused and only the
% authoritative final/split manifests are republished.
report=RUN_FULL_MATLAB_DATASET(1:workerCount);
end

function close_pool()
pool=gcp('nocreate');
if ~isempty(pool), delete(pool); end
end
