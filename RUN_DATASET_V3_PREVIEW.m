function report = RUN_DATASET_V3_PREVIEW(outputRoot)
%RUN_DATASET_V3_PREVIEW New protocol; requires CPU-only dated launcher contract.
repo=fileparts(mfilename('fullpath'));
addpath(fullfile(repo,'matlab_code','dataset_v3'), ...
    fullfile(repo,'matlab_code','cell_dataset'),fullfile(repo,'matlab_code','pilot_dataset'));
report=run_dataset_v3_preview(outputRoot);
end
