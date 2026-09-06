function report = RUN_FULL_MATLAB_DATASET(gpuIndices)
%RUN_FULL_MATLAB_DATASET One command for the approved complete dataset.
% Defaults to all six A40s. Completed owned stages are reused after restart.
if nargin<1 || isempty(gpuIndices), gpuIndices=1:6; end
repoRoot=fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot,'matlab_code','cell_dataset'), ...
    fullfile(repoRoot,'matlab_code','pilot_dataset'), ...
    fullfile(repoRoot,'matlab_code','Util'),fullfile(repoRoot,'matlab_code','Solver'));
report=run_cell_full_dataset(gpuIndices);
end
