function report = RUN_CELL_DATASET_PREVIEW(outputRoot)
%RUN_CELL_DATASET_PREVIEW Run this file in MATLAB to simulate five cell GTs.
% Safe default: only truth + previews. No GPU, illumination, RL or training.
% To change morphology, edit cell_make_truth.m and use a new output folder.
if nargin<1, outputRoot=''; end
repoRoot=fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot,'matlab_code','cell_dataset'));
report=run_cell_truth_preview(outputRoot);
end
