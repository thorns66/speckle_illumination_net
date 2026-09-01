function export_operator_reference(psfFile, outputFile, realVolumeFile)
% Export forwardProjectACC references for Python comparison.
% Add the original Code2.0/Util directory to the MATLAB path first.

if nargin < 3
    realVolumeFile = '';
end

requestedZUm = 10:10:100;
S = load(psfFile, 'H', 'Ht', 'CAindex', 'x3objspace');
if ~isfield(S, 'H') || ~isfield(S, 'Ht') || ~isfield(S, 'CAindex') || ~isfield(S, 'x3objspace')
    error('PSF file must contain H, Ht, CAindex, and x3objspace.');
end

zAllUm = double(S.x3objspace(:)) * 1e6;
depthIndices = zeros(size(requestedZUm));
for i = 1:numel(requestedZUm)
    matches = find(abs(zAllUm - requestedZUm(i)) < 1e-4);
    if numel(matches) ~= 1
        error('Depth %.6g um has %d matches.', requestedZUm(i), numel(matches));
    end
    depthIndices(i) = matches;
end

Hselected = single(S.H(:,:,:,:,depthIndices));
Htselected = single(S.Ht(:,:,:,:,depthIndices));
CAselected = S.CAindex(depthIndices,:);
rng(20260831);
randomVolume = rand(67, 73, numel(depthIndices), 'single');
randomSensor = forwardProjectACC(Hselected, randomVolume, CAselected);
randomBackprojection = backwardProjectACC(Htselected, randomSensor, CAselected);

hasRealReference = false;
realVolume = [];
realSensor = [];
if ~isempty(realVolumeFile)
    info = imfinfo(realVolumeFile);
    if numel(info) ~= numel(depthIndices)
        error('Real volume TIFF must contain exactly %d depth pages.', numel(depthIndices));
    end
    realVolume = zeros(info(1).Height, info(1).Width, numel(info), 'single');
    for i = 1:numel(info)
        realVolume(:,:,i) = im2single(imread(realVolumeFile, i));
    end
    realSensor = forwardProjectACC(Hselected, realVolume, CAselected);
    hasRealReference = true;
end

selectedIndicesOneBased = depthIndices;
selectedZUm = requestedZUm;
save(outputFile, 'randomVolume', 'randomSensor', 'randomBackprojection', ...
    'realVolume', 'realSensor', ...
    'hasRealReference', 'selectedIndicesOneBased', 'selectedZUm', '-v7.3');
end
