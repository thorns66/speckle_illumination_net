function Reconstruction_remainingDepths_it5_50_priority650um3
clear; close all; clc;
warning('on');

%% ================================================================
% 功能：
% 对已经完成“部分深度 + 50 张 LF 单图重建”的 FOV，
% 补充重建其余未覆盖深度。
%
% 例如：
% FOV01_(-160+100)
%
% 已经完成：
% -160 到 +100 um
%
% 本代码补充：
% -340 到 -180 um
% +120 到 +340 um
%
% 输入：
% good_fov_gray\切片\FOV\*.tif
%
% 输出：
% 仍保存到原来的：
% recon_it5_others\切片\FOV\single_rectified_50\img_001\
% ...
% recon_it5_others\切片\FOV\single_rectified_50\img_050\
%
% 新生成的 tif 因为深度范围不同，例如：
% depth_from-340to-180
% depth_from120to340
% 因此不会覆盖原来的 depth_from-160to100 文件。
%
% 处理顺序：
% 1. 650um_3 优先
% 2. 其他已完成部分深度重建的 FOV
%% ================================================================

%% ====== 总路径 ======
rootPath = 'D:\xy\20260611_brainSlices_screening';

inputRootAll = fullfile(rootPath, 'good_fov_gray');
saveRootAll  = fullfile(rootPath, 'recon_it5_others');

%% ====== 要处理的数据，650um_3 放在第一位 ======
selectedList = {
    '650um_3', 'FOV01_(-160+100)'
    '500um_3', 'FOV02_(-100+40)'
    '650um_1', 'FOV02_(-100+140)'
    '650um_1', 'FOV03(-120+140)'
};

%% ====== 函数路径 ======
codePath = fullfile(rootPath, 'code_used');

addpath(genpath(codePath));
addpath(genpath('D:\xy\Util'));
addpath(genpath('D:\xy\Solver'));

%% ====== PSF 路径 ======
psfHPath = ...
    'F:\xy\20250626-大深度微型系统\合并psf\merged-340_to_340_H.mat';

psfHtPath = ...
    'F:\xy\20250626-大深度微型系统\合并psf\merged-340_to_340_Ht.mat';

if ~exist(psfHPath, 'file')
    error('找不到 H 文件：%s', psfHPath);
end

if ~exist(psfHtPath, 'file')
    error('找不到 Ht 文件：%s', psfHtPath);
end

%% ====== 检查关键函数 ======
requiredFunctions = {
    'Reconstruction3D'
    'backwardProjectGPU'
    'forwardProjectGPU'
    'deconvRL'
};

for i = 1:length(requiredFunctions)
    if isempty(which(requiredFunctions{i}))
        error('找不到函数：%s', requiredFunctions{i});
    end
end

fprintf('当前使用的关键函数：\n');
fprintf('Reconstruction3D:    %s\n', which('Reconstruction3D'));
fprintf('backwardProjectGPU:  %s\n', which('backwardProjectGPU'));
fprintf('forwardProjectGPU:   %s\n', which('forwardProjectGPU'));
fprintf('deconvRL:            %s\n', which('deconvRL'));

%% ====== 重建参数 ======
iterNum = 5;

% 每个 FOV 使用 50 张 LF 图
targetFrameNum = 50;

% 完整物理深度
fullZStart = -340;
fullZEnd   = 340;

% 实际物理深度，共 35 层
depthValueAll = fullZStart:20:fullZEnd;

% 原始 PSF 文件中的对应索引
% 第 19 个原始索引被跳过
psfIndexAll = [1:18, 20:36];

if length(depthValueAll) ~= length(psfIndexAll)
    error('depthValueAll 与 psfIndexAll 长度不一致。');
end

% 已存在新深度结果时是否重新运行
% false：若检测到对应深度结果，就跳过
% true：重新运行，但 Reconstruction3D 可能覆盖相同深度文件
overwriteRemaining = false;

%% ====== 日志 ======
logCell = {
    'slice', ...
    'fov', ...
    'segment', ...
    'z_start', ...
    'z_end', ...
    'depth_num', ...
    'frame_num', ...
    'status', ...
    'message'
};

successSegment = 0;
failSegment = 0;
skipSegment = 0;

%% ====== 逐个 FOV 处理 ======
for n = 1:size(selectedList, 1)

    sliceName = selectedList{n, 1};
    fovName   = selectedList{n, 2};

    fprintf('\n====================================================\n');
    fprintf('开始处理：%s / %s\n', sliceName, fovName);
    fprintf('====================================================\n');

    inputFOVPath = fullfile(inputRootAll, sliceName, fovName);
    inputFOVPathWithSep = [inputFOVPath, filesep];

    savePathSingle = fullfile( ...
        saveRootAll, ...
        sliceName, ...
        fovName, ...
        'single_rectified_50' ...
    );

    if ~exist(inputFOVPath, 'dir')
        warning('找不到输入 FOV：%s', inputFOVPath);

        logCell(end+1,:) = {
            sliceName, fovName, '', NaN, NaN, 0, 0, ...
            'failed', 'missing input FOV'
        };

        failSegment = failSegment + 1;
        continue;
    end

    if ~exist(savePathSingle, 'dir')
        warning('找不到原来的 single_rectified_50：%s', savePathSingle);

        logCell(end+1,:) = {
            sliceName, fovName, '', NaN, NaN, 0, 0, ...
            'failed', 'missing single_rectified_50'
        };

        failSegment = failSegment + 1;
        continue;
    end

    %% ====== 获取输入图像 ======
    imgList = getImageList(inputFOVPath);
    imgNum = length(imgList);

    if imgNum == 0
        warning('没有找到输入图像：%s', inputFOVPath);
        continue;
    end

    frameNumToRun = min(targetFrameNum, imgNum);
    selectedImgIdx = makeEvenIndices(imgNum, frameNumToRun);

    fprintf('输入图像数量：%d\n', imgNum);
    fprintf('用于重建的图像数量：%d\n', frameNumToRun);

    %% ====== 获取已重建深度范围 ======
    [oldZStart, oldZEnd, hasRange] = ...
        parseDepthRangeFromName(fovName, fullZStart, fullZEnd);

    if ~hasRange
        warning('FOV 名称中没有深度范围：%s，跳过。', fovName);
        continue;
    end

    oldZStart = max(oldZStart, fullZStart);
    oldZEnd   = min(oldZEnd, fullZEnd);

    fprintf('原有重建范围：%d 到 %d um\n', oldZStart, oldZEnd);

    %% ====== 计算剩余深度段 ======
    remainingSegments = getRemainingSegments( ...
        fullZStart, ...
        fullZEnd, ...
        oldZStart, ...
        oldZEnd, ...
        20 ...
    );

    if isempty(remainingSegments)
        fprintf('该 FOV 已覆盖完整深度，无剩余深度。\n');
        continue;
    end

    fprintf('需要补充的深度段：\n');

    for seg = 1:size(remainingSegments, 1)
        fprintf('  第 %d 段：%d 到 %d um\n', ...
            seg, remainingSegments(seg,1), remainingSegments(seg,2));
    end

    %% ====== 每段剩余深度单独重建 ======
    for seg = 1:size(remainingSegments, 1)

        zStartSeg = remainingSegments(seg, 1);
        zEndSeg   = remainingSegments(seg, 2);

        segmentName = sprintf('remaining_%s_to_%s', ...
            zToText(zStartSeg), zToText(zEndSeg));

        fprintf('\n----------------------------------------------------\n');
        fprintf('开始补充深度段：%d 到 %d um\n', zStartSeg, zEndSeg);
        fprintf('----------------------------------------------------\n');

        %% ====== 选择该段对应 PSF ======
        depthMask = ...
            depthValueAll >= zStartSeg & ...
            depthValueAll <= zEndSeg;

        depthValueUse = depthValueAll(depthMask);
        depthIdxUse   = psfIndexAll(depthMask);

        if isempty(depthIdxUse)
            warning('当前深度段没有可用 PSF：%d 到 %d', ...
                zStartSeg, zEndSeg);

            logCell(end+1,:) = {
                sliceName, fovName, segmentName, ...
                zStartSeg, zEndSeg, 0, frameNumToRun, ...
                'failed', 'no PSF depth'
            };

            failSegment = failSegment + 1;
            continue;
        end

        % 使用实际 PSF 层对应的首尾深度
        zStartRecon = depthValueUse(1);
        zEndRecon   = depthValueUse(end);

        fprintf('实际补充深度：%d 到 %d um，共 %d 层。\n', ...
            zStartRecon, zEndRecon, length(depthIdxUse));

        %% ====== 检查该深度段是否已经存在 ======
        if ~overwriteRemaining
            firstOutputDir = fullfile(savePathSingle, 'img_001');

            if hasDepthResult( ...
                    firstOutputDir, ...
                    zStartRecon, ...
                    zEndRecon)

                fprintf('检测到该深度段已存在，跳过：%d 到 %d um\n', ...
                    zStartRecon, zEndRecon);

                logCell(end+1,:) = {
                    sliceName, fovName, segmentName, ...
                    zStartRecon, zEndRecon, length(depthIdxUse), ...
                    frameNumToRun, 'skipped', ...
                    'depth result already exists'
                };

                skipSegment = skipSegment + 1;
                continue;
            end
        end

        %% ====== 分层加载 PSF ======
        try
            fprintf('正在加载该深度段 PSF...\n');

            [H_use, Ht_use, CAindex_use] = ...
                loadPSFByMatfileRuns( ...
                    psfHPath, ...
                    psfHtPath, ...
                    depthIdxUse ...
                );

            fprintf('PSF 加载完成。\n');

        catch ME
            warning('PSF 加载失败：%s', ME.message);

            logCell(end+1,:) = {
                sliceName, fovName, segmentName, ...
                zStartRecon, zEndRecon, length(depthIdxUse), ...
                frameNumToRun, 'failed', ...
                ['PSF failed: ', ME.message]
            };

            failSegment = failSegment + 1;
            continue;
        end

        %% ====== 对 50 张 LF 分别重建该深度段 ======
        segmentStatus = 'success';
        message = '';

        try
            for j = 1:frameNumToRun

                originalIdx = selectedImgIdx(j);
                inputFileName = imgList(originalIdx).name;

                % 保持与原来的 img_001 ~ img_050 完全一致
                savePathJ = fullfile( ...
                    savePathSingle, ...
                    sprintf('img_%03d', j) ...
                );

                if ~exist(savePathJ, 'dir')
                    mkdir(savePathJ);
                end

                savePathJWithSep = [savePathJ, filesep];

                fprintf('\n正在补充第 %d / %d 个体积：%s\n', ...
                    j, frameNumToRun, inputFileName);

                Reconstruction3D( ...
                    H_use, ...
                    Ht_use, ...
                    CAindex_use, ...
                    inputFOVPathWithSep, ...
                    inputFileName, ...
                    savePathJWithSep, ...
                    iterNum, ...
                    zStartRecon, ...
                    zEndRecon ...
                );
            end

            successSegment = successSegment + 1;

            fprintf('\n补充深度段完成：%s / %s，%d 到 %d um\n', ...
                sliceName, fovName, zStartRecon, zEndRecon);

        catch ME
            segmentStatus = 'failed';
            message = ME.message;
            failSegment = failSegment + 1;

            fprintf(2, '\n当前深度段重建失败：%s / %s\n', ...
                sliceName, fovName);
            fprintf(2, '深度范围：%d 到 %d um\n', ...
                zStartRecon, zEndRecon);
            fprintf(2, '原因：%s\n', ME.message);
            fprintf(2, '%s\n', ...
                getReport(ME, 'extended', 'hyperlinks', 'off'));
        end

        logCell(end+1,:) = {
            sliceName, ...
            fovName, ...
            segmentName, ...
            zStartRecon, ...
            zEndRecon, ...
            length(depthIdxUse), ...
            frameNumToRun, ...
            segmentStatus, ...
            message
        };

        %% ====== 清理内存 ======
        clear H_use Ht_use CAindex_use;

        try
            g = gpuDevice;
            reset(g);
            fprintf('GPU 已 reset。\n');
        catch
        end
    end
end

%% ====== 保存日志 ======
logPath = fullfile( ...
    saveRootAll, ...
    'remaining_depth_reconstruction_it5_50_log.xlsx' ...
);

try
    writecell(logCell, logPath);
    fprintf('\n日志已保存：%s\n', logPath);
catch
    logPathTxt = fullfile( ...
        saveRootAll, ...
        'remaining_depth_reconstruction_it5_50_log.txt' ...
    );

    writeLogTxt(logCell, logPathTxt);
    fprintf('\n日志已保存：%s\n', logPathTxt);
end

fprintf('\n====================================================\n');
fprintf('剩余深度补充重建结束。\n');
fprintf('成功深度段：%d\n', successSegment);
fprintf('失败深度段：%d\n', failSegment);
fprintf('跳过深度段：%d\n', skipSegment);
fprintf('====================================================\n');

end


%% ================================================================
% 计算完整范围中未被原范围覆盖的连续深度段
%% ================================================================
function segments = getRemainingSegments( ...
    fullZStart, fullZEnd, oldZStart, oldZEnd, zStep)

segments = [];

% 原范围以下
lowerStart = fullZStart;
lowerEnd   = oldZStart - zStep;

if lowerStart <= lowerEnd
    segments(end+1,:) = [lowerStart, lowerEnd];
end

% 原范围以上
upperStart = oldZEnd + zStep;
upperEnd   = fullZEnd;

if upperStart <= upperEnd
    segments(end+1,:) = [upperStart, upperEnd];
end

end


%% ================================================================
% 用 matfile 分段读取 H / Ht
%% ================================================================
function [H_use, Ht_use, CAindex_use] = ...
    loadPSFByMatfileRuns(psfHPath, psfHtPath, depthIdxUse)

fprintf('原始 PSF 索引：');
fprintf('%d ', depthIdxUse);
fprintf('\n');

M_H = matfile(psfHPath);
H_use = readPSFByContinuousRuns(M_H, 'H', depthIdxUse);
clear M_H;

M_Ht = matfile(psfHtPath);
Ht_use = readPSFByContinuousRuns(M_Ht, 'Ht', depthIdxUse);

CAindexAll = M_Ht.CAindex(:,:);
CAindex_use = CAindexAll(depthIdxUse,:);

clear M_Ht CAindexAll;

end


%% ================================================================
% 按连续索引段读取 PSF
%% ================================================================
function A = readPSFByContinuousRuns(M, varName, depthIdxUse)

depthIdxUse = depthIdxUse(:)';
runCell = splitContinuousRuns(depthIdxUse);

A = [];

for r = 1:length(runCell)

    idxRun = runCell{r};

    fprintf('  读取 %s：%d 到 %d，共 %d 层\n', ...
        varName, idxRun(1), idxRun(end), length(idxRun));

    switch varName
        case 'H'
            temp = M.H(:,:,:,:,idxRun(1):idxRun(end));

        case 'Ht'
            temp = M.Ht(:,:,:,:,idxRun(1):idxRun(end));

        otherwise
            error('未知 PSF 变量：%s', varName);
    end

    if ~isa(temp, 'single')
        temp = single(temp);
    end

    if isempty(A)
        A = temp;
    else
        A = cat(5, A, temp);
    end

    clear temp;
end

end


%% ================================================================
% 把索引拆分成连续段
%% ================================================================
function runCell = splitContinuousRuns(idx)

idx = idx(:)';

if isempty(idx)
    runCell = {};
    return;
end

breakPos = find(diff(idx) ~= 1);

startPos = [1, breakPos + 1];
endPos   = [breakPos, length(idx)];

runCell = cell(length(startPos), 1);

for i = 1:length(startPos)
    runCell{i} = idx(startPos(i):endPos(i));
end

end


%% ================================================================
% 检查指定深度结果是否已经存在
%% ================================================================
function tf = hasDepthResult(folderPath, zStart, zEnd)

tf = false;

if ~exist(folderPath, 'dir')
    return;
end

tifList = [
    dir(fullfile(folderPath, '*.tif'));
    dir(fullfile(folderPath, '*.tiff'))
];

if isempty(tifList)
    return;
end

for i = 1:length(tifList)

    nameLower = lower(tifList(i).name);

    % 兼容常见命名：
    % depth_from-340to-180
    % depth_from-340_to_-180
    % from-340to-180
    numberTokens = regexp( ...
        nameLower, ...
        '-?\d+', ...
        'match' ...
    );

    if isempty(numberTokens)
        continue;
    end

    values = str2double(numberTokens);

    if any(values == zStart) && any(values == zEnd)
        tf = true;
        return;
    end
end

end


%% ================================================================
% 从 FOV 名称解析原有深度范围
%% ================================================================
function [zStart, zEnd, hasDepthRange] = ...
    parseDepthRangeFromName(fovName, defaultZStart, defaultZEnd)

zStart = defaultZStart;
zEnd   = defaultZEnd;
hasDepthRange = false;

token = regexp( ...
    fovName, ...
    '\((-?\d+)\s*([+-])\s*(\d+)\)', ...
    'tokens', ...
    'once' ...
);

if isempty(token)
    return;
end

zStart = str2double(token{1});
secondValue = str2double(token{3});

if strcmp(token{2}, '-')
    zEnd = -secondValue;
else
    zEnd = secondValue;
end

if isnan(zStart) || isnan(zEnd)
    zStart = defaultZStart;
    zEnd   = defaultZEnd;
    return;
end

if zStart > zEnd
    temp = zStart;
    zStart = zEnd;
    zEnd = temp;
end

hasDepthRange = true;

end


%% ================================================================
% 均匀抽取 50 张图
%% ================================================================
function idx = makeEvenIndices(totalNum, targetNum)

if targetNum >= totalNum
    idx = 1:totalNum;
    return;
end

idx = round(linspace(1, totalNum, targetNum));
idx = unique(idx, 'stable');

while length(idx) < targetNum
    missing = setdiff(1:totalNum, idx, 'stable');
    needNum = targetNum - length(idx);

    idx = [idx, missing(1:needNum)];
    idx = sort(idx);
end

idx = idx(1:targetNum);

end


%% ================================================================
% 读取图像列表
%% ================================================================
function imgList = getImageList(folderPath)

imgList = [];

if ~exist(folderPath, 'dir')
    return;
end

imgList = [
    dir(fullfile(folderPath, '*.tif'));
    dir(fullfile(folderPath, '*.tiff'));
    dir(fullfile(folderPath, '*.bmp'));
    dir(fullfile(folderPath, '*.png'))
];

if isempty(imgList)
    return;
end

[~, uniqueIdx] = unique({imgList.name});
imgList = imgList(uniqueIdx);

names = {imgList.name};
numbers = nan(length(names), 1);

for k = 1:length(names)
    tokens = regexp(names{k}, '\d+', 'match');

    if ~isempty(tokens)
        numbers(k) = str2double(tokens{end});
    end
end

if all(~isnan(numbers))
    [~, sortIdx] = sort(numbers);
else
    [~, sortIdx] = sort(names);
end

imgList = imgList(sortIdx);

end


%% ================================================================
% 深度数值转文字
%% ================================================================
function textValue = zToText(z)

if z < 0
    textValue = sprintf('m%d', abs(z));
else
    textValue = sprintf('p%d', z);
end

end


%% ================================================================
% 保存文本日志
%% ================================================================
function writeLogTxt(logCell, filePath)

fid = fopen(filePath, 'w');

if fid < 0
    warning('无法创建日志：%s', filePath);
    return;
end

for i = 1:size(logCell, 1)

    rowText = strings(1, size(logCell, 2));

    for j = 1:size(logCell, 2)
        rowText(j) = string(logCell{i,j});
    end

    fprintf(fid, '%s\n', strjoin(rowText, sprintf('\t')));
end

fclose(fid);

end