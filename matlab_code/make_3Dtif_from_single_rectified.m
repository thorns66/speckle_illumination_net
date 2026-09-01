function make_3Dtif_from_single_rectified
clear; close all; clc;

%% ================================================================
% 功能：
% 把 single_rectified 下面的 100 个单张重建体积按深度层重新堆叠
%
% 兼容两种情况：
% 情况 1：img_001 里面有多个深度层 tif
% 情况 2：img_001 里面只有 1 个多页 tif，每一页是一个深度层
%
% 输出：
% single_rectified\single_depth_3Dtif\
%
% 每个输出 tif 是一个 100 页的 3D tif：
% 第 1 页 = img_001 的某个深度层
% 第 2 页 = img_002 的同一个深度层
% ...
% 第 100 页 = img_100 的同一个深度层
%% ================================================================

%% ====== 总路径 ======
rootPath = 'D:\xy\20260611_brainSlices_screening';

% 如果处理原来的 800um_2 / 800um_3，用 recon_it5
reconRoot = fullfile(rootPath, 'recon_it5_others');

% 如果处理 800um_1 那批 recon_it5_others，把上一行注释掉，改用这一行：
% reconRoot = fullfile(rootPath, 'recon_it5_others');

%% ====== 要处理的视场 ======
selectedList = {
    '500um_3', 'FOV02_(-100+40)'
    '650um_1', 'FOV02_(-100+140)'
    '650um_1', 'FOV03(-120+140)'
    '650um_3', 'FOV01_(-160+100)'
};

% 如果处理 800um_1 那批，就改成类似：
% selectedList = {
%     '800um_1', 'FOV01_(-100+160)'
%     '800um_1', 'FOV03-200_(-120+260)'
% };

%% ====== 参数 ======
frameNumWanted = 100;

% 是否每张图单独归一化到 [0,1]
normalizeEachImage = true;

% 输出保存为 uint16
saveAsUint16 = true;

%% ====== 开始处理 ======
for idx = 1:size(selectedList, 1)

    sliceName = selectedList{idx, 1};
    fovName   = selectedList{idx, 2};

    fprintf('\n====================================================\n');
    fprintf('开始堆叠 3D tif：%s / %s\n', sliceName, fovName);
    fprintf('====================================================\n');

    %% ====== 输入：single_rectified ======
    singleRoot = fullfile(reconRoot, sliceName, fovName, 'single_rectified_50');

    if ~exist(singleRoot, 'dir')
        warning('找不到 single_rectified 文件夹：%s', singleRoot);
        continue;
    end

    %% ====== 输出：创建在 single_rectified 下面 ======
    savePath = fullfile(singleRoot, 'single_depth_3Dtif_50');

    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end

    fprintf('输入路径：%s\n', singleRoot);
    fprintf('输出路径：%s\n', savePath);

    %% ====== 获取 img_001 ~ img_100 文件夹 ======
    imgDirList = getImgDirList(singleRoot);

    if isempty(imgDirList)
        warning('没有找到 img_001/img_002 这类文件夹：%s', singleRoot);
        continue;
    end

    frameNum = min(frameNumWanted, length(imgDirList));

    if length(imgDirList) < frameNumWanted
        warning('只找到 %d 个 img_xxx 文件夹，不足 100 个。', length(imgDirList));
    end

    fprintf('将参与堆叠的散斑重建体积数量：%d\n', frameNum);

    %% ====== 判断每个 img_xxx 里面是“多页 tif”还是“多个单页深度 tif” ======
    firstImgDir = fullfile(singleRoot, imgDirList(1).name);
    firstTifList = getTifList(firstImgDir);

    if isempty(firstTifList)
        warning('第一个 img 文件夹里没有 tif：%s', firstImgDir);
        continue;
    end

    firstTifFile = fullfile(firstImgDir, firstTifList(1).name);
    firstInfo = imfinfo(firstTifFile);

    if length(firstTifList) == 1 && numel(firstInfo) > 1
        inputMode = 'multipage';
        templateVolumeName = firstTifList(1).name;
        depthNum = numel(firstInfo);
        fprintf('检测到输入模式：单个多页 3D tif。\n');
    else
        inputMode = 'separate_files';
        templateFileList = firstTifList;
        depthNum = length(templateFileList);
        fprintf('检测到输入模式：多个深度层 tif 文件。\n');
    end

    fprintf('检测到深度层数量：%d\n', depthNum);

    %% ====== 推断每个深度层对应的物理深度，用于输出文件命名 ======
    zList = inferDepthValues(fovName, depthNum);

    %% ====== 对每一个深度层进行堆叠 ======
    for d = 1:depthNum

        if ~isempty(zList)
            depthName = sprintf('depth_%s', zToLabel(zList(d)));
        else
            depthName = sprintf('depth_%03d', d);
        end

        outputFileName = fullfile(savePath, sprintf('%s_stack%03d.tif', depthName, frameNum));

        if exist(outputFileName, 'file')
            delete(outputFileName);
        end

        fprintf('\n[%d/%d] 正在生成：%s\n', d, depthNum, outputFileName);

        %% ====== 从 img_001 到 img_100 读取同一个深度层 ======
        for j = 1:frameNum

            imgDir = fullfile(singleRoot, imgDirList(j).name);

            switch inputMode

                case 'multipage'
                    inputFile = findVolumeFile(imgDir, templateVolumeName);
                    img = imread(inputFile, d);

                case 'separate_files'
                    currentList = getTifList(imgDir);

                    if length(currentList) < d
                        error('文件夹 %s 中深度层数量不足，无法读取第 %d 层。', imgDir, d);
                    end

                    inputFile = fullfile(imgDir, currentList(d).name);
                    img = imread(inputFile);

                otherwise
                    error('未知输入模式。');
            end

            img = double(img);
            img(isnan(img)) = 0;
            img(isinf(img)) = 0;

            if normalizeEachImage
                maxVal = max(img(:));
                if maxVal > 0
                    img = img ./ maxVal;
                end
            end

            if saveAsUint16
                img = img - min(img(:));
                maxVal = max(img(:));
                if maxVal > 0
                    img = img ./ maxVal;
                end
                img = uint16(img .* 65535);
            else
                img = double(img);
                img = max(min(img, 1), 0);
            end

            %% ====== 写入 3D tif，第 j 页对应第 j 个散斑重建体积 ======
            if j == 1
                imwrite(img, outputFileName, 'tif', 'Compression', 'none');
            else
                imwrite(img, outputFileName, 'tif', ...
                    'WriteMode', 'append', 'Compression', 'none');
            end

            fprintf('  已加入第 %03d / %03d 个体积\n', j, frameNum);
        end

        fprintf('已保存：%s\n', outputFileName);
    end

    fprintf('\n完成当前 FOV：%s / %s\n', sliceName, fovName);
end

fprintf('\n====================================================\n');
fprintf('全部 single_depth_3Dtif 生成完成。\n');
fprintf('====================================================\n');

end


%% ================================================================
% 获取 img_001、img_002 ... 文件夹
%% ================================================================
function imgDirList = getImgDirList(singleRoot)

imgDirList = dir(fullfile(singleRoot, 'img_*'));
imgDirList = imgDirList([imgDirList.isdir]);

if isempty(imgDirList)
    return;
end

names = {imgDirList.name};
nums = nan(length(names), 1);

for k = 1:length(names)
    tokens = regexp(names{k}, '\d+', 'match');
    if ~isempty(tokens)
        nums(k) = str2double(tokens{end});
    end
end

if all(~isnan(nums))
    [~, idxSort] = sort(nums);
else
    [~, idxSort] = sort(names);
end

imgDirList = imgDirList(idxSort);

end


%% ================================================================
% 获取 tif 文件列表
%% ================================================================
function tifList = getTifList(folderPath)

tifList = [ ...
    dir(fullfile(folderPath, '*.tif')); ...
    dir(fullfile(folderPath, '*.tiff')) ...
];

if isempty(tifList)
    return;
end

[~, idxUnique] = unique({tifList.name});
tifList = tifList(idxUnique);

names = {tifList.name};
nums = nan(length(names), 1);

for k = 1:length(names)
    tokens = regexp(names{k}, '-?\d+', 'match');
    if ~isempty(tokens)
        nums(k) = str2double(tokens{end});
    end
end

if all(~isnan(nums))
    [~, idxSort] = sort(nums);
else
    [~, idxSort] = sort(names);
end

tifList = tifList(idxSort);

end


%% ================================================================
% 在某个 img_xxx 目录里找对应的多页体积 tif
%% ================================================================
function inputFile = findVolumeFile(imgDir, templateVolumeName)

inputFile = fullfile(imgDir, templateVolumeName);

if exist(inputFile, 'file')
    return;
end

tifList = getTifList(imgDir);

if isempty(tifList)
    error('文件夹中没有 tif 文件：%s', imgDir);
end

if length(tifList) == 1
    inputFile = fullfile(imgDir, tifList(1).name);
    return;
end

% 如果有多个 tif，但找不到同名文件，则默认取第一个
inputFile = fullfile(imgDir, tifList(1).name);

end


%% ================================================================
% 推断深度值
%% ================================================================
function zList = inferDepthValues(fovName, depthNum)

zList = [];

%% 情况 1：文件夹名里有类似 (-100+160)
[zStart, zEnd, hasRange] = parseDepthRangeFromName(fovName);

if hasRange
    zList = round(linspace(zStart, zEnd, depthNum));
    return;
end

%% 情况 2：默认完整深度 -340 到 340
if depthNum == 35
    zList = -340:20:340;
    return;
end

%% 情况 3：无法判断，就用空，后面按 depth_001 命名
zList = [];

end


%% ================================================================
% 从名字里解析深度范围
% 例：FOV01_(-100+160) -> -100, 160
%% ================================================================
function [zStart, zEnd, hasRange] = parseDepthRangeFromName(nameStr)

zStart = NaN;
zEnd = NaN;
hasRange = false;

token = regexp(nameStr, '\((-?\d+)\s*([+-])\s*(\d+)\)', 'tokens', 'once');

if isempty(token)
    return;
end

zStart = str2double(token{1});
secondVal = str2double(token{3});

if strcmp(token{2}, '-')
    zEnd = -secondVal;
else
    zEnd = secondVal;
end

if isnan(zStart) || isnan(zEnd)
    return;
end

if zStart > zEnd
    tmp = zStart;
    zStart = zEnd;
    zEnd = tmp;
end

hasRange = true;

end


%% ================================================================
% 深度值转文件名标签
%% ================================================================
function label = zToLabel(z)

if z < 0
    label = sprintf('m%03d', abs(z));
else
    label = sprintf('p%03d', z);
end

end