function ImageRectification2_brainSlices_new()
warning('off');
close all;
clc;

%% ================================================================
% 功能：
% new_gray\数据组\*.tif
%       ↓
% new_rectified\数据组\*.tif
%
% 输入示例：
% D:\xy\20260611_brainSlices_screening\new_gray\500_3_fov2\001.tif
%
% 输出示例：
% D:\xy\20260611_brainSlices_screening\new_rectified\500_3_fov2\001.tif
%% ================================================================

%% 总路径
rootPath = 'D:\xy\20260611_brainSlices_screening';

grayRoot = fullfile(rootPath, 'new_gray');

rectifiedRoot = fullfile(rootPath, 'new_rectified');

if ~exist(rectifiedRoot, 'dir')
    mkdir(rectifiedRoot);
end

%% ================================================================
% 最新校正参数
%
% LFDisplay输出：
% (x-offset, y-offset, right-dx, right-dy, down-dx, down-dy)
%
% (704.438000, 600.828000, 48.000000, ...
%   2.900000, -3.200000, 48.000000)
%% ================================================================

xCenter = 704.438000;
yCenter = 600.828000;

rx = 48.000000;
ry = 2.900000;

dx = -3.200000;
dy = 48.000000;

%% 微透镜参数
Nnum = 49;

XcutLeft  = 1;
XcutRight = 0;

YcutUp   = 0;
YcutDown = 1;

%% 保留原始代码中的方向修正
ry = -ry;
dx = -dx;

fprintf('实际传入 ImageRect 的参数：\n');
fprintf('xCenter = %.6f\n', xCenter);
fprintf('yCenter = %.6f\n', yCenter);
fprintf('rx      = %.6f\n', rx);
fprintf('ry      = %.6f\n', ry);
fprintf('dx      = %.6f\n', dx);
fprintf('dy      = %.6f\n', dy);

%% Gamma
gamma = 1;

%% ================================================================
% 测试开关
%
% 只测试某一个目录时：
% onlyFolder = '650_3_fov1';
%
% 全部处理时：
% onlyFolder = '';
%% ================================================================

onlyFolder = '';

%% 检查输入目录
if ~exist(grayRoot, 'dir')
    error('找不到输入灰度图目录：\n%s', grayRoot);
end

%% 获取 new_gray 下的子目录
folderList = dir(grayRoot);
folderList = folderList([folderList.isdir]);
folderList = folderList( ...
    ~ismember({folderList.name}, {'.', '..'}) ...
);

if isempty(folderList)
    error('new_gray下面没有数据子目录：\n%s', grayRoot);
end

fprintf('\n发现 %d 个数据目录。\n', numel(folderList));

totalFolder = 0;
totalImageFound = 0;
totalImageSuccess = 0;
totalImageFailed = 0;

%% ================================================================
% 遍历各数据目录
%% ================================================================

for f = 1:numel(folderList)

    folderName = folderList(f).name;

    if ~isempty(onlyFolder) && ~strcmp(folderName, onlyFolder)
        continue;
    end

    inputPath = fullfile(grayRoot, folderName);

    savePath = fullfile(rectifiedRoot, folderName);

    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end

    imgList = getImageList(inputPath);
    imgNum = numel(imgList);

    if imgNum == 0
        warning('没有找到灰度图：%s', inputPath);
        continue;
    end

    fprintf('\n====================================================\n');
    fprintf('正在矫正目录：%s\n', folderName);
    fprintf('输入目录：%s\n', inputPath);
    fprintf('输出目录：%s\n', savePath);
    fprintf('图片数量：%d\n', imgNum);
    fprintf('====================================================\n');

    totalFolder = totalFolder + 1;
    totalImageFound = totalImageFound + imgNum;

    %% 逐张处理
    for i = 1:imgNum

        inputFileName = imgList(i).name;
        inputFullName = fullfile(inputPath, inputFileName);

        %% 读取图像
        try
            IMG_RAW = imread(inputFullName);
            IMG_RAW = im2double(IMG_RAW);
        catch ME
            warning('读取失败：%s\n原因：%s', ...
                inputFullName, ME.message);

            totalImageFailed = totalImageFailed + 1;
            continue;
        end

        %% 转换为二维灰度图
        if ndims(IMG_RAW) == 3

            if size(IMG_RAW, 3) == 3
                IMG_RAW = rgb2gray(IMG_RAW);

            elseif size(IMG_RAW, 3) == 4
                IMG_RAW = rgb2gray(IMG_RAW(:,:,1:3));

            else
                warning('通道数无法识别，跳过：%s', inputFullName);

                totalImageFailed = totalImageFailed + 1;
                continue;
            end
        end

        if ndims(IMG_RAW) ~= 2
            warning('不是二维灰度图，跳过：%s', inputFullName);

            totalImageFailed = totalImageFailed + 1;
            continue;
        end

        %% 检查图像尺寸
        if xCenter < 1 || xCenter > size(IMG_RAW, 2)
            warning( ...
                'xCenter超出图像宽度，跳过：%s\n图像宽度=%d，xCenter=%.3f', ...
                inputFullName, size(IMG_RAW,2), xCenter);

            totalImageFailed = totalImageFailed + 1;
            continue;
        end

        if yCenter < 1 || yCenter > size(IMG_RAW, 1)
            warning( ...
                'yCenter超出图像高度，跳过：%s\n图像高度=%d，yCenter=%.3f', ...
                inputFullName, size(IMG_RAW,1), yCenter);

            totalImageFailed = totalImageFailed + 1;
            continue;
        end

        %% 输入图像归一化
        IMG_RAW(~isfinite(IMG_RAW)) = 0;

        maxVal = max(IMG_RAW(:));

        if maxVal > 0
            IMG_RAW = IMG_RAW ./ maxVal;
        else
            warning('图片全黑，跳过：%s', inputFullName);

            totalImageFailed = totalImageFailed + 1;
            continue;
        end

        %% 执行矫正
        try
            IMG_Rect = ImageRect( ...
                IMG_RAW, ...
                xCenter, ...
                yCenter, ...
                rx, ...
                ry, ...
                dx, ...
                dy, ...
                Nnum, ...
                XcutLeft, ...
                XcutRight, ...
                YcutUp, ...
                YcutDown);
        catch ME
            warning('矫正失败：%s\n原因：%s', ...
                inputFullName, ME.message);

            totalImageFailed = totalImageFailed + 1;
            continue;
        end

        %% 清理非法值
        IMG_Rect(~isfinite(IMG_Rect)) = 0;

        %% 第一次归一化
        maxRect = max(IMG_Rect(:));

        if maxRect > 0
            IMG_Rect = IMG_Rect ./ maxRect;
        else
            warning('矫正结果全黑，跳过：%s', inputFullName);

            totalImageFailed = totalImageFailed + 1;
            continue;
        end

        %% Gamma处理
        IMG_Rect = IMG_Rect .^ gamma;

        %% 第二次归一化
        maxRect = max(IMG_Rect(:));

        if maxRect > 0
            IMG_Rect = IMG_Rect ./ maxRect;
        end

        %% 保存，沿用原文件名
        [~, baseName, ~] = fileparts(inputFileName);

        saveName = [baseName, '.tif'];

        saveFullName = fullfile(savePath, saveName);

        try
            imwrite(IMG_Rect, saveFullName, ...
                'tif', ...
                'Compression', 'none');

            totalImageSuccess = totalImageSuccess + 1;

        catch ME
            warning('保存失败：%s\n原因：%s', ...
                saveFullName, ME.message);

            totalImageFailed = totalImageFailed + 1;
            continue;
        end

        if mod(i, 20) == 0 || i == imgNum
            fprintf('已完成 %d / %d\n', i, imgNum);
        end
    end

    fprintf('目录完成：%s\n', folderName);
end

%% 总结
fprintf('\n====================================================\n');
fprintf('全部矫正结束。\n');
fprintf('处理目录数量：%d\n', totalFolder);
fprintf('发现图片总数：%d\n', totalImageFound);
fprintf('成功处理数量：%d\n', totalImageSuccess);
fprintf('失败或跳过数量：%d\n', totalImageFailed);
fprintf('输出根目录：\n%s\n', rectifiedRoot);
fprintf('====================================================\n');

end


%% ================================================================
% 读取图片列表
%% ================================================================
function imgList = getImageList(folderPath)

imgList = [];

if ~exist(folderPath, 'dir')
    return;
end

imgList = [ ...
    dir(fullfile(folderPath, '*.tif')); ...
    dir(fullfile(folderPath, '*.tiff')); ...
    dir(fullfile(folderPath, '*.bmp')); ...
    dir(fullfile(folderPath, '*.png')); ...
    dir(fullfile(folderPath, '*.jpg')); ...
    dir(fullfile(folderPath, '*.jpeg')) ...
];

if isempty(imgList)
    return;
end

%% 去重
[~, idxUnique] = unique(lower({imgList.name}), 'stable');
imgList = imgList(idxUnique);

%% 按文件名排序
[~, idxSort] = sort(lower({imgList.name}));
imgList = imgList(idxSort);

end


%% ================================================================
% ImageRect函数
%% ================================================================
function IMG_Rect = ImageRect(IMG_BW, xCenter, yCenter, rx, ry, dx, dy, M, ...
                              XcutLeft, XcutRight, YcutUp, YcutDown)

Mdiff = floor(M / 2);

%% Resample the image
Xresample = [ ...
    fliplr((xCenter + 1):-rx/M:1), ...
    ((xCenter + 1) + rx/M):rx/M:size(IMG_BW, 2) ...
];

Yresample = [ ...
    fliplr((yCenter + 1):-dy/M:1), ...
    ((yCenter + 1) + dy/M):dy/M:size(IMG_BW, 1) ...
];

[X, Y] = meshgrid( ...
    1:size(IMG_BW, 2), ...
    1:size(IMG_BW, 1));

[Xq, Yq] = meshgrid(Xresample, Yresample);

XqCenterInit = find(Xq(1,:) == (xCenter + 1), 1) - Mdiff;

XqInit = XqCenterInit ...
    - M * floor(XqCenterInit / M) ...
    + M;

YqCenterInit = find(Yq(:,1) == (yCenter + 1), 1) - Mdiff;

YqInit = YqCenterInit ...
    - M * floor(YqCenterInit / M) ...
    + M;

XresampleQ = Xresample(XqInit:end);
YresampleQ = Yresample(YqInit:end);

[Xqq, Yqq] = meshgrid(XresampleQ, YresampleQ);

numleft = size((xCenter + 1):-rx/M:1, 2);

numright = size( ...
    ((xCenter + 1) + rx/M):rx/M:size(IMG_BW, 2), ...
    2);

numup = size((yCenter + 1):-dy/M:1, 2);

numdown = size( ...
    ((yCenter + 1) + dy/M):dy/M:size(IMG_BW, 1), ...
    2);

Xresample_dx = (-numup + 1:numdown) * ry / M;

Xresample_dx = repmat( ...
    Xresample_dx(YqInit:end)', ...
    1, ...
    size(Xqq, 2));

Xresample_dy = (-numleft + 1:numright) * dx / M;

Xresample_dy = repmat( ...
    Xresample_dy(XqInit:end), ...
    size(Yqq, 1), ...
    1);

Xq_ = Xqq + Xresample_dx;
Yq_ = Yqq + Xresample_dy;

IMG_RESAMPLE = interp2( ...
    X, ...
    Y, ...
    IMG_BW, ...
    Xq_, ...
    Yq_, ...
    'linear', ...
    0);

IMG_RESAMPLE(~isfinite(IMG_RESAMPLE)) = 0;

cropHeight = M * floor( ...
    (size(IMG_RESAMPLE, 1) - YqInit) / M);

cropWidth = M * floor( ...
    (size(IMG_RESAMPLE, 2) - XqInit) / M);

if cropHeight <= 0 || cropWidth <= 0
    error('矫正后有效区域尺寸异常。');
end

IMG_RESAMPLE_crop1 = IMG_RESAMPLE( ...
    1:cropHeight, ...
    1:cropWidth);

%% Crop
XsizeML = size(IMG_RESAMPLE_crop1, 2) / M;
YsizeML = size(IMG_RESAMPLE_crop1, 1) / M;

if (XcutLeft + XcutRight) >= XsizeML
    error('X-cut范围大于或等于图像微透镜数量。');
end

if (YcutUp + YcutDown) >= YsizeML
    error('Y-cut范围大于或等于图像微透镜数量。');
end

Xrange = (1 + XcutLeft):(XsizeML - XcutRight);
Yrange = (1 + YcutUp):(YsizeML - YcutDown);

IMG_RESAMPLE_crop2 = IMG_RESAMPLE_crop1( ...
    ((Yrange(1) - 1) * M + 1):(Yrange(end) * M), ...
    ((Xrange(1) - 1) * M + 1):(Xrange(end) * M));

IMG_Rect = IMG_RESAMPLE_crop2;

end