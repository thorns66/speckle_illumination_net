function ImageRectification2_brainSlices
warning('off');
close all;
clc;

%% ================================================================
%  功能：
%  gray\切片\FOV\*.tif  →  rectified\切片\FOV\*.tif
%
%  你的目录结构应为：
%  D:\xy\20260611_brainSlices_screening\
%  ├─ gray\
%  │  ├─ 500um_2\FOV01\001.tif ...
%  │  └─ ...
%  └─ rectified\
%
%% ================================================================

%% ====== 总路径 ======
rootPath = 'D:\xy\20260611_brainSlices_screening';

grayRoot      = fullfile(rootPath, 'gray');
rectifiedRoot = fullfile(rootPath, 'rectified');

if ~exist(rectifiedRoot, 'dir')
    mkdir(rectifiedRoot);
end

%% ====== 新矫正参数 ======
% LFDisplay 输出：
% (x-offset, y-offset, right-dx, right-dy, down-dx, down-dy)
% (735.338000, 582.728000, 48.000000, -1.500000, 2.500000, 48.525000)

xCenter             = 735.338000;
yCenter             = 582.728000;
rx                  = 48.000000;
ry                  = -1.500000;
dx                  = 2.500000;
dy                  = 48.525000;

Nnum                = 49;
XcutLeft            = 1;
XcutRight           = 0;
YcutUp              = 0;
YcutDown            = 1;

% 保留原代码里的方向修正
ry = -ry;
dx = -dx;

gamma = 1;

%% ====== 测试开关 ======
% 如果只想测试一个视场，就填：
% onlySlice = '500um_2';
% onlyFOV   = 'FOV01';
%
% 如果要全部批量处理，就保持空字符串：
onlySlice = '';
onlyFOV   = '';

%% ====== 检查 gray 文件夹是否存在 ======
if ~exist(grayRoot, 'dir')
    error('找不到 gray 文件夹：%s', grayRoot);
end

%% ====== 遍历切片文件夹 ======
sliceList = dir(grayRoot);
sliceList = sliceList([sliceList.isdir]);
sliceList = sliceList(~ismember({sliceList.name}, {'.', '..'}));

if isempty(sliceList)
    error('gray 文件夹下面没有切片文件夹：%s', grayRoot);
end

totalFOV = 0;
totalImg = 0;

for s = 1:length(sliceList)

    sliceName = sliceList(s).name;

    if ~isempty(onlySlice) && ~strcmp(sliceName, onlySlice)
        continue;
    end

    sliceGrayPath = fullfile(grayRoot, sliceName);

    fovList = dir(sliceGrayPath);
    fovList = fovList([fovList.isdir]);
    fovList = fovList(~ismember({fovList.name}, {'.', '..'}));

    if isempty(fovList)
        warning('该切片下面没有 FOV 文件夹：%s', sliceGrayPath);
        continue;
    end

    for f = 1:length(fovList)

        fovName = fovList(f).name;

        if ~isempty(onlyFOV) && ~strcmp(fovName, onlyFOV)
            continue;
        end

        inputPath = fullfile(grayRoot, sliceName, fovName);
        savePath  = fullfile(rectifiedRoot, sliceName, fovName);

        if ~exist(savePath, 'dir')
            mkdir(savePath);
        end

        imgList = getImageList(inputPath);
        imgNum = length(imgList);

        if imgNum == 0
            warning('没有找到灰度图：%s', inputPath);
            continue;
        end

        fprintf('\n正在矫正：%s / %s，共 %d 张图\n', sliceName, fovName, imgNum);

        totalFOV = totalFOV + 1;
        totalImg = totalImg + imgNum;

        for i = 1:imgNum

            inputFileName = imgList(i).name;
            inputFullName = fullfile(inputPath, inputFileName);

            try
                IMG_RAW = im2double(imread(inputFullName));
            catch ME
                warning('读取失败：%s\n原因：%s', inputFullName, ME.message);
                continue;
            end

            % 如果是 RGB 图，先转灰度
            if size(IMG_RAW, 3) == 3
                IMG_RAW = rgb2gray(IMG_RAW);
            end

            % 如果是 RGBA，只取前三通道转灰度
            if size(IMG_RAW, 3) == 4
                IMG_RAW = rgb2gray(IMG_RAW(:,:,1:3));
            end

            % 如果仍然不是二维图，跳过
            if ndims(IMG_RAW) ~= 2
                warning('不是二维灰度图，跳过：%s', inputFullName);
                continue;
            end

            % 归一化，防止除以 0
            maxVal = max(IMG_RAW(:));
            if maxVal > 0
                IMG_RAW = IMG_RAW ./ maxVal;
            else
                warning('图片全黑，跳过：%s', inputFullName);
                continue;
            end

            % 矫正
            try
                IMG_Rect = ImageRect(IMG_RAW, xCenter, yCenter, rx, ry, dx, dy, ...
                                     Nnum, XcutLeft, XcutRight, YcutUp, YcutDown);
            catch ME
                warning('矫正失败：%s\n原因：%s', inputFullName, ME.message);
                continue;
            end

            % 清理 NaN / Inf
            IMG_Rect(isnan(IMG_Rect)) = 0;
            IMG_Rect(isinf(IMG_Rect)) = 0;

            % 归一化
            maxRect = max(IMG_Rect(:));
            if maxRect > 0
                IMG_Rect = IMG_Rect ./ maxRect;
            end

            % gamma，默认 gamma = 1，不改变图像
            IMG_Rect = IMG_Rect .^ gamma;

            maxRect = max(IMG_Rect(:));
            if maxRect > 0
                IMG_Rect = IMG_Rect ./ maxRect;
            end

            % 保存，沿用原文件名
            [~, baseName, ~] = fileparts(inputFileName);
            saveName = [baseName, '.tif'];
            saveFullName = fullfile(savePath, saveName);

            imwrite(IMG_Rect, saveFullName);

            if mod(i, 20) == 0 || i == imgNum
                fprintf('  已完成 %d / %d\n', i, imgNum);
            end
        end

        fprintf('完成：%s / %s\n', sliceName, fovName);
    end
end

fprintf('\n全部矫正完成。\n');
fprintf('处理 FOV 数量：%d\n', totalFOV);
fprintf('处理图像总数：%d\n', totalImg);

end


%% ================================================================
%  读取图片列表
%% ================================================================
function imgList = getImageList(folderPath)

imgList = [];

if ~exist(folderPath, 'dir')
    return;
end

% Windows 下大小写不敏感，所以不要同时写 *.tif 和 *.TIF，避免重复
imgList = [ ...
    dir(fullfile(folderPath, '*.tif')); ...
    dir(fullfile(folderPath, '*.tiff')); ...
    dir(fullfile(folderPath, '*.bmp')); ...
    dir(fullfile(folderPath, '*.png')) ...
];

if isempty(imgList)
    return;
end

% 去重
[~, idxUnique] = unique({imgList.name});
imgList = imgList(idxUnique);

% 按文件名排序
[~, idxSort] = sort({imgList.name});
imgList = imgList(idxSort);

end


%% ================================================================
%  原 ImageRect 函数：不要随便改
%% ================================================================
function IMG_Rect = ImageRect(IMG_BW, xCenter, yCenter, rx, ry, dx, dy, M, ...
                              XcutLeft, XcutRight, YcutUp, YcutDown)

Mdiff = floor(M/2);

%% Resample the image
Xresample = [fliplr((xCenter+1):-rx/M:1), ...
             ((xCenter+1)+rx/M:rx/M:size(IMG_BW,2))];

Yresample = [fliplr((yCenter+1):-dy/M:1), ...
             ((yCenter+1)+dy/M:dy/M:size(IMG_BW,1))];

[X, Y] = meshgrid((1:1:size(IMG_BW,2)), ...
                  (1:1:size(IMG_BW,1)));

[Xq, Yq] = meshgrid(Xresample, Yresample);

XqCenterInit = find(Xq(1,:) == (xCenter+1)) - Mdiff;
XqInit = XqCenterInit - M * floor(XqCenterInit / M) + M;

YqCenterInit = find(Yq(:,1) == (yCenter+1)) - Mdiff;
YqInit = YqCenterInit - M * floor(YqCenterInit / M) + M;

XresampleQ = Xresample(XqInit:end);
YresampleQ = Yresample(YqInit:end);

[Xqq, Yqq] = meshgrid(XresampleQ, YresampleQ);

numleft  = size((xCenter+1):-rx/M:1, 2);
numright = size(((xCenter+1)+rx/M:rx/M:size(IMG_BW,2)), 2);
numup    = size((yCenter+1):-dy/M:1, 2);
numdown  = size(((yCenter+1)+dy/M:dy/M:size(IMG_BW,1)), 2);

Xresample_dx = (-numup+1:1:numdown) * ry / M;
Xresample_dx = repmat(Xresample_dx(YqInit:end)', 1, size(Xqq,2));

Xresample_dy = (-numleft+1:1:numright) * dx / M;
Xresample_dy = repmat(Xresample_dy(XqInit:end), size(Yqq,1), 1);

Xq_ = Xqq + Xresample_dx;
Yq_ = Yqq + Xresample_dy;

IMG_RESAMPLE = interp2(X, Y, IMG_BW, Xq_, Yq_);

IMG_RESAMPLE(isnan(IMG_RESAMPLE)) = 0;

IMG_RESAMPLE_crop1 = IMG_RESAMPLE( ...
    1:1:M*floor((size(IMG_RESAMPLE,1)-YqInit)/M), ...
    1:1:M*floor((size(IMG_RESAMPLE,2)-XqInit)/M) ...
);

%% Crop the right portion
XsizeML = size(IMG_RESAMPLE_crop1, 2) / M;
YsizeML = size(IMG_RESAMPLE_crop1, 1) / M;

if (XcutLeft + XcutRight) >= XsizeML
    error('X-cut range is larger than the x-size of image');
end

if (YcutUp + YcutDown) >= YsizeML
    error('Y-cut range is larger than the y-size of image');
end

Xrange = (1 + XcutLeft) : (XsizeML - XcutRight);
Yrange = (1 + YcutUp)   : (YsizeML - YcutDown);

IMG_RESAMPLE_crop2 = IMG_RESAMPLE_crop1( ...
    ((Yrange(1)-1)*M+1) : (Yrange(end)*M), ...
    ((Xrange(1)-1)*M+1) : (Xrange(end)*M) ...
);

IMG_Rect = IMG_RESAMPLE_crop2;

end