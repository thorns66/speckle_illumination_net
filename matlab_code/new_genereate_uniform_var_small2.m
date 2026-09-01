clear;
close all;
clc;
warning('off');

%% =========================================================
% 功能：
%
% new_rectified\500_3_fov2\001.tif ...
% new_rectified\650_3_fov1\001.tif ...
% new_rectified\800_2_fov2\001.tif ...
%
% 分别计算：
%   1. uniform：所有帧累加后归一化
%   2. var：所有帧沿第三维计算总体方差
%   3. sqrt：方差开平方后归一化
%
% 输出到：
% new_uniform_var_gamma1_small\对应子目录\
%% =========================================================

%% 路径
rootPath = 'D:\xy\20260611_brainSlices_screening';

inputRoot = fullfile(rootPath, 'new_rectified');

outputRoot = fullfile( ...
    rootPath, ...
    'new_uniform_var_gamma1_small');

if ~exist(inputRoot, 'dir')
    error('找不到输入目录：\n%s', inputRoot);
end

if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
end

%% 参数
gamma = 1;

% 若只测试一个目录：
% onlyFolder = '650_3_fov1';
%
% 全部处理：
onlyFolder = '';

%% 获取输入子目录
folderList = dir(inputRoot);
folderList = folderList([folderList.isdir]);
folderList = folderList( ...
    ~ismember({folderList.name}, {'.', '..'}) ...
);

if isempty(folderList)
    error('输入目录下没有数据子目录：\n%s', inputRoot);
end

fprintf('共检测到 %d 个数据目录。\n', numel(folderList));

successFolderCount = 0;
failedFolderCount = 0;

%% =========================================================
% 遍历每个数据目录
%% =========================================================

for folderIndex = 1:numel(folderList)

    folderName = folderList(folderIndex).name;

    if ~isempty(onlyFolder) && ~strcmp(folderName, onlyFolder)
        continue;
    end

    inputPath = fullfile(inputRoot, folderName);
    savePath  = fullfile(outputRoot, folderName);

    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end

    imgList = getImageList(inputPath);

    if isempty(imgList)
        warning('目录中没有找到图像：\n%s', inputPath);
        failedFolderCount = failedFolderCount + 1;
        continue;
    end

    imgNum = numel(imgList);

    fprintf('\n====================================================\n');
    fprintf('正在处理：%s\n', folderName);
    fprintf('输入目录：%s\n', inputPath);
    fprintf('输出目录：%s\n', savePath);
    fprintf('图像数量：%d\n', imgNum);
    fprintf('gamma：%.3f\n', gamma);
    fprintf('====================================================\n');

    %% -----------------------------------------------------
    % 读取第一张图，确定尺寸
    %% -----------------------------------------------------

    firstPath = fullfile(inputPath, imgList(1).name);

    try
        firstImg = im2double(imread(firstPath));
    catch ME
        warning('第一张图读取失败：%s\n原因：%s', ...
            firstPath, ME.message);

        failedFolderCount = failedFolderCount + 1;
        continue;
    end

    if ndims(firstImg) == 3
        if size(firstImg, 3) == 3
            firstImg = rgb2gray(firstImg);
        elseif size(firstImg, 3) == 4
            firstImg = rgb2gray(firstImg(:,:,1:3));
        else
            warning('第一张图通道数异常：%s', firstPath);
            failedFolderCount = failedFolderCount + 1;
            continue;
        end
    end

    if ndims(firstImg) ~= 2
        warning('第一张图不是二维灰度图：%s', firstPath);
        failedFolderCount = failedFolderCount + 1;
        continue;
    end

    imageHeight = size(firstImg, 1);
    imageWidth  = size(firstImg, 2);

    fprintf('图像尺寸：%d × %d\n', imageHeight, imageWidth);

    %% -----------------------------------------------------
    % 分配三维数组
    %
    % 尺寸：
    % height × width × frame
    %% -----------------------------------------------------

    img_final = zeros( ...
        imageHeight, ...
        imageWidth, ...
        imgNum, ...
        'single');

    validFrameCount = 0;

    %% -----------------------------------------------------
    % 逐张读取
    %% -----------------------------------------------------

    for i = 1:imgNum

        inputFileName = imgList(i).name;
        inputFullName = fullfile(inputPath, inputFileName);

        try
            temp = im2double(imread(inputFullName));
        catch ME
            warning('读取失败，跳过：%s\n原因：%s', ...
                inputFullName, ME.message);
            continue;
        end

        %% 转灰度
        if ndims(temp) == 3

            if size(temp, 3) == 3
                temp = rgb2gray(temp);

            elseif size(temp, 3) == 4
                temp = rgb2gray(temp(:,:,1:3));

            else
                warning('通道数异常，跳过：%s', inputFullName);
                continue;
            end
        end

        if ndims(temp) ~= 2
            warning('不是二维图，跳过：%s', inputFullName);
            continue;
        end

        %% 检查尺寸一致性
        if size(temp, 1) ~= imageHeight || ...
           size(temp, 2) ~= imageWidth

            warning([ ...
                '图像尺寸不一致，跳过：%s\n' ...
                '预期：%d × %d，实际：%d × %d'], ...
                inputFullName, ...
                imageHeight, imageWidth, ...
                size(temp,1), size(temp,2));

            continue;
        end

        %% 清理 NaN / Inf
        temp(~isfinite(temp)) = 0;

        %% 单帧归一化
        maxVal = max(temp(:));

        if maxVal <= 0
            warning('图像全黑，跳过：%s', inputFullName);
            continue;
        end

        temp = temp ./ maxVal;

        %% Gamma
        temp = temp .^ gamma;

        %% Gamma 后再次归一化
        maxVal = max(temp(:));

        if maxVal > 0
            temp = temp ./ maxVal;
        end

        validFrameCount = validFrameCount + 1;

        img_final(:,:,validFrameCount) = single(temp);

        if mod(i, 20) == 0 || i == imgNum
            fprintf('已读取 %d / %d，当前有效帧 %d\n', ...
                i, imgNum, validFrameCount);
        end
    end

    %% 去掉未使用的预分配页面
    if validFrameCount == 0
        warning('目录中没有有效图像：%s', inputPath);
        clear img_final;
        failedFolderCount = failedFolderCount + 1;
        continue;
    end

    img_final = img_final(:,:,1:validFrameCount);

    fprintf('有效帧数量：%d / %d\n', ...
        validFrameCount, imgNum);

    %% =====================================================
    % 计算 uniform
    %
    % 与原代码一致：
    % uniform = sum(img_final, 3)
    %% =====================================================

    uniform = sum(img_final, 3, 'double');

    maxUniform = max(uniform(:));

    if maxUniform > 0
        uniform = uniform ./ maxUniform;
    end

    %% =====================================================
    % 计算方差
    %
    % var(...,1,3)：
    % 使用总体方差，即除以 N
    %% =====================================================

    var_img = var(double(img_final), 1, 3);

    maxVar = max(var_img(:));

    if maxVar > 0
        var_img = var_img ./ maxVar;
    end

    %% =====================================================
    % 计算 sqrt(var)
    %% =====================================================

    sqrt_img = sqrt(var_img);

    maxSqrt = max(sqrt_img(:));

    if maxSqrt > 0
        sqrt_img = sqrt_img ./ maxSqrt;
    end

    %% =====================================================
    % 保存
    %% =====================================================

    uniformName = sprintf( ...
        '%s_uniform.tif', ...
        folderName);

    varName = sprintf( ...
        '%s_var.tif', ...
        folderName);

    sqrtName = sprintf( ...
        '%s_sqrt.tif', ...
        folderName);

    uniformPath = fullfile(savePath, uniformName);
    varPath     = fullfile(savePath, varName);
    sqrtPath    = fullfile(savePath, sqrtName);

    try
        imwrite(uniform, uniformPath, ...
            'tif', ...
            'Compression', 'none');

        imwrite(var_img, varPath, ...
            'tif', ...
            'Compression', 'none');

        imwrite(sqrt_img, sqrtPath, ...
            'tif', ...
            'Compression', 'none');

    catch ME
        warning('保存失败：%s\n原因：%s', ...
            savePath, ME.message);

        clear img_final uniform var_img sqrt_img;
        failedFolderCount = failedFolderCount + 1;
        continue;
    end

    %% 保存处理信息
    infoFile = fullfile(savePath, 'processing_info.txt');

    fid = fopen(infoFile, 'w');

    if fid ~= -1
        fprintf(fid, 'Dataset: %s\n', folderName);
        fprintf(fid, 'Input folder: %s\n', inputPath);
        fprintf(fid, 'Gamma: %.6f\n', gamma);
        fprintf(fid, 'Found frames: %d\n', imgNum);
        fprintf(fid, 'Valid frames: %d\n', validFrameCount);
        fprintf(fid, 'Image height: %d\n', imageHeight);
        fprintf(fid, 'Image width: %d\n', imageWidth);
        fprintf(fid, 'Variance normalization: population, N\n');
        fclose(fid);
    end

    fprintf('已保存：\n');
    fprintf('  %s\n', uniformPath);
    fprintf('  %s\n', varPath);
    fprintf('  %s\n', sqrtPath);

    clear img_final uniform var_img sqrt_img;

    successFolderCount = successFolderCount + 1;
end

%% =========================================================
% 完成
%% =========================================================

fprintf('\n====================================================\n');
fprintf('全部处理结束。\n');
fprintf('成功目录数量：%d\n', successFolderCount);
fprintf('失败或跳过目录数量：%d\n', failedFolderCount);
fprintf('输出根目录：\n%s\n', outputRoot);
fprintf('====================================================\n');


%% =========================================================
% 获取图像列表
%% =========================================================
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
lowerNames = lower({imgList.name});
[~, idxUnique] = unique(lowerNames, 'stable');
imgList = imgList(idxUnique);

%% 自然数字排序
% 当前文件名是 001、002……，普通字符排序已经正确
[~, idxSort] = sort(lower({imgList.name}));
imgList = imgList(idxSort);

end