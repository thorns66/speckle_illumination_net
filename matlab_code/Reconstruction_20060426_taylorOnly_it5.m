function Reconstruction_20060426_taylorOnly_it5
clear; close all; clc;
warning('on');

%% ================================================================
% 功能：
% 只重建 Taylor / var 结果，迭代次数 5 次
%
% 只处理：
% 1. 800um_2 / FOV02
% 2. 800um_3 / FOV02
%
% 输入：
% recon_it5\切片\FOV\uniform_var_used\var_img.tif
%
% 输出：
% recon_it5\切片\FOV\taylor_var\
%
% 注意：
% 本代码不重建 uniform，不重建 100 张 single_rectified。
%% ================================================================

%% ====== 总路径 ======
rootPath = 'D:\xy\20260611_brainSlices_screening';

inputRectifiedRoot = fullfile(rootPath, 'rectified');
saveRoot = fullfile(rootPath, 'recon_it5');

%% ====== 只重建这两个最终筛选视场 ======
selectedList = {
    '800um_2', 'FOV02'
    '800um_3', 'FOV02'
};

%% ====== 当前代码路径 ======
codePath = fullfile(rootPath, 'code_used');

addpath(genpath(codePath));
addpath(genpath('D:\xy\Util'));
addpath(genpath('D:\xy\Solver'));

%% ====== PSF 路径 ======
psfHtPath = 'F:\xy\20250626-大深度微型系统\合并psf\merged-340_to_340_Ht.mat';
psfHPath  = 'F:\xy\20250626-大深度微型系统\合并psf\merged-340_to_340_H.mat';

if ~exist(psfHtPath, 'file')
    error('找不到 Ht 文件：%s', psfHtPath);
end

if ~exist(psfHPath, 'file')
    error('找不到 H 文件：%s', psfHPath);
end

%% ====== 检查函数 ======
if isempty(which('Reconstruction3D_speckle'))
    error('找不到 Reconstruction3D_speckle.m，请检查 code_used 是否已加入 MATLAB 路径。');
end

if isempty(which('backwardProjectGPU'))
    error('找不到 backwardProjectGPU.m，请检查 D:\xy\Util 是否已加入 MATLAB 路径。');
end

if isempty(which('forwardProjectGPU'))
    error('找不到 forwardProjectGPU.m，请检查 D:\xy\Util 是否已加入 MATLAB 路径。');
end

if isempty(which('deconvRL'))
    error('找不到 deconvRL.m，请检查 D:\xy\Solver 是否已加入 MATLAB 路径。');
end

fprintf('当前使用的关键函数：\n');
fprintf('Reconstruction3D_speckle: %s\n', which('Reconstruction3D_speckle'));
fprintf('backwardProjectGPU:       %s\n', which('backwardProjectGPU'));
fprintf('forwardProjectGPU:        %s\n', which('forwardProjectGPU'));
fprintf('deconvRL:                 %s\n', which('deconvRL'));

%% ====== 重建参数 ======
iterNum = 5;
zStart  = -340;
zEnd    = 340;

%% ====== 加载 PSF，并直接变成 Taylor 用的平方 PSF ======
fprintf('\n正在加载 Taylor 用 PSF...\n');

depthIdx = [1:18, 20:36];

%% 先加载 H，裁剪后转 single，然后平方
fprintf('正在加载 H，并生成 H.^2 ...\n');
S_H = load(psfHPath);

if ~isfield(S_H, 'H')
    error('PSF 文件中没有变量 H。');
end

H_speckle = single(S_H.H(:,:,:,:,depthIdx));
clear S_H;

H_speckle = H_speckle .^ 2;

fprintf('H.^2 准备完成。\n');

%% 再加载 Ht 和 CAindex，裁剪后转 single，然后平方
fprintf('正在加载 Ht 和 CAindex，并生成 Ht.^2 ...\n');
S_Ht = load(psfHtPath);

if ~isfield(S_Ht, 'Ht')
    error('PSF 文件中没有变量 Ht。');
end

if ~isfield(S_Ht, 'CAindex')
    error('PSF 文件中没有变量 CAindex。');
end

Ht_speckle = single(S_Ht.Ht(:,:,:,:,depthIdx));
CAindex_use = S_Ht.CAindex(depthIdx,:);

clear S_Ht;

Ht_speckle = Ht_speckle .^ 2;

fprintf('Ht.^2 和 CAindex 准备完成。\n');

%% 清理 GPU 显存
try
    g = gpuDevice;
    reset(g);
    fprintf('GPU 已 reset，显存已清理。\n');
catch
    fprintf('未检测到 GPU 或 GPU reset 失败，继续运行。\n');
end

%% ====== 日志 ======
logCell = {};
logCell(end+1,:) = {'slice', 'fov', 'var_status', 'message'};

successFOV = 0;
failFOV = 0;

%% ====== 逐个重建 Taylor / var ======
for idx = 1:size(selectedList, 1)

    sliceName = selectedList{idx, 1};
    fovName   = selectedList{idx, 2};

    fprintf('\n====================================================\n');
    fprintf('开始 Taylor / var 5 次迭代重建：%s / %s\n', sliceName, fovName);
    fprintf('====================================================\n');

    %% ====== var_img.tif 输入路径 ======
    saveFOVRoot = fullfile(saveRoot, sliceName, fovName);
    uniformVarPath = fullfile(saveFOVRoot, 'uniform_var_used');
    varFullName = fullfile(uniformVarPath, 'var_img.tif');

    %% 如果 var_img.tif 不存在，则从 rectified 的 100 张图重新生成
    if ~exist(varFullName, 'file')

        fprintf('没有找到 var_img.tif，准备从 100 张矫正图重新生成...\n');

        rectifiedPath = fullfile(inputRectifiedRoot, sliceName, fovName);

        if ~exist(rectifiedPath, 'dir')
            warning('找不到矫正图文件夹：%s', rectifiedPath);
            failFOV = failFOV + 1;
            logCell(end+1,:) = {sliceName, fovName, 'missing', 'missing rectified folder'};
            continue;
        end

        if ~exist(uniformVarPath, 'dir')
            mkdir(uniformVarPath);
        end

        try
            generateUniformVarFromImages(rectifiedPath, uniformVarPath);
        catch ME
            failFOV = failFOV + 1;
            logCell(end+1,:) = {sliceName, fovName, 'failed', ['generate var failed: ', ME.message]};
            fprintf(2, '生成 var_img.tif 失败：%s\n', ME.message);
            continue;
        end
    end

    if ~exist(varFullName, 'file')
        warning('仍然找不到 var_img.tif：%s', varFullName);
        failFOV = failFOV + 1;
        logCell(end+1,:) = {sliceName, fovName, 'missing', 'missing var_img.tif'};
        continue;
    end

    %% ====== Taylor 输出路径 ======
    savePathVar = fullfile(saveFOVRoot, 'taylor_var');

    % 清空旧的空目录或失败残留，避免混淆
    if exist(savePathVar, 'dir')
        rmdir(savePathVar, 's');
    end
    mkdir(savePathVar);

    uniformVarPathWithSep = [uniformVarPath, filesep];
    savePathVarWithSep    = [savePathVar, filesep];

    %% ====== Taylor / var 重建 ======
    try
        fprintf('正在重建 var_img.tif，迭代次数 = %d ...\n', iterNum);

        Reconstruction3D_speckle( ...
            H_speckle, ...
            Ht_speckle, ...
            CAindex_use, ...
            uniformVarPathWithSep, ...
            'var_img.tif', ...
            savePathVarWithSep, ...
            iterNum, ...
            zStart, ...
            zEnd ...
        );

        successFOV = successFOV + 1;
        logCell(end+1,:) = {sliceName, fovName, 'success', ''};

        fprintf('Taylor / var 重建完成：%s / %s\n', sliceName, fovName);

    catch ME
        failFOV = failFOV + 1;
        logCell(end+1,:) = {sliceName, fovName, 'failed', ME.message};

        fprintf(2, '\nTaylor / var 重建失败：%s / %s\n', sliceName, fovName);
        fprintf(2, '原因：%s\n', ME.message);
        fprintf(2, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
    end

    %% 每个 FOV 后清一下 GPU
    try
        g = gpuDevice;
        reset(g);
        fprintf('当前 FOV 结束，GPU 已 reset。\n');
    catch
    end
end

%% ====== 保存日志 ======
logPath = fullfile(saveRoot, 'reconstruction_taylorOnly_it5_log.xlsx');

try
    writecell(logCell, logPath);
    fprintf('\nTaylor-only 日志已保存：%s\n', logPath);
catch
    logPathTxt = fullfile(saveRoot, 'reconstruction_taylorOnly_it5_log.txt');
    fid = fopen(logPathTxt, 'w');

    for i = 1:size(logCell,1)
        fprintf(fid, '%s\t%s\t%s\t%s\n', ...
            logCell{i,1}, logCell{i,2}, logCell{i,3}, logCell{i,4});
    end

    fclose(fid);
    fprintf('\nExcel 日志保存失败，已保存 txt 日志：%s\n', logPathTxt);
end

fprintf('\n====================================================\n');
fprintf('Taylor-only 5 次迭代重建结束。\n');
fprintf('成功 FOV 数量：%d\n', successFOV);
fprintf('失败 FOV 数量：%d\n', failFOV);
fprintf('输出目录：%s\n', saveRoot);
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
    dir(fullfile(folderPath, '*.png')) ...
];

if isempty(imgList)
    return;
end

[~, idxUnique] = unique({imgList.name});
imgList = imgList(idxUnique);

names = {imgList.name};
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

imgList = imgList(idxSort);

end


%% ================================================================
% 从 100 张矫正图生成 uniform_img.tif 和 var_img.tif
%% ================================================================
function generateUniformVarFromImages(inputPath, savePath)

imgList = getImageList(inputPath);
imgNum = length(imgList);

if imgNum == 0
    error('无法生成 uniform/var，因为没有找到图像：%s', inputPath);
end

fprintf('正在从 %d 张图生成 uniform_img.tif 和 var_img.tif ...\n', imgNum);

firstImg = im2double(imread(fullfile(inputPath, imgList(1).name)));

if size(firstImg, 3) == 3
    firstImg = rgb2gray(firstImg);
end

if size(firstImg, 3) == 4
    firstImg = rgb2gray(firstImg(:,:,1:3));
end

firstImg(isnan(firstImg)) = 0;
firstImg(isinf(firstImg)) = 0;

[H, W] = size(firstImg);

imgStack = zeros(H, W, imgNum);

for i = 1:imgNum

    temp = im2double(imread(fullfile(inputPath, imgList(i).name)));

    if size(temp, 3) == 3
        temp = rgb2gray(temp);
    end

    if size(temp, 3) == 4
        temp = rgb2gray(temp(:,:,1:3));
    end

    temp(isnan(temp)) = 0;
    temp(isinf(temp)) = 0;

    if size(temp,1) ~= H || size(temp,2) ~= W
        error('图像尺寸不一致：%s', fullfile(inputPath, imgList(i).name));
    end

    maxVal = max(temp(:));
    if maxVal > 0
        temp = temp ./ maxVal;
    end

    imgStack(:,:,i) = temp;
end

uniform_img = mean(imgStack, 3);
uniform_img(isnan(uniform_img)) = 0;
uniform_img(isinf(uniform_img)) = 0;

if max(uniform_img(:)) > 0
    uniform_img = uniform_img ./ max(uniform_img(:));
end

var_img = var(imgStack, 1, 3);
var_img(isnan(var_img)) = 0;
var_img(isinf(var_img)) = 0;

if max(var_img(:)) > 0
    var_img = var_img ./ max(var_img(:));
end

imwrite(uniform_img, fullfile(savePath, 'uniform_img.tif'));
imwrite(var_img,     fullfile(savePath, 'var_img.tif'));

fprintf('uniform_img.tif 和 var_img.tif 已保存到：%s\n', savePath);

end