function Reconstruction_20060426_selectedFOV_it5
clear; close all; clc;
warning('off');

%% ================================================================
% 功能：
% 对最终筛选出来的 2 个好视场做 5 次迭代正式重建
%
% 本代码只处理：
% 1. 800um_2 / FOV02
% 2. 800um_3 / FOV02
%
% 输入：
% D:\xy\20260611_brainSlices_screening\rectified\800um_2\FOV02\100张矫正灰度图
% D:\xy\20260611_brainSlices_screening\rectified\800um_3\FOV02\100张矫正灰度图
%
% 输出：
% D:\xy\20260611_brainSlices_screening\recon_it5\800um_2\FOV02\uniform\
% D:\xy\20260611_brainSlices_screening\recon_it5\800um_2\FOV02\taylor_var\
% D:\xy\20260611_brainSlices_screening\recon_it5\800um_2\FOV02\single_rectified\
%
% D:\xy\20260611_brainSlices_screening\recon_it5\800um_3\FOV02\uniform\
% D:\xy\20260611_brainSlices_screening\recon_it5\800um_3\FOV02\taylor_var\
% D:\xy\20260611_brainSlices_screening\recon_it5\800um_3\FOV02\single_rectified\
%
% 每个 FOV 会做三类重建：
% 1. uniform_img.tif  5次迭代重建
% 2. var_img.tif      5次迭代重建
% 3. 100张矫正图      每张分别5次迭代重建
%% ================================================================

%% ====== 总路径 ======
rootPath = 'D:\xy\20260611_brainSlices_screening';

% 矫正后的灰度图路径
inputRectifiedRoot = fullfile(rootPath, 'rectified');

% 5 次迭代正式重建输出路径
saveRoot = fullfile(rootPath, 'recon_it5');

if ~exist(saveRoot, 'dir')
    mkdir(saveRoot);
end

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

%% ====== 检查重建函数 ======
if isempty(which('Reconstruction3D'))
    error('找不到 Reconstruction3D.m，请检查 code_used 是否已加入 MATLAB 路径。');
end

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
fprintf('Reconstruction3D:         %s\n', which('Reconstruction3D'));
fprintf('Reconstruction3D_speckle: %s\n', which('Reconstruction3D_speckle'));
fprintf('backwardProjectGPU:       %s\n', which('backwardProjectGPU'));
fprintf('forwardProjectGPU:        %s\n', which('forwardProjectGPU'));
fprintf('deconvRL:                 %s\n', which('deconvRL'));

%% ====== 加载 PSF，并尽量节省内存 ======
fprintf('\n正在加载 PSF...\n');

depthIdx = [1:18, 20:36];

%% 先加载 H，截取需要的深度后立刻清掉完整 H
fprintf('正在加载 H...\n');
S_H = load(psfHPath);

if ~isfield(S_H, 'H')
    error('PSF 文件中没有变量 H。');
end

H_use = single(S_H.H(:,:,:,:,depthIdx));
clear S_H;

fprintf('H 加载并裁剪完成。\n');

%% 再加载 Ht 和 CAindex，截取需要的深度后立刻清掉完整 Ht
fprintf('正在加载 Ht 和 CAindex...\n');
S_Ht = load(psfHtPath);

if ~isfield(S_Ht, 'Ht')
    error('PSF 文件中没有变量 Ht。');
end

if ~isfield(S_Ht, 'CAindex')
    error('PSF 文件中没有变量 CAindex。');
end

Ht_use = single(S_Ht.Ht(:,:,:,:,depthIdx));
CAindex_use = S_Ht.CAindex(depthIdx,:);

clear S_Ht;

fprintf('PSF 加载完成。\n');

%% 清理内存
drawnow;

%% ====== 重建参数 ======
iterNum = 5;
zStart  = -340;
zEnd    = 340;

%% ====== 检查输入路径 ======
if ~exist(inputRectifiedRoot, 'dir')
    error('找不到 rectified 文件夹：%s', inputRectifiedRoot);
end

%% ====== 日志初始化 ======
totalFOV = size(selectedList, 1);
successFOV = 0;
failFOV = 0;

logCell = {};
logCell(end+1,:) = {'slice', 'fov', 'image_num', 'uniform_status', 'var_status', 'single_status', 'message'};

%% ====== 逐个处理最终筛选视场 ======
for idx = 1:size(selectedList, 1)

    sliceName = selectedList{idx, 1};
    fovName   = selectedList{idx, 2};

    fprintf('\n====================================================\n');
    fprintf('开始 5 次迭代正式重建：%s / %s\n', sliceName, fovName);
    fprintf('====================================================\n');

    %% ====== 输入：100 张矫正灰度图 ======
    rectifiedPath = fullfile(inputRectifiedRoot, sliceName, fovName);
    rectifiedPathWithSep = [rectifiedPath, filesep];

    if ~exist(rectifiedPath, 'dir')
        warning('找不到该视场的矫正灰度图文件夹：%s', rectifiedPath);
        failFOV = failFOV + 1;
        logCell(end+1,:) = {sliceName, fovName, 0, 'not_run', 'not_run', 'missing', 'missing rectified FOV folder'};
        continue;
    end

    imgList = getImageList(rectifiedPath);
    imgNum = length(imgList);

    if imgNum == 0
        warning('该视场没有找到矫正灰度图：%s', rectifiedPath);
        failFOV = failFOV + 1;
        logCell(end+1,:) = {sliceName, fovName, imgNum, 'not_run', 'not_run', 'missing', 'missing rectified images'};
        continue;
    end

    if imgNum ~= 100
        warning('%s / %s 不是 100 张图，而是 %d 张，请检查。', sliceName, fovName, imgNum);
    end

    fprintf('找到矫正灰度图数量：%d\n', imgNum);

    %% ====== 输出路径 ======
    saveFOVRoot = fullfile(saveRoot, sliceName, fovName);

    savePathUniform = fullfile(saveFOVRoot, 'uniform');
    savePathVar     = fullfile(saveFOVRoot, 'taylor_var');
    savePathSingle  = fullfile(saveFOVRoot, 'single_rectified');
    savePathUV      = fullfile(saveFOVRoot, 'uniform_var_used');

    if ~exist(savePathUniform, 'dir')
        mkdir(savePathUniform);
    end

    if ~exist(savePathVar, 'dir')
        mkdir(savePathVar);
    end

    if ~exist(savePathSingle, 'dir')
        mkdir(savePathSingle);
    end

    if ~exist(savePathUV, 'dir')
        mkdir(savePathUV);
    end

    savePathUniform = [savePathUniform, filesep];
    savePathVar     = [savePathVar, filesep];

    %% ====== 从 100 张矫正图生成 uniform_img.tif 和 var_img.tif ======
    fprintf('\n正在生成 uniform_img.tif 和 var_img.tif ...\n');
    try
        generateUniformVarFromImages(rectifiedPath, savePathUV);
    catch ME
        warning('生成 uniform / var 失败：%s / %s\n原因：%s', sliceName, fovName, ME.message);
        failFOV = failFOV + 1;
        logCell(end+1,:) = {sliceName, fovName, imgNum, 'failed', 'failed', 'not_run', ['generate uniform/var failed: ', ME.message]};
        continue;
    end

    uniformVarPathWithSep = [savePathUV, filesep];

    uniformStatus = 'not_run';
    varStatus     = 'not_run';
    singleStatus  = 'not_run';
    message       = '';

    %% ====== 1. 均匀照明 5 次迭代重建 ======
    try
        fprintf('\n[1/3] 正在重建 uniform_img.tif，迭代次数 = %d ...\n', iterNum);

        Reconstruction3D( ...
            H_use, ...
            Ht_use, ...
            CAindex_use, ...
            uniformVarPathWithSep, ...
            'uniform_img.tif', ...
            savePathUniform, ...
            iterNum, ...
            zStart, ...
            zEnd ...
        );

        uniformStatus = 'success';
        fprintf('uniform 重建完成：%s / %s\n', sliceName, fovName);

    catch ME
        uniformStatus = 'failed';
        message = [message, ' uniform failed: ', ME.message];
        warning('uniform 重建失败：%s / %s\n原因：%s', sliceName, fovName, ME.message);
    end

    %% ====== 2. Taylor / var 5 次迭代重建 ======
    try
        fprintf('\n[2/3] 正在重建 var_img.tif，迭代次数 = %d ...\n', iterNum);

        Reconstruction3D_speckle( ...
            H_use.^2, ...
            Ht_use.^2, ...
            CAindex_use, ...
            uniformVarPathWithSep, ...
            'var_img.tif', ...
            savePathVar, ...
            iterNum, ...
            zStart, ...
            zEnd ...
        );

        varStatus = 'success';
        fprintf('var / Taylor 重建完成：%s / %s\n', sliceName, fovName);

    catch ME
        varStatus = 'failed';
        message = [message, ' var failed: ', ME.message];
        warning('var / Taylor 重建失败：%s / %s\n原因：%s', sliceName, fovName, ME.message);
    end

    %% ====== 3. 100 张矫正灰度图逐张 5 次迭代重建 ======
    try
        fprintf('\n[3/3] 正在逐张重建 %d 张矫正灰度图，迭代次数 = %d ...\n', imgNum, iterNum);

        for j = 1:imgNum

            inputFileName = imgList(j).name;

            savePathJ = fullfile(savePathSingle, sprintf('img_%03d', j));
            if ~exist(savePathJ, 'dir')
                mkdir(savePathJ);
            end
            savePathJ = [savePathJ, filesep];

            fprintf('正在重建第 %d / %d 张：%s\n', j, imgNum, inputFileName);

            Reconstruction3D( ...
                H_use, ...
                Ht_use, ...
                CAindex_use, ...
                rectifiedPathWithSep, ...
                inputFileName, ...
                savePathJ, ...
                iterNum, ...
                zStart, ...
                zEnd ...
            );
        end

        singleStatus = 'success';
        fprintf('100 张单独矫正图重建完成：%s / %s\n', sliceName, fovName);

    catch ME
        singleStatus = 'failed';
        message = [message, ' single rectified failed: ', ME.message];
        warning('单独矫正图重建失败：%s / %s\n原因：%s', sliceName, fovName, ME.message);
    end

    %% ====== 当前 FOV 统计 ======
    if strcmp(uniformStatus, 'success') && strcmp(varStatus, 'success') && strcmp(singleStatus, 'success')
        successFOV = successFOV + 1;
    else
        failFOV = failFOV + 1;
    end

    logCell(end+1,:) = {sliceName, fovName, imgNum, uniformStatus, varStatus, singleStatus, message};

    fprintf('\n完成当前 FOV：%s / %s\n', sliceName, fovName);
end

%% ====== 保存日志 ======
logPath = fullfile(saveRoot, 'reconstruction_it5_log.xlsx');

try
    writecell(logCell, logPath);
    fprintf('\n重建日志已保存：%s\n', logPath);
catch
    logPathTxt = fullfile(saveRoot, 'reconstruction_it5_log.txt');
    fid = fopen(logPathTxt, 'w');

    for i = 1:size(logCell,1)
        fprintf(fid, '%s\t%s\t%d\t%s\t%s\t%s\t%s\n', ...
            logCell{i,1}, logCell{i,2}, logCell{i,3}, logCell{i,4}, logCell{i,5}, logCell{i,6}, logCell{i,7});
    end

    fclose(fid);
    fprintf('\nExcel 日志保存失败，已保存 txt 日志：%s\n', logPathTxt);
end

fprintf('\n====================================================\n');
fprintf('全部 5 次迭代正式重建结束。\n');
fprintf('总 FOV 数量：%d\n', totalFOV);
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

% 去重
[~, idxUnique] = unique({imgList.name});
imgList = imgList(idxUnique);

% 尽量按文件名里的数字排序
names = {imgList.name};
nums = nan(length(names), 1);

for k = 1:length(names)
    token = regexp(names{k}, '\d+', 'match', 'once');
    if ~isempty(token)
        nums(k) = str2double(token);
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

%% ====== 计算 uniform 图 ======
uniform_img = mean(imgStack, 3);
uniform_img(isnan(uniform_img)) = 0;
uniform_img(isinf(uniform_img)) = 0;

if max(uniform_img(:)) > 0
    uniform_img = uniform_img ./ max(uniform_img(:));
end

%% ====== 计算 var 图 ======
var_img = var(imgStack, 1, 3);
var_img(isnan(var_img)) = 0;
var_img(isinf(var_img)) = 0;

if max(var_img(:)) > 0
    var_img = var_img ./ max(var_img(:));
end

%% ====== 保存 ======
imwrite(uniform_img, fullfile(savePath, 'uniform_img.tif'));
imwrite(var_img,     fullfile(savePath, 'var_img.tif'));

fprintf('uniform_img.tif 和 var_img.tif 已保存到：%s\n', savePath);

end