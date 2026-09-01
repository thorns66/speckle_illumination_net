function Reconstruction_goodFOV_gray_it5_others
clear; close all; clc;
warning('on');

%% ================================================================
% 功能：
% 重建 good_fov_gray 下面所有合适视场
%
% 只做两类 5 次迭代重建：
% 1. uniform_img.tif  → uniform
% 2. var_img.tif      → taylor_var
%
% 不再逐张重建 100 张矫正图。
%
% 输入：
% D:\xy\20260611_brainSlices_screening\good_fov_gray\切片\FOV\*.tif
%
% 输出：
% D:\xy\20260611_brainSlices_screening\recon_it5_others\切片\FOV\uniform\
% D:\xy\20260611_brainSlices_screening\recon_it5_others\切片\FOV\taylor_var\
% D:\xy\20260611_brainSlices_screening\recon_it5_others\切片\FOV\uniform_var_used\
%% ================================================================

%% ====== 总路径 ======
rootPath = 'D:\xy\20260611_brainSlices_screening';

inputRoot = fullfile(rootPath, 'good_fov_gray');
saveRoot  = fullfile(rootPath, 'recon_it5_others');

if ~exist(saveRoot, 'dir')
    mkdir(saveRoot);
end

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

%% ====== 重建参数 ======
iterNum = 5;
zStart  = -340;
zEnd    = 340;
depthIdx = [1:18, 20:36];

%% ====== 检查输入路径 ======
if ~exist(inputRoot, 'dir')
    error('找不到 good_fov_gray 文件夹：%s', inputRoot);
end

%% ====== 收集 good_fov_gray 下面所有 FOV ======
fovTable = collectFOVList(inputRoot);

if isempty(fovTable)
    error('good_fov_gray 下面没有找到任何 FOV 文件夹：%s', inputRoot);
end

fprintf('\n共找到 %d 个 FOV 需要重建。\n', size(fovTable, 1));

%% ====== 日志初始化 ======
logCell = {};
logCell(end+1,:) = {'slice', 'fov', 'image_num', 'uniform_var_status', 'uniform_status', 'taylor_status', 'message'};

%% ================================================================
% 第 1 阶段：先为所有 FOV 生成 uniform_img.tif 和 var_img.tif
%% ================================================================
fprintf('\n====================================================\n');
fprintf('第 1 阶段：生成 uniform_img.tif 和 var_img.tif\n');
fprintf('====================================================\n');

for idx = 1:size(fovTable, 1)

    sliceName = fovTable{idx, 1};
    fovName   = fovTable{idx, 2};

    inputFOVPath = fullfile(inputRoot, sliceName, fovName);
    saveFOVRoot  = fullfile(saveRoot, sliceName, fovName);
    savePathUV   = fullfile(saveFOVRoot, 'uniform_var_used');

    if ~exist(savePathUV, 'dir')
        mkdir(savePathUV);
    end

    imgList = getImageList(inputFOVPath);
    imgNum = length(imgList);

    fprintf('\n[%d/%d] 生成 uniform/var：%s / %s，图像数 = %d\n', ...
        idx, size(fovTable,1), sliceName, fovName, imgNum);

    if imgNum == 0
        warning('没有找到图像：%s', inputFOVPath);
        continue;
    end

    if imgNum ~= 100
        warning('%s / %s 不是 100 张图，而是 %d 张，请检查。', sliceName, fovName, imgNum);
    end

    try
        generateUniformVarFromImages(inputFOVPath, savePathUV);
    catch ME
        warning('生成 uniform/var 失败：%s / %s\n原因：%s', sliceName, fovName, ME.message);
    end
end

%% ================================================================
% 第 2 阶段：加载普通 PSF，重建所有 uniform
%% ================================================================
fprintf('\n====================================================\n');
fprintf('第 2 阶段：uniform 5 次迭代重建\n');
fprintf('====================================================\n');

fprintf('\n正在加载普通 PSF...\n');

S_H = load(psfHPath);
if ~isfield(S_H, 'H')
    error('PSF 文件中没有变量 H。');
end
H_use = single(S_H.H(:,:,:,:,depthIdx));
clear S_H;

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

fprintf('普通 PSF 加载完成。\n');

uniformStatusList = cell(size(fovTable,1), 1);
taylorStatusList  = cell(size(fovTable,1), 1);
uvStatusList      = cell(size(fovTable,1), 1);
messageList       = cell(size(fovTable,1), 1);
imgNumList        = zeros(size(fovTable,1), 1);

for idx = 1:size(fovTable, 1)

    sliceName = fovTable{idx, 1};
    fovName   = fovTable{idx, 2};

    inputFOVPath = fullfile(inputRoot, sliceName, fovName);
    imgList = getImageList(inputFOVPath);
    imgNum = length(imgList);
    imgNumList(idx) = imgNum;

    saveFOVRoot = fullfile(saveRoot, sliceName, fovName);
    savePathUV  = fullfile(saveFOVRoot, 'uniform_var_used');
    savePathUniform = fullfile(saveFOVRoot, 'uniform');

    if ~exist(savePathUniform, 'dir')
        mkdir(savePathUniform);
    end

    uniformVarPathWithSep = [savePathUV, filesep];
    savePathUniformWithSep = [savePathUniform, filesep];

    uniformFile = fullfile(savePathUV, 'uniform_img.tif');
    varFile     = fullfile(savePathUV, 'var_img.tif');

    if exist(uniformFile, 'file') && exist(varFile, 'file')
        uvStatusList{idx} = 'success';
    else
        uvStatusList{idx} = 'failed';
    end

    uniformStatusList{idx} = 'not_run';
    taylorStatusList{idx}  = 'not_run';
    messageList{idx}       = '';

    fprintf('\n[%d/%d] uniform 重建：%s / %s\n', idx, size(fovTable,1), sliceName, fovName);

    if ~exist(uniformFile, 'file')
        uniformStatusList{idx} = 'missing';
        messageList{idx} = [messageList{idx}, ' missing uniform_img.tif;'];
        warning('缺少 uniform_img.tif：%s', uniformFile);
        continue;
    end

    try
        Reconstruction3D( ...
            H_use, ...
            Ht_use, ...
            CAindex_use, ...
            uniformVarPathWithSep, ...
            'uniform_img.tif', ...
            savePathUniformWithSep, ...
            iterNum, ...
            zStart, ...
            zEnd ...
        );

        uniformStatusList{idx} = 'success';
        fprintf('uniform 重建完成：%s / %s\n', sliceName, fovName);

    catch ME
        uniformStatusList{idx} = 'failed';
        messageList{idx} = [messageList{idx}, ' uniform failed: ', ME.message, ';'];
        fprintf(2, '\nuniform 重建失败：%s / %s\n原因：%s\n', sliceName, fovName, ME.message);
    end
end

clear H_use Ht_use CAindex_use;
try
    g = gpuDevice;
    reset(g);
    fprintf('\n普通 PSF 阶段结束，GPU 已 reset。\n');
catch
end

%% ================================================================
% 第 3 阶段：加载 Taylor 平方 PSF，重建所有 taylor_var
%% ================================================================
fprintf('\n====================================================\n');
fprintf('第 3 阶段：Taylor / var 5 次迭代重建\n');
fprintf('====================================================\n');

fprintf('\n正在加载 Taylor 用 PSF...\n');

S_H = load(psfHPath);
if ~isfield(S_H, 'H')
    error('PSF 文件中没有变量 H。');
end
H_speckle = single(S_H.H(:,:,:,:,depthIdx));
clear S_H;
H_speckle = H_speckle .^ 2;

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

fprintf('Taylor 用 PSF 加载完成。\n');

for idx = 1:size(fovTable, 1)

    sliceName = fovTable{idx, 1};
    fovName   = fovTable{idx, 2};

    saveFOVRoot = fullfile(saveRoot, sliceName, fovName);
    savePathUV  = fullfile(saveFOVRoot, 'uniform_var_used');
    savePathVar = fullfile(saveFOVRoot, 'taylor_var');

    if exist(savePathVar, 'dir')
        rmdir(savePathVar, 's');
    end
    mkdir(savePathVar);

    uniformVarPathWithSep = [savePathUV, filesep];
    savePathVarWithSep = [savePathVar, filesep];

    varFile = fullfile(savePathUV, 'var_img.tif');

    fprintf('\n[%d/%d] Taylor / var 重建：%s / %s\n', idx, size(fovTable,1), sliceName, fovName);

    if ~exist(varFile, 'file')
        taylorStatusList{idx} = 'missing';
        messageList{idx} = [messageList{idx}, ' missing var_img.tif;'];
        warning('缺少 var_img.tif：%s', varFile);
        continue;
    end

    try
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

        taylorStatusList{idx} = 'success';
        fprintf('Taylor / var 重建完成：%s / %s\n', sliceName, fovName);

    catch ME
        taylorStatusList{idx} = 'failed';
        messageList{idx} = [messageList{idx}, ' taylor failed: ', ME.message, ';'];
        fprintf(2, '\nTaylor / var 重建失败：%s / %s\n原因：%s\n', sliceName, fovName, ME.message);
        fprintf(2, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
    end

    try
        g = gpuDevice;
        reset(g);
        fprintf('当前 FOV 结束，GPU 已 reset。\n');
    catch
    end
end

clear H_speckle Ht_speckle CAindex_use;

%% ====== 汇总日志 ======
successFOV = 0;
failFOV = 0;

for idx = 1:size(fovTable, 1)

    sliceName = fovTable{idx, 1};
    fovName   = fovTable{idx, 2};

    if strcmp(uniformStatusList{idx}, 'success') && strcmp(taylorStatusList{idx}, 'success')
        successFOV = successFOV + 1;
    else
        failFOV = failFOV + 1;
    end

    logCell(end+1,:) = { ...
        sliceName, ...
        fovName, ...
        imgNumList(idx), ...
        uvStatusList{idx}, ...
        uniformStatusList{idx}, ...
        taylorStatusList{idx}, ...
        messageList{idx} ...
    };
end

%% ====== 保存日志 ======
logPath = fullfile(saveRoot, 'reconstruction_it5_others_log.xlsx');

try
    writecell(logCell, logPath);
    fprintf('\n重建日志已保存：%s\n', logPath);
catch
    logPathTxt = fullfile(saveRoot, 'reconstruction_it5_others_log.txt');
    fid = fopen(logPathTxt, 'w');

    for i = 1:size(logCell,1)
        fprintf(fid, '%s\t%s\t%d\t%s\t%s\t%s\t%s\n', ...
            logCell{i,1}, logCell{i,2}, logCell{i,3}, logCell{i,4}, logCell{i,5}, logCell{i,6}, logCell{i,7});
    end

    fclose(fid);
    fprintf('\nExcel 日志保存失败，已保存 txt 日志：%s\n', logPathTxt);
end

fprintf('\n====================================================\n');
fprintf('good_fov_gray 的 5 次迭代重建结束。\n');
fprintf('总 FOV 数量：%d\n', size(fovTable,1));
fprintf('成功 FOV 数量：%d\n', successFOV);
fprintf('失败 FOV 数量：%d\n', failFOV);
fprintf('输出目录：%s\n', saveRoot);
fprintf('====================================================\n');

end


%% ================================================================
% 收集 good_fov_gray 下面所有 切片/FOV
%% ================================================================
function fovTable = collectFOVList(inputRoot)

fovTable = {};

sliceList = dir(inputRoot);
sliceList = sliceList([sliceList.isdir]);
sliceList = sliceList(~ismember({sliceList.name}, {'.', '..'}));

for s = 1:length(sliceList)

    sliceName = sliceList(s).name;
    slicePath = fullfile(inputRoot, sliceName);

    fovList = dir(slicePath);
    fovList = fovList([fovList.isdir]);
    fovList = fovList(~ismember({fovList.name}, {'.', '..'}));

    for f = 1:length(fovList)
        fovName = fovList(f).name;
        fovPath = fullfile(slicePath, fovName);

        imgList = getImageList(fovPath);

        if ~isempty(imgList)
            fovTable(end+1,:) = {sliceName, fovName};
        end
    end
end

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
% 从矫正图生成 uniform_img.tif 和 var_img.tif
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

imgStack = zeros(H, W, imgNum, 'single');

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

    imgStack(:,:,i) = single(temp);
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

%% ====== 保存 ======
uniform_img = double(uniform_img);
var_img     = double(var_img);

uniform_img = max(min(uniform_img, 1), 0);
var_img     = max(min(var_img, 1), 0);

imwrite(uniform_img, fullfile(savePath, 'uniform_img.tif'));
imwrite(var_img,     fullfile(savePath, 'var_img.tif'));

fprintf('uniform_img.tif 和 var_img.tif 已保存到：%s\n', savePath);

end