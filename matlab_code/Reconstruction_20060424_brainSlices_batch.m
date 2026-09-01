function Reconstruction_20060424_brainSlices_batch
clear; close all; clc;
warning('off');

%% ================================================================
% 功能：
% 批量重建所有 FOV
%
% 输入：
% D:\xy\20260611_brainSlices_screening\uniform_var\切片\FOV\uniform_img.tif
% D:\xy\20260611_brainSlices_screening\uniform_var\切片\FOV\var_img.tif
%
% 输出：
% D:\xy\20260611_brainSlices_screening\recon_it3\切片\FOV\uniform\
% D:\xy\20260611_brainSlices_screening\recon_it3\切片\FOV\taylor_var\
%% ================================================================

%% ====== 总路径 ======
rootPath = 'D:\xy\20260611_brainSlices_screening';

inputRoot = fullfile(rootPath, 'uniform_var');
saveRoot  = fullfile(rootPath, 'recon_it3');

if ~exist(saveRoot, 'dir')
    mkdir(saveRoot);
end

%% ====== 当前代码路径 ======
codePath = fullfile(rootPath, 'code_used');

% 加入本次任务代码
addpath(genpath(codePath));

% 加入原始重建工具函数
addpath(genpath('D:\xy\Util'));
addpath(genpath('D:\xy\Solver'));

%% ====== PSF 路径 ======
% 这里沿用你原代码里的 PSF 路径。
% 如果服务器上 PSF 不在 F 盘，就只需要改这里。
psfHtPath = 'F:\xy\20250626-大深度微型系统\合并psf\merged-340_to_340_Ht.mat';
psfHPath  = 'F:\xy\20250626-大深度微型系统\合并psf\merged-340_to_340_H.mat';

if ~exist(psfHtPath, 'file')
    error('找不到 Ht 文件：%s', psfHtPath);
end

if ~exist(psfHPath, 'file')
    error('找不到 H 文件：%s', psfHPath);
end

%% ====== 检查重建函数 ======
if exist('Reconstruction3D', 'file') ~= 2
    error('找不到 Reconstruction3D.m，请把它或 Util 文件夹加入 MATLAB 路径。');
end

if exist('backwardProjectGPU', 'file') ~= 2
    error('找不到 backwardProjectGPU.m，请检查 D:\xy\Util 是否已加入 MATLAB 路径。');
end

if exist('forwardProjectGPU', 'file') ~= 2
    error('找不到 forwardProjectGPU.m，请检查 D:\xy\Util 是否已加入 MATLAB 路径。');
end

if exist('Reconstruction3D_speckle', 'file') ~= 2
    error('找不到 Reconstruction3D_speckle.m，请把它或 Util 文件夹加入 MATLAB 路径。');
end

if exist('deconvRL', 'file') ~= 2
    error('找不到 deconvRL.m，请把完整 Util 文件夹加入 MATLAB 路径。');
end

%% ====== 加载 PSF ======
fprintf('正在加载 PSF...\n');

load(psfHtPath);   % 需要包含 Ht 和 CAindex
load(psfHPath);    % 需要包含 H

if ~exist('H', 'var')
    error('PSF 文件中没有变量 H。');
end

if ~exist('Ht', 'var')
    error('PSF 文件中没有变量 Ht。');
end

if ~exist('CAindex', 'var')
    error('PSF 文件中没有变量 CAindex。');
end

fprintf('PSF 加载完成。\n');

%% ====== 沿用原代码的深度索引 ======
depthIdx = [1:18, 20:36];

H_use       = H(:,:,:,:,depthIdx);
Ht_use      = Ht(:,:,:,:,depthIdx);
CAindex_use = CAindex(depthIdx,:);

clear H Ht CAindex;

%% ====== 重建参数 ======
iterNum = 3;
zStart  = -340;
zEnd    = 340;

%% ====== 如果只想测试一个 FOV，就在这里填 ======
% 例如：
% onlySlice = '500um_2';
% onlyFOV   = 'FOV01';
%
% 全部重建时保持空：
onlySlice = '';
onlyFOV   = '';

%% ====== 检查输入路径 ======
if ~exist(inputRoot, 'dir')
    error('找不到 uniform_var 文件夹：%s', inputRoot);
end

%% ====== 遍历所有切片和 FOV ======
sliceList = dir(inputRoot);
sliceList = sliceList([sliceList.isdir]);
sliceList = sliceList(~ismember({sliceList.name}, {'.', '..'}));

if isempty(sliceList)
    error('uniform_var 文件夹下面没有切片文件夹：%s', inputRoot);
end

totalFOV = 0;
successFOV = 0;
failFOV = 0;

logCell = {};
logCell(end+1,:) = {'slice', 'fov', 'uniform_status', 'var_status', 'message'};

for s = 1:length(sliceList)

    sliceName = sliceList(s).name;

    if ~isempty(onlySlice) && ~strcmp(sliceName, onlySlice)
        continue;
    end

    slicePath = fullfile(inputRoot, sliceName);

    fovList = dir(slicePath);
    fovList = fovList([fovList.isdir]);
    fovList = fovList(~ismember({fovList.name}, {'.', '..'}));

    if isempty(fovList)
        warning('该切片下面没有 FOV 文件夹：%s', slicePath);
        continue;
    end

    for f = 1:length(fovList)

        fovName = fovList(f).name;

        if ~isempty(onlyFOV) && ~strcmp(fovName, onlyFOV)
            continue;
        end

        totalFOV = totalFOV + 1;

        fprintf('\n====================================================\n');
        fprintf('开始重建：%s / %s\n', sliceName, fovName);
        fprintf('====================================================\n');

        inputPath = fullfile(inputRoot, sliceName, fovName);
        inputPath = [inputPath, filesep];

        uniformFile = 'uniform_img.tif';
        varFile     = 'var_img.tif';

        uniformFullName = fullfile(inputPath, uniformFile);
        varFullName     = fullfile(inputPath, varFile);

        if ~exist(uniformFullName, 'file')
            warning('缺少 uniform_img.tif：%s', uniformFullName);
            failFOV = failFOV + 1;
            logCell(end+1,:) = {sliceName, fovName, 'missing', 'not_run', 'missing uniform_img.tif'};
            continue;
        end

        if ~exist(varFullName, 'file')
            warning('缺少 var_img.tif：%s', varFullName);
            failFOV = failFOV + 1;
            logCell(end+1,:) = {sliceName, fovName, 'not_run', 'missing', 'missing var_img.tif'};
            continue;
        end

        savePathUniform = fullfile(saveRoot, sliceName, fovName, 'uniform');
        savePathVar     = fullfile(saveRoot, sliceName, fovName, 'taylor_var');

        if ~exist(savePathUniform, 'dir')
            mkdir(savePathUniform);
        end

        if ~exist(savePathVar, 'dir')
            mkdir(savePathVar);
        end

        savePathUniform = [savePathUniform, filesep];
        savePathVar     = [savePathVar, filesep];

        uniformStatus = 'not_run';
        varStatus     = 'not_run';
        message       = '';

        %% ====== 1. 均匀照明重建 ======
        try
            fprintf('正在重建 uniform_img.tif ...\n');

            Reconstruction3D( ...
                H_use, ...
                Ht_use, ...
                CAindex_use, ...
                inputPath, ...
                uniformFile, ...
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

        %% ====== 2. Taylor / var 散斑照明重建 ======
        try
            fprintf('正在重建 var_img.tif ...\n');

            Reconstruction3D_speckle( ...
                H_use.^2, ...
                Ht_use.^2, ...
                CAindex_use, ...
                inputPath, ...
                varFile, ...
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

        if strcmp(uniformStatus, 'success') && strcmp(varStatus, 'success')
            successFOV = successFOV + 1;
        else
            failFOV = failFOV + 1;
        end

        logCell(end+1,:) = {sliceName, fovName, uniformStatus, varStatus, message};

        fprintf('完成当前 FOV：%s / %s\n', sliceName, fovName);
    end
end

%% ====== 保存日志 ======
logPath = fullfile(saveRoot, 'reconstruction_log.xlsx');

try
    writecell(logCell, logPath);
    fprintf('\n重建日志已保存：%s\n', logPath);
catch
    logPathTxt = fullfile(saveRoot, 'reconstruction_log.txt');
    fid = fopen(logPathTxt, 'w');

    for i = 1:size(logCell,1)
        fprintf(fid, '%s\t%s\t%s\t%s\t%s\n', ...
            logCell{i,1}, logCell{i,2}, logCell{i,3}, logCell{i,4}, logCell{i,5});
    end

    fclose(fid);
    fprintf('\nExcel 日志保存失败，已保存 txt 日志：%s\n', logPathTxt);
end

fprintf('\n====================================================\n');
fprintf('全部批量重建结束。\n');
fprintf('总 FOV 数量：%d\n', totalFOV);
fprintf('成功 FOV 数量：%d\n', successFOV);
fprintf('失败 FOV 数量：%d\n', failFOV);
fprintf('输出目录：%s\n', saveRoot);
fprintf('====================================================\n');

end