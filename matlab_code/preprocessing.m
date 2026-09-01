clear;
close all;
clc;

%% ==========================================================
% 路径
%% ==========================================================

rootPath = 'D:\xy\20260611_brainSlices_screening';

rawRoot  = fullfile(rootPath,'new_raw');
grayRoot = fullfile(rootPath,'new_gray');

if ~exist(grayRoot,'dir')
    mkdir(grayRoot);
end

%% ==========================================================
% 获取三个数据文件夹
%% ==========================================================

folderList = dir(rawRoot);
folderList = folderList([folderList.isdir]);
folderList = folderList(~ismember({folderList.name},{'.','..'}));

%% ==========================================================
% 开始转换
%% ==========================================================

for k = 1:length(folderList)

    folderName = folderList(k).name;

    inputPath = fullfile(rawRoot,folderName);

    savePath = fullfile(grayRoot,folderName);

    if ~exist(savePath,'dir')
        mkdir(savePath);
    end

    imgList = getImageList(inputPath);

    if isempty(imgList)
        warning('没有找到图片：%s',inputPath);
        continue;
    end

    fprintf('\n===============================\n');
    fprintf('%s\n',folderName);
    fprintf('共 %d 张图片\n',length(imgList));

    for i = 1:length(imgList)

        inputFullName = fullfile(inputPath,imgList(i).name);

        img = im2double(imread(inputFullName));

        % RGB -> Gray
        if size(img,3)==3
            img = rgb2gray(img);
        end

        % 归一化
        maxVal = max(img(:));

        if maxVal>0
            img = img./maxVal;
        end

        saveName = sprintf('%03d.tif',i);

        imwrite(img,fullfile(savePath,saveName));

        if mod(i,50)==0 || i==length(imgList)
            fprintf('%4d/%4d\n',i,length(imgList));
        end

    end

end

fprintf('\n===============================\n');
fprintf('全部灰度转换完成！\n');
fprintf('输出目录：\n%s\n',grayRoot);

%% ==========================================================
% 获取图片列表
%% ==========================================================

function imgList = getImageList(folderPath)

imgList = [];

if ~exist(folderPath,'dir')
    return;
end

imgList = [ ...
    dir(fullfile(folderPath,'*.bmp')); ...
    dir(fullfile(folderPath,'*.png')); ...
    dir(fullfile(folderPath,'*.jpg')); ...
    dir(fullfile(folderPath,'*.jpeg')); ...
    dir(fullfile(folderPath,'*.tif')); ...
    dir(fullfile(folderPath,'*.tiff'))];

% 去重
[~,idx] = unique({imgList.name});
imgList = imgList(idx);

% 按文件名排序
[~,idx] = sort({imgList.name});
imgList = imgList(idx);

end