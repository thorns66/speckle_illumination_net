function merge_depth_tif_to_3D()

clear;
clc;
close all;

%%==============================================================
% 把很多单层tif按深度顺序合并成一个3D tif
%
% 输出：
% all_FOVxx.tif
%%==============================================================

folderList = {

'D:\xy\20260611_brainSlices_screening\recon_it5\800um_2\FOV02\single_rectified\single_depth_3Dtif\AlgoRIM_it_4_w_0p0005_r_4'

'D:\xy\20260611_brainSlices_screening\recon_it5\800um_3\FOV02\single_rectified\single_depth_3Dtif\AlgoRIM_it_4_w_0p0005_r_4'

'D:\xy\20260611_brainSlices_screening\recon_it5_others\500um_3\FOV02_(-100+40)\single_rectified_50\single_depth_3Dtif_50\AlgoRIM_it_4_w_0p0005_r_4'

'D:\xy\20260611_brainSlices_screening\recon_it5_others\650um_1\FOV02_(-100+140)\single_rectified_50\single_depth_3Dtif_50\AlgoRIM_it_4_w_0p0005_r_4'

'D:\xy\20260611_brainSlices_screening\recon_it5_others\650um_1\FOV03(-120+140)\single_rectified_50\single_depth_3Dtif_50\AlgoRIM_it_4_w_0p0005_r_4'

'D:\xy\20260611_brainSlices_screening\recon_it5_others\650um_3\FOV01_(-160+100)\single_rectified_50\single_depth_3Dtif_50\AlgoRIM_it_4_w_0p0005_r_4'

};

%%==============================================================

for k = 1:length(folderList)

    folder = folderList{k};

    fprintf('\n========================================\n');
    fprintf('%s\n',folder);

    tifList = dir(fullfile(folder,'*.tif'));

    if isempty(tifList)
        warning('没有tif.');
        continue;
    end

    %%-----------------------
    % 提取深度
    %%-----------------------
    depth = zeros(length(tifList),1);

    for i=1:length(tifList)

        name=tifList(i).name;

        token=regexp(name,'([mp])(\d+)','tokens','once');

        if isempty(token)
            error('文件名无法识别：%s',name);
        end

        v=str2double(token{2});

        if token{1}=='m'
            depth(i)=-v;
        else
            depth(i)=v;
        end

    end

    [depth,idx]=sort(depth);

    tifList=tifList(idx);

    %%-----------------------
    % 输出名称
    %%-----------------------

    token=regexp(folder,'FOV[^\\\/]*','match','once');

    if isempty(token)
        token='Unknown';
    end

    outName=fullfile(folder,['all_' token '.tif']);

    if exist(outName,'file')
        delete(outName);
    end

    fprintf('输出：%s\n',outName);

    %%-----------------------
    % 开始堆叠
    %%-----------------------

    for i=1:length(tifList)

        img=imread(fullfile(folder,tifList(i).name));

        if i==1

            imwrite(img,outName,...
                'tif',...
                'Compression','none');

        else

            imwrite(img,outName,...
                'tif',...
                'WriteMode','append',...
                'Compression','none');

        end

        fprintf('%3d/%3d   z=%4d   %s\n',...
            i,...
            length(tifList),...
            depth(i),...
            tifList(i).name);

    end

    fprintf('完成。\n');

end

fprintf('\n===============================\n');
fprintf('全部完成！\n');
fprintf('===============================\n');

end