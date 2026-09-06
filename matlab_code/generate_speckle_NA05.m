%生成psf
clear all
close all
% 定义参数
NA = 0.05;            % 数值孔径
lambda = 488e-9;      % 波长 (550nm, 绿色光)
pixel_size = 4.5e-6/4;
sampling = 260;
fov = sampling*pixel_size;  % 视场大小（2微米）
% 计算空间频率限制
k0 = 2 * pi / lambda;  % 波数
% 创建坐标系
[x, y] = meshgrid([-fov/2:pixel_size:fov/2-pixel_size], [-fov/2:pixel_size:fov/2-pixel_size]);
r = sqrt(x.^2 + y.^2);  % 距离
% 定义理想点扩散函数（使用Airy盘的衍射极限公式）
psf = (besselj(1, NA * k0 * r) ./ (k0 * r)).^2 * (k0^2 / pi);
psf = psf./max(psf(:));
psf(r == 0) = 1;  % 修正中心值

% 显示PSF
figure;
imagesc(psf);
colormap hot;
colorbar;
title('Point Spread Function (PSF)');
axis equal tight;
imwrite(psf,['psf_NA',num2str(NA),'.tif']);
fwhm = length(find(psf(sampling/2+1,:)>=0.5))*pixel_size*10^6;
% imwrite(imresize(psf,2),'psf_up2_NA1.49_pixelsize0.05lambda.png');
% imwrite(imresize(psf,4),'psf_up4_NA1.49_pixelsize0.05lambda.png');
%% 生成512像素的散斑
clear all
close all
% 定义参数
NA = 0.05;            % 数值孔径
lambda = 488e-9;      % 波长 (550nm, 绿色光)
pixel_size = 4.5e-6/4;
sampling = 260;
fov = sampling*pixel_size;  % 视场大小（2微米）

k0 = 2 * pi / lambda;  % 波数
% 创建坐标系
[x, y] = meshgrid([-fov/2:pixel_size:fov/2], [-fov/2:pixel_size:fov/2]);
r = sqrt(x.^2 + y.^2);  % 距离
% 定义理想点扩散函数（使用Airy盘的衍射极限公  式）
psf = (besselj(1, NA * k0 * r) ./ (k0 * r)).^2 * (k0^2 / pi);
psf = psf./max(psf(:));
psf(r == 0) = 1;  % 修正中心值

%

phase = rand(sampling,sampling,100).*2*pi;
SLM = exp(1j*phase);
SLM = padarray(SLM, [sampling/2, sampling/2], 0);
phase_fft = fftshift(fft2(SLM));
%figure,subplot(1,2,1),imshow(abs(phase_fft(:,:,1)),[]);
%subplot(1,2,2),imshow(angle(phase_fft(:,:,1)),[]);


% Cutoff frequency based on NA and wavelength
f0 = NA / lambda;        % Cutoff frequency: f0 = NA / lambda
% Frequency grid parameters
L = 2*fov;                 % Size of the frequency grid (or image size)
du = pixel_size;               % Sampling interval (adjust based on image scaling)
% Frequency coordinates
fu = -1/(2*du):1/L:1/(2*du)-1/L; % Frequency coordinates in u direction
fv = fu;                            % Frequency coordinates in v direction
[Fu, Fv] = meshgrid(fu, fv);        % Create 2D frequency grid
r = sqrt(Fu.^2 + Fv.^2);  % 距离
H = zeros(size(r,1),size(r,2));
% Coherent Transfer Function (CTF)
H((r / f0)<=1)=1;  % Circular aperture cutoff in frequency domain
% imwrite(H,'H.png');
figure,mesh(H)
% 计算相干脉冲响应函数 (CPSF)
speckle = phase_fft.*H;
speckle = ifft2(ifftshift(speckle));
speckle = abs(speckle).^2;
% figure
for a = 1:100
    speckle_final(:,:,a) = speckle(sampling/2+1:sampling/2+sampling,sampling/2+1:sampling/2+sampling,a);
    speckle_final(:,:,a) = speckle_final(:,:,a)./max(max(speckle_final(:,:,a)));
%     imshow(speckle_final(:,:,a)),title(num2str(a)),pause(0.1);
    imwrite(speckle_final(:,:,a),['./speckle_random_NA0.05/speckle_random_',num2str(a),'.png']);
    temp = speckle_final(:,:,a);
    corr_speckle = fftshift(ifft2(abs(fft2(ifftshift(temp-mean(mean(temp))))).^2));
    corr_speckle  = corr_speckle./max(max(corr_speckle));

%     plot(corr_speckle(size(corr_speckle,1)/2+1,size(corr_speckle,1)/2+1-50:size(corr_speckle,1)/2+1+50));

    fwhm(a) = length(find(corr_speckle(size(corr_speckle,1)/2+1,:)>0.5))*4.5/4;
end
% figure;
% imagesc(H);
%
% addpath('./Util/');
% addpath('./Solver/');
% % load('F:\zq\2.0\Code\SIsoftware\PSFmatrix\PSFmatrix_M22.22NA0.5MLPitch125fml2754from-100to100zspacing10Nnum23lambda461n1.0a0_8.333b0_2.778.mat');
% load('H_-30.mat');
% forwardFUN =  @(Xguess) forwardProjectACC( H, Xguess, [369,691] );
% % H = H(:,:,:,:,8);
% % save('H_-30.mat','H');
% sample = imread('USAF_small.bmp');
% sample = double(sample);
% sample = sample./max(sample(:));
% for a = 1:100
%     img = speckle_final(:,:,a).*sample;
%     img = forwardFUN(img);
%     img = double(img);
%     img = img./max(img(:));
%     img = imnoise(img,'poisson');
%     img = imnoise(img,'gaussian',0,0.00001);
%     img = img./max(img(:));
%     img_final(:,:,a) = img;
%
%     imwrite(img,['./img_NA0.5/img_',num2str(a),'.png']);
% end
% img = sum(img_final,3);
% img = img./max(img(:));
% imwrite(img,['./img_NA0.5/uniform.png']);
%
% var_img = var(img_final,1,3);
% var_img = var_img./max(var_img(:));
% imwrite(var_img,['./img_NA0.5/var.png']);
% % corr_obtain = fftshift(ifft2(fftshift(abs(fftshift(fft2(fftshift(speckle_final(:,:,a)-mean(mean(speckle_final(:,:,a))))))).^2)));
% % corr_obtain = corr_obtain./max(max(corr_obtain));
% % figure,plot(psf(513,513-100:513+100));hold on
% % plot(corr_obtain(513,513-100:513+100));