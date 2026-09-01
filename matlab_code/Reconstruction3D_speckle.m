% function [] = Reconstruction3D(PSFfile,inputFilePath,inputFileName,savePath)
function [] = Reconstruction3D_speckle(H,Ht,CAindex,inputFilePath,inputFileName,savePath,it,depth_z1,depth_z2)
warning('off');
addpath('./Util/');
addpath('./Solver/');
eqtol = 1e-10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

indpIter =  1;                          %% Decide whether or not to run frame-by-frame independent reconstruction
GPUcompute = 1;                        %% Decide whether or not to compute on GP
whichSolver = 1;                     %% Iterations method. Current version supports only Richardson-Lucy Iteration
maxIter = it;                    %% Number of iteration per each frame. Large number of iteration results in higher resolution/contrast at the price of computation time and pronounced artifacts
FirstFrame = '1';              %% If the data is a time series in .mat format, user can decide the range for reconstruction as [FirstFrame:DecimationRatio:LastFrame]
LastFrame = '1';                %% If the data is a time series in .mat format, user can decide the range for reconstruction as [FirstFrame:DecimationRatio:LastFrame]
DecimationRatio = '1';    %% If the data is a time series in .mat format, user can decide the range for reconstruction as [FirstFrame:DecimationRatio:LastFrame]
saturate = 0;                         %% Output will be automatically normalized to the full scale. User can decide to "amplify" the results for better visualization
saturationGain = '1';      %% Output will be automatically normalized to the full scale. User can decide to "amplify" the results for better visualization
contrast = '0.95';                  %% Contrast adjustment value used if one wants to use the result from last frame as the initial guess for the next frame. contrast<1 is often required to avoid artifact. Low value will result in low contrast in the result.
edgeSuppress = 0;                   %% Since border area in the reconstruction result is often subject to artifacts, user can simply assign zeros to the region by turning this option on.
useDiskVariable = 0;             %% Result can be directly saved the result on disk in a frame-by-frame manner. Recommended for large data. SSD is strongly recommended.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% PREPARE PARALLAL COMPUTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This part is not necessary for MATLAB 2014a or later version
% if matlabpool('size') == 0 % checking to see if my pool is already open
%     matlabpool open
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Load Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load(['../PSFmatrix/' PSFfile]);
if class(H)=='double',
    H = single(H);
    Ht = single(Ht);
end
% disp(['Successfully loaded PSF matrix : ' PSFfile]);
disp(['Size of PSF matrix is : ' num2str(size(H)) ]);

if strcmp( inputFileName(end-3:end), '.tif')
    LFmovie = im2double(imread([inputFilePath inputFileName]));
    FirstFrame = 1;
    LastFrame = 1;
    DecimationRatio = 1;
elseif strcmp( inputFileName(end-3:end), '.mat')
    load([inputFilePath inputFileName]);  
else
    disp('input should be a single TIFF file or a mat file (both need to be rectified');
end

disp(['Successfully loaded input data']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% Check data size  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lightFieldResolution = [size(LFmovie,1), size(LFmovie,2)];
global volumeResolution ;
volumeResolution = [size(LFmovie,1)  size(LFmovie,2)  size(H,5)];
disp(['Image size is ' num2str(volumeResolution(1)) 'X' num2str(volumeResolution(2))]); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% Prepare for GPU computation %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if GPUcompute,
    Nnum = size(H,3);
    backwardFUN = @(projection) backwardProjectGPU(Ht, projection );
    forwardFUN = @(Xguess) forwardProjectGPU( H, Xguess );
    
    global zeroImageEx;
    global exsize;
    xsize = [volumeResolution(1), volumeResolution(2)];
    msize = [size(H,1), size(H,2)];
    mmid = floor(msize/2);
    exsize = xsize + mmid;  
    exsize = [ min( 2^ceil(log2(exsize(1))), 128*ceil(exsize(1)/128) ), min( 2^ceil(log2(exsize(2))), 128*ceil(exsize(2)/128) ) ];    
    zeroImageEx = gpuArray(zeros(exsize, 'single'));
    disp(['FFT size is ' num2str(exsize(1)) 'X' num2str(exsize(2))]); 
else
    forwardFUN =  @(Xguess) forwardProjectACC( H, Xguess, CAindex );
    backwardFUN = @(projection) backwardProjectACC(Ht, projection, CAindex );
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% RUN Reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ReconFrames = [FirstFrame:DecimationRatio: min(LastFrame, size(LFmovie,3))];
numFrames =  length(ReconFrames);
if useDiskVariable,
    movie3Drecon = zeros([volumeResolution(1), volumeResolution(2), volumeResolution(3), 2], 'uint8');
    save([savePath 'Recon3D_'  inputFileName(1:end-4)], 'movie3Drecon', '-v7.3');      
    m = matfile([savePath 'Recon3D_'  inputFileName(1:end-4)], 'Writable', true);
    disp(['Use disk variable']);
else
    movie3Drecon = zeros([volumeResolution(1), volumeResolution(2), volumeResolution(3), numFrames], 'uint16');
end

k = 1;

for frame = ReconFrames,
    disp(['Volume reconstruction of Frame # ' num2str(k) ' / ' num2str(numFrames) ' is ongoing...']);
    LFIMG = single(LFmovie(:,:,frame));        
    tic; Htf = backwardFUN(LFIMG); ttime = toc;
    disp(['  iter ' num2str(0) ' | ' num2str(maxIter) ', took ' num2str(ttime) ' secs']);
    
    %%% Determined initial guess
    if k==1,
        Xguess = Htf;
    elseif indpIter,
        Xguess = Htf;
    else
       ; 
    end
    
    if k>1,
        if ~indpIter,
            Xguess = contrastAdjust(Xguess, contrast);
            maxIter = 1;
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Xguess = deconvRL(forwardFUN, backwardFUN, Htf, maxIter, Xguess );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%% Adjust the full scale of the reconstruction result
    if k==1,
        MV3Dgain = gather(0.66/max(Xguess(:))); %% full scale is inferred from the reconstruction result of the first frame
    end
    
    if GPUcompute,
       XguessCPU = gather(Xguess);
    else
       XguessCPU = Xguess;
    end    
    
    
    %%% Save the results on disk (only if used specivied to use disk variable
    if useDiskVariable,
        Xvolume = uint8(round(255*MV3Dgain*XguessCPU));  
        if edgeSuppress,                 
            Xvolume( (1:1*Nnum), :,:) = 0;
            Xvolume( (end-1*Nnum+1:end), :,:) = 0;
            Xvolume( :,(1:1*Nnum), :) = 0;
            Xvolume( :,(end-1*Nnum+1:end), :) = 0;
        end
        tic;
        m.movie3Drecon(:,:,:,k) = Xvolume;
        ttime = toc;
        disp(['Writing time: ' num2str(ttime) ' secs']);
    else
        Xvolume = uint16(round(65535*MV3Dgain*XguessCPU));
        if edgeSuppress,            
            Xvolume( (1:1*Nnum), :,:) = 0;
            Xvolume( (end-1*Nnum+1:end), :,:) = 0;
            Xvolume( :,(1:1*Nnum), :) = 0;
            Xvolume( :,(end-1*Nnum+1:end), :) = 0;
        end
        movie3Drecon(:,:,:,k) = Xvolume;
    end

    
    k = k + 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Post-processing the result...');
if GPUcompute,
    Xguess = gather(Xguess);
end

%XguessFinal = sqrt(double(Xvolume));
XguessFinal = (double(Xvolume));
if saturate,
    XguessSAVE1 = uint16(round(1*65535*XguessFinal/max(XguessFinal(:))));
    XguessSAVE2 = uint16(round(saturationGain*65535*XguessFinal/max(XguessFinal(:))));    
    XguessVIEW = uint16(round(saturationGain*65535*XguessFinal/max(XguessFinal(:))));
else    
    XguessSAVE1 = uint16(round(1*65535*XguessFinal/max(XguessFinal(:))));
    XguessVIEW = XguessSAVE1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Save the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imwrite( squeeze(XguessSAVE1(:,:,1)), [savePath 'Recon3D_var_' inputFileName(1:end-4) '_depth_from',num2str(depth_z1),'to',num2str(depth_z2),'.tif']);
for k = 2:size(XguessSAVE1,3),
    imwrite(squeeze(XguessSAVE1(:,:,k)),  [savePath 'Recon3D_var_' inputFileName(1:end-4) '_depth_from',num2str(depth_z1),'to',num2str(depth_z2),'.tif'], 'WriteMode', 'append');
end
if saturate,
    imwrite( squeeze(XguessSAVE2(:,:,2)), [savePath 'Recon3D_saturate_' inputFileName(1:end-4) '.tif']);
    for k = 2:size(XguessSAVE2,3)
        imwrite(squeeze(XguessSAVE2(:,:,k)),  [savePath 'Recon3D_saturate_' inputFileName(1:end-4) '.tif'], 'WriteMode', 'append');
    end        
end




XguessFinal = sqrt(double(Xvolume));
% XguessFinal = (double(Xvolume));
if saturate,
    XguessSAVE1 = uint16(round(1*65535*XguessFinal/max(XguessFinal(:))));
    XguessSAVE2 = uint16(round(saturationGain*65535*XguessFinal/max(XguessFinal(:))));    
    XguessVIEW = uint16(round(saturationGain*65535*XguessFinal/max(XguessFinal(:))));
else    
    XguessSAVE1 = uint16(round(1*65535*XguessFinal/max(XguessFinal(:))));
    XguessVIEW = XguessSAVE1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Save the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imwrite( squeeze(XguessSAVE1(:,:,1)), [savePath 'Recon3D_sqrt_' inputFileName(1:end-4) '_depth_from',num2str(depth_z1),'to',num2str(depth_z2),'.tif']);
for k = 2:size(XguessSAVE1,3),
    imwrite(squeeze(XguessSAVE1(:,:,k)),  [savePath 'Recon3D_sqrt_' inputFileName(1:end-4) '_depth_from',num2str(depth_z1),'to',num2str(depth_z2),'.tif'], 'WriteMode', 'append');
end
if saturate,
    imwrite( squeeze(XguessSAVE2(:,:,2)), [savePath 'Recon3D_saturate_' inputFileName(1:end-4) '.tif']);
    for k = 2:size(XguessSAVE2,3)
        imwrite(squeeze(XguessSAVE2(:,:,k)),  [savePath 'Recon3D_saturate_' inputFileName(1:end-4) '.tif'], 'WriteMode', 'append');
    end        
end
% if strcmp( inputFileName(end-3:end), '.tif')    
%     save([savePath 'Recon3D_'  inputFileName(1:end-4)], 'XguessSAVE1', '-v7.3');      
% elseif strcmp( inputFileName(end-3:end), '.mat')
%     if useDiskVariable,
%         ;
%     else
%         save([savePath 'Recon3D_'  inputFileName(1:end-4)], 'movie3Drecon', '-v7.3');  
%     end
% 
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(['Volume reconstruction complete.']);