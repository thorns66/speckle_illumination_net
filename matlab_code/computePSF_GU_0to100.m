clear all
close all
warning('off');
addpath('./Util/');
savePath = '../PSFmatrix/';
if exist(savePath)==7,
   ; 
else
   mkdir(savePath); 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M =         4;
NA =        0.15;
MLPitch =   220*1e-6;
Nnum =      49;
OSR =       1;
n =         1;
fml =       4000*1e-6;
lambda =    532*1e-9;
zmax =      100*1e-6;
zmin =      0*1e-6;
zspacing =  10*1e-6;
a0 =        -11*1e-3;
b0 =        2.9333*1e-3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eqtol = 1e-10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% PREPARE PARALLAL COMPUTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if matlabpool('size') == 0 % checking to see if my pool is already open
%     matlabpool open
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%% SIM PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%
k = 2*pi*n/lambda; %% k
k0 = 2*pi*1/lambda; %% k
d = fml;   %% optical distance between the microlens and the sensor
ftl = 20e-3;        %% focal length of tube lens%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%×¢ÒâÕâ¸ö
fobj = ftl/M;  %% focal length of objective lens
fnum_obj = M/(2*NA); %% f-number of objective lens (imaging-side)
fnum_ml = b0/MLPitch; %% f-number of micrl lens
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% DEFINE OBJECT SPACE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if mod(Nnum,2)==0,
   error(['Nnum should be an odd number']); 
end
pixelPitch = MLPitch/Nnum; %% pitch of virtual pixels

x1objspace = [0]; 
x2objspace = [0];
x3objspace = [zmin:zspacing:zmax];
objspace = ones(length(x1objspace),length(x2objspace),length(x3objspace));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p3max = max(abs(x3objspace));
x1testspace = (pixelPitch/OSR)* [0:1: Nnum*OSR*100];
x2testspace = [0];   
[psfLine] = calcPSFFT_Corrected(p3max, fobj, NA, x1testspace, pixelPitch/OSR, lambda, d, M, n);
outArea = find(psfLine<0.01);
if isempty(outArea),
   error('Estimated PSF size exceeds the limit');   
end
IMGSIZE_REF = ceil(outArea(1)*2/(OSR*Nnum));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% OTHER SIMULATION PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%
disp(['Size of PSF ~= ' num2str(IMGSIZE_REF) ' [microlens pitch]' ]);
IMG_HALFWIDTH = max( Nnum*(IMGSIZE_REF + 1), 2*Nnum);
disp(['Size of IMAGE = ' num2str(IMG_HALFWIDTH*2*OSR+1) 'X' num2str(IMG_HALFWIDTH*2*OSR+1) '' ]);
x1space = (pixelPitch/OSR)*[-IMG_HALFWIDTH*OSR:1:IMG_HALFWIDTH*OSR]; 
x2space = (pixelPitch/OSR)*[-IMG_HALFWIDTH*OSR:1:IMG_HALFWIDTH*OSR]; 
x1length = length(x1space);
x2length = length(x2space);

x1MLspace = (pixelPitch/OSR)* [-(Nnum*OSR-1)/2 : 1 : (Nnum*OSR-1)/2];
x2MLspace = (pixelPitch/OSR)* [-(Nnum*OSR-1)/2 : 1 : (Nnum*OSR-1)/2];
x1MLdist = length(x1MLspace);
x2MLdist = length(x2MLspace);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%% FIND NON-ZERO POINTS %%%%%%%%%%%%%%%%%%%%%%%%%%
validpts = find(objspace>eqtol);
numpts = length(validpts);
[p1indALL p2indALL p3indALL] = ind2sub( size(objspace), validpts);
p1ALL = x1objspace(p1indALL)';
p2ALL = x2objspace(p2indALL)';
p3ALL = x3objspace(p3indALL)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%% DEFINE ML ARRAY %%%%%%%%%%%%%%%%%%%%%%%%% 
MLARRAY = calcML(fml, k0, x1MLspace, x2MLspace, x1space, x2space); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%% Alocate Memory for storing PSFs %%%%%%%%%%%   
LFpsfWAVE_STACK = zeros(x1length, x2length, numpts);
psfWAVE_STACK = zeros(x1length, x2length, numpts);
disp(['Start Calculating PSF...']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%% PROJECTION FROM SINGLE POINT %%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
centerPT = ceil(length(x1space)/2);
halfWidth =  Nnum*(IMGSIZE_REF + 0 )*OSR;
centerArea = (  max((centerPT - halfWidth),1)   :   min((centerPT + halfWidth),length(x1space))     );

disp(['Computing PSFs (1/3)']);
for eachpt=1:numpts,        
    p1 = p1ALL(eachpt);
    p2 = p2ALL(eachpt);
    p3 = p3ALL(eachpt);
    
    IMGSIZE_REF_IL = ceil(IMGSIZE_REF*( abs(p3)/p3max));
    halfWidth_IL =  max(Nnum*(IMGSIZE_REF_IL + 0 )*OSR, 2*Nnum*OSR);
    centerArea_IL = (  max((centerPT - halfWidth_IL),1)   :   min((centerPT + halfWidth_IL),length(x1space))     );
    disp(['size of center area = ' num2str(length(centerArea_IL)) 'X' num2str(length(centerArea_IL)) ]);    
    
    %% excute PSF computing funcion
    [psfWAVE LFpsfWAVE] = calcPSF_Corrected(p1, p2, p3, fobj, NA, x1space, x2space, pixelPitch/OSR, lambda, MLARRAY, d, M, n,  centerArea_IL , a0 , b0);
    psfWAVE_STACK(:,:,eachpt)  = psfWAVE;
    LFpsfWAVE_STACK(:,:,eachpt)= LFpsfWAVE;    



end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%% Compute Light Field PSFs (light field) %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
x1objspace = (pixelPitch/M)*[-floor(Nnum/2):1:floor(Nnum/2)];
x2objspace = (pixelPitch/M)*[-floor(Nnum/2):1:floor(Nnum/2)];
XREF = ceil(length(x1objspace)/2);
YREF = ceil(length(x1objspace)/2);
CP = ( (centerPT-1)/OSR+1 - halfWidth/OSR :1: (centerPT-1)/OSR+1 + halfWidth/OSR  );
H = zeros( length(CP), length(CP), length(x1objspace), length(x2objspace), length(x3objspace) );

disp(['Computing LF PSFs (2/3)']);
 tic
for i=1:length(x1objspace)*length(x2objspace)*length(x3objspace) ,
   
    [a, b, c] = ind2sub([length(x1objspace) length(x2objspace) length(x3objspace)], i);  
     psfREF = psfWAVE_STACK(:,:,c);  
        
     %%====================================================================
    
  
    psfSHIFT= im_shift2(psfREF, OSR*(a-XREF), OSR*(b-YREF) );    
        
    [f1,dx1,x1]                        = fresnel2D(psfSHIFT.*MLARRAY, pixelPitch/OSR, b0,lambda);
     
     %%====================================================================
        
     f1= im_shift2(f1, -OSR*(a-XREF), -OSR*(b-YREF) );
     
     xmin =  max( centerPT  - halfWidth, 1);
     xmax =  min( centerPT  + halfWidth, size(f1,1) );
     ymin =  max( centerPT  - halfWidth, 1);
     ymax =  min( centerPT  + halfWidth, size(f1,2) );
 
     f1_AP = zeros(size(f1));
     f1_AP( (xmin:xmax), (ymin:ymax) ) = f1( (xmin:xmax), (ymin:ymax) );
     [f1_AP_resize, x1shift, x2shift] = pixelBinning(abs(f1_AP.^2), OSR);           
     f1_CP = f1_AP_resize( CP - x1shift, CP-x2shift );

     H(:,:,a,b,c) = f1_CP;
     if mod(i,50)==0
     toc
     disp([num2str(i),'/',num2str(length(x1objspace)*length(x2objspace)*length(x3objspace))])
     end
     

end
H = H/max(H(:));

x1space = (pixelPitch/1)*[-IMG_HALFWIDTH*1:1:IMG_HALFWIDTH*1];
x2space = (pixelPitch/1)*[-IMG_HALFWIDTH*1:1:IMG_HALFWIDTH*1]; 
x1space = x1space(CP);
x2space = x2space(CP);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%% Clear variables that are no longer necessary %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear LFpsfWAVE_STACK;
clear LFpsfWAVE_VIEW;
clear psfWAVE_STACK;
clear psfWAVE_VIEW;
clear LFpsfWAVE;
clear PSF_AP;
clear PSF_AP_resize;
clear PSF_CP;
clear f1;
clear f1_AP;
clear f1_AP_resize;
clear f1_CP;
clear psfREF;
clear psfSHIFT;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tol = 0.005;
for i=1:size(H,5),
   H4Dslice = H(:,:,:,:,i);
   H4Dslice(find(H4Dslice< (tol*max(H4Dslice(:))) )) = 0;
   H(:,:,:,:,i) = H4Dslice;   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = single(H);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%% Estimate PSF size again  %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
centerCP = ceil(length(CP)/2);
CAindex = zeros(length(x3objspace),2);
for i=1:length(x3objspace),
    IMGSIZE_REF_IL = ceil(IMGSIZE_REF*( abs(x3objspace(i))/p3max));
    halfWidth_IL =  max(Nnum*(IMGSIZE_REF_IL + 0 ), 2*Nnum);
    CAindex(i,1) = max( centerCP - halfWidth_IL , 1);
    CAindex(i,2) = min( centerCP + halfWidth_IL , size(H,1));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
fileName = ['NEW_modifyfobj_PSFmatrix_'  'M' num2str(M)  'NA' num2str(NA) 'MLPitch' num2str(MLPitch) 'fml' num2str(fml) 'from' num2str(zmin) 'to' num2str(zmax)  'zspacing' num2str(zspacing) 'Nnum' num2str(Nnum)  'lambda' num2str(lambda) 'n' num2str(n) 'a0_' num2str(a0) 'b0_' num2str(b0) '.mat' ];


disp(['Saving PSF matrix file...']);
save([savePath fileName] , 'H', 'CAindex' ,'-v7.3');
disp(['PSF computation complete.']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%% Calculate Ht (transpose for backprojection) %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
disp(['Computing Transpose (3/3)']);
Ht = calcHt(H);
H = single(H);
Ht = single(Ht);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
disp(['Saving PSF matrix file...']);
save([savePath fileName] , 'H','Ht', 'CAindex', 'settingPSF', 'OSR', 'fobj', 'd', 'NA', 'objspace', 'M', 'MLARRAY', 'zspacing','x1objspace', 'x2objspace', 'x3objspace', 'pixelPitch', 'x1space','x2space', 'CP','Nnum' ,'a0', 'b0' ,'-v7.3');
disp(['PSF computation complete.']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


