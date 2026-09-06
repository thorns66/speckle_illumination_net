function [psf, LFpsf] = calcPSF_Corrected(p1, p2, p3, fobj, NA, x1space, x2space, scale, lambda, MLARRAY, fml, M, n, centerArea, a0, b0)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = 2*pi*n/lambda; %波数
alpha = asin(NA/n);
x1length = length(x1space); % x1space = (pixelPitch/OSR)*[-IMG_HALFWIDTH*OSR:1:IMG_HALFWIDTH*OSR];
x2length = length(x2space);
zeroline = zeros(1, length(x2space) );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pattern = zeros(x1length, x2length);
centerPT = ceil(length(x1space)/2);
% centerArea以center point为中心取halfWidth（PSF）半边
% parfor并行执行，计算NIP处的波函数Ui(x,p)
for a=centerArea(1):centerPT,
    patternLine = zeroline;
    for b=a:centerPT, % 对称，这里只去一半计算，即左上角（1/4）的右上角（三角形）
        x1 = x1space(a);
        x2 = x2space(b);
        xL2normsq = (((x1+M*p1)^2+(x2+M*p2)^2)^0.5)/M;

        v = k*xL2normsq*sin(alpha);  % 归一化的径向坐标
        u = 4*k*(p3*1)*(sin(alpha/2)^2); % 归一化的轴向坐标
        Koi = M/((fobj*lambda)^2)*exp(-i*u/(4*(sin(alpha/2)^2)));
        intgrand = @(theta) (sqrt(cos(theta))) .* (1+cos(theta))  .*  (exp((i*u/2)* (sin(theta/2).^2) / (sin(alpha/2)^2)))  .*  (besselj(0, sin(theta)/sin(alpha)*v))  .*  (sin(theta));
        I0 = integral(@(theta)intgrand (theta),0,alpha);  % 公式1

        patternLine(1,b) =  Koi*I0;
    end
    pattern(a,:) = patternLine;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
patternA = pattern( (1:centerPT), (1:centerPT) ); % 左上角（1/4）
patternAt = fliplr(patternA); % fliplr左右翻转矩阵

pattern3D = zeros(size(pattern,1), size(pattern,2), 4);
pattern3D(:,:,1) = pattern;
pattern3D( (1:centerPT), (centerPT:end),1 ) = patternAt; % 右上角（1/4）
pattern3D(:,:,2) = rot90( pattern3D(:,:,1) , -1);
pattern3D(:,:,3) = rot90( pattern3D(:,:,1) , -2);
pattern3D(:,:,4) = rot90( pattern3D(:,:,1) , -3);

pattern = max(pattern3D,[],3); %一个三角形旋转4次，完成整个大小
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



        Step_value          = ( a0/10 );
        Distance_Steps      = floor(a0/Step_value);
        Distance_Remainder  = a0 - Step_value*Distance_Steps;
        f1                  = pattern;

        % 公式2,得到的f1为MLA波前
        for m=1:Distance_Steps
            [f1,dx1,x1]                    = fresnel2D(f1.*1, scale, Step_value,lambda); % dx1 = scale;x1=[-Nx/2:Nx/2-1]*dx1;
        end
            [pattern,dx1,x1]               = fresnel2D(f1.*1, scale, Distance_Remainder,lambda);






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% CALCAULTED  LF PSF %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[f1,dx1,x1]  = fresnel2D(pattern.*MLARRAY, scale, b0,lambda); % 公式4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
psf = pattern;
LFpsf = f1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%