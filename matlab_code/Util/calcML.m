function MLARRAY = calcML(fml, k, x1MLspace, x2MLspace, x1space, x2space)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1length = length(x1space);
x2length = length(x2space);
x1MLdist = length(x1MLspace);
x2MLdist = length(x2MLspace);
x1center = find(x1space==0); % x1中点
x2center = find(x2space==0);
x1centerALL = [  (x1center: -x1MLdist:1)  (x1center + x1MLdist: x1MLdist :x1length)]; % 以x1MLdist为间隔，在x1范围内找到所有中心点
x1centerALL = sort(x1centerALL); % 如果中心点与中心微透镜中心点相同，返回值为其他微透镜中心点在图像上的index
x2centerALL = [  (x2center: -x2MLdist:1)  (x2center + x2MLdist: x2MLdist :x2length)];
x2centerALL = sort(x2centerALL);

zeroline = zeros(1, length(x2space) );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
patternML = zeros( length(x1MLspace), length(x2MLspace) );
patternMLcp = zeros( length(x1MLspace), length(x2MLspace) );
for a=1:length(x1MLspace),
    for b=1:length(x2MLspace),
        x1 = x1MLspace(a); % 一个微透镜上点的实际坐标
        x2 = x2MLspace(b);
        xL2norm = x1^2 + x2^2;


        patternML(a,b) = exp(-i*k/(2*fml)*xL2norm);  %公式3的exp，公式3有宽度为d的矩形函数，即一个微透镜进行计算
        patternMLcp(a,b) = exp(-0.05*i*k/(2*fml)*xL2norm);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MLspace = zeros( length(x1space), length(x2space) );
MLcenters = MLspace;
for a=1:length(x1centerALL),
    for b=1:length(x2centerALL),
        MLcenters( x1centerALL(a), x2centerALL(b)) = 1; % 微透镜中心标记为1
    end
end
MLARRAY = conv2(MLcenters, patternML, 'same'); %对每一个微透镜中心都进行公式3卷积，即卷积梳状函数
MLARRAYcp = conv2(MLcenters, patternMLcp, 'same');

MLARRAYcpANG = angle(MLARRAYcp);
MLARRAYcpANG = MLARRAYcpANG - min(min(MLARRAYcpANG)) + 0.0;
MLARRAYcpANGnorm = MLARRAYcpANG/max(max(MLARRAYcpANG));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%