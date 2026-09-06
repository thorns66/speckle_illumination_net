function [Xguess] = deconvRL(forwardFUN, backwardFUN, Htf, maxIter, Xguess )

for i=1:maxIter,
    tic;
    HXguess = forwardFUN(Xguess);
    HXguessBack = backwardFUN(HXguess);
    errorBack = Htf./HXguessBack; % 由于一些问题造成的误差比例的倒数
    Xguess = Xguess.*errorBack; % 减轻前一次的错误
    Xguess(find(isnan(Xguess))) = 0;
    ttime = toc;
    disp(['  iter ' num2str(i) ' | ' num2str(maxIter) ', took ' num2str(ttime) ' secs']);
end
