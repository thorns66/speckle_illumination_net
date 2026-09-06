function Xguess=pilot_deconv_rl_nonnegative(forwardFUN,backwardFUN,Htf,maxIter)
%PILOT_DECONV_RL_NONNEGATIVE Numerical fallback on the physical cone.
% Exact nonnegative convolution cannot be negative. Undefined 0/0 or a
% positive numerator over a zero numerical denominator receives zero update
% because that voxel has no supported response in the current iterate.
Xguess=max(Htf,0);
for iteration=1:maxIter
    started=tic;
    projected=max(forwardFUN(Xguess),0);
    denominator=max(backwardFUN(projected),0);
    ratio=zeros(size(Htf),'like',Htf);
    valid=denominator>0&isfinite(denominator)&isfinite(Htf);
    ratio(valid)=Htf(valid)./denominator(valid);
    Xguess=max(Xguess.*ratio,0);
    Xguess(~isfinite(Xguess))=0;
    fprintf('  positivity fallback iter %d | %d, took %.4g secs\n', ...
        iteration,maxIter,toc(started));
end
end
