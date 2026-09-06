function [Xguess] = deconvLSM(forwardFUN, backwardFUN, Htf, LFIMG, maxIter, Xguess)

for i=1:maxIter
    tic;
    HXguess = forwardFUN(Xguess);
%     Pderivative_Xguess = 2*backwardProjectupdate(H, HXguess-LFIMG, Nnum);   % calculate partial derivatives
    Pderivative_Xguess = 2*backwardFUN(HXguess-LFIMG);
    nominator = (reshape(forwardFUN(Pderivative_Xguess),[],1))'*reshape(HXguess-LFIMG,[],1);
    denominator = (reshape(forwardFUN(Pderivative_Xguess),[],1))'*(reshape(forwardFUN(Pderivative_Xguess),[],1));
    step_size = nominator/denominator;
%     step_size = 0.004;
    Xguess = Xguess-0.2*step_size*Pderivative_Xguess;
    Xguess(find(isnan(Xguess))) = 0;
    ttime = toc;
    disp(['  iter ' num2str(i) ' | ' num2str(maxIter) ', took ' num2str(ttime) ' secs']);
end
