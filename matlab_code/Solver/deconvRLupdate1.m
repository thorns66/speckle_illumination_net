function [Xguess] = deconvRLupdate1(forwardFUN, backwardFUN, Htf, maxIter, Xguess, inputFileName, savePath)

for i=1:maxIter,
    tic;
    HXguess = forwardFUN(Xguess);
    HXguessBack = backwardFUN(HXguess);
    errorBack = Htf./HXguessBack;
    Xguess = Xguess.*errorBack;
    Xguess(find(isnan(Xguess))) = 0;
    ttime = toc;
    disp(['  iter ' num2str(i) ' | ' num2str(maxIter) ', took ' num2str(ttime) ' secs']);

    %%% Save the results on disk (only if used specivied to use disk variable
    MV3Dgain = gather(0.66/max(Xguess(:)));
    Xvolume = uint16(round(65535*MV3Dgain*Xguess));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Post-processing the result...');
    XguessFinal = double(Xvolume);
    XguessSAVE1 = uint16(round(1*65535*XguessFinal/max(XguessFinal(:))));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Save the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    imwrite( squeeze(XguessSAVE1(:,:,1)), [savePath '30_Recon3D_' inputFileName(1:end-4) '_Iter' num2str(i) '.tif']);
    for kk = 2:size(XguessSAVE1,3),
        imwrite(squeeze(XguessSAVE1(:,:,kk)),  [savePath '30_Recon3D_' inputFileName(1:end-4) '_Iter' num2str(i) '.tif'], 'WriteMode', 'append');
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ÐÞ¸Ä
    if strcmp( inputFileName(end-3:end), '.tif')
         save([savePath '30_Recon3D_'  inputFileName(1:end-4) '_Iter' num2str(i) '.mat'], 'XguessSAVE1', '-v7.3');
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
