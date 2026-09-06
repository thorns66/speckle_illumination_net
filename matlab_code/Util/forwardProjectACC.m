function TOTALprojection = forwardProjectACC( H, realspace, CAindex)
% 每一层的映射结果叠加得到最后的结果
% CAindex = zeros(length(x3objspace),2);

Nnum = size(H,3);
zerospace = zeros(  size(realspace,1),   size(realspace,2), 'single');
TOTALprojection = zerospace;




for aa=1:Nnum,
    for bb=1:Nnum,
        for cc=1:size(realspace,3),



            Hs = squeeze(H( CAindex(cc,1):CAindex(cc,2), CAindex(cc,1):CAindex(cc,2) ,aa,bb,cc)); % CAindex(cc,1):CAindex(cc,2)：CA大小
            tempspace = zerospace;
            tempspace( (aa:Nnum:end), (bb:Nnum:end) ) = realspace( (aa:Nnum:end), (bb:Nnum:end), cc); % 要把真实物按照微透镜分，才能用PSF
            projection = conv2(tempspace, Hs, 'same');
            TOTALprojection = TOTALprojection + projection;
        end
    end
end
