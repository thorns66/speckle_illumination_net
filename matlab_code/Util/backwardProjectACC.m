function Backprojection = backwardProjectACC(Ht, projection, CAindex )

x3length = size(Ht,5);
Nnum = size(Ht,3);
Backprojection = zeros(size(projection, 1), size(projection, 2), x3length);
zeroSlice = zeros(  size(projection,1) , size(projection, 2));



for cc=1:x3length,
    tempSliceBack = zeroSlice;
    for aa=1:Nnum,
        for bb=1:Nnum,

            Hts = squeeze(Ht( CAindex(cc,1):CAindex(cc,2), CAindex(cc,1):CAindex(cc,2) ,aa,bb,cc)); % 一个物点的PSF转置
            tempSlice = zeroSlice;
            tempSlice( (aa:Nnum:end) , (bb:Nnum:end) ) = projection( (aa:Nnum:end) , (bb:Nnum:end) ); % 光场图片每一个宏像素的同样位置
            tempSliceBack = tempSliceBack + conv2(tempSlice, Hts, 'same'); % 一个深度下物的图像

        end
    end
    Backprojection(:,:,cc) = Backprojection(:,:,cc) + tempSliceBack;
end
