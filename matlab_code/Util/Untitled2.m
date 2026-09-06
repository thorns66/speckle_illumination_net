H = 1:1:80;
H = reshape(H,[2,2,2,2,5]);
projection = zeros(6,6);
projection(3,3) = 1;
x3length = size(H,5);
Backprojection = zeros(size(projection, 1), size(projection, 1), x3length );

Nnum = 2;
for aa=1:Nnum,
    for bb=1:Nnum,
        for cc=1:x3length,

            Ht = imrotate( H(:,:,aa,bb,cc), 180);
            disp(Ht);
            tempSlice = conv2(projection, Ht, 'same');
            disp(tempSlice);
            Backprojection((aa:Nnum:end) , (bb:Nnum:end),cc) = Backprojection((aa:Nnum:end) , (bb:Nnum:end),cc) + tempSlice( (aa:Nnum:end) , (bb:Nnum:end) );
            disp(Backprojection)
        end
    end
end
