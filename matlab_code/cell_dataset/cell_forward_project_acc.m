function sensor = cell_forward_project_acc(H,volume,CAindex)
%CELL_FORWARD_PROJECT_ACC Exact sum of original per-depth ACC projections.
validateattributes(volume,{'single','double'},{'real','finite','nonnegative','3d'});
assert(size(volume,3)==size(H,5)&&size(CAindex,1)==size(volume,3), ...
    'cells:ForwardShape','H, volume and CAindex depth counts differ.');
sensor=zeros(size(volume,1),size(volume,2),'single');
for depth=1:size(volume,3)
    if any(volume(:,:,depth)>0,'all')
        contribution=forwardProjectACC(H(:,:,:,:,depth),volume(:,:,depth),CAindex(depth,:));
        sensor=sensor+single(contribution);
    end
end
end
