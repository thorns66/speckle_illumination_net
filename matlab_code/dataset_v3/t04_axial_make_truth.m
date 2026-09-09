function sample = t04_axial_make_truth(cfg)
%T04_AXIAL_MAKE_TRUTH Finite-area XY strips, thin Gaussian Z; no GPU or RNG.
regions=t04_axial_regions(cfg); b=cfg.axial_board;
pitch=cfg.object_pixel_pitch_um; v=zeros(260,260,100,'single');
for k=1:numel(regions)
    r=regions{k}; center=r.center_xy_um;
    wx=max(0,min(cfg.x_um+pitch/2,center(1)+r.length_um/2)- ...
        max(cfg.x_um-pitch/2,center(1)-r.length_um/2))/pitch;
    wy=max(0,min(cfg.y_um+pitch/2,center(2)+r.width_um/2)- ...
        max(cfg.y_um-pitch/2,center(2)-r.width_um/2))/pitch;
    % Analytic pixel integration preserves 40x6 um area at every subpixel phase.
    mask=single(wy(:)*wx(:)');
    for j=1:numel(r.z_um)
        dz=cfg.fine_z_um-r.z_um(j);
        profile=exp(-dz.^2/(2*b.axial_sigma_um^2));
        profile(abs(dz)>b.truncation_sigma*b.axial_sigma_um)=0;
        density=mask.*reshape(single(r.amplitudes(j)*profile),1,1,[]);
        v=max(v,density);
    end
end
peak=double(max(v(:))); assert(peak>0,'t04:Empty','Empty axial board.');
fine=v/peak; coarse=zeros(260,260,10,'single');
for k=1:10
    selected=cfg.fine_z_um>=cfg.z_um(k)-5 & cfg.fine_z_um<cfg.z_um(k)+5;
    coarse(:,:,k)=mean(fine(:,:,selected),3);
end
meta=struct('normalization','one global fine-volume maximum; never per layer, region or target', ...
    'raw_global_peak',peak,'geometry_units','um','geometry_axis_order','XYZ', ...
    'array_axis_order','YXZ','geometry_revision',cfg.geometry_revision, ...
    'coarse_voxel_semantics','mean density in centered 10 um slabs', ...
    'fine_grid_semantics','1 um geometry quadrature, NOT optical resolution', ...
    'xy_voxel_semantics','analytic rectangle pixel-area average', ...
    'compositing','maximum union of non-overlapping 3D supports', ...
    'geometry',{regions});
sample=struct('cfg',cfg,'ground_truth',coarse,'ground_truth_fine',fine,'meta',meta);
end
