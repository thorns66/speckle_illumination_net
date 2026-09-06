function mesh = cell_plot_truth(sample, previewDir)
%CELL_PLOT_TRUTH Native layers, fine/native MIPs, and metric-aspect 3D views.
% No AI rendering, per-layer contrast scaling, or anisotropic Z stretching.
cfg=sample.cfg; g=sample.ground_truth; f=sample.ground_truth_fine;
mesh=cell_surface(sample);
fig=figure('Visible','off','Color','w','Position',[50,50,1800,760]);
clean=onCleanup(@() close(fig));
layout=tiledlayout(fig,2,5,'TileSpacing','compact','Padding','compact');
limit=double(max(g(:)));
for k=1:10
    ax=nexttile(layout);
    imagesc(ax,cfg.x_um,cfg.y_um,g(:,:,k),[0,limit]);
    axis(ax,'image'); colormap(ax,gray(256));
    xlabel(ax,'X (um)'); ylabel(ax,'Y (um)');
    title(ax,sprintf('z = %g um | mass %.1f%%',cfg.z_um(k),100*sample.metrics.mass_fraction(k)));
end
title(layout,sprintf('%s | %s | native 10 um slabs, ONE shared intensity scale', ...
    cfg.sample_id,cfg.description),'Interpreter','none');
exportgraphics(fig,fullfile(previewDir,'layers_native.png'),'Resolution',130);
clear clean;

fig=figure('Visible','off','Color','w','Position',[50,50,1400,920]);
clean=onCleanup(@() close(fig));
layout=tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
for fineMode=[true,false]
    if fineMode, value=f; zz=cfg.fine_z_um; label='Fine geometry (1 um Z)';
    else, value=g; zz=cfg.z_um; label='Native slabs (10 um Z)'; end
    scale=double(max(value(:)));
    ax=nexttile(layout); show_mip(ax,cfg.x_um,cfg.y_um,max(value,[],3),scale,'X','Y');
    title(ax,[label ' | XY max projection']);
    ax=nexttile(layout); show_mip(ax,cfg.x_um,zz,squeeze(max(value,[],1))',scale,'X','Depth Z');
    title(ax,[label ' | XZ max projection']);
    ax=nexttile(layout); show_mip(ax,cfg.y_um,zz,squeeze(max(value,[],2))',scale,'Y','Depth Z');
    title(ax,[label ' | YZ max projection']);
end
title(layout,sprintf('%s | MIPs: equal micrometre scale on both axes, not isotropic voxels',cfg.sample_id));
exportgraphics(fig,fullfile(previewDir,'projections_physical.png'),'Resolution',140);
clear clean;

fig=figure('Visible','off','Color','w','Position',[50,50,1450,740]);
clean=onCleanup(@() close(fig));
layout=tiledlayout(fig,1,2,'TileSpacing','loose','Padding','loose');
ax=nexttile(layout); cell_draw_surface(ax,mesh,cfg); view(ax,[-38,24]);
title(ax,{[cfg.sample_id ' | ' cfg.description],'Fine geometry | physical aspect 1:1:1'},'Interpreter','none');
ax=nexttile(layout); cell_draw_surface(ax,mesh,cfg); view(ax,[42,19]);
title(ax,{'Second viewpoint | Z is NOT stretched','Surface at 12% global fine peak; color = depth'});
exportgraphics(fig,fullfile(previewDir,'view3d_physical.png'),'Resolution',140);
savefig(fig,fullfile(previewDir,'view3d_physical.fig'));
clear clean;

fig=figure('Visible','off','Color','w','Position',[50,50,1000,360]);
clean=onCleanup(@() close(fig));
bar(cfg.z_um,sample.metrics.mass_fraction,0.7); xlabel('Depth (um)'); ylabel('Fraction of total mass');
title(sprintf('%s | native GT mass (not normalized per layer)',cfg.sample_id)); grid on;
exportgraphics(fig,fullfile(previewDir,'axial_mass.png'),'Resolution',120);
end

function show_mip(ax,x,y,value,scale,xname,yname)
imagesc(ax,x,y,value,[0,scale]); axis(ax,'image'); colormap(ax,gray(256));
xlabel(ax,[xname ' (um)']); ylabel(ax,[yname ' (um)']);
set(ax,'YDir','reverse');
end

function mesh=cell_surface(sample)
fv=isosurface(sample.ground_truth_fine,0.12);
assert(~isempty(fv.vertices),'cells:Surface','No surface at the frozen threshold.');
% Reduce only the DISPLAY mesh, never the saved truth volume.
if size(fv.faces,1)>90000, fv=reducepatch(fv,90000/size(fv.faces,1)); end
fv.vertices(:,1)=(fv.vertices(:,1)-1)*sample.cfg.object_pixel_pitch_um;
fv.vertices(:,2)=(fv.vertices(:,2)-1)*sample.cfg.object_pixel_pitch_um;
fv.vertices(:,3)=(fv.vertices(:,3)-1)*sample.cfg.fine_dz_um+sample.cfg.fine_z_um(1);
mesh=fv;
end
