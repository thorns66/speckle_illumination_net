function cell_draw_surface(ax, mesh, cfg)
%CELL_DRAW_SURFACE Plot vertices already expressed in physical micrometres.
patch(ax,'Faces',mesh.faces,'Vertices',mesh.vertices, ...
    'FaceVertexCData',mesh.vertices(:,3),'FaceColor','interp', ...
    'EdgeColor','none','FaceLighting','gouraud','AmbientStrength',0.6);
colormap(ax,turbo(256)); clim(ax,[10,100]);
xlim(ax,[0,cfg.x_um(end)]); ylim(ax,[0,cfg.y_um(end)]); zlim(ax,[5,105]);
daspect(ax,[1,1,1]); set(ax,'ZDir','reverse');
xlabel(ax,'X (um)'); ylabel(ax,'Y (um)'); zlabel(ax,'Depth (um)');
grid(ax,'on'); box(ax,'on'); view(ax,[-38,24]); camlight(ax,'headlight');
end
