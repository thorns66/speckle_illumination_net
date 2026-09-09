function regions = t04_axial_regions(cfg)
%T04_AXIAL_REGIONS Frozen four-row test matrix in physical coordinates.
b=cfg.axial_board; ncols=numel(b.separations_um); regions=cell(1,4*ncols); rowNames='ABCD';
assert(numel(b.x_centers_um)==ncols && numel(b.single_depths_um)==ncols, ...
    't04:Layout','Each spacing must have its own column and single-layer control.');
families={'shallow_equal','deep_equal','weak_deep','single_control'};
for row=1:4
    for col=1:ncols
        center=[b.x_centers_um(col),b.y_centers_um(row)];
        switch row
            case 1, depths=[b.shallow_base_um,b.shallow_base_um+b.separations_um(col)]; amps=[1,1];
            case 2, depths=[b.deep_base_um,b.deep_base_um+b.separations_um(col)]; amps=[1,1];
            case 3, depths=[b.weak_base_um,b.weak_base_um+b.separations_um(col)]; amps=[1,b.weak_ratio];
            case 4, depths=b.single_depths_um(col); amps=1;
        end
        separation=[];
        if numel(depths)==2, separation=diff(depths); end
        regions{(row-1)*ncols+col}=struct('kind','axial_line_region', ...
            'region_id',sprintf('%c%d',rowNames(row),col),'row',row,'column',col, ...
            'family',families{row},'center_xy_um',center,'z_um',depths, ...
            'amplitudes',amps,'separation_um',separation, ...
            'length_um',b.line_length_um,'width_um',b.line_width_um, ...
            'axial_sigma_um',b.axial_sigma_um, ...
            'roi_bounds_xy_um',[center-b.roi_half_size_xy_um,center+b.roi_half_size_xy_um]);
    end
end
end
