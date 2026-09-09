function truth = priority_validation_make_truth(family, depthUm)
%PRIORITY_VALIDATION_MAKE_TRUTH Fixed local-structure validation phantoms.
% Geometry is sampled on the existing 1 um fine-Z quadrature and converted
% to ten 10 um slab means.  Amplitudes are never normalized per scene.

family = validatestring(family, {'points','lines','axial_pairs','zero'});
pitch = 220/49/4;
x = (0:259)*pitch; y = x; z = 5.5:1:104.5;
volume = zeros(numel(y),numel(x),numel(z),'single');
targets = struct([]);

switch family
    case 'points'
        validateattributes(depthUm,{'numeric'},{'scalar'}); assert(ismember(depthUm,[20,40,60,80,90]),'priority:Depth');
        gridX=[48,96,144,192,240]; gridY=[55,110,165,220];
        cells=zeros(20,2); position=1;
        for row=1:numel(gridY)
            for column=1:numel(gridX)
                cells(position,:)=[gridX(column),gridY(row)]; position=position+1;
            end
        end
        for item=1:4
            amplitude=1-0.5*mod(item,2);
            add_point([cells(item,:),depthUm],amplitude,item,'single');
        end
        combinations={};
        for separation=[4,6,8,12]
            for axisName={'x','y'}
                for ratio=[1,0.5]
                    combinations(end+1,:)={separation,axisName{1},ratio}; %#ok<AGROW>
                end
            end
        end
        for item=1:16
            center=cells(item+4,:); separation=combinations{item,1};
            axisName=combinations{item,2}; ratio=combinations{item,3};
            direction=[1,0]; if strcmp(axisName,'y'), direction=[0,1]; end
            add_point([center-direction*separation/2,depthUm],1,item+4,'pair_a');
            add_point([center+direction*separation/2,depthUm],ratio,item+4,'pair_b');
        end
    case 'lines'
        validateattributes(depthUm,{'numeric'},{'scalar'}); assert(ismember(depthUm,[20,40,60,80,90]),'priority:Depth');
        gridX=[55,115,175,235]; gridY=[35,77,119,161,203,245];
        combinations={};
        for gap=[0,4,8,12]
            for angle=[0,45,90]
                for amplitude=[1,0.5]
                    combinations(end+1,:)={gap,angle,amplitude}; %#ok<AGROW>
                end
            end
        end
        item=1;
        for row=1:numel(gridY)
            for column=1:numel(gridX)
                gap=combinations{item,1}; angle=combinations{item,2}; amplitude=combinations{item,3};
                center=[gridX(column),gridY(row),depthUm];
                direction=[cosd(angle),sind(angle),0]; halfLength=16;
                if gap==0
                    add_segment(center-halfLength*direction,center+halfLength*direction,amplitude,item);
                else
                    add_segment(center-halfLength*direction,center-gap/2*direction,amplitude,item);
                    add_segment(center+gap/2*direction,center+halfLength*direction,amplitude,item);
                end
                lineTarget=struct('cell_id',item,'kind','line','center_xyz_um',center, ...
                    'length_um',32,'gap_um',gap,'angle_deg',angle,'amplitude',amplitude, ...
                    'sigma_um',1); if isempty(targets), targets=lineTarget; else, targets(end+1)=lineTarget; end %#ok<AGROW>
                item=item+1;
            end
        end
    case 'axial_pairs'
        gridX=[65,145,225]; gridY=[55,115,175,235];
        cells=zeros(12,2); position=1;
        for row=1:numel(gridY)
            for column=1:numel(gridX)
                cells(position,:)=[gridX(column),gridY(row)]; position=position+1;
            end
        end
        item=1;
        for deep=[50,60,70]
            for ratio=[1,0.5]
                add_point([cells(item,:),30],1,item,'axial_a');
                add_point([cells(item,:),deep],ratio,item,'axial_b');
                item=item+1;
            end
        end
        for singleDepth=[30,50,60,70]
            add_point([cells(item,:),singleDepth],1,item,'single_control'); item=item+1;
        end
        % Cells 11 and 12 intentionally remain empty controls.
    case 'zero'
        % Deliberately empty input control.
end

zUm=10:10:100; coarse=zeros(260,260,10,'single');
for layer=1:10
    selected=z>=zUm(layer)-5 & z<zUm(layer)+5;
    coarse(:,:,layer)=mean(volume(:,:,selected),3);
end
truth=struct('family',family,'depth_um',depthUm,'ground_truth',coarse, ...
    'ground_truth_fine',volume,'z_um',zUm,'fine_z_um',z,'x_um',x,'y_um',y, ...
    'object_pixel_pitch_um',pitch,'targets',targets,'array_axis_order','YXZ', ...
    'normalization','fixed analytic amplitudes; no per-scene normalization', ...
    'coarse_voxel_semantics','mean density in centered 10 um slabs');

    function add_point(center,amplitude,cellId,role)
        sigma=1; radius=3*sigma;
        ix=find(abs(x-center(1))<=radius); iy=find(abs(y-center(2))<=radius); iz=find(abs(z-center(3))<=radius);
        [xx,yy,zz]=meshgrid(x(ix)-center(1),y(iy)-center(2),z(iz)-center(3));
        r2=xx.^2+yy.^2+zz.^2; density=amplitude*exp(-r2/(2*sigma^2)); density(r2>radius^2)=0;
        volume(iy,ix,iz)=max(volume(iy,ix,iz),single(density));
        pointTarget=struct('cell_id',cellId,'kind','point','center_xyz_um',center, ...
            'amplitude',amplitude,'sigma_um',sigma,'role',role);
        if isempty(targets), targets=pointTarget; else, targets(end+1)=pointTarget; end %#ok<AGROW>
    end

    function add_segment(first,last,amplitude,cellId)
        sigma=1; low=min(first,last)-3*sigma; high=max(first,last)+3*sigma;
        ix=find(x>=low(1)&x<=high(1)); iy=find(y>=low(2)&y<=high(2)); iz=find(z>=low(3)&z<=high(3));
        [xx,yy,zz]=meshgrid(x(ix),y(iy),z(iz)); delta=last-first;
        t=((xx-first(1))*delta(1)+(yy-first(2))*delta(2)+(zz-first(3))*delta(3))/sum(delta.^2);
        t=min(max(t,0),1);
        r2=(xx-first(1)-t*delta(1)).^2+(yy-first(2)-t*delta(2)).^2+(zz-first(3)-t*delta(3)).^2;
        density=amplitude*exp(-r2/(2*sigma^2)); density(r2>9*sigma^2)=0;
        volume(iy,ix,iz)=max(volume(iy,ix,iz),single(density));
        if cellId<0, error('priority:Cell','Unreachable cell id'); end
    end
end
