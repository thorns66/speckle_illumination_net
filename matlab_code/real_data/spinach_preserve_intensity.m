function spinach_preserve_intensity(manifestPath)
% CPU-only grayscale and rectification, with fixed units across all frames.
cfg=jsondecode(fileread(manifestPath));
assert(isempty(getenv('CUDA_VISIBLE_DEVICES')),'This preprocessing is CPU only');
p=cfg.calibration;
for f=1:numel(cfg.fields)
 field=cfg.fields(f); mkdir(field.gray_dir); mkdir(field.rectified_dir);
 records=struct([]);
 for k=1:numel(field.source_files)
  original=imread(field.source_files{k});
  assert(isa(original,'uint8') && ndims(original)==3 && size(original,3)==3);
  gray=rgb2gray(im2double(original)); % Matches supplied preprocessing, no max normalization.
  rect=rectify_original_geometry(gray,p);
  assert(all(isfinite(rect(:))) && min(rect(:))>=0 && max(rect(:))<=1);
  assert(all(mod(size(rect),p.Nnum)==0));
  write_float_tiff(fullfile(field.gray_dir,sprintf('frame_%03d.tif',k)),single(gray));
  write_float_tiff(fullfile(field.rectified_dir,sprintf('frame_%03d.tif',k)),single(rect));
  records(k).index_one_based=k;
  records(k).original_shape=size(original);
  records(k).rectified_shape=size(rect);
  records(k).raw_rgb_max=double(max(original(:)));
  records(k).raw_rgb_saturated_fraction=mean(original(:)==255);
  records(k).gray_max=max(gray(:)); records(k).gray_mean=mean(gray(:));
  records(k).rectified_max=max(rect(:)); records(k).rectified_mean=mean(rect(:));
  if k==1 && strcmp(field.field_id,'55')
   % Reproduce the supplied legacy chain to establish the source-file mapping.
   legacyGray=double(im2uint8(gray/max(gray(:))))/255;
   legacy=rectify_original_geometry(legacyGray,p);
   legacy=double(im2uint8(legacy/max(legacy(:))))/255;
   previous=double(imread(cfg.legacy_first_frame))/255;
   assert(isequal(size(previous),size(legacy)));
   difference=legacy-previous;
   records(k).legacy_replay_relative_l2=norm(difference(:))/max(norm(previous(:)),eps);
   records(k).legacy_replay_max_abs=max(abs(difference(:)));
   records(k).legacy_replay_exact_fraction=mean(difference(:)==0);
  end
  if mod(k,20)==0, fprintf('PREPROCESS field=%s frames=%d/100\n',field.field_id,k);end
 end
 file=fopen(fullfile(field.rectified_dir,'preprocessing_record.json'),'w');
 fprintf(file,'%s',jsonencode(struct('complete',true,'records',records,'policy', ...
  'uint8 RGB /255; MATLAB rgb2gray; supplied linear geometry; float32 TIFF; no frame max, gamma, denoise or inferred dark subtraction')));
 fclose(file);
end
fprintf('PREPROCESS_COMPLETE\n');
end

function write_float_tiff(path,value)
assert(~isfile(path),'Refusing to overwrite an existing processed frame');
t=Tiff(path,'w'); cleanup=onCleanup(@()close(t));
tag.ImageLength=size(value,1);tag.ImageWidth=size(value,2);
tag.Photometric=Tiff.Photometric.MinIsBlack;tag.BitsPerSample=32;
tag.SamplesPerPixel=1;tag.RowsPerStrip=min(64,size(value,1));
tag.PlanarConfiguration=Tiff.PlanarConfiguration.Chunky;
tag.SampleFormat=Tiff.SampleFormat.IEEEFP;tag.Compression=Tiff.Compression.None;
t.setTag(tag);t.write(value);
end

function IMG_Rect=rectify_original_geometry(IMG_BW,p)
% Preserve supplied ImageRectification2.m indexing/cropping exactly.
xCenter=p.xCenter;yCenter=p.yCenter;rx=p.rx;ry=-p.ry;dx=-p.dx;dy=p.dy;
M=p.Nnum;Mdiff=floor(M/2);
Xresample=[fliplr((xCenter+1):-rx/M:1) ((xCenter+1)+rx/M:rx/M:size(IMG_BW,2))];
Yresample=[fliplr((yCenter+1):-dy/M:1) ((yCenter+1)+dy/M:dy/M:size(IMG_BW,1))];
[X,Y]=meshgrid(1:size(IMG_BW,2),1:size(IMG_BW,1));
[Xq,Yq]=meshgrid(Xresample,Yresample);
XqCenterInit=find(Xq(1,:)==(xCenter+1))-Mdiff;
XqInit=XqCenterInit-M*floor(XqCenterInit/M)+M;
YqCenterInit=find(Yq(:,1)==(yCenter+1))-Mdiff;
YqInit=YqCenterInit-M*floor(YqCenterInit/M)+M;
[Xqq,Yqq]=meshgrid(Xresample(XqInit:end),Yresample(YqInit:end));
numleft=size((xCenter+1):-rx/M:1,2);
numright=size(((xCenter+1)+rx/M:rx/M:size(IMG_BW,2)),2);
numup=size((yCenter+1):-dy/M:1,2);
numdown=size(((yCenter+1)+dy/M:dy/M:size(IMG_BW,1)),2);
Xresample_dx=[-numup+1:0 1:numdown]*ry/M;
Xresample_dx=repmat(Xresample_dx(YqInit:end)',1,size(Xqq,2));
Xresample_dy=[-numleft+1:0 1:numright]*dx/M;
Xresample_dy=repmat(Xresample_dy(XqInit:end),size(Yqq,1),1);
IMG_RESAMPLE=interp2(X,Y,IMG_BW,Xqq+Xresample_dx,Yqq+Xresample_dy);
crop1=IMG_RESAMPLE(1:M*floor((size(IMG_RESAMPLE,1)-YqInit)/M), ...
 1:M*floor((size(IMG_RESAMPLE,2)-XqInit)/M));
XsizeML=size(crop1,2)/M;YsizeML=size(crop1,1)/M;
assert(p.XcutLeft+p.XcutRight<XsizeML && p.YcutUp+p.YcutDown<YsizeML);
Xrange=1+p.XcutLeft:XsizeML-p.XcutRight;
Yrange=1+p.YcutUp:YsizeML-p.YcutDown;
IMG_Rect=crop1((Yrange(1)-1)*M+1:Yrange(end)*M,(Xrange(1)-1)*M+1:Xrange(end)*M);
end
