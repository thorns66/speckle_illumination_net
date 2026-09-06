function [rawIntensity, normalizedIntensity, meta] = cell_generate_illumination_3d(seed,frameCount,zUm)
%CELL_GENERATE_ILLUMINATION_3D Propagate one complex speckle field through Z.
% At z=0 this is the active generate_speckle_NA05.m expression. All depths
% share the same pupil coefficients; only the angular-spectrum phase varies.
validateattributes(seed,{'numeric'},{'scalar','integer','>=',0,'<=',2^32-1});
validateattributes(frameCount,{'numeric'},{'scalar','integer','positive'});
validateattributes(zUm,{'numeric'},{'real','finite','vector','nonempty'});
oldRng=rng; cleanup=onCleanup(@() rng(oldRng)); %#ok<NASGU>
rng(double(seed),'twister');
NA=0.05; lambda=488e-9; pixelSize=4.5e-6/4; sampling=260;
padded=2*sampling; fov=sampling*pixelSize; L=2*fov;
frequency=-1/(2*pixelSize):1/L:1/(2*pixelSize)-1/L;
[fx,fy]=meshgrid(frequency,frequency);
radial=sqrt(fx.^2+fy.^2); pupil=radial<=NA/lambda;
kz=sqrt(max(0,lambda^-2-fx.^2-fy.^2));
transfer=zeros(padded,padded,numel(zUm),'like',complex(single(0)));
for depth=1:numel(zUm)
    transfer(:,:,depth)=single(pupil).*exp(1i*single(2*pi*(zUm(depth)*1e-6).*kz));
end
rawIntensity=zeros(sampling,sampling,numel(zUm),frameCount,'single');
normalizedIntensity=zeros(size(rawIntensity),'single');
crop=sampling/2+1:sampling/2+sampling;
framePeaks=zeros(1,frameCount);
for frame=1:frameCount
    phase=rand(sampling,sampling).*2*pi;
    slm=padarray(exp(1i*phase),[sampling/2,sampling/2],0);
    pupilField=single(fftshift(fft2(slm))).*single(pupil);
    for depth=1:numel(zUm)
        field=ifft2(ifftshift(pupilField.*transfer(:,:,depth)));
        rawIntensity(:,:,depth,frame)=single(abs(field(crop,crop)).^2);
    end
    framePeaks(frame)=double(max(rawIntensity(:,:,:,frame),[],'all'));
    assert(isfinite(framePeaks(frame))&&framePeaks(frame)>0,'cells:Illumination','Empty speckle frame.');
    normalizedIntensity(:,:,:,frame)=rawIntensity(:,:,:,frame)/single(framePeaks(frame));
end
meta=struct('producer','cell_generate_illumination_3d/1','seed',double(seed), ...
    'rng_algorithm','twister','frame_count',double(frameCount),'z_um',double(zUm), ...
    'reference_z_um',0,'NA',NA,'wavelength_nm',lambda*1e9, ...
    'pixel_pitch_um',pixelSize*1e6,'sampling',sampling,'padded_sampling',padded, ...
    'pupil_nonzero_count',nnz(pupil),'frame_global_yxz_peaks',framePeaks, ...
    'axes','YXZF','propagation','band-limited angular spectrum exp(+i 2pi z kz)', ...
    'depth_coupling','same complex pupil field at every z; not independent layers', ...
    'normalization','normalizedIntensity uses one maximum over complete YXZ frame', ...
    'noise_model','none');
end
