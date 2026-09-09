function three_way_generate_na_mismatch(repoRoot,outputRoot)
% Paired acquisition: exact original MATLAB phase sequence, changed pupil NA.
addpath(fullfile(repoRoot,'matlab_code','cell_dataset'),fullfile(repoRoot,'matlab_code','pilot_dataset'));
destination=fullfile(outputRoot,'illumination','mismatch_acquisition_na04479.mat');
if isfile(destination),return;end
source=fullfile(repoRoot,'outputs','priority_validation_20260907','shared_illumination','repeat_01.mat');
saved=load(source,'illumination_meta');seed=saved.illumination_meta.seed;
zUm=10:10:100;frameCount=100;NA=0.04479;lambda=488e-9;pixelSize=4.5e-6/4;sampling=260;
padded=2*sampling;L=padded*pixelSize;
frequency=-1/(2*pixelSize):1/L:1/(2*pixelSize)-1/L;
[fx,fy]=meshgrid(frequency,frequency);radial=sqrt(fx.^2+fy.^2);
pupil=radial<=NA/lambda;referencePupil=radial<=0.05/lambda;
fixedPowerGain=nnz(referencePupil)/nnz(pupil);
kz=sqrt(max(0,lambda^-2-fx.^2-fy.^2));
transfer=zeros(padded,padded,10,'like',complex(single(0)));
referenceTransfer=transfer;
for depth=1:10
 phaseFactor=exp(1i*single(2*pi*(zUm(depth)*1e-6).*kz));
 transfer(:,:,depth)=single(pupil).*phaseFactor;
 referenceTransfer(:,:,depth)=single(referencePupil).*phaseFactor;
end
oldRng=rng;cleanup=onCleanup(@()rng(oldRng));rng(double(seed),'twister');
raw=zeros(sampling,sampling,10,frameCount,'single');crop=sampling/2+1:sampling/2+sampling;
referenceCheck=zeros(sampling,sampling,10,'single');
for frame=1:frameCount
 phase=rand(sampling,sampling).*2*pi;
 slm=padarray(exp(1i*phase),[sampling/2,sampling/2],0);
 spectral=single(fftshift(fft2(slm)));
 for depth=1:10
  field=ifft2(ifftshift((spectral.*single(pupil)).*transfer(:,:,depth)));
  raw(:,:,depth,frame)=single(abs(field(crop,crop)).^2)*single(fixedPowerGain);
  if frame==1
   referenceField=ifft2(ifftshift((spectral.*single(referencePupil)).*referenceTransfer(:,:,depth)));
   referenceCheck(:,:,depth)=single(abs(referenceField(crop,crop)).^2);
  end
 end
end
reference=matfile(source);expected=single(reference.illumination_raw(:,:,:,1));
error=norm(double(referenceCheck(:))-double(expected(:)))/norm(double(expected(:)));
assert(error<1e-6,'threeway:PhasePairing','Original acquisition phase reproduction failed');
meta=struct('NA',NA,'reference_NA',0.05,'seed',seed,'pixel_pitch_um',pixelSize*1e6, ...
 'wavelength_nm',lambda*1e9,'frame_count',frameCount,'z_um',zUm,'fixed_power_gain',fixedPowerGain, ...
 'normalization','one fixed pupil-area gain; no per-frame normalization', ...
 'paired_phase_reference_relative_l2',error,'source',source);
pilot_atomic_save(destination,struct('illumination_raw',raw,'illumination_meta',meta));
fprintf('Paired NA mismatch illumination complete; reference error=%g\n',error);
end
