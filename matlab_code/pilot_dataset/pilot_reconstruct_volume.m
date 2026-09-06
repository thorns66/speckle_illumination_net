function [Xraw, diagnostic] = pilot_reconstruct_volume(psf, sensor, iterations, mode)
%PILOT_RECONSTRUCT_VOLUME Original GPU backprojection + deconvRL core.
% mode='mean' uses H/Ht; mode='taylor' uses H.^2/Ht.^2. The original
% solver is unchanged. After gather, negative FFT residues are set to zero
% only when both their peak-relative magnitude and total negative mass pass
% the explicit numerical-negligibility bounds; material negatives still fail.
mode = validatestring(mode, {'mean','taylor'}, mfilename, 'mode');
validateattributes(sensor, {'single','double','uint8','uint16'}, ...
    {'real','finite','nonempty','nonsparse','2d'}, mfilename, 'sensor');
validateattributes(iterations, {'numeric'}, {'scalar','integer','nonnegative'}, ...
    mfilename, 'iterations');
assert(isequal(size(sensor), [260,260]), 'pilot:SensorShape', ...
    'The frozen pilot protocol requires a 260-by-260 sensor image.');
if strcmp(mode, 'taylor')
    H = psf.H .^ 2;
    Ht = psf.Ht .^ 2;
else
    H = psf.H;
    Ht = psf.Ht;
end
assert(size(H,5) == 10 && isequal(size(H), size(Ht)), 'pilot:ReconDepths', ...
    'Reconstruction must use all ten 10--100 um layers.');
global volumeResolution zeroImageEx exsize;
volumeResolution = [size(sensor,1), size(sensor,2), size(H,5)];
msize = [size(H,1), size(H,2)];
mmid = floor(msize / 2);
candidate = [size(sensor,1), size(sensor,2)] + mmid;
exsize = [min(2^ceil(log2(candidate(1))), 128*ceil(candidate(1)/128)), ...
    min(2^ceil(log2(candidate(2))), 128*ceil(candidate(2)/128))];
zeroImageEx = gpuArray.zeros(exsize, 'single');
forwardFUN = @(x) forwardProjectGPU(H, x);
backwardFUN = @(y) backwardProjectGPU(Ht, y);
sensorSingle = single(sensor); % exactly as the original wrapper
t0 = tic;
Htf = backwardFUN(sensorSingle);
backwardSeconds = toc(t0);
Xguess = Htf;
t1 = tic;
Xguess = deconvRL(forwardFUN, backwardFUN, Htf, double(iterations), Xguess);
iterationSeconds = toc(t1);
Xraw = single(gather(Xguess));
nonfiniteCount = sum(~isfinite(Xraw(:)));
assert(nonfiniteCount == 0, 'pilot:NonfiniteReconstruction', ...
    'Original MATLAB solver produced NaN/Inf; refusing to publish it.');
positivityFallbackUsed=false; rejectedOriginal=''; fallbackSeconds=0;
try
    [Xraw, roundoff] = pilot_sanitize_roundoff(Xraw, mode);
catch exception
    if ~strcmp(exception.identifier,'pilot:MaterialNegativeReconstruction'), rethrow(exception); end
    % Exact convolution of nonnegative PSFs and estimates is nonnegative.
    % FFT convolution can produce signed residues near zero; allowing those
    % values into the multiplicative division can amplify them for sparse
    % inputs. Re-run only rejected cases with the physical cone enforced at
    % every operator boundary. This is numerical stabilization, not a prior.
    assert(min(H,[],'all')>=0&&min(Ht,[],'all')>=0, ...
        'pilot:PositivityFallback','Cannot use positivity fallback with signed kernels.');
    rejectedOriginal=exception.message; positivityFallbackUsed=true; t2=tic;
    positiveForward=@(x) max(forwardProjectGPU(H,max(x,0)),0);
    positiveBackward=@(y) max(backwardProjectGPU(Ht,max(y,0)),0);
    HtfPositive=positiveBackward(sensorSingle);
    Xguess=pilot_deconv_rl_nonnegative(positiveForward,positiveBackward, ...
        HtfPositive,double(iterations));
    Xraw=single(gather(Xguess)); fallbackSeconds=toc(t2);
    nonfiniteCount=sum(~isfinite(Xraw(:)));
    assert(nonfiniteCount==0,'pilot:NonfiniteReconstruction', ...
        'Positivity-preserving fallback produced NaN/Inf.');
    [Xraw,roundoff]=pilot_sanitize_roundoff(Xraw,[mode ' positivity fallback']);
end
diagnostic = struct('mode', mode, 'iterations', double(iterations), ...
    'z_um', psf.z_um, 'backward_seconds', backwardSeconds, ...
    'iteration_seconds', iterationSeconds, 'minimum', double(min(Xraw(:))), ...
    'maximum', double(max(Xraw(:))), 'nonfinite_count', nonfiniteCount, ...
    'solver_update', 'X *= Hty / backward(forward(X)); X0=Hty; clear NaN only', ...
    'post_gather_roundoff_cleanup', roundoff, ...
    'positivity_fallback_used',positivityFallbackUsed, ...
    'positivity_fallback_seconds',fallbackSeconds, ...
    'rejected_original_path_message',rejectedOriginal, ...
    'positivity_fallback_policy',['only after original path fails both-gate sanitizer; ' ...
        'clamp FFT operator outputs to the exact nonnegative physical cone and rerun']);
end
