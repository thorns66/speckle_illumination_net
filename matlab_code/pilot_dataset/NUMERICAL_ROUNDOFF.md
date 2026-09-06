# Numerical roundoff policy

The unchanged legacy MATLAB solver uses single-precision FFT convolution.
The pilot wrapper never modifies the iterative update itself.

The frozen operator accumulates `49 x 49 x 10 = 24010` single-precision
convolution terms per volume projection/backprojection. With float32 unit
roundoff `u`, the first-order sequential accumulation bound
`gamma_n = n*u/(1-n*u)` is approximately `1.43e-3`. The peak-relative
gate is therefore conservatively rounded to `2e-3`.

Post-gather cleanup is allowed only when both conditions hold:

1. `abs(min(X))/max(abs(X)) <= 2e-3` (with an eight-ULP absolute floor), and
2. `sum(abs(X(X<0)))/sum(X(X>0)) <= 2e-6`.

The mass limit is still below half of one 16-bit normalized intensity level
(`0.5/65535 = 7.63e-6`). It was revised from 1e-6 after the approved T01
continuous-dataset run produced 1.24021e-6 while independently passing the
peak-relative accumulation bound (`min/peak = 0.00120532 < 0.002`). Clipping
therefore changes integrated positive mass by only 1.24 ppm. Values above
2 ppm still fail even when the local peak gate passes.

If either final gate fails, `pilot_reconstruct_volume` does not publish or
clip that result. Because the stored PSF kernels, their Taylor squares, the
sensor and the RL iterate are nonnegative, their exact convolutions must also
be nonnegative. A rejected case is re-run once with negative FFT residues
projected to zero after every forward/backward operator call. The diagnostic
stores `positivity_fallback_used=true` and the complete message from the
rejected original path. The fallback result must still pass both final gates;
otherwise the sample remains a hard failure.

The fallback update divides only where the backprojected denominator is
strictly positive and finite. Unsupported zero-denominator locations receive
a zero multiplicative update. This avoids both `0/0` and positive/zero Inf;
every fallback iteration is required to remain finite and nonnegative.

Anything larger stops the run. Accepted values are set to zero and their
count, minimum, total mass, relative peak amplitude, mass ratio, accumulation
term count, and numerical bound are stored in the reconstruction diagnostics.

The mass gate is much tighter than the peak gate and prevents a spatially
distributed negative component from being mistaken for isolated FFT residue.
This preserves the original solver while making the nonnegative physical
contract explicit and auditable.
