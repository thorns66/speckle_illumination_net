# MATLAB and PSF audit

## Source of truth

The reconstruction wrappers in `matlab_code/Reconstruction3D.m` and
`matlab_code/Reconstruction3D_speckle.m` call these original utility files:

```text
/home2/xyx/mycode/my_net/Code2.0/Util/forwardProjectGPU.m
/home2/xyx/mycode/my_net/Code2.0/Util/forwardProjectACC.m
/home2/xyx/mycode/my_net/Code2.0/Util/backwardProjectGPU.m
/home2/xyx/mycode/my_net/Code2.0/Util/backwardProjectACC.m
/home2/xyx/mycode/my_net/Code2.0/Util/calcHt.m
/home2/xyx/mycode/my_net/Code2.0/Util/conv2FFT.m
```

`forwardProjectACC` loops in the order phase row `aa`, phase column `bb`, and
depth `cc`. It places one depth slice only at MATLAB indices
`aa:Nnum:end, bb:Nnum:end`, convolves with `H(:,:,aa,bb,cc)` using zero-boundary
`conv2(...,'same')`, then sums. Python uses `aa-1::Nnum, bb-1::Nnum`.

The GPU path uses `conv2FFT`. Its PSF center is `1 + floor(size(H)/2)` and the
result is cropped back to the input sensor size. The reference Python path uses
full linear convolution followed by the same center crop. The present kernel
support is odd (197 x 197), so ACC and GPU alignment agree without even-kernel
ambiguity.

The exact mathematical adjoint follows `backwardProject.m`: convolve the sensor
with the 180-degree rotated phase kernel and retain the corresponding object
phase. `calcHt.m` encodes that mapping into a second periodic kernel library for
the historical `backwardProjectGPU/ACC` implementation. Python exposes both a
direct exact adjoint and `backward_with_matlab_ht` for exported comparisons.

## Current MAT file

```text
format: MATLAB v7.3 HDF5
MATLAB H shape:  [197, 197, 49, 49, 13]
MATLAB Ht shape: [197, 197, 49, 49, 13]
Nnum: 49
CAindex: [13, 2], every row [1, 197]
x3objspace: [10,20,30,40,50,60,70,80,90,100,110,120,130] um
selected zero-based indices: [0,1,2,3,4,5,6,7,8,9]
selected physical depths: [10,20,30,40,50,60,70,80,90,100] um
canonical selected H shape: [10,49,49,197,197]
```

HDF5 exposes dimensions in reverse MATLAB order as `[13,49,49,197,197]`.
The loader explicitly maps raw `[Z,phase_col,phase_row,kernel_col,kernel_row]`
to canonical `[Z,phase_row,phase_col,kernel_row,kernel_col]`; it does not infer
orientation from equal dimensions.

The file name and scalar `zspacing` report 14.6154 um, but `x3objspace` contains
13 values at exactly 10 um spacing. The physical vector is the only value that
also agrees with the requested 10:10:100 window. V1 therefore logs this conflict
and uses `x3objspace` for exact matching. It never silently uses `zspacing`.

## Remaining external check

This machine has no MATLAB or Octave executable. The independent NumPy formula,
reference-vs-optimized comparison, finite-difference autograd check, and direct
adjoint identity pass locally. The supplied MATLAB export script still needs to
be run in a MATLAB environment to establish the final cross-language error and
to compare the historical Ht boundary behavior. Long real-volume optimization
must wait for that check.
