# Fixed-50-um H2 frame-cross-fit result

## Purpose

This diagnostic fixes the object to the known 50 um layer and optimizes only its
nonnegative lateral shape. It is an oracle upper-bound test, not a deployable
depth prior. Five independent 50/50 frame splits were run for 200 updates on five
A40 GPUs. The training half supplies the sensor variance target and the disjoint
half selects the checkpoint. The untouched raw-VAR anchor is an admissible step
-1 checkpoint.

## Result

All five holdout curves decreased monotonically enough that the selected
checkpoint was step 199 in every fold. Holdout log-Huber loss improved by
20.3--21.0%. The fixed-reference-center FTC changed from 7.775 um for raw VAR to
6.860 um in every fold, a nominal 11.8% improvement.

That apparent gain fails the visual-fidelity gates:

| Measurement | Raw VAR | Five-fold range | Mean ensemble |
|---|---:|---:|---:|
| Fixed-center FTC (um; lower is better) | 7.775 | 6.860 | 6.860 |
| Laplacian energy / VAR | 1.000 | 5.274--5.524 | 5.175 |
| Radial-barb energy / VAR | 1.000 | 1.561--1.886 | 1.709 |
| Annular intensity correlation / VAR | 1.000 | 0.772--0.785 | 0.785 |
| Annular gradient cosine / VAR | 1.000 | 0.605--0.614 | 0.622 |

Median pairwise Pearson correlation between fold reconstructions is 0.976, while
the median symmetric relative L2 difference is 0.214. Mean and median ensembling
do not remove the artifacts. Therefore the barbs are not mainly independent
50-frame sampling noise; they are a repeatable direction preferred by the
misspecified/ill-conditioned inverse objective.

## Physical interpretation

The historical synthetic frames are uint8, every frame reaches exactly 255,
23.2% of pixels are quantized to zero, and the frame-integrated intensity has an
11.7% coefficient of variation. The supplied MATLAB generator independently
normalizes every speckle realization by its maximum. Its example image-forward
path also normalizes every sensor frame by its maximum before noise, adds noise,
and normalizes by the maximum again. These nonlinear, realization-dependent
operations violate the linear covariance relation used by H2 and covariance
sketch losses.

An exact sensor-max-normalized covariance scan with 4096 simulated patterns also
failed the depth gate: the 50-um data selected 60 um for both full and split
targets, with a negative true-depth margin. Using the generated true lateral Cs
therefore does not repair the present dataset.

The three-iteration Richardson--Lucy VAR anchor acts as early regularization.
Continuing a scale-free H2 inverse fit reduces the held-out sensor variance loss,
but amplifies poorly constrained high frequencies to compensate systematic data
and forward-model mismatch. Frame splitting detects independent noise overfit;
it cannot reject a shared systematic mismatch.

## Decision

The numerical 6.860-um value is not a credible resolution improvement. Do not
promote the fixed-depth H2 optimizer, its mean/median ensemble, or the current
covariance sketch into the reconstruction network. Keep raw VAR as the lateral
quality reference and `no_mean` as the depth-capable production baseline.

The next valid resolution experiment must first regenerate floating-point sensor
frames with one preserved physical photometric scale, known ground truth, and no
per-frame max normalization or image-file quantization. Add calibrated Poisson and
read noise in count units, then compare H2 and covariance methods on independent
speckles and a deliberately mismatched reconstruction model. Only methods that
improve ground-truth lateral metrics and depth metrics while passing the artifact
gates should be transferred to real data.

Authoritative machine-readable results are in
`outputs/fixed_depth_lateral_50um_h2_crossfit5/crossfit_summary.json` and the
per-fold `summary.json` / `resolution_best/resolution_metrics.json` files.
