# Fixed-50-um lateral-only diagnostic

## Question

How much lateral-resolution headroom remains if the true depth is supplied as an
oracle and all reconstructed mass is forced onto the 50 um layer?

This is an upper-bound diagnostic, not a deployable prior for real samples with
unknown three-dimensional support.

## Controls

- Exactly 200 Adam updates for every run.
- Only one positive, unit-mass 260x260 image is optimized; all other depth layers
  are exactly zero.
- Checkpoints are selected by physical loss, never by FTC or the reference image.
- Covariance training uses 16 low-pass sensor probes in rotating batches of four.
- Covariance validation uses an independent set of 16 probes.
- The generated-Cs factors are also independent: 1024 for training and 2048 for
  validation.
- The same fixed-reference-center FTC, radial-barb, Laplacian, phase and annular
  similarity evaluation is applied after training.

## Main measurements

The corrected raw-VAR reference cutoff is 7.775 um. The current 400-step E0
`no_mean` reconstruction measures 11.799 um at the fixed VAR center. Its much
better registered-only value is not accepted as the main result because the fitted
center moves by 5.70 original pixels.

| Result | Fixed-center FTC (um) | Laplacian / VAR | Radial barb / VAR | Intensity corr. / VAR | Interpretation |
|---|---:|---:|---:|---:|---|
| Raw VAR | 7.775 | 1.000 | 1.000 | 1.000 | Reference |
| E0 `no_mean` | 11.799 | 1.737 | 1.668 | 0.321 | Coarser and less clean |
| H2, E0 anchor, B=2 | 2.012 | 9.295 | 2.665 | 0.185 | Severe false resolution |
| Covariance, E0 anchor, B=1 | 10.153 | 2.084 | 2.519 | 0.308 | Better than E0, worse than VAR |
| H2, VAR anchor, B=0.5 | 6.860 | 6.579 | 2.795 | 0.764 | False high-frequency gain |
| Covariance, VAR anchor, B=0.25 | 7.775 | 1.300 | 1.282 | 0.921 | Most faithful, no FTC gain |
| Covariance, VAR anchor, B=0.5 | 7.683 | 1.305 | 1.683 | 0.781 | Only 1.18% FTC gain with artifacts |
| Joint, VAR anchor, B=0.5 | 7.226 | 6.755 | 2.958 | 0.767 | False high-frequency gain |

All physical holdout curves continued improving through step 199. Consequently,
the artifact increases are not explained by choosing a late checkpoint using the
reference; they are directions preferred by the present inverse objective.

## Decision

The unconstrained numerical FTC ceiling is 2.012 um, but it is invalid: it is
obtained by injecting high-frequency texture that raises normalized Laplacian
energy to 9.295 times VAR and destroys annular similarity.

Under a structural-fidelity requirement, the demonstrated useful improvement over
raw VAR is zero. The most faithful covariance result retains the same 7.775 um
cutoff while already increasing Laplacian and radial-barb measures by 30% and 28%.
The 1.18% covariance cutoff change is below a credible gain given its 68% barb
increase and 2.06-pixel fitted-center shift.

Therefore, further tuning this fixed-depth pixel optimizer, H2 loss, or current
covariance sketch is not justified as a lateral super-resolution direction. Keep
`no_mean` for the axial reconstruction baseline, but do not claim that its true
layer has higher lateral resolution than raw VAR. Any next lateral-resolution
effort should first add genuinely constraining measurements or correct the
measurement/forward statistics, and must beat raw VAR under the fixed-center and
artifact gates.
