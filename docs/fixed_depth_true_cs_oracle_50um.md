# Fixed-depth 50 um true-Cs covariance diagnosis

Date: 2026-09-04

## Provenance correction added after the six-group audit

The historical values below are retained, but the interpretation of 5.4–5.6 um as a population-true-Cs ceiling is superseded. Those oracle targets and predictions shared the covariance of the same finite illumination bank within each training/validation split. Independent validation banks did not remove this within-split sharing. The oracle uses image truth to construct target statistics (not truth-image metrics for checkpoint selection).

The replication matches the old B=0.5, sigma=4 image to relative L2 2.49e-7. With deterministic stationary Cs, the same complete-vector 200-step setup instead yields FTC 7.226 um and relative L2 0.2950; with independent 1024-pattern target/model banks the best is 7.409 um / 0.3226. Thus the stronger resolution claim cannot be carried over to unknown-speckle real acquisitions. See [the full provenance audit](covariance_oracle_provenance_audit_20260904.md). Production no_mean remains unchanged.

## Question

With the object fixed at the known 50 um layer and only the 2D lateral image optimized for 200 steps:

1. Does the true speckle covariance contain useful lateral super-resolution information?
2. How much improvement is possible in a noise-free covariance oracle?
3. Can the same covariance-vector loss be used with the present 100-frame data?

The object depth is fixed only for this diagnostic. It is not a deployable depth prior.

## Controlled data

- Float32 linear sensor measurements.
- No realization-wise sensor or speckle max normalization.
- No quantization and no camera noise.
- One system-wide illumination scale.
- Independent speckle realizations for data, Cs modeling, and holdout validation.
- The Siemens-star object is known only for final evaluation, never for checkpoint selection.

The historical MATLAB-generated data do not satisfy these conditions: each speckle is max-normalized, the sensor path contains realization-wise max-normalization, and the saved frames are uint8 with about 23.17% zeros. Results below therefore measure physical potential, not immediate compatibility with the historical data.

## H2 baseline versus noise-free true-Cs oracle

All optimized checkpoints were selected by independent physical holdout loss. Smaller FTC and relative L2 are better. Peak ratio is the reconstructed maximum divided by the truth maximum after unit-mass normalization.

| Method | FTC (um) | Unit-mass relative L2 | Pearson | Peak ratio | Support leak |
|---|---:|---:|---:|---:|---:|
| RL3 anchor | 7.317 | 0.321 | 0.892 | 1.649 | 0.138 |
| H2, B=0.5 | 6.860 | 0.335 | 0.880 | 1.914 | 0.082 |
| H2, B=2 | 6.494 | 0.378 | 0.854 | 2.924 | 0.068 |
| true Cs, B=0.25, probe sigma=8 | 6.037 | 0.289 | 0.913 | 1.345 | 0.113 |
| true Cs, B=0.5, probe sigma=4 | 5.580 | 0.273 | 0.922 | 1.345 | 0.099 |
| true Cs, B=0.5, probe sigma=8 | 5.580 | 0.274 | 0.921 | 1.370 | 0.100 |
| true Cs, B=0.5, probe sigma=16 | 5.763 | 0.276 | 0.920 | 1.430 | 0.101 |
| true Cs, B=1, probe sigma=8 | 5.488 | 0.264 | 0.927 | 1.367 | 0.091 |
| true Cs, B=2, probe sigma=8 | **5.397** | **0.259** | **0.930** | 1.372 | **0.087** |

The B=2 oracle is the numerical maximum in this 200-step sweep:

- FTC improves 26.3% versus the anchor and 16.9% versus the most permissive H2 result.
- Direct truth error improves 19.2% versus the anchor and 31.5% versus H2 B=2.
- Peak overshoot falls from 1.649 at the anchor and 2.924 under H2 B=2 to 1.372.
- Support leakage falls 36.9% versus the anchor.

The cleaner compromise is B=0.5 with probe sigma 4: FTC 5.580 um, zero measured center shift, relative L2 0.273, peak ratio 1.345, and substantially better target-harmonic phase consistency than B=2. The oracle reconstructions still contain residual grain and are far from the sampled binary truth's numerical FTC. The 5.397 um value is therefore an experimental upper bound for the tested optimizer, not an optical resolution claim.

Conclusion: true Cs covariance has real lateral reconstruction value. In contrast, H2 obtains a smaller FTC by adding incorrect high-frequency content: both direct truth error and peak overshoot worsen.

## Frame-count sweep

Six independent linear-float speckle sequences were generated. For each sequence, 16 fixed low-pass covariance-vector probes compared the empirical covariance action against the expected actions of the truth, anchor, two H2 reconstructions, and the best true-Cs oracle reconstruction.

| Frames | Best oracle solution ranks first | Median empirical/expected truth-action correlation |
|---:|---:|---:|
| 50 | 2/6 | 0.347 |
| 100 | 3/6 | 0.463 |
| 200 | 5/6 | 0.586 |
| 400 | 5/6 | 0.707 |
| 800 | 6/6 | 0.794 |
| 1600 | 6/6 | 0.854 |

At 100 frames, covariance can consistently reject the two H2 artifact solutions, but cannot reliably distinguish the improved oracle solution from the untouched anchor. Stable six-of-six ranking begins at 800 frames for this raw estimator.

## 100-frame covariance guardrail experiment

The exact true-Cs model was combined with H2 using covariance gradient ratios 0.01, 0.03, 0.1, and 0.3. A covariance-only group and a B=2 joint group were also run for 200 steps.

- The covariance-only holdout selected step -1, i.e. the untouched anchor.
- Every optimized joint group reduced H2 loss but increased independent covariance holdout error.
- B=0.5 joint truth errors were 0.340--0.346, all worse than the anchor's 0.321.
- B=2 joint truth error was 0.388 and peak ratio was 2.978.
- No tested covariance weight acted as a useful gradient guardrail at 100 frames.

Conclusion: with 100 frames, raw covariance-vector matching may be used as a diagnostic veto or early-stop signal, but not as a training driver. Adding a small weight does not solve the sampling problem.

## Decision

1. Keep the current no_mean reconstruction as the production baseline.
2. Do not integrate raw 100-frame covariance-vector matching into production training.
3. Continue true-Cs work because the noise-free oracle shows a credible 5.4--5.6 um lateral ceiling with better truth fidelity than H2.
4. The next research target is sample efficiency, not Set/gate tuning: translation/patch averaging, compressed covariance with shrinkage, reliable local lags, and more raw frames.
5. Simulation and real acquisition must preserve linear photometry. Per-frame max normalization, uint8 quantization, clipping, or undocumented exposure changes invalidate the absolute covariance model.
6. Any future covariance candidate must pass independent frame splits and truth-fidelity checks before being connected to the full 3D network.

## Outputs

- Controlled dataset: `outputs/linear_float_oracle_50um/dataset/`
- H2 controls: `outputs/linear_float_oracle_50um/h2_b0p5/`, `h2_b2/`
- Noise-free true-Cs oracle sweep: `outputs/linear_float_oracle_50um/oracle_cov_*/`
- Frame-count sweeps: `outputs/linear_float_oracle_50um/frame_count_seed*.json`
- 100-frame guardrail sweep: `outputs/linear_float_oracle_50um/empirical_guard_*/`
