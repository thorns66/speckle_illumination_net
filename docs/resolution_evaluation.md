# Siemens-star resolution evaluation

`tools/evaluate_resolution.py` evaluates only the configured layer matching
`--true-depth-um`. It follows the supplied `analyze_resolution_NM.m` Fourier
threshold contrast (FTC) definition, but it does not assume that the center
`(11,11)` from that particular MATLAB example is valid for another image.

## Center calibration and two reported comparisons

The default `--center-mode reference_calibrated` searches resolved outer annuli
of the VAR reference. Its score combines:

- amplitude of the expected tenth angular harmonic;
- phase agreement of that harmonic across radii;
- concentration of energy in the tenth rather than neighbouring harmonics.

That VAR-derived center is frozen for the **native shared-center** comparison.
This result contains both radial contrast and any reconstruction displacement.
The program also estimates the reconstruction center independently for
diagnostics and reports a **center-registered radial-contrast** comparison.
The latter removes star-center displacement and is closer to a pure resolution
comparison, although it still cannot replace evaluation against an ideal target.

The JSON report includes center displacement and a +/-2 evaluation-pixel center
sensitivity range. A large displacement or sensitivity range is a geometric
warning and must not be hidden inside a single resolution number.

## MATLAB mapping retained

- Each selected image is peak-normalized and bicubically upsampled by 2.
- The evaluation-grid pixel size is `5.2 / 8.93 = 0.58230683 um`.
- Each radius uses 1,000 nearest-neighbour samples over `0..pi/2`.
- FTC is `2*abs(FFT[10])/abs(FFT[0])`, corresponding to MATLAB element 11.
- The line-pair period is `radius * pixel_size * 2*pi / 40`.
- The cutoff threshold is FTC `0.1`; a smaller period is better.

Both the literal MATLAB reverse scan and a robust result are written. The robust
result applies a nine-radius median filter and requires five consecutive
below-threshold radii. The program also fixes two bugs in the supplied script:
it clears the angular profile for every radius and uses complete quarter arcs.

## Current 50 um VAR comparison

```bash
PYTHONPATH=. python -u -m tools.evaluate_resolution \
  --config configs/depth50_n100_no_mean_loss.yaml \
  --true-depth-um 50 \
  --reference data/分辨率图仿真/processed_statistics/reconstruction/z_10_to_100_n10_it3/Recon3D_img_detph50_var.tif \
  --output-dir outputs/depth50_n100_no_mean_loss/resolution_evaluation_50um_raw_var
```

If `--reference` is omitted, the configured `F_var` anchor is used. Each run
writes `resolution_metrics.json`, separate shared-center and center-registered
FTC tables/plots, normalized true-depth TIFFs, image comparison, and cutoff
overlays.

The old fixed-center result (`VAR=13.903 um`, `no_mean=7.500 um`) is invalid for
the current data: at `(22,22)` the clear VAR spokes leak from the intended tenth
harmonic mainly into the ninth. Without an ideal/ground-truth Siemens-star file,
the corrected report is a relative VAR comparison and must not be described as
absolute optical resolution.
