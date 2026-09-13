# Mean-anchor network architecture figure

- Purpose: explain how Mean-RL3 anchors a reconstruction refined by Taylor, Mean and Set features, followed by nonnegative mapping and E3 intensity calibration. No performance claim is made.
- Workflow: existing Python/matplotlib project figure workflow; all rendering uses this backend, with editable SVG/PDF and 600-dpi PNG/TIFF.
- Layout: 183-mm-wide schematic-led figure. Panel a explains feature extraction and multiscale decoding; panel b explains the Mean anchor and calibrated reconstruction; panel c explains training-only physical constraints.
- Evidence: configurable_anchor_lfm_net.py, variance_anchored_lfm_net.py, encoder3d.py, set_encoder.py, gated_fusion.py, decoder3d.py, v5_mixed_real_anchor_experiment.py, v3_compare_experiment.py and current Mean-800 configuration.
- Scientific requirements: distinguish image pixels from object voxels; preserve the axial dimension through all scales; Set uses shared 2D CNN, feature mean/std pooling and depth-conditioned lifting (not attention); Taylor input is sqrt after RL3; feature RMS normalization does not replace the physical Mean anchor; residual is signed and the positive mapping is not ReLU; E3 brightness uses only the input ten-frame mean; complementary 90 frames provide training targets only.
- Image integrity: all block/volume icons are abstract vector geometry, not reconstructed samples or empirical observations. No illustrative icon may imply improved reconstruction quality.
- Export and QA: 5-pt minimum rendered glyphs, embedded fonts, SVG editable text, aligned panel plot areas, source validator, final PDF text and collision audit, direct visual inspection. This is a publication-style architecture schematic, not a claim of acceptance by a named journal.
- Scope: CPU drawing only; do not modify any training source, checkpoint, process or configuration.
