# Image-led Mean architecture revision

The supplied references are used for style only: microscopy image stacks, volumetric projections, colored feature planes, explicit three-scale fusion and visually dominant output. Their variance anchor, 100-frame inference, softplus mapping and experimental roadmap are not the current computation.

The revised figure explains one claim: input Mean-RL3 anchors a shape corrected using all three feature branches and then calibrated using E3. Panel a shows the input and RL3 statistics; b shows feature extraction/fusion/decoding; c shows training-only physics; d shows the Mean anchor, positive residual map and calibrated volume.

Data mapping: saved ten input frames, input/target means and variances, saved Mean-RL3 and Taylor-RL3-sqrt volumes and the saved older Mean-anchor output from the same fixed spinach-root subset. Use all ten planes for depth projection. Individual full-frame thumbnails are illustrative representatives, not independent replicates. The source is an older illustrative example, not the ongoing 800-step experiment. Record exact files and global display transformations. Cubes contain orthogonal MIPs, not claimed volumetric ray-casting or learned feature activations. Feature plates are abstract colored geometry.

Backend is the established Python/matplotlib workflow. Produce a 240-mm editing master and a 183-mm journal-layout version. Fonts on the 183-mm version must remain ≥5 pt after script scaling. Export editable SVG/PDF plus 600-dpi PNG/TIFF; perform source, rendered text, panel alignment, collision and visual checks. Preserve the first figure version. CPU only; no training mutations.
