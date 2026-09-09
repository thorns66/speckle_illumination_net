# Variance-Anchored Self-Supervised Multi-Speckle LFM V1

> **当前基线已更新为 E3＋100% mean 结构梯度上限，默认使用第 400 步 final。** 详见 [当前基线记录](CURRENT_BASELINE.md) 和 [模型及来源清单](CURRENT_BASELINE.json)。下文保留历史架构说明；其中旧的“当前基线/正式基线”表述以该记录为准。

The original per-volume V1 path below remains available as the no_mean
baseline. A separate shared multi-object, 10-frame-input/90-frame-constraint
training path is now runnable; its frozen protocol, arbitrary 1--6 A40 GPU
selection, resume, monitoring, and inference commands are documented in
[`docs/multivolume_shared_training.md`](docs/multivolume_shared_training.md).

This repository reconstructs one 3D volume at a time from multiple raw
light-field speckle measurements. It is an untrained, per-volume optimization
method, not supervised dataset training.

## Physics contract

The PSF library is periodic and shift variant:

```text
MATLAB H shape: [kernel_row, kernel_col, phase_row, phase_col, Z]
Canonical Torch shape: [Z, phase_row, phase_col, kernel_row, kernel_col]
```

For each depth and each of the 49 x 49 virtual-pixel phases, the forward model
keeps object samples at `phase_row::49, phase_col::49`, convolves that sparse
slice with its own 2D kernel using MATLAB `conv2(..., 'same')` alignment, and
sums all results. The 49 x 49 axes are positions within one microlens period;
they are not 2,401 microlenses. No PSF normalization or downsampling is done by
the Python loader.

The mean loss uses `H(g_hat)`. The V1 Taylor variance approximation uses
`H2(g_hat**2)`, where `H2 = H**2` element by element, plus the configured shot
and read-noise terms. This approximation omits finite-width speckle covariance
cross terms and is not the full covariance model. The interface is isolated in
`VariancePhysicsModel` so a future finite-Cs model can replace it.

See `docs/physics_audit.md` for the source audit and the measured MAT metadata.

## Model

`F_var` is the high-resolution reconstruction anchor because it is already a
non-negative 3D estimate from the measured sensor variance and squared-PSF RL.
The variance encoder therefore supplies the primary 3D features.

`g_mean` is an auxiliary 3D prior. It supplies stable low-frequency intensity,
support, and depth structure, but it does not replace the variance anchor.

The Set branch applies one shared 2D CNN to every centered raw frame, aggregates
nonlinear features with mean and standard deviation, and lifts them with the
absolute physical z coordinate. It only proposes a residual correction. There
is no frame-index encoding, so changing frame order does not change its result.

At scale `l`, fusion is

```text
F_phys_l = Conv1x1(concat(V_l, M_l))
G_l      = sigmoid(Conv1x1(concat(F_phys_l, S_l)))
F_l      = F_phys_l + sigmoid(raw_alpha_l) * G_l * Conv1x1(S_l)
```

Each voxel has one gate shared across feature channels. Alpha starts near 0.05.
The decoder predicts only `R_theta`; its final convolution is zero initialized.
Let `a = beta * F_var`. The nonnegative output uses an anchor-scaled ELU
correction:

```text
g_hat = a + R_theta                               if R_theta >= 0
g_hat = a * exp(R_theta / max(a, positivity_eps)) if R_theta < 0
```

This gives `g_hat = a` exactly when the zero-initialized residual is zero. It
keeps positive corrections linear and bounds negative corrections smoothly at
zero. `beta0` is initialized analytically from `H(F_var)` and the measured
sensor mean, then constrained to a configurable range around that value.
See `docs/positive_output_parameterization.md` for the alternatives and
gradient analysis.

## Data

The loader expects grayscale multi-page TIFF files:

```text
raw speckle: N x H x W
g_mean:      Z x H x W
F_var:       Z x H x W
```

The default `matlab_im2double` mode divides integer TIFFs by their type maximum,
matching MATLAB. Physics-space tensors keep this scale. Branch-only feature
normalization is applied to the variance and mean encoder inputs and never
changes sensor losses. Residual speckle frames enter the shared Set CNN without
explicit per-frame RMS normalization.

All raw statistics use exactly the configured frame selection. Set
`offline_statistics_num_frames` to the N used to generate the matching
`g_mean` and `F_var`; training refuses a declared mismatch. Frame indices in
YAML are zero based.

To move from N=100 to N=50/20/10/5, create matching mean and variance
reconstructions from that same selected subset, update all three TIFF paths,
`num_speckle_frames`, `offline_statistics_num_frames`, and optionally
`frame_indices`. Do not reuse N=100 anchors with a smaller Set input.

## Run

The existing `my_net` environment already contains the runtime dependencies.

```bash
PYTHONPATH=. /home2/xyx/miniconda3/envs/my_net/bin/python inspect_psf.py \
  psf/NEW_modifyfobj_PSFmatrix_M4NA0.15MLPitch220fml4000OSR3chunk05from10to130zspacing14.6154Nnum49lambda532n1a0_-11b0_2.9333.mat

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. python -u train_volume.py \
  --config configs/depth50_n100_no_mean_loss.yaml --device cuda:0 --init random
```

The production configuration is the variance-only (`no_mean`) objective. It keeps
the mean reconstruction as an input feature but removes the mean data term
from optimization (`lambda_mean: 0`).

Warm start loads network weights but initializes beta from the new volume and
does not load old optimizer momentum:

```bash
PYTHONPATH=. /home2/xyx/miniconda3/envs/my_net/bin/python train_volume.py \
  --config configs/depth50_n100_no_mean_loss.yaml --init checkpoint \
  --checkpoint outputs/previous_fov/checkpoint_best.pt
```

`operator_mode: reference` is the literal loop implementation.
`operator_mode: optimized` batches phases and uses differentiable FFT linear
convolution. `operator_phase_chunk_size` trades memory for launch overhead.
`physics_use_checkpoint: true` recomputes `H(g_hat)` and `H2(g_hat**2)` during
backpropagation instead of retaining their FFT intermediates. This reduces
peak GPU memory without changing the forward model or reconstruction target.

The final reconstruction TIFF/NumPy volume, gate arrays/figures, MIPs, depth
layers, metrics, and `checkpoint_best.pt` all refer to the same minimum-total-
loss step. Per-step reconstruction snapshots and `losses.csv` remain as the
optimization trace and are named by step.

## Validation and benchmarks

```bash
PYTHONPATH=. /home2/xyx/miniconda3/envs/my_net/bin/python -m unittest discover -s tests -v
PYTHONPATH=. /home2/xyx/miniconda3/envs/my_net/bin/python tests/run_optimization_smoke.py
PYTHONPATH=. /home2/xyx/miniconda3/envs/my_net/bin/python benchmark_operator.py --device cuda:0
PYTHONPATH=. /home2/xyx/miniconda3/envs/my_net/bin/python benchmark_train_step.py --device cuda:0
```

The external MATLAB comparison is skipped until a fixture exists. In MATLAB,
add the original `Code2.0/Util` directory to the path and call
`physics/matlab_reference/export_operator_reference.m`. Then set
`MATLAB_REFERENCE_MAT` and `PSF_PATH` before running the test.

The default real-data files under `data/` are placeholders. Long 260 x 260,
N=100 optimization should start only after the external MATLAB operator check
has passed. Full-FOV patch physics is intentionally not implemented because a
correct patch operator requires PSF halo and modulo-49 origin handling.

## Ablations

All ablations are YAML-only:

```text
A anchor only:                 use_network_refinement=false
B F_var refinement:           variance=true, mean=false, set=false
C F_var + Mean:               variance=true, mean=true,  set=false
D F_var + Mean + Set:         set=true, gate=false
E F_var + Mean + Set + Gate:  set=true, gate=true
```

`lambda_mean` and `lambda_var` may independently be set to zero for loss
ablations. No supervised, adversarial, Transformer, blind-PSF, or depth-axis
downsampling path is present in V1.
