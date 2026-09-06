"""Match the first CNN image step to a pixel control without truth/holdout.

Uses completed step-0 optimizer states; no physics forward, no image fitting.
The chosen scalar LR is fixed before the new 200-update trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.audit_covariance_convolutional_tangent import sha256
from tools.diagnose_covariance_oracle_provenance import FrozenOffsetResidual
from tools.diagnose_fixed_depth_lateral import bounded_anchor_shape


def first_adam_direction(names, optimizer):
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("Require one optimizer group")
    group = optimizer["param_groups"][0]
    if len(names) != len(group["params"]) or group.get("weight_decay", 0) != 0 or group.get("amsgrad", False):
        raise ValueError("Unsupported optimizer configuration")
    beta1, beta2 = group["betas"]
    result = {}
    for name, index in zip(names, group["params"]):
        state = optimizer["state"][index]
        if float(state["step"]) != 1:
            raise ValueError("Only completed first-update states may calibrate")
        m = state["exp_avg"].double() / (1-beta1)
        v = state["exp_avg_sq"].double() / (1-beta2)
        result[name] = -m / (v.sqrt() + group["eps"])
        if not torch.isfinite(result[name]).all():
            raise FloatingPointError("Nonfinite first Adam direction")
    return result, float(group["lr"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/linear_float_oracle_50um/n800_cnn_empirical")
    parser.add_argument("--cnn", default="cnn_seed20260901")
    parser.add_argument("--pixel", default="pixels_b2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    cnn, pixel = root / args.cnn, root / args.pixel
    initial = torch.load(cnn/"state_initial.pt", map_location="cpu", weights_only=True)
    first = torch.load(cnn/"state_step000.pt", map_location="cpu", weights_only=True)
    pixel_first = torch.load(pixel/"state_step000.pt", map_location="cpu", weights_only=True)
    if initial["step"] != -1 or first["step"] != 0 or pixel_first["step"] != 0:
        raise ValueError("Require initial and completed first-step states")
    settings = first["arguments"]
    for key in ("dataset", "target_source", "cs_type", "bound", "probe_sigma", "train_probe_batch"):
        if settings[key] != pixel_first["arguments"][key]:
            raise ValueError(f"Unmatched control setting: {key}")
    if settings["shape_model"] != "cnn" or settings["target_source"] != "frames":
        raise ValueError("This calibration is for the observed-frame CNN diagnostic")
    cnn_initial = np.load(cnn/"reconstruction_initial.npy")
    pixel_initial = np.load(pixel/"reconstruction_initial.npy")
    np.testing.assert_array_equal(cnn_initial, pixel_initial)
    anchor = torch.tensor(cnn_initial[4], dtype=torch.float64)
    anchor /= anchor.sum()
    actual_cnn_first = torch.tensor(np.load(cnn/"reconstruction_step000.npy")[4], dtype=torch.float64)
    actual_cnn_first /= actual_cnn_first.sum()
    actual_pixel_first = torch.tensor(np.load(pixel/"reconstruction_step000.npy")[4], dtype=torch.float64)
    actual_pixel_first /= actual_pixel_first.sum()
    radius = float((actual_pixel_first - anchor).norm())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Control image radius must be finite and positive")
    model = FrozenOffsetResidual(anchor.float(), settings["seed"])
    model.load_state_dict(initial["residual_model"])
    model = model.to(device=device, dtype=torch.float32)
    parameters = {name: p.detach() for name, p in model.named_parameters()}
    buffers = {name: b.detach() for name, b in model.named_buffers()}
    direction, original_lr = first_adam_direction(list(parameters), first["optimizer"])
    # Interpolate the actual saved update, including its original FP32
    # rounding, rather than applying a different CPU arithmetic path.
    post_parameters = {name: first["residual_model"][name].to(parameters[name]) for name in parameters}
    anchor_device = anchor.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        cpu_offset = torch.func.functional_call(model, (parameters, buffers), (), strict=True).detach()
        initial_raw_error = float(cpu_offset.abs().max())
        if initial_raw_error > 1e-6:
            raise AssertionError(f"Initial backend replay differs by {initial_raw_error}; use the original CUDA backend")
        def image_at(lr):
            if lr == 0:
                params = parameters
            elif lr == original_lr:
                params = post_parameters
            else:
                params = {name: p + (lr/original_lr)*(post_parameters[name]-p)
                          for name, p in parameters.items()}
            # Remove only the verified sub-micro-unit backend replay offset.
            # This is fixed, not recomputed as the trial parameters change.
            raw = torch.func.functional_call(model, (params, buffers), (), strict=True) - cpu_offset
            return bounded_anchor_shape(anchor_device, raw, settings["bound"])[0].double().cpu()
        replay = image_at(original_lr)
        replay_error = float((replay-actual_cnn_first).norm()/actual_cnn_first.norm())
        if replay_error > 1e-6:
            raise AssertionError(f"Cannot replay original CUDA step: {replay_error}")
        original_radius = float((replay-anchor).norm())
        lo, hi = 0., original_lr
        if original_radius <= radius:
            raise ValueError("No oversized CNN step; do not shrink it using this tool")
        # Match image norm only, without reading truth, frames or validation.
        for _ in range(24):
            middle = (lo+hi)/2
            candidate_radius = float((image_at(middle)-anchor).norm())
            if candidate_radius > radius:
                hi = middle
            else:
                lo = middle
        calibrated_lr = (lo+hi)/2
        matched = image_at(calibrated_lr)
        matched_radius = float((matched-anchor).norm())
    if abs(matched_radius/radius-1) > .001:
        raise AssertionError("Output-step matching failed")
    output.mkdir(parents=True)
    volume = np.zeros_like(cnn_initial)
    volume[4] = matched.numpy().astype(np.float32)
    np.save(output/"counterfactual_matched_first_step.npy", volume)
    report = {
        "complete": True, "arguments": vars(args), "seed": settings["seed"],
        "original_network_lr": original_lr, "calibrated_network_lr": calibrated_lr,
        "pixel_image_step_norm": radius, "original_cnn_image_step_norm": original_radius,
        "original_to_pixel_radius_ratio": original_radius/radius,
        "matched_image_step_norm": matched_radius,
        "matched_radius_relative_error": matched_radius/radius-1,
        "original_cuda_step_replay_relative_l2": replay_error,
        "initial_raw_offset_max_abs": initial_raw_error,
        "replay_device": str(device), "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "inputs_sha256": {str(path): sha256(path) for path in (
            cnn/"state_initial.pt", cnn/"state_step000.pt", pixel/"state_step000.pt",
            cnn/"reconstruction_initial.npy", pixel/"reconstruction_step000.npy")},
        "scope": "Only original TRAIN gradient/Adam state, actual saved first parameter update and control image-step norm. No truth image, holdout score, scene gain or optimization. Saved image is a one-step counterfactual, not a reconstruction result.",
        "decision": "One fixed LR matched to the pixel first-step image radius. This removes one initial step-size confound, not a guarantee of equal subsequent optimization speed or reconstruction success.",
    }
    (output/"calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
