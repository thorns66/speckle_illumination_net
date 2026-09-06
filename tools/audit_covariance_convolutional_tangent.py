"""Audit convolutional tangent geometry from cached Cs gradients.

An isolated 2D encoder/decoder reuses existing blocks. This is NOT the NTK of
the production three-branch network, and no reconstruction is optimized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import dctn
from torch import nn

from models.blocks import ConvBlock2D
from models.set_encoder import SharedFrameEncoder


class DiagnosticReadoutNet(nn.Module):
    """2D analogue of the current three-scale decoder, with zero readout."""

    def __init__(self, channels=(16, 32, 64), *, zero_readout=True):
        super().__init__()
        c0, c1, c2 = channels
        self.encoder = SharedFrameEncoder(tuple(channels))
        self.deep = ConvBlock2D(c2, c2)
        self.decode1 = ConvBlock2D(c2 + c1, c1)
        self.decode0 = ConvBlock2D(c1 + c0, c0)
        self.head = nn.Conv2d(c0, 1, 1)
        if zero_readout:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def features(self, value):
        l0, l1, l2 = self.encoder(value)
        value = self.deep(l2)
        value = F.interpolate(value, size=l1.shape[-2:], mode="bilinear", align_corners=False)
        value = self.decode1(torch.cat((value, l1), dim=1))
        value = F.interpolate(value, size=l0.shape[-2:], mode="bilinear", align_corners=False)
        return self.decode0(torch.cat((value, l0), dim=1))

    def forward(self, value):
        return self.head(self.features(value))


def finite_array(value):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite audit input")
    return value


def cosine(a, b):
    a, b = finite_array(a).ravel(), finite_array(b).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / denom, -1., 1.)) if denom > 0 else None


def agreement(values):
    pairs = (("train", "population"), ("holdout", "population"), ("train", "holdout"))
    result = {f"{a}_{b}_cosine": cosine(values[a], values[b]) for a, b in pairs}
    result["direction_gate_pass"] = all(v is not None and v >= .5 for v in result.values())
    return result


def feature_geometry(features):
    features = finite_array(features)
    if features.ndim != 3:
        raise ValueError("Features must have shape C,H,W")
    matrix = features.reshape(len(features), -1).T
    # Quotient by the constant-log-residual null direction at R=0.
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * (singular[0] if len(singular) else 0.)
    rank = int(np.count_nonzero(singular > tolerance))
    basis = u[:, :rank]
    return matrix, basis, singular[:rank]


def shape_tangent(anchor, direction):
    anchor, direction = finite_array(anchor).ravel(), finite_array(direction).ravel()
    if np.any(anchor < 0) or not np.isclose(anchor.sum(), 1):
        raise ValueError("Anchor must have unit nonnegative mass")
    return anchor * (direction - np.dot(anchor, direction))


def frequency_fraction(value, shape, cut=16):
    coefficients = dctn(finite_array(value).reshape(shape), norm="ortho")
    total = float(np.sum(coefficients ** 2))
    if total == 0:
        return None
    coarse = float(np.sum(coefficients[:cut, :cut] ** 2))
    return max(0., min(1., 1. - coarse / total))


def geometry_report(features, gradients, anchor):
    matrix, basis, singular = feature_geometry(features)
    shape = np.shape(anchor)
    if matrix.shape[0] != np.size(anchor):
        raise ValueError("Feature and anchor sizes disagree")
    if set(gradients) != {"train", "holdout", "population"}:
        raise ValueError("Exactly three target gradients required")
    raw = {}
    for name, value in gradients.items():
        if np.shape(value) != shape:
            raise ValueError("Gradient and anchor sizes disagree")
        gradient = finite_array(value).ravel()
        raw[name] = gradient - gradient.mean()
    parameter = {name: matrix.T @ value for name, value in raw.items()}
    projected = {name: basis @ (basis.T @ value) for name, value in raw.items()}
    kernel = {name: matrix @ value for name, value in parameter.items()}
    # First Adam step with empty history: bias-corrected m=g, v=g^2.
    adam = {name: matrix @ (value / (np.abs(value) + 1e-8)) for name, value in parameter.items()}
    image_kernel = {name: shape_tangent(anchor, value) for name, value in kernel.items()}
    image_adam = {name: shape_tangent(anchor, value) for name, value in adam.items()}
    # Direction comparison alone is insufficient: a preconditioned train step
    # must descend the ORIGINAL held-out/population objectives.
    descent = {}
    for method, directions in (("gradient_descent", kernel), ("initial_adam", adam),
                               ("orthogonal_projection_not_network", projected)):
        direction = directions["train"]
        descent[method] = {f"{name}_descent_cosine": cosine(direction, value)
                           for name, value in raw.items()}
        descent[method]["all_first_order_descents"] = all(
            value is not None and value > 0 for value in descent[method].values())
    result = {
        "rank": len(singular),
        "feature_condition_number": float(singular[0] / singular[-1]) if len(singular) else None,
        "raw_gradient": agreement(raw),
        "parameter_gradient": agreement(parameter),
        "orthogonal_span_projection": agreement(projected),
        "raw_gradient_descent_update": agreement(kernel),
        "image_gradient_descent_update": agreement(image_kernel),
        "raw_initial_adam_update": agreement(adam),
        "image_initial_adam_update": agreement(image_adam),
        "train_update_cross_objective_descent": descent,
        "span_retained_gradient_energy": {
            name: float(np.dot(projected[name], projected[name]) / max(np.dot(value, value), 1e-300))
            for name, value in raw.items()},
        "frequency": {
            "dct_coarse_side": 16,
            "raw_population_fine_fraction": frequency_fraction(raw["population"], shape),
            "projected_population_fine_fraction": frequency_fraction(projected["population"], shape),
            "kernel_population_fine_fraction": frequency_fraction(kernel["population"], shape)},
    }
    result["conservative_initial_gate_pass"] = all(result[key]["direction_gate_pass"] for key in (
        "parameter_gradient", "image_gradient_descent_update", "image_initial_adam_update"))
    result["conservative_initial_gate_pass"] &= all(
        descent[key]["all_first_order_descents"] for key in ("gradient_descent", "initial_adam"))
    return result


def make_network_input(anchor, kind, seed, *, context=40, channels=(16, 32, 64),
                       device="cpu", zero_readout=True):
    anchor = finite_array(anchor)
    if anchor.ndim != 2 or min(anchor.shape) <= context or context < 0:
        raise ValueError("Invalid anchor or reflect context")
    if kind not in ("anchor", "noise"):
        raise ValueError("Unknown input kind")
    # Initialize on CPU inside fork_rng, never change caller RNG or use truth.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = DiagnosticReadoutNet(channels, zero_readout=zero_readout).eval().to(device)
    if kind == "anchor":
        value = torch.tensor(anchor, dtype=torch.float32)
    else:
        generator = torch.Generator().manual_seed(20261040)
        value = torch.randn(anchor.shape, generator=generator)
    value = value / value.square().mean().sqrt().clamp_min(1e-6)
    value = value[None, None].to(device)
    if context:
        value = F.pad(value, (context,) * 4, mode="reflect")
    return model, value


def extract_features(anchor, kind, seed, *, context=40, channels=(16, 32, 64), device="cpu"):
    model, value = make_network_input(anchor, kind, seed, context=context,
                                      channels=channels, device=device)
    with torch.no_grad():
        features = model.features(value)
        output = model.head(features)
        if torch.count_nonzero(output):
            raise AssertionError("Expected exactly zero readout")
        if context:
            features = features[..., context:-context, context:-context]
    return features[0].cpu().numpy().copy()


def network_tangent_components(model, value, gradients, *, context=0):
    """Exact finite-network VJP/JVP without a dense Jacobian or any update.

    The proposed diagnostic residual is f(theta)-f(theta_initial), with the
    second term frozen. Its initial value is exactly zero; its derivative is
    just that of f. We differentiate f, not a parameter-dependent subtraction.
    """
    parameters = {name: p.detach() for name, p in model.named_parameters()}

    def output(params):
        result = torch.func.functional_call(model, params, (value,), strict=True)[0, 0]
        if context:
            result = result[context:-context, context:-context]
        return result - result.mean()

    initial_output, pullback = torch.func.vjp(output, parameters)
    frozen_offset = initial_output.detach().clone()
    raw, parameter, kernel, adam, groups = {}, {}, {}, {}, {}
    shapes = {np.shape(g) for g in gradients.values()}
    if shapes != {tuple(initial_output.shape)}:
        raise ValueError("Gradient shape disagrees with network output")
    for name, gradient in gradients.items():
        gradient = finite_array(gradient)
        gradient = gradient - gradient.mean()
        raw[name] = gradient.ravel()
        upstream = torch.as_tensor(gradient, device=value.device, dtype=value.dtype)
        param_gradient = {key: item.detach() for key, item in pullback(upstream)[0].items()}
        parameter[name] = np.concatenate([item.cpu().numpy().ravel() for item in param_gradient.values()])
        group_energy = {}
        for key, item in param_gradient.items():
            group = key.split(".")[0]
            group_energy[group] = group_energy.get(group, 0.) + float(item.double().square().sum())
        total = sum(group_energy.values())
        groups[name] = {key: energy / max(total, 1e-300) for key, energy in group_energy.items()}
        adam_parameter = {key: item / (item.abs() + 1e-8) for key, item in param_gradient.items()}
        with torch.no_grad():
            _, gd_direction = torch.func.jvp(output, (parameters,), (param_gradient,))
            _, adam_direction = torch.func.jvp(output, (parameters,), (adam_parameter,))
        kernel[name] = finite_array(gd_direction.cpu().numpy()).ravel()
        adam[name] = finite_array(adam_direction.cpu().numpy()).ravel()
    # VJP/JVP duality checks PSD and cross-target symmetry in the actual run.
    gram = np.array([[np.dot(finite_array(parameter[a]), finite_array(parameter[b]))
                      for b in gradients] for a in gradients])
    cross = np.array([[np.dot(raw[a], kernel[b]) for b in gradients] for a in gradients])
    scales = np.sqrt(np.maximum(np.diag(gram)[:, None] * np.diag(gram)[None, :], 1e-300))
    duality_error = float(np.max(np.abs(gram - cross) / scales))
    if not np.isfinite(duality_error) or duality_error > 2e-4:
        raise AssertionError(f"VJP/JVP duality failed: {duality_error}")
    return {
        "raw": raw, "parameter": parameter, "kernel": kernel, "adam": adam,
        "parameter_group_gradient_energy_fraction": groups,
        "duality_max_relative_error": duality_error,
        "parameter_count": sum(p.numel() for p in parameters.values()),
        "initial_residual_max_abs": float((initial_output.detach() - frozen_offset).abs().max()),
        "frozen_offset_rms": float(frozen_offset.square().mean().sqrt()),
    }


def full_geometry_report(model, value, gradients, anchor, *, context=40):
    if set(gradients) != {"train", "holdout", "population"}:
        raise ValueError("Exactly three target gradients required")
    components = network_tangent_components(model, value, gradients, context=context)
    raw, parameter, kernel, adam = (components[key] for key in ("raw", "parameter", "kernel", "adam"))
    image_kernel = {key: shape_tangent(anchor, val) for key, val in kernel.items()}
    image_adam = {key: shape_tangent(anchor, val) for key, val in adam.items()}
    descent = {}
    for method, directions in (("gradient_descent", kernel), ("initial_adam", adam)):
        direction = directions["train"]
        descent[method] = {f"{key}_descent_cosine": cosine(direction, val) for key, val in raw.items()}
        descent[method]["all_first_order_descents"] = all(
            val is not None and val > 0 for val in descent[method].values())
    report = {
        "raw_gradient": agreement(raw),
        "parameter_gradient": agreement(parameter),
        "raw_gradient_descent_update": agreement(kernel),
        "image_gradient_descent_update": agreement(image_kernel),
        "raw_initial_adam_update": agreement(adam),
        "image_initial_adam_update": agreement(image_adam),
        "train_update_cross_objective_descent": descent,
        "frequency": {
            "dct_coarse_side": 16,
            "raw_population_fine_fraction": frequency_fraction(raw["population"], np.shape(anchor)),
            "kernel_population_fine_fraction": frequency_fraction(kernel["population"], np.shape(anchor)),
        },
        **{key: val for key, val in components.items() if key not in ("raw", "parameter", "kernel", "adam")},
    }
    report["conservative_initial_gate_pass"] = all(report[key]["direction_gate_pass"] for key in (
        "parameter_gradient", "image_gradient_descent_update", "image_initial_adam_update"))
    report["conservative_initial_gate_pass"] &= all(
        descent[key]["all_first_order_descents"] for key in ("gradient_descent", "initial_adam"))
    return report


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/linear_float_oracle_50um")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tangent-mode", choices=("readout", "full"), default="readout")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    root = Path(args.root)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    seeds = (20260901, 20260902, 20260903)
    datasets = {"n100": root / "dataset", "n800": root / "frame_count_nested/n800"}
    rows, provenance = [], {}
    started = time.perf_counter()
    for name, dataset in datasets.items():
        cache = root / "covariance_gradient_convergence" / name
        prior = json.loads((cache / "report.json").read_text())
        if not prior["complete"] or Path(prior["arguments"]["dataset"]).resolve() != dataset.resolve():
            raise ValueError("Incomplete or wrong cached gradient dataset")
        anchor_path = dataset / "anchor_rl3.npy"
        anchor = np.load(anchor_path)[4].astype(np.float32)
        if not np.isfinite(anchor).all() or anchor.max() <= 0:
            raise ValueError("Invalid anchor")
        # Match cached gradient initial shape exactly (float32 normalization).
        anchor = np.maximum(anchor, anchor.max() * 1e-8)
        anchor /= anchor.sum()
        provenance[name] = {"anchor_sha256": sha256(anchor_path), "gradient_sha256": {}}
        gradients = {}
        for probes in (256, 512):
            for mode in ("per_probe", "global_fixed"):
                path = cache / f"gradients_k{probes}_{mode}.npz"
                with np.load(path) as archive:
                    gradients[probes, mode] = {key: archive[key] for key in archive.files}
                provenance[name]["gradient_sha256"][path.name] = sha256(path)
        for kind in ("anchor", "noise"):
            for seed in seeds:
                if args.tangent_mode == "readout":
                    features = extract_features(anchor, kind, seed, device=args.device)
                else:
                    model, value = make_network_input(anchor, kind, seed, device=args.device, zero_readout=False)
                for (probes, mode), values in gradients.items():
                    geometry = (geometry_report(features, values, anchor) if args.tangent_mode == "readout"
                                else full_geometry_report(model, value, values, anchor))
                    row = {"dataset": name, "input": kind, "network_seed": seed,
                           "probes": probes, "loss": mode,
                           **geometry}
                    rows.append(row)
                    print(json.dumps({key: row[key] for key in (
                        "dataset", "input", "network_seed", "probes", "loss",
                        "conservative_initial_gate_pass",
                        "parameter_gradient", "train_update_cross_objective_descent")}), flush=True)
    report = {
        "complete": True, "arguments": vars(args), "rows": rows, "provenance": provenance,
        "elapsed_s": time.perf_counter() - started,
        "scope": "No reconstruction updates. Fixed 50um 2D diagnostic, not arbitrary 3D recovery.",
        "architecture": "Isolated SharedFrameEncoder + 2D analogue of ResidualDecoder3D, channels16/32/64, reflect40, zero 1x1 readout. No production model changes.",
        "null_direction": "Constant log residual removed at R=0. Image tangents include anchor-weighted unit-mass normalization.",
        "geometry": "J^T grad, JJ^T grad, and orthogonal span projection are reported separately. Only zero-readout head has nonzero initial parameter gradients. Span projection is not a CNN update.",
        "gate": "All three cosines >=0.5 in parameter/GD-image/first-Adam-image spaces plus positive train-step descent against all three original objectives. Engineering prescreen, NOT necessary/sufficient condition for 200-step success.",
        "limitations": "Cached 256/512 nested sensor probes reuse the same acquired frames; not exact convergence. Anchor comes from training frames. Population gradient is oracle evidence, not deployable data. Initial rank16 cannot establish later network capacity. No truth image or FTC used to construct features or choose a seed.",
    }
    if args.tangent_mode == "full":
        report["architecture"] = "Same isolated 2D encoder/decoder, channels16/32/64, reflect40. Random NONZERO readout plus FROZEN initial-output subtraction gives zero initial residual while activating the backbone. Not the production network."
        report["geometry"] = "Full finite-network VJP/JVP: all parameter gradients, JJ^T action, first-Adam J action, and original-objective descent. No dense Jacobian, fitted image, or span projection. Initial output offset is fixed, NOT recomputed during differentiation."
        report["limitations"] = "Same acquired frames and nested 256/512 sensor probes; not exact convergence. Anchor comes from training frames. Population gradient is oracle evidence, not deployable data. Full-initial-Jacobian audit does not prove 200-step outcomes. No truth image, depth fitting, mean backprop, or production model changes."
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
