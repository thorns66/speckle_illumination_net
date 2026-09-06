"""Full-size value/gradient audit of the compressed covariance implementation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tools.diagnose_cs_information import CachedCs, load_operator, model_action
from tools.diagnose_projected_covariance_likelihood import fixed_sensor_basis, precompute_adjoint, projected_model_covariance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--dataset", default="outputs/linear_float_oracle_50um/frame_count_nested/n800")
    parser.add_argument("--config", default="configs/depth50_n100_no_mean_loss.yaml")
    parser.add_argument("--output", default="outputs/linear_float_oracle_50um/projected_covariance_likelihood/full_size_equivalence.json")
    args = parser.parse_args()
    device = torch.device(args.device); torch.cuda.set_device(device)
    dataset = Path(args.dataset)
    operator = load_operator(Path(args.config).resolve(), device)
    metadata = json.loads((dataset / "metadata.json").read_text())
    cs = CachedCs(torch.from_numpy(np.load(dataset / "cs_kernel.npy")).to(device)
                  * metadata["model_pattern_variance_mean"], (260, 260))
    q = torch.from_numpy(fixed_sensor_basis(3, (260, 260), 4, 20260921)).to(device)
    back = precompute_adjoint(operator, q)
    shape = torch.from_numpy(np.load(dataset / "anchor_rl3.npy")[4]).to(device)
    shape = (shape / shape.sum()).requires_grad_()
    compact = projected_model_covariance(shape, back, cs)
    full_action = model_action(shape, back, cs, operator, checkpoint_enabled=True)
    full = q.flatten(1) @ full_action.flatten(1).T
    value_error = float((compact - full).norm() / full.norm())
    scale = full.detach().norm()
    grad_compact = torch.autograd.grad((compact / scale).square().sum(), shape, retain_graph=True)[0]
    grad_full = torch.autograd.grad((full / scale).square().sum(), shape)[0]
    grad_error = float((grad_compact - grad_full).norm() / grad_full.norm())
    with torch.no_grad():
        projected = operator(shape.detach()[None, None, None])[0, 0]
        left = (q * projected).flatten(1).sum(1)
        right = (back[:, 0, 0] * shape.detach()).flatten(1).sum(1)
        adjoint_error = float((left - right).norm() / left.norm())
    report = {"image_shape": [260, 260], "phase_period": operator.phase_period,
              "kernel_shape": list(operator.kernel_shape), "covariance_value_relative_l2": value_error,
              "gradient_relative_l2": grad_error, "adjoint_inner_product_relative_l2": adjoint_error,
              "passed": max(value_error, grad_error, adjoint_error) < 2e-5}
    Path(args.output).write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Full-size projected covariance equivalence failed")


if __name__ == "__main__":
    main()
