from __future__ import annotations

import argparse
import json

import torch

from losses.self_supervised_losses import TaylorH2VarianceModel, compute_self_supervised_loss
from models.variance_anchored_lfm_net import VarianceAnchoredLFMNet
from physics.lfm_operator import LFMOperator
from tests.operator_fixtures import toy_psf


def run(steps: int) -> dict[str, float]:
    torch.manual_seed(81)
    h = toy_psf(z=2, period=2, kh=3, kw=3, dtype=torch.float32)
    operator = LFMOperator(h, mode="optimized", phase_chunk_size=2)
    ground_truth = torch.rand((1, 1, 2, 8, 9)) * 0.8
    measured_mean = operator(ground_truth).detach()
    measured_variance = operator.forward_squared(ground_truth.square()).detach()
    f_var = (ground_truth * 0.7).detach()
    g_mean = (ground_truth * 0.85).detach()
    raw = torch.randn((1, 4, 1, 8, 9))
    residual = raw - raw.mean(dim=1, keepdim=True)
    z_values = torch.tensor([10.0, 20.0])
    model = VarianceAnchoredLFMNet(
        var_channels=(4, 4, 4),
        mean_channels=(4, 4, 4),
        set_channels=(4, 4, 4),
        decoder_channels=(4, 4, 4),
        set_frame_chunk_size=2,
    )
    model.initialize_beta(f_var, measured_mean, operator)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    variance_model = TaylorH2VarianceModel(operator)
    values = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(f_var, g_mean, residual, z_values)
        loss = compute_self_supervised_loss(
            output.reconstruction,
            measured_mean,
            measured_variance,
            operator,
            variance_model,
            lambda_tv=1e-6,
        )
        if not torch.isfinite(loss.total):
            raise RuntimeError("Smoke optimization produced a non-finite loss")
        values.append(float(loss.total.detach()))
        loss.total.backward()
        optimizer.step()
    return {
        "steps": steps,
        "initial_loss": values[0],
        "final_loss": values[-1],
        "minimum_loss": min(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 5, 20])
    args = parser.parse_args()
    print(json.dumps([run(steps) for steps in args.steps], indent=2))


if __name__ == "__main__":
    main()
