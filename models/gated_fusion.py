from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class GatedSetFusion(nn.Module):
    def __init__(
        self,
        var_channels: tuple[int, int, int],
        mean_channels: tuple[int, int, int],
        set_channels: tuple[int, int, int],
        output_channels: tuple[int, int, int],
        *,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha_init < 1.0:
            raise ValueError("alpha_init must lie strictly between zero and one")
        self.physics = nn.ModuleList()
        self.set_projection = nn.ModuleList()
        self.gate = nn.ModuleList()
        for v, m, s, out in zip(var_channels, mean_channels, set_channels, output_channels):
            self.physics.append(nn.Conv3d(v + m, out, 1))
            self.set_projection.append(nn.Conv3d(s, out, 1))
            self.gate.append(nn.Conv3d(out + s, 1, 1))
        raw_alpha = math.log(alpha_init / (1.0 - alpha_init))
        self.raw_alpha = nn.Parameter(torch.full((3,), raw_alpha))

    def forward(
        self,
        variance_features: tuple[Tensor, Tensor, Tensor],
        mean_features: tuple[Tensor, Tensor, Tensor],
        set_features: tuple[Tensor, Tensor, Tensor] | None,
        *,
        use_set: bool,
        use_gate: bool,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor], Tensor]:
        outputs: list[Tensor] = []
        gates: list[Tensor] = []
        alphas = torch.sigmoid(self.raw_alpha)
        for scale in range(3):
            physics = self.physics[scale](
                torch.cat([variance_features[scale], mean_features[scale]], dim=1)
            )
            if use_set:
                if set_features is None:
                    raise ValueError("Set features are required when use_set=True")
                set_feature = set_features[scale]
                correction = self.set_projection[scale](set_feature)
                if use_gate:
                    gate = torch.sigmoid(self.gate[scale](torch.cat([physics, set_feature], dim=1)))
                else:
                    gate = torch.ones_like(physics[:, :1])
                output = physics + alphas[scale] * gate * correction
            else:
                gate = torch.zeros_like(physics[:, :1])
                output = physics
            outputs.append(output)
            gates.append(gate)
        return tuple(outputs), tuple(gates), alphas  # type: ignore[return-value]
