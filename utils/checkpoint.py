from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(
    path: str | Path,
    *,
    model_state: dict[str, torch.Tensor],
    step: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model_state,
        "step": int(step),
        "metrics": metrics,
        "config": config,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    torch.save(payload, path)


def load_model_checkpoint(model: nn.Module, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=strict)
    return payload
