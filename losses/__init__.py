"""Sensor-domain self-supervised objectives."""

from .self_supervised_losses import (
    LossBreakdown,
    TaylorH2VarianceModel,
    VariancePhysicsModel,
    compute_self_supervised_loss,
)

__all__ = [
    "LossBreakdown",
    "TaylorH2VarianceModel",
    "VariancePhysicsModel",
    "compute_self_supervised_loss",
]
