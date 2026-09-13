"""LFM network variant whose reconstruction anchor is selected by configuration."""
from __future__ import annotations

from typing import Any

from torch import Tensor

from .variance_anchored_lfm_net import ModelOutput, VarianceAnchoredLFMNet


ANCHORS = {"taylor_sqrt", "mean_rl3"}


class ConfigurableAnchorLFMNet(VarianceAnchoredLFMNet):
    """Keep all feature branches while choosing the volume corrected by the decoder."""

    def __init__(self, *, reconstruction_anchor: str = "taylor_sqrt", **kwargs: Any) -> None:
        if reconstruction_anchor not in ANCHORS:
            raise ValueError(
                "reconstruction_anchor must be 'taylor_sqrt' or 'mean_rl3', "
                f"got {reconstruction_anchor!r}"
            )
        super().__init__(**kwargs)
        self.reconstruction_anchor = reconstruction_anchor

    def forward(
        self,
        f_var: Tensor,
        g_mean: Tensor,
        residual_frames: Tensor,
        z_values_um: Tensor,
        *,
        var_feature_volume: Tensor | None = None,
        detail_strength: float = 1.0,
        beta0: Tensor | None = None,
    ) -> ModelOutput:
        variance_input = f_var if var_feature_volume is None else var_feature_volume
        anchor = f_var if self.reconstruction_anchor == "taylor_sqrt" else g_mean
        return super().forward(
            anchor,
            g_mean,
            residual_frames,
            z_values_um,
            var_feature_volume=variance_input,
            detail_strength=detail_strength,
            beta0=beta0,
        )


def model_from_config(config: dict[str, Any]) -> ConfigurableAnchorLFMNet:
    """Build with the same module order and defaults as the historical factory."""
    model = config["model"]
    ablation = config["ablation"]
    return ConfigurableAnchorLFMNet(
        reconstruction_anchor=model.get("reconstruction_anchor", "taylor_sqrt"),
        var_channels=model["var_channels"],
        mean_channels=model["mean_channels"],
        set_channels=model["set_channels"],
        decoder_channels=model["decoder_channels"],
        set_frame_chunk_size=model["set_frame_chunk_size"],
        alpha_init=model["alpha_init"],
        beta_range=model["beta_range"],
        positivity_eps=model.get("positivity_eps", 1e-8),
        z_scale_um=model["z_scale_um"],
        set_use_checkpoint=model.get("set_use_checkpoint", True),
        set_encoder_type=model.get("set_encoder_type", "mean_std"),
        set_transformer_heads=model.get("set_transformer_heads", 4),
        set_transformer_inducing_points=model.get("set_transformer_inducing_points", 16),
        set_transformer_layers=model.get("set_transformer_layers", 2),
        use_role_separated_detail=model.get("use_role_separated_detail", False),
        detail_lowpass_passes=model.get("detail_lowpass_passes", 2),
        detail_lowpass_auxiliary_features=model.get("detail_lowpass_auxiliary_features", True),
        detail_application=model.get("detail_application", "legacy_additive"),
        detail_log_modulation_bound=model.get("detail_log_modulation_bound", 0.15),
        network_context_pad_xy=model.get("network_context_pad_xy", 0),
        network_context_pad_mode=model.get("network_context_pad_mode", "reflect"),
        anti_alias_downsampling=model.get("anti_alias_downsampling", False),
        coarse_application=model.get("coarse_application", "legacy_anchor_positive"),
        coarse_log_residual_bound=model.get("coarse_log_residual_bound", 4.0),
        lateral_log_residual_bound=model.get("lateral_log_residual_bound", 0.5),
        axial_logit_scale=model.get("axial_logit_scale", 1.0),
        use_variance_branch=ablation["use_variance_branch"],
        use_mean_branch=ablation["use_mean_branch"],
        use_set_branch=ablation["use_set_branch"],
        use_gate=ablation["use_gate"],
        use_network_refinement=ablation.get("use_network_refinement", True),
        activation_checkpoint_segments=model.get("activation_checkpoint_segments", False),
    )
