import torch

from models.variance_anchored_lfm_net import VarianceAnchoredLFMNet


def tiny_model(**overrides):
    options = {
        "var_channels": (4, 8, 12),
        "mean_channels": (4, 8, 12),
        "set_channels": (4, 8, 12),
        "decoder_channels": (4, 8, 12),
        "set_frame_chunk_size": 2,
    }
    options.update(overrides)
    return VarianceAnchoredLFMNet(**options)


def tiny_inputs(batch=1, frames=5, z=10, height=17, width=19):
    generator = torch.Generator().manual_seed(1234)
    f_var = torch.rand((batch, 1, z, height, width), generator=generator)
    g_mean = torch.rand((batch, 1, z, height, width), generator=generator)
    residual = torch.randn((batch, frames, 1, height, width), generator=generator)
    residual = residual - residual.mean(dim=1, keepdim=True)
    z_values = torch.arange(1, z + 1, dtype=torch.float32) * 10
    return f_var, g_mean, residual, z_values
