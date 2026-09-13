import copy
import unittest

import torch

from models.configurable_anchor_lfm_net import ConfigurableAnchorLFMNet
from tests.model_fixtures import tiny_inputs


def model(anchor="mean_rl3"):
    return ConfigurableAnchorLFMNet(
        reconstruction_anchor=anchor,
        var_channels=(4, 8, 12),
        mean_channels=(4, 8, 12),
        set_channels=(4, 8, 12),
        decoder_channels=(4, 8, 12),
        set_frame_chunk_size=2,
    )


class MeanAnchorTest(unittest.TestCase):
    def test_zero_head_selects_mean_anchor(self):
        instance = model()
        f_var, g_mean, residual, z = tiny_inputs()
        output = instance(f_var, g_mean, residual, z)
        torch.testing.assert_close(output.reconstruction, instance.beta * g_mean)
        torch.testing.assert_close(output.residual, torch.zeros_like(output.residual))

    def test_legacy_default_selects_taylor_anchor(self):
        instance = model("taylor_sqrt")
        f_var, g_mean, residual, z = tiny_inputs()
        output = instance(f_var, g_mean, residual, z)
        torch.testing.assert_close(output.reconstruction, instance.beta * f_var)

    def test_anchor_choice_does_not_change_parameters_or_features(self):
        torch.manual_seed(20260901)
        mean = model("mean_rl3")
        torch.manual_seed(20260901)
        taylor = model("taylor_sqrt")
        for name, value in mean.state_dict().items():
            torch.testing.assert_close(value, taylor.state_dict()[name])
        f_var, g_mean, residual, z = tiny_inputs()
        mean_output = mean(f_var, g_mean, residual, z)
        taylor_output = taylor(f_var, g_mean, residual, z)
        for left, right in zip(mean_output.variance_features, taylor_output.variance_features):
            torch.testing.assert_close(left, right)
        for left, right in zip(mean_output.mean_features, taylor_output.mean_features):
            torch.testing.assert_close(left, right)

    def test_signed_correction_backpropagates(self):
        instance = model()
        f_var, g_mean, residual, z = tiny_inputs()
        with torch.no_grad():
            instance.decoder.head.weight.fill_(0.01)
        output = instance(f_var, g_mean, residual, z)
        self.assertTrue(torch.isfinite(output.reconstruction).all())
        self.assertTrue(bool((output.reconstruction >= 0).all()))
        output.reconstruction.mean().backward()
        self.assertIsNotNone(instance.decoder.head.weight.grad)


if __name__ == "__main__":
    unittest.main()

