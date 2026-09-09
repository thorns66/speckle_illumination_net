import unittest

import torch

from tests.model_fixtures import tiny_inputs, tiny_model


class AnchorInitializationTest(unittest.TestCase):
    def test_zero_head_keeps_step_zero_at_variance_anchor(self):
        model = tiny_model()
        f_var, g_mean, residual, z_values = tiny_inputs()
        output = model(f_var, g_mean, residual, z_values)
        torch.testing.assert_close(output.residual, torch.zeros_like(output.residual))
        torch.testing.assert_close(output.reconstruction, model.beta * f_var)

    def test_anchor_only_ablation_disables_refinement(self):
        model = tiny_model(use_network_refinement=False)
        f_var, g_mean, residual, z_values = tiny_inputs()
        output = model(f_var, g_mean, residual, z_values)
        torch.testing.assert_close(output.residual, torch.zeros_like(output.residual))

    def test_raw_feature_volume_does_not_replace_sqrt_anchor(self):
        model = tiny_model()
        f_var, g_mean, residual, z_values = tiny_inputs()
        output = model(
            f_var,
            g_mean,
            residual,
            z_values,
            var_feature_volume=f_var.square(),
        )
        sqrt_output = model(f_var, g_mean, residual, z_values)
        torch.testing.assert_close(output.reconstruction, model.beta * f_var)
        self.assertFalse(
            torch.equal(output.variance_features[0], sqrt_output.variance_features[0])
        )


if __name__ == "__main__":
    unittest.main()
