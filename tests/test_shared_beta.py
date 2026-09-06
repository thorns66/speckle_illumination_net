import unittest

import torch

from tests.model_fixtures import tiny_inputs, tiny_model


class SharedBetaTest(unittest.TestCase):
    def test_per_item_analytic_base_gets_one_shared_fractional_correction(self):
        model = tiny_model().eval()
        f_var, g_mean, residual, z_values = tiny_inputs(batch=2)
        beta0 = torch.tensor([2.0, 5.0])
        with torch.no_grad():
            model.raw_beta.fill_(0.7)
            output = model(f_var, g_mean, residual, z_values, beta0=beta0)
        correction = 1.0 + model.beta_range * torch.tanh(model.raw_beta)
        torch.testing.assert_close(output.beta, beta0 * correction)
        torch.testing.assert_close(
            output.reconstruction,
            output.beta[:, None, None, None, None] * f_var,
        )
        torch.testing.assert_close(output.beta[1] / output.beta[0], beta0[1] / beta0[0])

    def test_legacy_single_volume_path_is_unchanged(self):
        model = tiny_model().eval()
        inputs = tiny_inputs()
        with torch.no_grad():
            output = model(*inputs)
        torch.testing.assert_close(output.reconstruction, model.beta * inputs[0])
        self.assertEqual(output.beta.ndim, 0)

    def test_bad_batch_beta_is_rejected(self):
        model = tiny_model()
        inputs = tiny_inputs(batch=2)
        with self.assertRaises(ValueError):
            model(*inputs, beta0=torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
