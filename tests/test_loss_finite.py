import unittest

import torch

from losses.self_supervised_losses import TaylorH2VarianceModel, compute_self_supervised_loss
from physics.lfm_operator import LFMOperator
from tests.model_fixtures import tiny_inputs, tiny_model
from tests.operator_fixtures import toy_psf


class LossFiniteTest(unittest.TestCase):
    def test_all_losses_and_gradients_are_finite(self):
        h = toy_psf(z=2, period=2, kh=3, kw=3, dtype=torch.float32)
        operator = LFMOperator(h, mode="optimized", phase_chunk_size=2)
        model = tiny_model()
        f_var, g_mean, residual, _ = tiny_inputs(z=2, height=8, width=9, frames=3)
        z_values = torch.tensor([10.0, 20.0])
        measured_mean = torch.rand((1, 1, 8, 9))
        measured_var = torch.rand((1, 1, 8, 9))
        output = model(f_var, g_mean, residual, z_values)
        variance_model = TaylorH2VarianceModel(operator)
        losses = compute_self_supervised_loss(
            output.reconstruction,
            measured_mean,
            measured_var,
            operator,
            variance_model,
        )
        losses.total.backward()
        self.assertTrue(torch.isfinite(losses.total))
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
