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

    def test_physics_checkpoint_preserves_outputs_and_gradients(self):
        h = toy_psf(z=2, period=2, kh=3, kw=5, dtype=torch.float64)
        operator = LFMOperator(h, mode="optimized", phase_chunk_size=2)
        variance_model = TaylorH2VarianceModel(operator, alpha_noise=0.2, sigma_read=0.03)
        generator = torch.Generator().manual_seed(20260901)
        reconstruction = torch.rand(
            (1, 1, 2, 6, 7), generator=generator, dtype=torch.float64
        )
        measured_mean = torch.rand(
            (1, 1, 6, 7), generator=generator, dtype=torch.float64
        )
        measured_var = torch.rand(
            (1, 1, 6, 7), generator=generator, dtype=torch.float64
        )

        baseline_input = reconstruction.clone().requires_grad_(True)
        checkpoint_input = reconstruction.clone().requires_grad_(True)
        baseline = compute_self_supervised_loss(
            baseline_input,
            measured_mean,
            measured_var,
            operator,
            variance_model,
            physics_use_checkpoint=False,
        )
        checkpointed = compute_self_supervised_loss(
            checkpoint_input,
            measured_mean,
            measured_var,
            operator,
            variance_model,
            physics_use_checkpoint=True,
        )
        baseline.total.backward()
        checkpointed.total.backward()

        torch.testing.assert_close(checkpointed.predicted_mean, baseline.predicted_mean)
        torch.testing.assert_close(checkpointed.predicted_variance, baseline.predicted_variance)
        torch.testing.assert_close(checkpointed.total, baseline.total)
        torch.testing.assert_close(checkpoint_input.grad, baseline_input.grad)


if __name__ == "__main__":
    unittest.main()
