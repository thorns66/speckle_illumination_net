from __future__ import annotations

import unittest

import torch

from tools.diagnose_fixed_depth_lateral import (
    bounded_anchor_shape,
    covariance_vector_loss,
    diagonal_variance_loss,
    hessian_schatten2,
)


class FixedDepthLateralTest(unittest.TestCase):
    def test_bounded_parameterization_is_positive_unit_mass_and_bounded(self) -> None:
        anchor = torch.rand(11, 13).clamp_min(1e-5)
        raw = torch.linspace(-100.0, 100.0, anchor.numel()).reshape_as(anchor)
        shape, correction = bounded_anchor_shape(anchor, raw, 0.5)
        self.assertTrue(bool((shape > 0.0).all()))
        self.assertTrue(torch.allclose(shape.sum(), torch.tensor(1.0), atol=1e-6))
        self.assertLessEqual(float(correction.abs().max()), 0.5)

    def test_covariance_loss_has_finite_nonzero_gradient(self) -> None:
        prediction = torch.randn(4, 9, 10, requires_grad=True)
        target = torch.randn(4, 9, 10)
        loss, values = covariance_vector_loss(prediction, target, correlation_weight=1.0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        self.assertGreater(float(torch.linalg.vector_norm(prediction.grad)), 0.0)
        self.assertTrue(bool(torch.isfinite(values["scale"])))

    def test_diagonal_loss_is_scale_invariant_after_analytic_fit(self) -> None:
        target = torch.rand(1, 10, 12).clamp_min(1e-4)
        prediction = target / 7.0
        loss, scale = diagonal_variance_loss(prediction, target)
        self.assertLess(float(loss), 1e-10)
        self.assertAlmostEqual(float(scale), 7.0, places=4)

    def test_curvature_of_constant_correction_is_epsilon_floor(self) -> None:
        correction = torch.full((8, 9), 0.2)
        value = hessian_schatten2(correction)
        self.assertAlmostEqual(float(value), 1e-6, places=8)


if __name__ == "__main__":
    unittest.main()
