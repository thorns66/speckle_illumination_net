from __future__ import annotations

import unittest

import torch

from tools.diagnose_covariance_oracle_provenance import FixedCurvaturePenalty
from tools.diagnose_fixed_depth_lateral import hessian_schatten2


class FixedCurvaturePenaltyTest(unittest.TestCase):
    def raw(self):
        generator = torch.Generator().manual_seed(19)
        return torch.randn(9, 11, generator=generator, dtype=torch.float64).requires_grad_()

    def test_disabled_and_warmup_are_exact_gradient_noops(self):
        for ratio, step in ((0.0, 50), (0.03, 19)):
            raw = self.raw()
            raw.square().mean().backward()
            expected = raw.grad.clone()
            state = FixedCurvaturePenalty(ratio)
            metrics = state.apply(raw, step)
            self.assertTrue(torch.equal(raw.grad, expected))
            self.assertIsNone(state.weight)
            self.assertEqual(metrics['weighted_curvature_loss'], 0.0)

    def test_gradient_ratio_and_accumulation_match_joint_loss(self):
        raw = self.raw()
        data = raw.square().mean()
        regularizer = hessian_schatten2(raw)
        grad_data = torch.autograd.grad(data, raw, retain_graph=True)[0]
        grad_reg = torch.autograd.grad(regularizer, raw)[0]
        # Represents all chunks of the covariance objective already accumulated.
        raw.grad = grad_data.clone()
        state = FixedCurvaturePenalty(0.01)
        metrics = state.apply(raw, 20)
        self.assertAlmostEqual(metrics['curvature_gradient_ratio'], 0.01, places=12)
        torch.testing.assert_close(raw.grad, grad_data + state.weight*grad_reg,
                                   rtol=1e-12, atol=1e-12)
        reference = raw.detach().clone().requires_grad_()
        (reference.square().mean()+state.weight*hessian_schatten2(reference)).backward()
        torch.testing.assert_close(raw.grad, reference.grad, rtol=1e-12, atol=1e-12)

    def test_weight_remains_fixed_when_data_scale_changes(self):
        raw = self.raw()
        raw.square().mean().backward()
        state = FixedCurvaturePenalty(0.03)
        state.apply(raw, 20)
        weight = state.weight
        raw.grad = None
        (raw.square().mean()*7).backward()
        metrics = state.apply(raw, 21)
        self.assertEqual(state.weight, weight)
        self.assertAlmostEqual(metrics['curvature_gradient_ratio'], 0.03/7, places=12)

    def test_zero_nonfinite_and_skipped_calibration_fail_closed(self):
        raw = self.raw()
        for ratio in (-1.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                FixedCurvaturePenalty(ratio)
        raw.grad = torch.zeros_like(raw)
        with self.assertRaises(FloatingPointError):
            FixedCurvaturePenalty(0.01).apply(raw, 20)
        raw.grad.fill_(1)
        with self.assertRaises(RuntimeError):
            FixedCurvaturePenalty(0.01).apply(raw, 21)
        raw.grad.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            FixedCurvaturePenalty(0.01).apply(raw, 20)
        constant = torch.zeros_like(raw, requires_grad=True)
        constant.grad = torch.ones_like(raw)
        with self.assertRaises(FloatingPointError):
            FixedCurvaturePenalty(0.01).apply(constant, 20)

    def test_curvature_preserves_right_angle_rotations_and_affine_ramps(self):
        raw = self.raw()
        for turns in range(4):
            torch.testing.assert_close(hessian_schatten2(raw),
                                       hessian_schatten2(torch.rot90(raw, turns)),
                                       rtol=1e-12, atol=1e-12)
        y, x = torch.meshgrid(torch.arange(9, dtype=torch.float64),
                              torch.arange(11, dtype=torch.float64), indexing='ij')
        ramp = (2*x-3*y+1).requires_grad_()
        self.assertAlmostEqual(float(hessian_schatten2(ramp)), 1e-6, places=14)
        grad = torch.autograd.grad(hessian_schatten2(ramp), ramp)[0]
        self.assertEqual(float(grad.norm()), 0.0)


if __name__ == '__main__':
    unittest.main()
