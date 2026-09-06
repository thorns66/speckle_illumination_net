import unittest

import torch

from tools.audit_covariance_optimizer_geometry import optimizer_direction, shape_tangent
from tools.diagnose_fixed_depth_lateral import bounded_anchor_shape


class OptimizerGeometryTest(unittest.TestCase):
    def test_adam_formula_matches_torch_and_does_not_mutate_history(self):
        torch.manual_seed(77)
        raw = torch.nn.Parameter(torch.randn(5, 7, dtype=torch.float64))
        optimizer = torch.optim.Adam([raw], lr=.01, foreach=False)
        for step in range(23):
            gradient = torch.randn_like(raw)*(.3 if step % 2 else 2.)
            state = optimizer.state.get(raw) or {
                'step': 0, 'exp_avg': torch.zeros_like(raw), 'exp_avg_sq': torch.zeros_like(raw)}
            previous_m, previous_v = state['exp_avg'].clone(), state['exp_avg_sq'].clone()
            delta, _ = optimizer_direction(gradient, state, optimizer.param_groups[0], 'adam')
            torch.testing.assert_close(previous_m, state['exp_avg'], rtol=0, atol=0)
            torch.testing.assert_close(previous_v, state['exp_avg_sq'], rtol=0, atol=0)
            previous_raw = raw.detach().clone()
            raw.grad = gradient.clone()
            optimizer.step()
            torch.testing.assert_close(raw, previous_raw+delta, rtol=1e-12, atol=1e-12)

    def test_shape_tangent_matches_autograd_and_preserves_mass(self):
        torch.manual_seed(78)
        anchor = torch.rand(7, 9, dtype=torch.float64); anchor /= anchor.sum()
        raw = torch.randn_like(anchor); direction = torch.randn_like(anchor)
        shape = bounded_anchor_shape(anchor, raw, 2.)[0]
        _, expected = torch.autograd.functional.jvp(
            lambda value: bounded_anchor_shape(anchor, value, 2.)[0], raw, direction)
        actual = shape_tangent(shape, raw, direction, 2.)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertLess(abs(float(actual.sum())), 1e-14)

    def test_directions_can_be_matched_without_truth(self):
        torch.manual_seed(79)
        anchor = torch.rand(7, 9, dtype=torch.float64); anchor /= anchor.sum()
        raw = torch.randn_like(anchor)*.1
        shape = bounded_anchor_shape(anchor, raw, 2.)[0]
        gradient = torch.randn_like(raw)
        state = {'step': 0, 'exp_avg': torch.zeros_like(raw), 'exp_avg_sq': torch.zeros_like(raw)}
        group = {'lr': .01, 'betas': (.9, .999), 'eps': 1e-8}
        reference = optimizer_direction(gradient,state,group,'adam')[0]
        norm = shape_tangent(shape,raw,reference,2.).norm()
        for variant in ('gd','isotropic_momentum','damped_adam'):
            direction = optimizer_direction(gradient,state,group,variant)[0]
            direction *= norm/shape_tangent(shape,raw,direction,2.).norm()
            self.assertAlmostEqual(float(shape_tangent(shape,raw,direction,2.).norm()),
                                   float(norm), places=14)

    def test_nonfinite_and_unsupported_updates_fail(self):
        gradient = torch.ones(3, 3)
        state = {'step': 0, 'exp_avg': torch.zeros_like(gradient), 'exp_avg_sq': torch.zeros_like(gradient)}
        group = {'lr': .01, 'betas': (.9, .999), 'eps': 1e-8}
        with self.assertRaises(ValueError):
            optimizer_direction(gradient,state,{**group,'weight_decay': .1},'adam')
        with self.assertRaises(ValueError):
            optimizer_direction(gradient,state,group,'unknown')
        with self.assertRaises(FloatingPointError):
            optimizer_direction(gradient*float('nan'),state,group,'adam')


if __name__ == '__main__':
    unittest.main()
