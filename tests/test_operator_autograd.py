import unittest

import torch

from physics.lfm_operator import LFMOperator
from tests.operator_fixtures import toy_psf


class OperatorAutogradTest(unittest.TestCase):
    def test_h2_gradient_matches_finite_difference(self):
        h = toy_psf(z=1, period=2, kh=3, kw=5)
        operator = LFMOperator(h, mode="optimized", phase_chunk_size=2)
        x = torch.rand((1, 1, 1, 5, 6), dtype=h.dtype, requires_grad=True)
        weights = torch.randn((1, 1, 5, 6), dtype=h.dtype)
        loss = (operator.forward_squared(x.square()) * weights).sum()
        loss.backward()
        index = (0, 0, 0, 2, 3)
        analytic = x.grad[index].item()
        eps = 1e-6
        with torch.no_grad():
            plus = x.detach().clone()
            minus = x.detach().clone()
            plus[index] += eps
            minus[index] -= eps
            numeric = (
                (operator.forward_squared(plus.square()) * weights).sum()
                - (operator.forward_squared(minus.square()) * weights).sum()
            ).item() / (2 * eps)
        self.assertAlmostEqual(analytic, numeric, delta=1e-6 * max(1.0, abs(numeric)))


if __name__ == "__main__":
    unittest.main()
