import unittest

import torch

from physics.lfm_operator import LFMOperator
from tests.operator_fixtures import toy_psf


class AdjointTest(unittest.TestCase):
    def test_inner_product_identity(self):
        h = toy_psf()
        x = torch.randn((1, 1, 2, 10, 13), dtype=h.dtype)
        y = torch.randn((1, 1, 10, 13), dtype=h.dtype)
        for mode in ("reference", "optimized"):
            operator = LFMOperator(h, mode=mode, phase_chunk_size=4)
            lhs = (operator(x) * y).sum()
            rhs = (x * operator.adjoint(y)).sum()
            relative = (lhs - rhs).abs() / torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-12)
            self.assertLess(relative.item(), 1e-10)

    def test_matlab_ht_optimized_matches_literal_loop(self):
        h = toy_psf()
        ht = toy_psf() * 0.7
        sensor = torch.randn((1, 1, 10, 13), dtype=h.dtype)
        operator = LFMOperator(h, ht, mode="optimized", phase_chunk_size=4)
        expected = operator.backward_with_matlab_ht(sensor, mode="reference")
        actual = operator.backward_with_matlab_ht(sensor, mode="optimized")
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
