import unittest

import torch

from physics.lfm_operator import LFMOperator
from tests.operator_fixtures import toy_psf


class H2OperatorTest(unittest.TestCase):
    def test_h2_is_elementwise_squared_psf(self):
        h = toy_psf()
        g_squared = torch.rand((1, 1, 2, 9, 14), dtype=h.dtype)
        actual = LFMOperator(h, mode="optimized", phase_chunk_size=4).forward_squared(g_squared)
        expected = LFMOperator(h.square(), mode="reference")(g_squared)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
