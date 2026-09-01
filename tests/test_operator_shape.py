import unittest

import torch

from physics.lfm_operator import LFMOperator
from tests.operator_fixtures import toy_psf


class OperatorShapeTest(unittest.TestCase):
    def test_rectangular_shape_reference_and_optimized(self):
        h = toy_psf()
        volume = torch.rand((2, 1, 2, 11, 13), dtype=h.dtype)
        for mode in ("reference", "optimized"):
            operator = LFMOperator(h, mode=mode, phase_chunk_size=4)
            self.assertEqual(operator(volume).shape, (2, 1, 11, 13))
            self.assertEqual(operator.adjoint(operator(volume)).shape, volume.shape)


if __name__ == "__main__":
    unittest.main()
