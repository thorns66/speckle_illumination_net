import os
import unittest

import numpy as np
import torch

from physics.lfm_operator import LFMOperator
from tests.operator_fixtures import numpy_matlab_forward, toy_psf


class OperatorMatlabFormulaTest(unittest.TestCase):
    def test_matches_independent_matlab_conv2_formula(self):
        h = toy_psf()
        volume = torch.rand((1, 1, 2, 12, 15), dtype=h.dtype)
        expected = numpy_matlab_forward(volume.numpy()[0, 0], h.numpy())
        for mode in ("reference", "optimized"):
            actual = LFMOperator(h, mode=mode, phase_chunk_size=4)(volume)[0, 0]
            np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-10, atol=1e-10)

    @unittest.skipUnless(
        os.environ.get("MATLAB_REFERENCE_MAT") and os.environ.get("PSF_PATH"),
        "MATLAB export fixture and PSF_PATH not provided",
    )
    def test_external_matlab_export(self):
        import h5py

        from physics.psf_loader import load_psf

        with h5py.File(os.environ["MATLAB_REFERENCE_MAT"], "r") as fixture:
            volume = torch.from_numpy(np.asarray(fixture["randomVolume"]).transpose(0, 2, 1))[None, None]
            expected = np.asarray(fixture["randomSensor"]).T
            expected_back = np.asarray(fixture["randomBackprojection"]).transpose(0, 2, 1)
            z_values = np.asarray(fixture["selectedZUm"]).reshape(-1)
        psf = load_psf(os.environ["PSF_PATH"], z_values, load_h=True, load_ht=True)
        h = torch.from_numpy(psf.H)
        ht = torch.from_numpy(psf.Ht)
        operator = LFMOperator(h, ht, mode="optimized", phase_chunk_size=32)
        actual = operator(volume)[0, 0].numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        actual_back = operator.backward_with_matlab_ht(torch.from_numpy(expected)[None, None])
        np.testing.assert_allclose(actual_back[0, 0].numpy(), expected_back, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
