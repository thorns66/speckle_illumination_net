import unittest

import numpy as np
from scipy.fft import dctn, idctn

from tools.audit_covariance_gradient_convergence import tangent_report
from tools.scan_covariance_sketch_depth import _lowpass_probes


class TangentAuditTests(unittest.TestCase):
    def test_dct_coefficients_are_gradient_chain_rule(self):
        rng = np.random.default_rng(12)
        gradient = rng.normal(size=(10, 12))
        coefficient = np.zeros_like(gradient); coefficient[2, 3] = 1
        direction = idctn(coefficient, type=2, norm="ortho")
        self.assertAlmostEqual(np.sum(gradient * direction), dctn(gradient, type=2, norm="ortho")[2, 3], places=12)

    def test_projection_removes_high_frequency_noise(self):
        signal = np.zeros((16, 16)); signal[1, 1] = 1
        noise = np.zeros_like(signal); noise[12, 12] = 5
        inverse = lambda value: idctn(value, type=2, norm="ortho")
        rows = tangent_report({"train": inverse(signal+noise), "holdout": inverse(signal-noise), "population": inverse(signal)}, cuts=(4, 16))
        self.assertTrue(rows[0]["direction_gate_pass"])
        self.assertFalse(rows[1]["direction_gate_pass"])
        self.assertAlmostEqual(rows[0]["retained_gradient_energy"]["train"], 1/26)

    def test_probe_prefixes_identical(self):
        short = _lowpass_probes(16, (16, 16), sigma=0, seed=20260983)
        long = _lowpass_probes(64, (16, 16), sigma=0, seed=20260983)
        np.testing.assert_array_equal(short, long[:16])


if __name__ == "__main__":
    unittest.main()
