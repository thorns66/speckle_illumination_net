import unittest

import numpy as np
from matplotlib.colors import PowerNorm

from utils.display_normalization import normalize_display_volume


class DisplayNormalizationTest(unittest.TestCase):
    def test_whole_volume_scalar_preserves_depth_and_does_not_mutate(self):
        value = np.array([[[1.0, 2.0]], [[3.0, 6.0]]])
        original = value.copy()
        normalized, maximum = normalize_display_volume(value)
        self.assertEqual(maximum, 6)
        np.testing.assert_array_equal(value, original)
        np.testing.assert_allclose(normalized, value / 6)
        np.testing.assert_allclose(
            normalized.sum((1, 2)) / normalized.sum(), value.sum((1, 2)) / value.sum()
        )
        self.assertAlmostEqual(normalized[0].max(), 1 / 3)

    def test_global_intensity_scale_invariance_and_zero(self):
        value = np.arange(8, dtype=float).reshape(2, 2, 2)
        np.testing.assert_allclose(
            normalize_display_volume(value)[0],
            normalize_display_volume(value * 0.001)[0],
        )
        zero, maximum = normalize_display_volume(np.zeros((2, 2, 2)))
        self.assertEqual(maximum, 0)
        self.assertTrue(np.array_equal(zero, np.zeros_like(zero)))

    def test_invalid_inputs(self):
        for value in (
            np.ones((2, 2)),
            np.array([[[-1.0]]]),
            np.array([[[np.nan]]]),
            np.array([[[np.inf]]]),
            np.zeros((0, 2, 2)),
        ):
            with self.assertRaises(ValueError):
                normalize_display_volume(value)

    def test_shared_gamma_enhances_display_without_changing_data(self):
        normalized, _ = normalize_display_volume(np.array([[[0.0, 0.01, 1.0]]]))
        before = normalized.copy()
        displayed = PowerNorm(gamma=0.5, vmin=0, vmax=1)(normalized)
        self.assertAlmostEqual(displayed[0, 0, 1], 0.1)
        np.testing.assert_array_equal(normalized, before)


if __name__ == "__main__":
    unittest.main()
