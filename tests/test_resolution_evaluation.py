import unittest

import numpy as np

from utils.resolution import (
    compute_ftc_curve,
    fourier_contrast,
    matlab_round_positive,
    robust_cutoff_estimate,
)
from utils.star_center import calibrate_star_center, harmonic_profile_metrics


class ResolutionEvaluationTest(unittest.TestCase):
    def test_matlab_positive_rounding(self):
        values = np.array([1.49, 1.50, 2.50, 3.01])
        np.testing.assert_array_equal(matlab_round_positive(values), [1, 2, 3, 3])

    def test_fourier_contrast_recovers_known_harmonic(self):
        count = 1000
        harmonic = 10
        index = np.arange(count)
        samples = 1.0 + 0.4 * np.cos(2.0 * np.pi * harmonic * index / count)
        self.assertAlmostEqual(fourier_contrast(samples, harmonic), 0.4, places=12)

    def test_robust_cutoff_rejects_isolated_outer_dip(self):
        radii = np.arange(1, 41, dtype=np.float64)
        resolution = radii * 0.1
        ftc = np.where(radii < 12, 0.02, 0.5)
        ftc[34] = 0.01
        cutoff, _ = robust_cutoff_estimate(
            radii,
            resolution,
            ftc,
            threshold=0.1,
            smoothing_window=1,
            consecutive_below=4,
        )
        self.assertEqual(cutoff.status, "crossing")
        self.assertEqual(cutoff.radius_px, 12.0)
        self.assertAlmostEqual(cutoff.resolution_um, 1.2)

    def test_curve_uses_only_complete_quarter_arcs(self):
        size = 80
        y, x = np.mgrid[1 : size + 1, 1 : size + 1]
        theta = np.arctan2(y - 10.0, x - 10.0)
        image = 1.0 + 0.5 * np.cos(40.0 * theta)
        curve = compute_ftc_curve(
            image,
            center_xy_1based=(10.0, 10.0),
            max_radius_px=100,
            angular_samples=400,
            smoothing_window=3,
            consecutive_below=2,
        )
        self.assertEqual(curve.radii_px[-1], 70.0)
        np.testing.assert_allclose(curve.valid_fraction, 1.0)
        self.assertTrue(np.isfinite(curve.ftc).all())

    def test_center_calibration_recovers_shifted_synthetic_star(self):
        size = 160
        true_center = (25.0, 31.0)
        y, x = np.mgrid[1 : size + 1, 1 : size + 1]
        theta = np.arctan2(y - true_center[1], x - true_center[0])
        image = 1.0 + 0.6 * np.cos(40.0 * theta)
        calibration = calibrate_star_center(
            image,
            initial_center_xy_1based=(30.0, 30.0),
            search_radius_px=8,
            harmonic=10,
            angular_samples=600,
            calibration_radii_px=np.array([50.0, 70.0, 90.0]),
        )
        np.testing.assert_allclose(calibration.center_xy_1based, true_center, atol=1.0)
        self.assertGreater(calibration.phase_coherence, 0.98)
        self.assertGreater(calibration.harmonic_purity, 0.8)

    def test_wrong_center_reduces_target_harmonic_purity(self):
        size = 160
        center = (25.0, 31.0)
        y, x = np.mgrid[1 : size + 1, 1 : size + 1]
        theta = np.arctan2(y - center[1], x - center[0])
        image = 1.0 + 0.6 * np.cos(40.0 * theta)
        radii = np.array([50.0, 70.0, 90.0])
        _, _, correct_purity = harmonic_profile_metrics(
            image,
            center_xy_1based=center,
            radii_px=radii,
            angular_samples=600,
        )
        _, _, wrong_purity = harmonic_profile_metrics(
            image,
            center_xy_1based=(35.0, 41.0),
            radii_px=radii,
            angular_samples=600,
        )
        self.assertGreater(correct_purity, wrong_purity)


if __name__ == "__main__":
    unittest.main()
