import unittest

import numpy as np

from utils.artifact_quality import (
    compare_annular_structure,
    compute_visual_artifact_metrics,
)


class ArtifactQualityTest(unittest.TestCase):
    @staticmethod
    def _star(add_barbs: bool) -> tuple[np.ndarray, tuple[float, float]]:
        size = 192
        center = (20.0, 24.0)
        yy, xx = np.mgrid[1 : size + 1, 1 : size + 1]
        dx = xx - center[0]
        dy = yy - center[1]
        radius = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        image = 1.0 + 0.55 * np.cos(40.0 * theta)
        if add_barbs:
            image = image + 0.25 * np.cos(0.55 * radius) * np.cos(28.0 * theta)
        return np.clip(image, 0.0, None), center

    def test_barbs_increase_radial_and_off_harmonic_energy(self):
        clean, center = self._star(add_barbs=False)
        barbed, _ = self._star(add_barbs=True)
        options = {
            "center_xy_1based": center,
            "inner_radius_px": 30.0,
            "outer_radius_px": 120.0,
            "angular_samples": 600,
        }
        clean_metrics = compute_visual_artifact_metrics(clean, **options)
        barbed_metrics = compute_visual_artifact_metrics(barbed, **options)
        self.assertGreater(
            barbed_metrics.radial_to_tangential_gradient_energy,
            clean_metrics.radial_to_tangential_gradient_energy,
        )
        self.assertGreater(
            barbed_metrics.off_harmonic_angular_energy_fraction,
            clean_metrics.off_harmonic_angular_energy_fraction,
        )

    def test_identity_similarity_is_exact(self):
        clean, center = self._star(add_barbs=False)
        similarity = compare_annular_structure(
            clean,
            clean,
            center_xy_1based=center,
            inner_radius_px=30.0,
            outer_radius_px=120.0,
        )
        self.assertAlmostEqual(similarity.intensity_correlation, 1.0, places=12)
        self.assertAlmostEqual(similarity.affine_normalized_rmse, 0.0, places=12)
        self.assertAlmostEqual(similarity.gradient_cosine_similarity, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
