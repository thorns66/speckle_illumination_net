import numpy as np

from tools.evaluate_validation_iteration_audit import (
    gradient_cosine,
    high_frequency_fraction,
    v01_line_metrics,
)


def test_gradient_cosine_identical_and_inverted_edges():
    image = np.zeros((32, 32), dtype=float)
    image[8:24, 10:20] = 1
    assert np.isclose(gradient_cosine(image, image), 1)
    assert gradient_cosine(1 - image, image) < 0


def test_high_frequency_fraction_detects_checkerboard():
    smooth = np.ones((32, 32), dtype=float)
    checkerboard = (np.indices((32, 32)).sum(0) % 2).astype(float)
    assert high_frequency_fraction(checkerboard) > high_frequency_fraction(smooth)


def test_fixed_two_line_metric_recovers_peaks_and_valley():
    offsets = np.linspace(-12, 12, 241)
    expected = np.array([-3.0, 3.0])
    profile = np.exp(-((offsets + 3) / 0.7) ** 2) + np.exp(-((offsets - 3) / 0.7) ** 2)
    spec = {"offset_um": offsets, "expected_peak_offsets_um": expected}
    metrics = v01_line_metrics(profile, spec)
    assert metrics["v01_two_line_valley_contrast"] > 0.99
    assert np.isclose(metrics["v01_observed_peak_separation_um"], 6.0)
    assert metrics["v01_mean_abs_peak_localization_error_um"] < 1e-12
