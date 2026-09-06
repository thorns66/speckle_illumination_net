"""Numerical guardrails for the reporting-only fixed-profile measurement."""
import numpy as np

from tools.report_complete_rl5_comparison import apparent_fwhm


def test_gaussian_fwhm_interpolated():
    x = np.linspace(-8, 8, 1601)
    y = np.exp(-x ** 2 / (2 * 1.3 ** 2))
    expected = 2 * np.sqrt(2 * np.log(2)) * 1.3
    assert abs(apparent_fwhm(x, y) - expected) < 1e-4


def test_gain_does_not_change_width():
    x = np.arange(-8, 9) * .6
    y = np.exp(-x ** 2 / 2)
    assert np.isclose(apparent_fwhm(x, y), apparent_fwhm(x, y * 127))


def test_missing_crossings_are_not_fabricated():
    x = np.arange(7)
    for y in [np.ones(7), np.zeros(7), np.arange(7), np.array([0, 0, 1, 3, 2, 2, 2])]:
        assert np.isnan(apparent_fwhm(x, y))
