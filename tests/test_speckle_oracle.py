from __future__ import annotations

import unittest

import torch

from physics.speckle_oracle import (
    SpeckleGeneratorConfig,
    analytic_intensity_covariance,
    ensemble_covariance_action,
    generate_speckle_ensemble,
    stationary_covariance_action,
)


class SpeckleOracleTest(unittest.TestCase):
    def test_generator_is_reproducible_and_normalized(self):
        config = SpeckleGeneratorConfig(sampling=8, seed=17)
        first = generate_speckle_ensemble(5, config, device="cpu", batch_size=2)
        second = generate_speckle_ensemble(5, config, device="cpu", batch_size=5)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first.amax(dim=(-2, -1)), torch.ones(5))

    def test_analytic_covariance_is_center_normalized_and_symmetric(self):
        config = SpeckleGeneratorConfig(sampling=7)
        kernel = analytic_intensity_covariance(config)
        self.assertEqual(tuple(kernel.shape), (13, 13))
        self.assertEqual(float(kernel[6, 6]), 1.0)
        torch.testing.assert_close(kernel, torch.flip(kernel, dims=(-2, -1)))

    def test_ensemble_action_matches_explicit_covariance(self):
        generator = torch.Generator().manual_seed(19)
        patterns = torch.rand((11, 4, 5), generator=generator, dtype=torch.float64)
        inputs = torch.rand((3, 4, 5), generator=generator, dtype=torch.float64)
        centered = patterns.reshape(11, -1)
        centered -= centered.mean(dim=0, keepdim=True)
        explicit = centered.T @ centered / 10
        expected = (explicit @ inputs.reshape(3, -1).T).T.reshape_as(inputs)
        actual = ensemble_covariance_action(inputs, patterns)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_stationary_action_is_symmetric_and_psd(self):
        kernel = torch.tensor(
            [[0.05, 0.1, 0.05], [0.1, 1.0, 0.1], [0.05, 0.1, 0.05]],
            dtype=torch.float64,
        )
        generator = torch.Generator().manual_seed(23)
        x = torch.randn((2, 5, 6), generator=generator, dtype=torch.float64)
        y = torch.randn((2, 5, 6), generator=generator, dtype=torch.float64)
        cx, _ = stationary_covariance_action(x, kernel)
        cy, _ = stationary_covariance_action(y, kernel)
        torch.testing.assert_close((x * cy).sum(), (cx * y).sum(), rtol=1e-11, atol=1e-11)
        self.assertGreaterEqual(float((x * cx).sum()), -1e-10)


if __name__ == "__main__":
    unittest.main()
