"""A matched finite-bank oracle can distinguish population-indistinguishable objects."""
import unittest

import torch


class MatchedCovarianceOracleBiasTest(unittest.TestCase):
    def test_population_ambiguity_can_disappear_in_a_known_bank(self):
        # One detector sums two voxels; independent unit-variance illumination.
        h = torch.ones((1, 2), dtype=torch.float64)
        objects = torch.eye(2, dtype=torch.float64)
        population = torch.eye(2, dtype=torch.float64)
        bank = torch.tensor([[1., 2.], [1., -2.], [-1., 2.], [-1., -2.]], dtype=torch.float64)
        sampled = bank.T @ bank / (len(bank)-1)

        def sensor_variance(g, covariance):
            return (h * g) @ covariance @ (h * g).T

        torch.testing.assert_close(sensor_variance(objects[0], population),
                                   sensor_variance(objects[1], population))
        self.assertGreater(float((sensor_variance(objects[0], sampled)
                                  - sensor_variance(objects[1], sampled)).square()), 0.)

    def test_extra_squared_loss_decays_with_bank_count(self):
        # For independent standard Gaussian coordinates the expected squared
        # difference of unbiased sample variances is exactly 4/(K-1).
        rng = torch.Generator().manual_seed(20260904)
        for count in (16, 64, 256):
            patterns = torch.randn((4000, count, 2), generator=rng, dtype=torch.float64)
            variance = patterns.var(dim=1, unbiased=True)
            loss = (variance[:, 0]-variance[:, 1]).square().mean()
            self.assertLess(abs(float(loss)/(4/(count-1))-1), .10)


if __name__ == "__main__":
    unittest.main()
