from __future__ import annotations
import unittest
import numpy as np
import torch
from tools.audit_projected_covariance_identifiability import closest_bounded_shape
from tools.diagnose_projected_covariance_likelihood import gaussian_covariance_divergence


class CovarianceCapacityAuditTest(unittest.TestCase):
    def test_box_simplex_feasibility_and_error(self):
        anchor = np.array([[1., 2., 3.], [4., 2., 1.]]); anchor /= anchor.sum()
        truth = np.array([[0., 0., 1.], [2., 1., 0.]]); truth /= truth.sum()
        previous = float("inf")
        for bound in (0.1, 0.5, 1.0):
            result, scale = closest_bounded_shape(anchor, truth, bound)
            self.assertAlmostEqual(float(result.sum()), 1., places=12)
            self.assertTrue(np.all(result >= scale*anchor - 1e-12))
            self.assertTrue(np.all(result <= scale*np.exp(2*bound)*anchor + 1e-12))
            error = float(np.linalg.norm(result-truth))
            self.assertLessEqual(error, previous + 1e-9)
            self.assertLessEqual(error, float(np.linalg.norm(anchor-truth)) + 1e-9)
            previous = error

    def test_known_gaussian_divergence_and_common_scale(self):
        prediction = torch.tensor([[2., .2], [.2, 1.]], dtype=torch.float64)
        target = torch.tensor([[1., .1], [.1, 1.5]], dtype=torch.float64)
        scale = torch.trace(target)/2
        p = prediction/scale + 1e-3*torch.eye(2, dtype=torch.float64)
        t = target/scale + 1e-3*torch.eye(2, dtype=torch.float64)
        expected = .25*(torch.trace(torch.linalg.solve(p,t)) + torch.logdet(p)-torch.logdet(t)-2)
        actual = gaussian_covariance_divergence(prediction,target,1e-3)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        for factor in (1e-12,1e12):
            torch.testing.assert_close(gaussian_covariance_divergence(prediction*factor,target*factor,1e-3),
                                       actual,rtol=1e-12,atol=1e-12)


if __name__ == "__main__":
    unittest.main()
