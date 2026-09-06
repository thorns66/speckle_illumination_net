import itertools
import unittest

import numpy as np

from tools.audit_covariance_sampling_risk import sampling_risk


class CovarianceRiskTests(unittest.TestCase):
    def test_non_gaussian_full_enumeration(self):
        a = np.array([[1., 0.], [.4, 2.]])
        points = np.array(list(itertools.product((-1., 1.), repeat=2))) @ a.T
        covariance = points.T @ points / len(points)
        trace = np.trace(covariance)
        frobenius = np.sum(covariance**2)
        fourth = np.mean(np.sum(points**2, axis=1)**2)
        for count in (2, 3, 4):
            errors = []
            for indices in itertools.product(range(4), repeat=count):
                samples = points[list(indices)]
                samples = samples - samples.mean(0)
                estimate = samples.T @ samples / (count - 1)
                errors.append(np.sum((estimate - covariance)**2))
            self.assertAlmostEqual(np.mean(errors), sampling_risk(count, trace, frobenius, fourth), places=10)

    def test_gaussian_formula(self):
        trace, frobenius = 7., 21.
        fourth = trace**2 + 2*frobenius
        self.assertAlmostEqual(sampling_risk(100, trace, frobenius, fourth), (trace**2 + frobenius)/99)


if __name__ == "__main__":
    unittest.main()
