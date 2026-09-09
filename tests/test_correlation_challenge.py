import unittest
import numpy as np
from tools.correlation_selection import correlation_matrix, local_search, score, select_extremes


class CorrelationSelectionTests(unittest.TestCase):
    def test_affine_invariance_and_degenerate(self):
        frames = np.random.default_rng(1).normal(size=(10, 20, 20))
        corr = correlation_matrix(frames)
        np.testing.assert_allclose(corr, correlation_matrix(frames * 3 + 10), atol=1e-14)
        np.testing.assert_allclose(corr, corr.T)
        np.testing.assert_allclose(np.diag(corr), 1)
        with self.assertRaises(ValueError):
            correlation_matrix(np.ones((3, 5, 5)))

    def test_search_reproducible_and_local_optimum(self):
        corr = correlation_matrix(np.random.default_rng(9).normal(size=(100, 30)))
        a = select_extremes(corr, count=30, starts=3)
        self.assertEqual(a, select_extremes(corr, count=30, starts=3))
        self.assertLessEqual(a['low']['score'], min(a['random_scores']))
        self.assertGreaterEqual(a['high']['score'], max(a['random_scores']))
        for group in ('low', 'high'):
            selected = np.array(a[group]['input_indices']) - 1
            complement = a[group]['holdout_indices']
            self.assertEqual(len(set(selected)), 10)
            self.assertEqual(sorted(a[group]['input_indices'] + complement), list(range(1,101)))
            new, value = local_search(corr, selected, group == 'high')
            np.testing.assert_array_equal(new, selected)
            self.assertAlmostEqual(value, score(corr, selected))

    def test_absolute_correlation(self):
        frames = np.array([[1., 2., 3.], [3., 2., 1.]])
        self.assertAlmostEqual(score(correlation_matrix(frames), [0, 1]), 1.)


if __name__ == '__main__':
    unittest.main()
