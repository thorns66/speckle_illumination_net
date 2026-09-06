import unittest

import numpy as np
import torch

from tools.audit_population_probe_modes import dct_modes
from tools.scan_covariance_sketch_depth import _empirical_action


class PopulationProbeTests(unittest.TestCase):
    def test_basis_has_dc_and_is_orthonormal(self):
        q, frequencies = dct_modes((12, 16), 65)
        self.assertEqual(frequencies[0], (0, 0))
        np.testing.assert_allclose(q[0], 1 / np.sqrt(192), rtol=1e-6)
        np.testing.assert_allclose(q.reshape(65, -1).astype(np.float64) @ q.reshape(65, -1).astype(np.float64).T, np.eye(65), atol=1e-6)
        np.testing.assert_allclose(q[1:].mean((1, 2)), 0, atol=1e-7)

    def test_thin_grid_and_invalid_count(self):
        q, _ = dct_modes((1, 80), 65)
        self.assertEqual(q.shape, (65, 1, 80))
        with self.assertRaises(ValueError):
            dct_modes((2, 2), 5)

    def test_dc_is_covariance_with_bucket_not_mean(self):
        rng = np.random.default_rng(20260904)
        frames = rng.normal(size=(50, 8, 10)).astype(np.float32)
        q, _ = dct_modes((8, 10), 1)
        observed = _empirical_action(frames, q)[0]
        centered = frames - frames.mean(0)
        bucket = centered.sum((1, 2)) / np.sqrt(80)
        expected = np.einsum("nhw,n->hw", centered, bucket) / 49
        np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-7)
        offset = rng.normal(size=(8, 10)).astype(np.float32)
        np.testing.assert_allclose(_empirical_action(frames + offset, q)[0], observed, rtol=1e-5, atol=1e-6)

    def test_probe_chunk_gradient_matches_full(self):
        torch.manual_seed(2)
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        probes = torch.randn(32, 8, dtype=x.dtype)
        targets = torch.randn(32, dtype=x.dtype)
        full = torch.autograd.grad(((probes @ x - targets)**2).mean(), x)[0]
        accumulated = torch.zeros_like(x)
        for start in range(0, 32, 4):
            loss = ((probes[start:start+4] @ x - targets[start:start+4])**2).mean()
            accumulated += torch.autograd.grad(loss * (4 / 32), x)[0]
        torch.testing.assert_close(full, accumulated)


if __name__ == "__main__":
    unittest.main()
