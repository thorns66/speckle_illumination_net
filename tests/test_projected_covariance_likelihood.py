from __future__ import annotations

import unittest
import numpy as np
import torch

from tools.diagnose_projected_covariance_likelihood import (
    empirical_projected_covariance,
    gaussian_covariance_divergence,
    projected_model_covariance,
)
from tools.diagnose_cs_information import CachedCs


class ProjectedCovarianceLikelihoodTest(unittest.TestCase):
    def test_empirical_covariance(self):
        rng = np.random.default_rng(2)
        frames = rng.normal(size=(9, 3, 4)).astype(np.float32)
        basis = np.eye(12, dtype=np.float32)[:4].reshape(4, 3, 4)
        got = empirical_projected_covariance(frames, basis)
        expected = np.cov(frames.reshape(9, 12)[:, :4], rowvar=False, ddof=1)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_divergence_zero_at_match_and_gradient(self):
        matrix = torch.tensor([[2.0, 0.3], [0.3, 0.8]], dtype=torch.float64)
        matrix.requires_grad_()
        loss = gaussian_covariance_divergence(matrix, matrix.detach(), 1e-3)
        self.assertAlmostEqual(float(loss), 0.0, places=12)
        torch.testing.assert_close(torch.autograd.grad(loss, matrix)[0], torch.zeros_like(matrix), atol=1e-12, rtol=0)

    def test_projected_model_covariance_explicit_and_gradient(self):
        dtype = torch.float64
        cs = CachedCs(torch.tensor([[0.2, 0.3, 0.2], [0.3, 1.0, 0.3], [0.2, 0.3, 0.2]], dtype=dtype), (3, 4))
        rng = torch.Generator().manual_seed(4)
        shape = torch.rand((3, 4), generator=rng, dtype=dtype, requires_grad=True)
        back = torch.randn((5, 1, 1, 3, 4), generator=rng, dtype=dtype)
        actual = projected_model_covariance(shape, back, cs)
        eye = torch.eye(12, dtype=dtype).reshape(12, 3, 4)
        covariance = cs.action(eye).flatten(1).T
        b = back[:, 0, 0].flatten(1).T
        diagonal = torch.diag(shape.flatten())
        expected = b.T @ diagonal @ covariance @ diagonal @ b
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), shape, retain_graph=True)[0],
                                   torch.autograd.grad(expected.square().sum(), shape)[0], rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
