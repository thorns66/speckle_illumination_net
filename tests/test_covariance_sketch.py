from __future__ import annotations

import unittest

import torch

from physics.covariance_sketch import (
    empirical_covariance_action,
    stationary_covariance_action,
    theoretical_covariance_action,
)
from physics.lfm_operator import LFMOperator
from tests.operator_fixtures import toy_psf


class CovarianceSketchTest(unittest.TestCase):
    def test_empirical_action_matches_explicit_covariance(self):
        generator = torch.Generator().manual_seed(20260901)
        frames = torch.randn((7, 4, 5), generator=generator, dtype=torch.float64)
        probes = torch.randn((3, 4, 5), generator=generator, dtype=torch.float64)
        centered = frames.reshape(7, -1)
        centered = centered - centered.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / 6
        expected = (covariance @ probes.reshape(3, -1).T).T.reshape_as(probes)
        actual = empirical_covariance_action(frames, probes)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_stationary_action_is_symmetric_and_psd(self):
        kernel = torch.tensor(
            [[0.05, 0.15, 0.05], [0.15, 1.0, 0.15], [0.05, 0.15, 0.05]],
            dtype=torch.float64,
        )
        generator = torch.Generator().manual_seed(20260901)
        x = torch.randn((1, 1, 2, 5, 6), generator=generator, dtype=torch.float64)
        y = torch.randn((1, 1, 2, 5, 6), generator=generator, dtype=torch.float64)
        cx = stationary_covariance_action(x, kernel)
        cy = stationary_covariance_action(y, kernel)
        torch.testing.assert_close((x * cy).sum(), (cx * y).sum(), rtol=1e-11, atol=1e-11)
        self.assertGreaterEqual(float((x * cx).sum().item()), -1e-10)

    def test_operator_chain_matches_explicit_matrix_and_gradient(self):
        dtype = torch.float64
        operator = LFMOperator(toy_psf(z=2, period=2, kh=3, kw=3, dtype=dtype), mode="reference")
        height, width = 4, 5
        voxel_count = 2 * height * width
        sensor_count = height * width

        volume_basis = torch.eye(voxel_count, dtype=dtype).reshape(
            voxel_count, 1, 2, height, width
        )
        h_matrix = operator(volume_basis)[:, 0].reshape(voxel_count, sensor_count).T
        cs_basis = stationary_covariance_action(volume_basis, torch.tensor(
            [[0.1, 0.2, 0.1], [0.2, 1.0, 0.2], [0.1, 0.2, 0.1]], dtype=dtype
        ))
        cs_matrix = cs_basis.reshape(voxel_count, voxel_count).T

        generator = torch.Generator().manual_seed(20260902)
        g = torch.rand((1, 1, 2, height, width), generator=generator, dtype=dtype, requires_grad=True)
        probes = torch.randn((3, height, width), generator=generator, dtype=dtype)
        actual = theoretical_covariance_action(
            g, probes, operator, torch.tensor(
                [[0.1, 0.2, 0.1], [0.2, 1.0, 0.2], [0.1, 0.2, 0.1]], dtype=dtype
            )
        )
        diagonal_g = torch.diag(g.reshape(-1))
        covariance = h_matrix @ diagonal_g @ cs_matrix @ diagonal_g @ h_matrix.T
        expected = (covariance @ probes.reshape(3, -1).T).T.reshape_as(probes)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

        actual.square().sum().backward()
        actual_gradient = g.grad.detach().clone()
        explicit_g = g.detach().clone().requires_grad_(True)
        diagonal_explicit = torch.diag(explicit_g.reshape(-1))
        explicit_covariance = h_matrix @ diagonal_explicit @ cs_matrix @ diagonal_explicit @ h_matrix.T
        explicit = (explicit_covariance @ probes.reshape(3, -1).T).T
        explicit.square().sum().backward()
        torch.testing.assert_close(actual_gradient, explicit_g.grad, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
