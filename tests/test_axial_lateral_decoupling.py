import unittest

import torch

from models.output_parameterizations import axial_lateral_decoupled_map
from tests.model_fixtures import tiny_inputs, tiny_model


class AxialLateralDecoupledMapTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(20260901)
        self.anchor = torch.rand((1, 1, 4, 6, 8), generator=generator) + 0.05
        self.scale = self.anchor.mean().detach()

    def test_zero_residual_has_uniform_axial_mass_and_preserves_layer_shapes(self):
        reconstruction, axial, lateral = axial_lateral_decoupled_map(
            self.anchor, torch.zeros_like(self.anchor), lateral_log_bound=0.5
        )
        expected_axial = torch.full_like(axial, 0.25)
        torch.testing.assert_close(axial, expected_axial, atol=1e-7, rtol=0)
        torch.testing.assert_close(lateral, torch.zeros_like(lateral), atol=0, rtol=0)
        torch.testing.assert_close(
            reconstruction.sum(), self.anchor.sum(), atol=2e-5, rtol=1e-6
        )
        expected_shape = self.anchor / self.anchor.sum(dim=(-2, -1), keepdim=True)
        actual_shape = reconstruction / reconstruction.sum(dim=(-2, -1), keepdim=True)
        torch.testing.assert_close(actual_shape, expected_shape, atol=2e-6, rtol=2e-6)

    def test_axial_offsets_only_change_layer_mass(self):
        offsets = torch.tensor([-1.0, 0.5, 1.5, -0.25]).view(1, 1, 4, 1, 1)
        residual = self.scale * offsets.expand_as(self.anchor)
        reconstruction, axial, lateral = axial_lateral_decoupled_map(
            self.anchor,
            residual,
            lateral_log_bound=0.5,
            axial_logit_scale=2.0,
        )
        expected_axial = torch.softmax(2.0 * offsets, dim=-3)
        torch.testing.assert_close(axial, expected_axial, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(lateral, torch.zeros_like(lateral), atol=2e-7, rtol=0)
        actual_mass = reconstruction.sum(dim=(-2, -1), keepdim=True) / reconstruction.sum()
        torch.testing.assert_close(actual_mass, axial, atol=1e-7, rtol=1e-6)

    def test_zero_mean_lateral_residual_cannot_change_axial_mass(self):
        yy, xx = torch.meshgrid(torch.arange(6), torch.arange(8), indexing="ij")
        checker = (2 * ((xx + yy) % 2) - 1).to(torch.float32)
        residual = self.scale * checker.view(1, 1, 1, 6, 8).expand_as(self.anchor)
        reconstruction, axial, lateral = axial_lateral_decoupled_map(
            self.anchor, residual, lateral_log_bound=0.5
        )
        torch.testing.assert_close(axial, torch.full_like(axial, 0.25), atol=1e-7, rtol=0)
        self.assertGreater(float(lateral.abs().sum()), 0.0)
        layer_mass = reconstruction.sum(dim=(-2, -1))
        torch.testing.assert_close(
            layer_mass, torch.full_like(layer_mass, self.anchor.sum() / 4), atol=2e-5, rtol=1e-6
        )

    def test_beta_scaling_changes_only_total_mass(self):
        generator = torch.Generator().manual_seed(3)
        residual = torch.randn(self.anchor.shape, generator=generator) * self.scale
        first = axial_lateral_decoupled_map(self.anchor, residual, lateral_log_bound=1.0)
        second = axial_lateral_decoupled_map(
            3.0 * self.anchor, 3.0 * residual, lateral_log_bound=1.0
        )
        torch.testing.assert_close(second[0], 3.0 * first[0], atol=2e-5, rtol=2e-6)
        torch.testing.assert_close(second[1], first[1], atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(second[2], first[2], atol=1e-7, rtol=1e-6)

    def test_b_zero_blocks_lateral_changes_but_keeps_axial_gradient(self):
        residual = torch.zeros_like(self.anchor, requires_grad=True)
        reconstruction, _, lateral = axial_lateral_decoupled_map(
            self.anchor, residual, lateral_log_bound=0.0
        )
        z_weight = torch.arange(1, 5, dtype=reconstruction.dtype).view(1, 1, 4, 1, 1)
        (reconstruction * z_weight).sum().backward()
        self.assertTrue(torch.isfinite(residual.grad).all())
        self.assertGreater(float(residual.grad.abs().sum()), 0.0)
        torch.testing.assert_close(lateral, torch.zeros_like(lateral), atol=0, rtol=0)
        spatial_std = residual.grad.flatten(-2).std(dim=-1)
        torch.testing.assert_close(spatial_std, torch.zeros_like(spatial_std), atol=2e-6, rtol=0)

    def test_lateral_gradient_has_zero_layer_mean(self):
        residual = torch.zeros_like(self.anchor, requires_grad=True)
        reconstruction, _, _ = axial_lateral_decoupled_map(
            self.anchor, residual, lateral_log_bound=0.5
        )
        spatial_weight = torch.linspace(0.0, 1.0, 48).reshape(1, 1, 1, 6, 8)
        normalized_shape = reconstruction / reconstruction.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-12)
        (normalized_shape * spatial_weight).sum().backward()
        self.assertTrue(torch.isfinite(residual.grad).all())
        torch.testing.assert_close(
            residual.grad.mean(dim=(-2, -1)),
            torch.zeros_like(residual.grad.mean(dim=(-2, -1))),
            atol=2e-6,
            rtol=0,
        )

    def test_zero_anchor_and_extreme_residual_are_finite(self):
        zero = torch.zeros_like(self.anchor)
        extreme = torch.linspace(-1e6, 1e6, zero.numel()).reshape_as(zero)
        reconstruction, axial, lateral = axial_lateral_decoupled_map(
            zero, extreme, lateral_log_bound=0.5, axial_logit_scale=2.0
        )
        self.assertTrue(torch.isfinite(reconstruction).all())
        self.assertTrue(torch.isfinite(axial).all())
        self.assertTrue(torch.isfinite(lateral).all())
        torch.testing.assert_close(reconstruction, torch.zeros_like(reconstruction), atol=0, rtol=0)


class AxialLateralModelIntegrationTest(unittest.TestCase):
    def test_model_exposes_diagnostics_and_backpropagates(self):
        torch.manual_seed(20260901)
        model = tiny_model(
            coarse_application="axial_lateral_decoupled",
            lateral_log_residual_bound=0.5,
            axial_logit_scale=1.0,
        )
        inputs = tiny_inputs(frames=4, z=6, height=12, width=12)
        output = model(*inputs)
        self.assertIsNotNone(output.axial_mass_fraction)
        self.assertIsNotNone(output.lateral_log_modulation)
        self.assertEqual(output.reconstruction.shape, inputs[0].shape)
        self.assertTrue(torch.isfinite(output.reconstruction).all())
        z_weight = torch.arange(1, 7, dtype=output.reconstruction.dtype).view(1, 1, 6, 1, 1)
        (output.reconstruction * z_weight).sum().backward()
        gradients = [value.grad for value in model.parameters() if value.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertTrue(any(float(value.abs().sum()) > 0.0 for value in gradients))


if __name__ == "__main__":
    unittest.main()
