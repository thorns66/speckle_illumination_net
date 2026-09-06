import unittest

import torch

from losses.self_supervised_losses import (
    compute_multiband_reliability,
    multiband_log_variance_loss,
)
from models.variance_anchored_lfm_net import bounded_log_multiplicative_map
from tests.model_fixtures import tiny_inputs, tiny_model


class MultibandVarianceLossTest(unittest.TestCase):
    def test_identical_splits_are_fully_reliable(self):
        generator = torch.Generator().manual_seed(20260901)
        value = torch.rand((1, 1, 17, 19), generator=generator) + 0.1
        weights, correlations = compute_multiband_reliability(value, value, levels=2)
        torch.testing.assert_close(correlations, torch.ones_like(correlations), atol=1e-6, rtol=0)
        torch.testing.assert_close(weights, torch.ones_like(weights), atol=1e-6, rtol=0)

    def test_unreliable_anticorrelated_bands_are_rejected(self):
        yy, xx = torch.meshgrid(torch.arange(16), torch.arange(18), indexing="ij")
        pattern = ((xx + yy) % 2).float()[None, None]
        first = torch.exp(pattern)
        second = torch.exp(1.0 - pattern)
        weights, correlations = compute_multiband_reliability(
            first, second, levels=2, threshold=0.5
        )
        self.assertTrue(bool((correlations < 0.5).any()))
        self.assertTrue(bool((weights[correlations < 0.5] == 0).all()))

    def test_band_loss_is_differentiable(self):
        generator = torch.Generator().manual_seed(20260901)
        target = torch.rand((1, 1, 13, 15), generator=generator) + 0.1
        prediction = (target * 0.8 + 0.05 * torch.rand(target.shape, generator=generator)).requires_grad_(True)
        weights = torch.tensor([1.0, 0.7])
        loss = multiband_log_variance_loss(prediction, target, weights, levels=2)
        loss.backward()
        self.assertGreater(float(loss), 0.0)
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)


class RoleSeparatedDetailTest(unittest.TestCase):
    def test_common_parameter_initialization_is_preserved(self):
        torch.manual_seed(20260901)
        baseline = tiny_model(use_role_separated_detail=False)
        torch.manual_seed(20260901)
        candidate = tiny_model(use_role_separated_detail=True)
        baseline_parameters = dict(baseline.named_parameters())
        candidate_parameters = dict(candidate.named_parameters())
        for name, parameter in baseline_parameters.items():
            torch.testing.assert_close(
                parameter,
                candidate_parameters[name],
                atol=0.0,
                rtol=0.0,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_detail_output_and_ramp_are_finite(self):
        torch.manual_seed(20260901)
        model = tiny_model(use_role_separated_detail=True, detail_lowpass_passes=2)
        inputs = tiny_inputs(frames=4)
        output_off = model(*inputs, detail_strength=0.0)
        output_on = model(*inputs, detail_strength=1.0)
        self.assertIsNotNone(output_on.detail_residual)
        self.assertTrue(torch.isfinite(output_off.reconstruction).all())
        self.assertTrue(torch.isfinite(output_on.reconstruction).all())
        self.assertEqual(output_off.detail_strength, 0.0)
        self.assertEqual(output_on.detail_strength, 1.0)

    def test_mass_preserving_detail_keeps_every_depth_mass(self):
        torch.manual_seed(20260901)
        model = tiny_model(
            use_role_separated_detail=True,
            detail_lowpass_auxiliary_features=False,
            detail_application="mass_preserving_multiplicative",
            detail_log_modulation_bound=0.2,
        )
        assert model.detail_head is not None
        with torch.no_grad():
            model.detail_head.head.weight.normal_(mean=0.0, std=0.1)
        inputs = tiny_inputs(frames=4)
        output_off = model(*inputs, detail_strength=0.0)
        output_on = model(*inputs, detail_strength=1.0)
        mass_off = output_off.reconstruction.sum(dim=(-2, -1))
        mass_on = output_on.reconstruction.sum(dim=(-2, -1))
        torch.testing.assert_close(mass_on, mass_off, atol=2e-5, rtol=2e-6)
        self.assertGreater(
            float((output_on.reconstruction - output_off.reconstruction).abs().sum()),
            0.0,
        )

    def test_network_context_is_cropped_and_differentiable(self):
        torch.manual_seed(20260901)
        model = tiny_model(
            use_role_separated_detail=True,
            detail_lowpass_auxiliary_features=False,
            detail_application="mass_preserving_multiplicative",
            network_context_pad_xy=4,
            network_context_pad_mode="reflect",
        )
        inputs = tiny_inputs(frames=4, height=17, width=19)
        output = model(*inputs, detail_strength=1.0)
        self.assertEqual(output.reconstruction.shape, inputs[0].shape)
        self.assertEqual(output.residual.shape, inputs[0].shape)
        self.assertEqual(output.gates[0].shape[-2:], (17, 19))
        self.assertEqual(output.gates[1].shape[-2:], (9, 10))
        self.assertEqual(output.gates[2].shape[-2:], (5, 5))
        output.reconstruction.mean().backward()
        gradients = [value.grad for value in model.parameters() if value.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_detail_strength_validation(self):
        model = tiny_model(use_role_separated_detail=True)
        with self.assertRaisesRegex(ValueError, "detail_strength"):
            model(*tiny_inputs(), detail_strength=1.1)


class BoundedLogMultiplicativeTest(unittest.TestCase):
    def test_zero_residual_reproduces_positive_anchor(self):
        anchor = torch.tensor([[[[[0.0, 0.25, 2.0]]]]])
        reconstruction = bounded_log_multiplicative_map(
            anchor,
            torch.zeros_like(anchor),
            bound=2.0,
            eps=1e-6,
        )
        torch.testing.assert_close(
            reconstruction,
            anchor.clamp_min(1e-6),
            atol=0.0,
            rtol=0.0,
        )

    def test_log_correction_is_bounded_and_differentiable(self):
        anchor = torch.ones((1, 1, 2, 3, 4))
        residual = torch.linspace(-100.0, 100.0, anchor.numel()).reshape_as(anchor)
        residual.requires_grad_(True)
        reconstruction = bounded_log_multiplicative_map(anchor, residual, bound=2.0)
        log_ratio = torch.log(reconstruction / anchor)
        self.assertLessEqual(float(log_ratio.max()), 2.0 + 1e-6)
        self.assertGreaterEqual(float(log_ratio.min()), -2.0 - 1e-6)
        reconstruction.mean().backward()
        self.assertIsNotNone(residual.grad)
        self.assertTrue(torch.isfinite(residual.grad).all())


if __name__ == "__main__":
    unittest.main()
