import unittest

import torch

from models.set_transformer_encoder import SetTransformerEncoder
from tests.model_fixtures import tiny_inputs, tiny_model


class SetTransformerTest(unittest.TestCase):
    def test_common_module_initialization_matches_mean_std_baseline(self):
        torch.manual_seed(20260901)
        baseline = tiny_model(set_encoder_type="mean_std")
        torch.manual_seed(20260901)
        candidate = tiny_model(
            set_encoder_type="set_transformer",
            set_transformer_heads=4,
            set_transformer_inducing_points=4,
            set_transformer_layers=2,
        )
        baseline_parameters = dict(baseline.named_parameters())
        candidate_parameters = dict(candidate.named_parameters())
        common_names = [
            name
            for name in baseline_parameters
            if not name.startswith("set_encoder.")
        ]
        self.assertTrue(common_names)
        for name in common_names:
            torch.testing.assert_close(
                baseline_parameters[name],
                candidate_parameters[name],
                rtol=0.0,
                atol=0.0,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_frame_permutation_preserves_features_gates_and_reconstruction(self):
        torch.manual_seed(20260901)
        model = tiny_model(
            set_encoder_type="set_transformer",
            set_transformer_heads=4,
            set_transformer_inducing_points=4,
            set_transformer_layers=2,
            set_use_checkpoint=False,
        ).eval()
        f_var, g_mean, residual, z_values = tiny_inputs(frames=6)
        permutation = torch.tensor([5, 2, 0, 4, 1, 3])
        with torch.no_grad():
            first = model(f_var, g_mean, residual, z_values)
            second = model(f_var, g_mean, residual[:, permutation], z_values)
        assert first.set_features is not None and second.set_features is not None
        for a, b in zip(first.set_features, second.set_features):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        for a, b in zip(first.gates, second.gates):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(
            first.reconstruction, second.reconstruction, rtol=3e-5, atol=3e-6
        )

    def test_attention_parameters_receive_finite_nonzero_gradients(self):
        torch.manual_seed(20260901)
        model = tiny_model(
            set_encoder_type="set_transformer",
            set_transformer_heads=4,
            set_transformer_inducing_points=4,
            set_transformer_layers=2,
            set_use_checkpoint=False,
        ).train()
        output = model(*tiny_inputs(frames=5))
        assert output.set_features is not None
        # The reconstruction head is intentionally zero-initialized, so the
        # first full-model step updates only that head. Test the Set branch
        # directly here; end-to-end upstream gradients become nonzero next step.
        loss = sum(feature.square().mean() for feature in output.set_features)
        loss.backward()
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith("set_encoder.transformers") and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_head_count_must_divide_every_scale(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            SetTransformerEncoder((6, 10, 14), heads=4)


if __name__ == "__main__":
    unittest.main()
