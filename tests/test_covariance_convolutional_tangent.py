import unittest

import numpy as np
import torch

from tools.audit_covariance_convolutional_tangent import (
    DiagnosticReadoutNet, extract_features, feature_geometry, geometry_report, shape_tangent,
    make_network_input, network_tangent_components,
)


class ConvolutionalTangentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_zero_readout_chain_rule_and_inactive_backbone(self):
        torch.manual_seed(13)
        net = DiagnosticReadoutNet((4, 8, 12)).double()
        value = torch.randn(1, 1, 12, 12, dtype=torch.double)
        gradient = torch.randn(12, 12, dtype=torch.double)
        gradient -= gradient.mean()
        features = net.features(value)
        loss = (net.head(features)[0, 0] * gradient).sum()
        loss.backward()
        matrix, _, _ = feature_geometry(features[0].detach().numpy())
        expected = matrix.T @ gradient.numpy().ravel()
        np.testing.assert_allclose(net.head.weight.grad.detach().numpy().ravel(), expected, atol=1e-12)
        self.assertLess(abs(float(net.head.bias.grad)), 1e-12)
        for name, parameter in net.named_parameters():
            if not name.startswith("head."):
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(int(torch.count_nonzero(parameter.grad)), 0)
        # Head-only JVP equals the analytical centered JJ^T action.
        step = torch.tensor(expected.reshape(1, -1, 1, 1))
        update = torch.nn.functional.conv2d(features.detach(), step)[0, 0].numpy().ravel()
        update -= update.mean()
        np.testing.assert_allclose(update, matrix @ expected, rtol=1e-12, atol=1e-11)

    def test_psd_span_and_feature_scaling_are_distinct(self):
        rng = np.random.default_rng(3)
        features = rng.normal(size=(3, 5, 6))
        matrix, basis, _ = feature_geometry(features)
        gradient = rng.normal(size=30)
        self.assertGreaterEqual(gradient @ matrix @ matrix.T @ gradient, 0)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
        projected = basis @ (basis.T @ gradient)
        np.testing.assert_allclose(basis @ (basis.T @ projected), projected, atol=1e-12)
        scaled = features * np.array([1, 10, 100])[:, None, None]
        other_matrix, other_basis, _ = feature_geometry(scaled)
        np.testing.assert_allclose(other_basis @ (other_basis.T @ gradient), projected, atol=1e-12)
        self.assertFalse(np.allclose(matrix @ (matrix.T @ gradient),
                                    other_matrix @ (other_matrix.T @ gradient)))

    def test_image_tangent_matches_normalized_exp_jvp(self):
        rng = np.random.default_rng(12)
        anchor = rng.uniform(.1, 2, size=20); anchor /= anchor.sum()
        direction = rng.normal(size=20)
        a = torch.tensor(anchor, dtype=torch.double)
        raw = torch.zeros_like(a)
        def normalized(x):
            u = a * torch.exp(.5 * torch.tanh(x / .5))
            return u / u.sum()
        _, actual = torch.autograd.functional.jvp(normalized, raw, torch.tensor(direction))
        np.testing.assert_allclose(actual.numpy(), shape_tangent(anchor, direction), atol=1e-15)
        self.assertAlmostEqual(float(actual.sum()), 0, places=14)

    def test_features_repeat_without_mutating_rng(self):
        anchor = np.full((16, 16), 1/256)
        state = torch.random.get_rng_state().clone()
        first = extract_features(anchor, "noise", 51, context=4, channels=(4, 8, 12))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        second = extract_features(anchor, "noise", 51, context=4, channels=(4, 8, 12))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (4, 16, 16))

    def test_rank_zero_nonfinite_and_conflicting_gradients(self):
        rng = np.random.default_rng(1)
        features = rng.normal(size=(3, 8, 8))
        gradient = rng.normal(size=(8, 8))
        anchor = np.full((8, 8), 1/64)
        values = {"train": gradient, "holdout": -gradient, "population": gradient}
        report = geometry_report(features, values, anchor)
        self.assertFalse(report["conservative_initial_gate_pass"])
        self.assertLess(report["train_update_cross_objective_descent"]["gradient_descent"]["holdout_descent_cosine"], 0)
        empty = geometry_report(np.ones_like(features), values, anchor)
        self.assertEqual(empty["rank"], 0)
        self.assertFalse(empty["conservative_initial_gate_pass"])
        with self.assertRaises(ValueError):
            feature_geometry(features * np.nan)

    def test_full_vjp_jvp_matches_explicit_jacobian(self):
        torch.manual_seed(32)
        net = DiagnosticReadoutNet((2, 4, 4), zero_readout=False).double()
        value = torch.randn(1, 1, 8, 8, dtype=torch.double)
        rng = np.random.default_rng(82)
        gradients = {name: rng.normal(size=(4, 4)) for name in ("train", "holdout", "population")}
        before = {name: p.detach().clone() for name, p in net.named_parameters()}
        components = network_tangent_components(net, value, gradients, context=2)
        params = {name: p.detach() for name, p in net.named_parameters()}
        def output(p):
            result = torch.func.functional_call(net, p, (value,), strict=True)[0, 0, 2:-2, 2:-2]
            return result - result.mean()
        jacobian = torch.func.jacrev(output)(params)
        matrix = np.concatenate([part.detach().numpy().reshape(16, -1) for part in jacobian.values()], axis=1)
        for name, gradient in gradients.items():
            g = (gradient - gradient.mean()).ravel()
            expected_param = matrix.T @ g
            np.testing.assert_allclose(components["parameter"][name], expected_param, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(components["kernel"][name], matrix @ expected_param, rtol=1e-10, atol=1e-10)
            adam_param = expected_param / (np.abs(expected_param) + 1e-8)
            np.testing.assert_allclose(components["adam"][name], matrix @ adam_param, rtol=1e-8, atol=1e-7)
            self.assertLess(components["parameter_group_gradient_energy_fraction"][name]["head"], .99)
        self.assertEqual(components["initial_residual_max_abs"], 0)
        self.assertLess(components["duality_max_relative_error"], 1e-10)
        for name, p in net.named_parameters():
            self.assertTrue(torch.equal(p, before[name]))
            self.assertIsNone(p.grad)

    def test_full_zero_readout_reduces_to_feature_kernel(self):
        anchor = np.full((16, 16), 1/256)
        net, value = make_network_input(anchor, "anchor", 4, context=4, channels=(4, 8, 12))
        net = net.double(); value = value.double()
        features = net.features(value)[0, :, 4:-4, 4:-4].detach().numpy()
        matrix, _, _ = feature_geometry(features)
        rng = np.random.default_rng(2)
        gradients = {name: rng.normal(size=(16, 16)) for name in ("train", "holdout", "population")}
        components = network_tangent_components(net, value, gradients, context=4)
        for name, gradient in gradients.items():
            g = (gradient - gradient.mean()).ravel()
            np.testing.assert_allclose(components["kernel"][name], matrix @ (matrix.T @ g), rtol=1e-10, atol=1e-11)
            self.assertAlmostEqual(components["parameter_group_gradient_energy_fraction"][name]["head"], 1.)


if __name__ == "__main__":
    unittest.main()
