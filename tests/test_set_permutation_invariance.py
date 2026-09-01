import unittest

import torch

from tests.model_fixtures import tiny_inputs, tiny_model


class SetPermutationInvarianceTest(unittest.TestCase):
    def test_shuffle_frames_preserves_aggregate_and_gate(self):
        model = tiny_model().eval()
        f_var, g_mean, residual, z_values = tiny_inputs(frames=6)
        permutation = torch.tensor([5, 2, 0, 4, 1, 3])
        with torch.no_grad():
            first = model(f_var, g_mean, residual, z_values)
            second = model(f_var, g_mean, residual[:, permutation], z_values)
        for a, b in zip(first.set_features, second.set_features):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        for a, b in zip(first.gates, second.gates):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
