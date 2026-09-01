import unittest

from tests.model_fixtures import tiny_inputs, tiny_model


class GateTest(unittest.TestCase):
    def test_gate_alpha_ranges_and_shapes(self):
        model = tiny_model(alpha_init=0.05)
        output = model(*tiny_inputs())
        self.assertTrue(((output.alphas >= 0) & (output.alphas <= 1)).all())
        self.assertEqual(output.alphas.shape, (3,))
        for gate, feature in zip(output.gates, output.variance_features):
            self.assertEqual(gate.shape[0], feature.shape[0])
            self.assertEqual(gate.shape[1], 1)
            self.assertEqual(gate.shape[2], feature.shape[2])
            self.assertTrue(((gate >= 0) & (gate <= 1)).all())


if __name__ == "__main__":
    unittest.main()
