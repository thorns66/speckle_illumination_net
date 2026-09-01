import unittest

from tests.model_fixtures import tiny_inputs, tiny_model


class NetworkShapeTest(unittest.TestCase):
    def test_arbitrary_rectangular_size_is_cropped_back(self):
        model = tiny_model()
        inputs = tiny_inputs(height=17, width=19)
        output = model(*inputs)
        self.assertEqual(output.reconstruction.shape, (1, 1, 10, 17, 19))
        self.assertEqual(output.residual.shape, output.reconstruction.shape)
        self.assertEqual(len(output.gates), 3)

if __name__ == "__main__":
    unittest.main()
