import unittest

from tests.model_fixtures import tiny_inputs, tiny_model


class NoDepthDownsampleTest(unittest.TestCase):
    def test_all_encoder_scales_keep_z(self):
        model = tiny_model()
        output = model(*tiny_inputs())
        for pyramid in (output.variance_features, output.mean_features, output.set_features):
            self.assertIsNotNone(pyramid)
            for feature in pyramid:
                self.assertEqual(feature.shape[2], 10)


if __name__ == "__main__":
    unittest.main()
