import unittest

import torch

from models.blocks import blur_lateral_2d, blur_lateral_3d
from tests.model_fixtures import tiny_inputs, tiny_model


class AntiAliasTest(unittest.TestCase):
    def test_binomial_blur_suppresses_checkerboard(self):
        yy, xx = torch.meshgrid(torch.arange(16), torch.arange(18), indexing="ij")
        checkerboard = ((xx + yy) % 2).float() * 2.0 - 1.0
        image = checkerboard[None, None]
        volume = checkerboard[None, None, None].repeat(1, 2, 3, 1, 1)
        self.assertLess(float(blur_lateral_2d(image).square().mean()), 0.02)
        self.assertLess(float(blur_lateral_3d(volume).square().mean()), 0.02)

    def test_anti_aliased_model_preserves_output_shapes_and_gradients(self):
        torch.manual_seed(20260901)
        model = tiny_model(
            anti_alias_downsampling=True,
            network_context_pad_xy=4,
            network_context_pad_mode="replicate",
        )
        inputs = tiny_inputs(frames=4, height=17, width=19)
        output = model(*inputs)
        self.assertEqual(output.reconstruction.shape, inputs[0].shape)
        output.reconstruction.mean().backward()
        gradients = [value.grad for value in model.parameters() if value.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))


if __name__ == "__main__":
    unittest.main()
