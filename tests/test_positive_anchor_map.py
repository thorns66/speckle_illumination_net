import unittest

import torch

from models.variance_anchored_lfm_net import anchor_preserving_positive_map


class PositiveAnchorMapTest(unittest.TestCase):
    def test_zero_residual_returns_anchor_exactly(self):
        anchor = torch.tensor([0.0, 1e-6, 0.1, 1.0])
        output = anchor_preserving_positive_map(anchor, torch.zeros_like(anchor))
        torch.testing.assert_close(output, anchor, rtol=0.0, atol=0.0)

    def test_signed_corrections_remain_nonnegative(self):
        anchor = torch.tensor([0.0, 0.1, 1.0, 1.0])
        residual = torch.tensor([0.2, -0.1, -2.0, 0.5])
        output = anchor_preserving_positive_map(anchor, residual)
        self.assertTrue(torch.all(output >= 0))
        torch.testing.assert_close(output[[0, 3]], torch.tensor([0.2, 1.5]))
        self.assertLess(float(output[1]), float(anchor[1]))
        self.assertLess(float(output[2]), float(anchor[2]))

    def test_initial_residual_gradient_is_identity(self):
        anchor = torch.tensor([0.0, 1e-6, 0.1, 1.0])
        residual = torch.zeros_like(anchor, requires_grad=True)
        anchor_preserving_positive_map(anchor, residual).sum().backward()
        torch.testing.assert_close(residual.grad, torch.ones_like(residual))

    def test_negative_branch_is_continuous_near_zero(self):
        anchor = torch.tensor([0.1, 1.0], dtype=torch.float64)
        delta = 1e-6
        below = anchor_preserving_positive_map(anchor, torch.full_like(anchor, -delta))
        at_zero = anchor_preserving_positive_map(anchor, torch.zeros_like(anchor))
        above = anchor_preserving_positive_map(anchor, torch.full_like(anchor, delta))
        torch.testing.assert_close(at_zero - below, torch.full_like(anchor, delta), rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(above - at_zero, torch.full_like(anchor, delta), rtol=1e-4, atol=1e-8)

    def test_gradient_descent_can_add_and_suppress_signal(self):
        anchor = torch.tensor([0.0, 0.2, 0.8])
        target = torch.tensor([0.3, 0.1, 1.0])
        residual = torch.nn.Parameter(torch.zeros_like(anchor))
        optimizer = torch.optim.Adam([residual], lr=0.05)
        initial_loss = None
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            output = anchor_preserving_positive_map(anchor, residual)
            loss = (output - target).square().mean()
            if initial_loss is None:
                initial_loss = float(loss.detach())
            loss.backward()
            optimizer.step()

        output = anchor_preserving_positive_map(anchor, residual)
        self.assertIsNotNone(initial_loss)
        self.assertLess(float((output - target).square().mean()), initial_loss * 1e-3)
        self.assertTrue(torch.all(output >= 0))


if __name__ == "__main__":
    unittest.main()
