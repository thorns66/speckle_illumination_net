from __future__ import annotations

import unittest
import torch

from physics.lfm_operator import LFMOperator
from tests.operator_fixtures import toy_psf
from tools.diagnose_cs_information import CachedCs, model_action, observation_loss, probe_seed, vector_loss


class CsInformationTest(unittest.TestCase):
    def test_seed_domains_disjoint(self):
        banks = [{probe_seed(20260901, domain, step, chunk) for step in range(200) for chunk in range(128)}
                 for domain in ("train_object", "holdout_object", "final_object")]
        self.assertEqual(len(set.union(*banks)), sum(map(len, banks)))

    def test_ustat_value_and_gradient_match_explicit_pairs(self):
        generator = torch.Generator().manual_seed(3)
        x = torch.rand((6, 4, 5), generator=generator, dtype=torch.float64, requires_grad=True)
        target = torch.rand((4, 5), generator=generator, dtype=torch.float64)
        mask = target > 0.3
        actual, _, _ = observation_loss(x, target, mask, True)
        residual = x[:, mask] - target[mask]
        pairs = [((residual[i] * residual[j]).mean() / target[mask].square().mean())
                 for i in range(6) for j in range(6) if i != j]
        expected = torch.stack(pairs).mean()
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(actual, x, retain_graph=True)[0],
                                   torch.autograd.grad(expected, x)[0], rtol=1e-12, atol=1e-12)

    def test_vector_scale_equivariance(self):
        g = torch.Generator().manual_seed(4)
        x = torch.randn((3, 4, 5), generator=g, dtype=torch.float64)
        y = torch.randn((3, 4, 5), generator=g, dtype=torch.float64)
        base = vector_loss(x, y)
        for scale in (1e-12, 1e-6, 1e6):
            for actual, expected in zip(vector_loss(x * scale, y * scale), base):
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_cached_action_explicit_value_gradient_checkpoint(self):
        dtype = torch.float64
        height, width = 4, 5
        cs = CachedCs(torch.tensor([[0.1, 0.2, 0.1], [0.2, 1, 0.2], [0.1, 0.2, 0.1]], dtype=dtype), (height, width))
        operator = LFMOperator(toy_psf(z=1, period=2, kh=3, kw=3, dtype=dtype), mode="reference")
        basis = torch.eye(height * width, dtype=dtype).reshape(-1, height, width)
        h = operator(basis[:, None, None])[:, 0].flatten(1).T
        c = cs.action(basis).flatten(1).T
        torch.testing.assert_close(c, c.T, rtol=1e-12, atol=1e-12)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(c).min()), -1e-12)
        rng = torch.Generator().manual_seed(5)
        shape = torch.rand((height, width), generator=rng, dtype=dtype, requires_grad=True)
        q = torch.randn((3, height, width), generator=rng, dtype=dtype)
        back = operator.adjoint(q[:, None]).detach()
        actual = model_action(shape, back, cs, operator)
        diagonal = torch.diag(shape.flatten())
        expected = (h @ diagonal @ c @ diagonal @ h.T @ q.flatten(1).T).T.reshape_as(q)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        grad = torch.autograd.grad(actual.square().sum(), shape, retain_graph=True)[0]
        torch.testing.assert_close(grad, torch.autograd.grad(expected.square().sum(), shape)[0], rtol=1e-9, atol=1e-9)
        other = model_action(shape, back, cs, operator, checkpoint_enabled=False)
        torch.testing.assert_close(actual, other, rtol=0, atol=0)
        torch.testing.assert_close(grad, torch.autograd.grad(other.square().sum(), shape)[0], rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
