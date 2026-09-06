import unittest

import numpy as np
import torch

from tools.localized_covariance import localized_probe_bank
from tools.scan_covariance_sketch_depth import _lowpass_probes


class LocalizedCovarianceTest(unittest.TestCase):
    def test_global_control_is_identical(self):
        q, windows = localized_probe_bank(5, (12, 14), seed=3, window_sigma=0)
        np.testing.assert_array_equal(q, _lowpass_probes(5, (12,14), sigma=0, seed=3))
        np.testing.assert_array_equal(windows, np.ones_like(q))

    def test_reproducible_unit_norm_zero_dc_and_coverage(self):
        q, w = localized_probe_bank(16, (32,32), seed=4, window_sigma=5)
        q2, w2 = localized_probe_bank(16, (32,32), seed=4, window_sigma=5)
        np.testing.assert_array_equal(q, q2); np.testing.assert_array_equal(w,w2)
        np.testing.assert_allclose(np.linalg.norm(q.reshape(16,-1),axis=1),1,atol=1e-6)
        np.testing.assert_allclose(q.sum(axis=(1,2)),0,atol=3e-6)
        self.assertGreater(float(w.sum(axis=0).min()),.5)
        self.assertTrue(np.isfinite(q).all())

    def test_explicit_covariance_value_gradient_and_psd(self):
        rng = torch.Generator().manual_seed(5)
        h = torch.randn((20,16), generator=rng, dtype=torch.float64)
        samples = torch.randn((9,16), generator=rng, dtype=torch.float64)
        samples -= samples.mean(0)
        cs = samples.T@samples/(len(samples)-1)
        g = torch.rand(16, generator=rng, dtype=torch.float64, requires_grad=True)
        w = torch.rand(20, generator=rng, dtype=torch.float64)
        r = torch.randn(20, generator=rng, dtype=torch.float64)
        q = w*r
        cy = (h*g)@cs@(h*g).T
        explicit = w*(cy@q)
        composed = w*(h@(g*(cs@(g*(h.T@q)))))
        torch.testing.assert_close(composed, explicit, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(composed.square().sum(),g,retain_graph=True)[0],
                                   torch.autograd.grad(explicit.square().sum(),g)[0],rtol=1e-12,atol=1e-12)
        self.assertGreater(float(torch.linalg.eigvalsh(w[:,None]*cy.detach()*w[None]).min()),-1e-10)
        frames = (samples*g.detach())@h.T
        empirical = w*((frames@q)@frames/(len(frames)-1))
        torch.testing.assert_close(empirical, composed.detach(), rtol=1e-12,atol=1e-12)

    def test_invalid_bank_rejected(self):
        with self.assertRaises(ValueError):
            localized_probe_bank(5, (12,12), seed=3, window_sigma=2)


if __name__ == "__main__":
    unittest.main()
