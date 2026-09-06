from __future__ import annotations
import unittest
import torch
from tools.audit_covariance_bures import bures_covariance_distance


class BuresTest(unittest.TestCase):
    def test_diagonal_reduces_to_standard_deviation_distance(self):
        p = torch.tensor([1.,4.,9.],dtype=torch.float64)
        t = torch.tensor([4.,9.,16.],dtype=torch.float64)
        actual = bures_covariance_distance(torch.diag(p),torch.diag(t),0.)
        expected = ((p.sqrt()-t.sqrt())**2).sum()/t.sum()
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)

    def test_match_repeated_eigenvalues_has_zero_gradient(self):
        p = torch.eye(4,dtype=torch.float64,requires_grad=True)
        value = bures_covariance_distance(p,p.detach(),1e-6)
        self.assertLess(abs(float(value)),1e-12)
        torch.testing.assert_close(torch.autograd.grad(value,p)[0],torch.zeros_like(p),rtol=0,atol=1e-10)

    def test_common_scale_invariance_but_not_free_model_gain(self):
        t = torch.tensor([[2.,.2],[.2,1.]],dtype=torch.float64)
        p = t*4
        original = bures_covariance_distance(p,t,0.)
        self.assertAlmostEqual(float(original),1.,places=10)
        torch.testing.assert_close(original,bures_covariance_distance(p*37,t*37,0.))
        self.assertGreater(float(original),float(bures_covariance_distance(t,t,0.)))

    def test_orthogonal_invariance_and_noncommuting_gradient(self):
        torch.manual_seed(61)
        a = torch.randn(4,4,dtype=torch.float64)
        b = torch.randn(4,4,dtype=torch.float64)
        p = (a@a.T+torch.eye(4)).requires_grad_()
        t = b@b.T+torch.eye(4)
        q,_ = torch.linalg.qr(torch.randn(4,4,dtype=torch.float64))
        torch.testing.assert_close(bures_covariance_distance(p,t),
                                  bures_covariance_distance(q@p@q.T,q@t@q.T),rtol=1e-10,atol=1e-10)
        self.assertTrue(torch.autograd.gradcheck(lambda x:bures_covariance_distance(x,t),p,eps=1e-6,atol=1e-5,rtol=1e-4))

    def test_invalid_input_rejected_and_target_detached(self):
        p = torch.eye(3,dtype=torch.float64,requires_grad=True)
        t = (2*torch.eye(3,dtype=torch.float64)).requires_grad_()
        value = bures_covariance_distance(p,t)
        value.backward()
        self.assertIsNone(t.grad)
        with self.assertRaises(ValueError):
            bures_covariance_distance(-p,t)
        with self.assertRaises(ValueError):
            bures_covariance_distance(p,torch.zeros_like(t))


if __name__=="__main__":
    unittest.main()
