import unittest

import torch
import torch.nn.functional as F

from physics.finite_phase_cs import DenseFinitePhaseCs, dense_phase_covariance, finite_phase_transfer
from physics.speckle_oracle import SpeckleGeneratorConfig, circular_pupil
from tools.audit_finite_speckle_stationarity import exact_moments


class DenseFinitePhaseCsTest(unittest.TestCase):
    def test_entire_covariance_matches_enumerated_phase_population(self):
        config = SpeckleGeneratorConfig(sampling=2, numerical_aperture=.2)
        angles = torch.cartesian_prod(*[torch.arange(4,dtype=torch.float64)]*4)*(torch.pi/2)
        source = torch.polar(torch.ones_like(angles), angles)
        b = finite_phase_transfer(config)
        intensity = (source@b.T.to(torch.complex128)).abs().square()
        centered = intensity-intensity.mean(0)
        expected = centered.T@centered/len(centered)
        cs, mean = dense_phase_covariance(b)
        torch.testing.assert_close(cs, expected, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(mean, intensity.mean(0), rtol=1e-12, atol=1e-14)
        self.assertGreater(float(torch.linalg.eigvalsh(cs).min()),-1e-13)

    def test_transfer_matches_original_fft_and_lag_formulas(self):
        config = SpeckleGeneratorConfig(sampling=8)
        b = finite_phase_transfer(config)
        rng = torch.Generator().manual_seed(99)
        phase = torch.rand((3,8,8),generator=rng,dtype=torch.float64)*(2*torch.pi)
        source = torch.polar(torch.ones_like(phase),phase)
        pupil = torch.fft.ifftshift(circular_pupil(config,device="cpu",dtype=torch.float64))
        expected = torch.fft.ifft2(torch.fft.fft2(F.pad(source,(4,4,4,4)))*pupil)[:,4:12,4:12]
        actual = (source.flatten(1)@b.T.to(torch.complex128)).reshape_as(expected)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-13)
        cs, mean = dense_phase_covariance(b)
        refmean, refvar, _ = exact_moments(config)
        torch.testing.assert_close(mean.reshape(8,8),refmean,rtol=1e-12,atol=1e-14)
        torch.testing.assert_close(cs.diag().reshape(8,8),refvar,rtol=1e-12,atol=1e-14)
        for dy,dx in ((1,0),(0,2),(3,3)):
            _, ref, _ = exact_moments(config,(dy,dx))
            indices = torch.arange(64).reshape(8,8)[:8-dy,:8-dx].flatten()
            values = cs[indices,indices+dy*8+dx].reshape(8-dy,8-dx)
            torch.testing.assert_close(values,ref[:8-dy,:8-dx],rtol=1e-11,atol=1e-14)

    def test_memory_saving_build_and_action_gradient(self):
        b = finite_phase_transfer(SpeckleGeneratorConfig(sampling=4,numerical_aperture=.1))
        expected, emean = dense_phase_covariance(b)
        changed = b.clone()
        actual, mean = dense_phase_covariance(changed,row_chunk=3,consume_transfer=True,source_chunk=3)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-14)
        torch.testing.assert_close(mean,emean)
        torch.testing.assert_close(changed,b.square())
        value = torch.arange(32,dtype=torch.float64).reshape(2,4,4).requires_grad_()
        cs = DenseFinitePhaseCs(actual,(4,4),system_mean=.2)
        prediction = cs.action(value)
        reference = (value.flatten(1)@expected.T/.2**2).reshape_as(value)
        torch.testing.assert_close(prediction,reference)
        torch.testing.assert_close(torch.autograd.grad(prediction.square().sum(),value,retain_graph=True)[0],
                                   torch.autograd.grad(reference.square().sum(),value)[0])


    def test_full_lfm_chain_checkpoint_and_chunked_gradient(self):
        from physics.lfm_operator import LFMOperator
        from tools.diagnose_cs_information import model_action
        torch.manual_seed(72)
        side = 4
        transfer = finite_phase_transfer(SpeckleGeneratorConfig(sampling=side, numerical_aperture=.1))
        covariance, _ = dense_phase_covariance(transfer)
        cs = DenseFinitePhaseCs(covariance, (side, side), system_mean=.2)
        operator = LFMOperator(torch.rand((1, 2, 2, 3, 3), dtype=torch.float64),
                               phase_chunk_size=2)
        probes = torch.randn((7, side, side), dtype=torch.float64)
        back = operator.adjoint(probes[:, None]).detach()
        shape = torch.rand((side, side), dtype=torch.float64, requires_grad=True)
        basis = torch.eye(side * side, dtype=torch.float64).reshape(-1, 1, 1, side, side)
        h = operator(basis).flatten(1).T
        theoretical = h @ torch.diag(shape.flatten()) @ (covariance/.2**2) @ torch.diag(shape.flatten()) @ h.T
        reference = (probes.flatten(1) @ theoretical.T).reshape_as(probes)
        expected_gradient = torch.autograd.grad(reference.square().mean(), shape)[0]
        for checkpoint in (False, True):
            result = model_action(shape, back, cs, operator, checkpoint_enabled=checkpoint)
            torch.testing.assert_close(result, reference, rtol=1e-10, atol=1e-12)
            actual_gradient = torch.autograd.grad(result.square().mean(), shape)[0]
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-12)
        accumulated = torch.zeros_like(shape)
        for start in range(0, len(probes), 4):
            result = model_action(shape, back[start:start+4], cs, operator)
            loss = result.square().mean() * len(result) / len(probes)
            accumulated += torch.autograd.grad(loss, shape)[0]
        torch.testing.assert_close(accumulated, expected_gradient, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
