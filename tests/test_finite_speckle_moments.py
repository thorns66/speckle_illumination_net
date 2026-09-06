from __future__ import annotations
import unittest
import torch
import torch.nn.functional as F
from physics.speckle_oracle import SpeckleGeneratorConfig, circular_pupil
from tools.audit_finite_speckle_stationarity import exact_moments


class FiniteSpeckleMomentsTest(unittest.TestCase):
    def test_four_phase_population_matches_exact_second_and_fourth_moments(self):
        config=SpeckleGeneratorConfig(sampling=2,numerical_aperture=0.2)
        phases=torch.cartesian_prod(*[torch.arange(4,dtype=torch.float64)]*4).reshape(-1,2,2)*(torch.pi/2)
        source=F.pad(torch.polar(torch.ones_like(phases),phases),(1,1,1,1))
        pupil=torch.fft.ifftshift(circular_pupil(config,device="cpu",dtype=torch.float64))
        field=torch.fft.ifft2(torch.fft.fft2(source)*pupil)
        intensity=field.abs().square()[:,1:3,1:3]
        mean,var,_=exact_moments(config)
        torch.testing.assert_close(intensity.mean(0),mean,rtol=1e-12,atol=1e-12)
        centered=intensity-intensity.mean(0)
        torch.testing.assert_close(centered.square().mean(0),var,rtol=1e-12,atol=1e-12)
        _,lag,_=exact_moments(config,(1,0))
        torch.testing.assert_close((centered[:,:-1]*centered[:,1:]).mean(0),lag[:-1],rtol=1e-12,atol=1e-12)


if __name__=="__main__":
    unittest.main()
