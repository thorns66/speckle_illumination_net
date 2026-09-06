from __future__ import annotations
import unittest
import torch
from physics.speckle_oracle import ensemble_covariance_action
from tools.diagnose_covariance_oracle_provenance import (
    FiniteEnsembleCs, FrozenOffsetResidual, checked_gradient_norm,
)


class OracleProvenanceTest(unittest.TestCase):
    def test_paired_probe_interval_is_reproducible_and_relative(self):
        import numpy as np
        from tools.summarize_covariance_oracle_provenance import paired_probe_comparison
        anchor = np.arange(1., 9.)
        same = paired_probe_comparison(anchor, anchor)
        self.assertEqual(same["probe_only_bootstrap_95_percent"], [0., 0.])
        half = paired_probe_comparison(anchor*.5, anchor)
        self.assertEqual(half["change_from_anchor_percent"], -50.)
        self.assertEqual(half["probe_only_bootstrap_95_percent"], [-50., -50.])
        self.assertEqual(half, paired_probe_comparison(anchor*.5, anchor))
        with self.assertRaises(ValueError):
            paired_probe_comparison([1., np.nan], [1., 2.])

    def test_cached_ensemble_action_and_gradient(self):
        rng=torch.Generator().manual_seed(12)
        patterns=torch.rand((7,4,5),dtype=torch.float64,generator=rng)
        value=torch.rand((3,4,5),dtype=torch.float64,generator=rng,requires_grad=True)
        actual=FiniteEnsembleCs(patterns).action(value)
        expected=ensemble_covariance_action(value,patterns)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(actual.square().sum(),value,retain_graph=True)[0],
                                   torch.autograd.grad(expected.square().sum(),value)[0],rtol=1e-12,atol=1e-12)

    def test_bank_offset_does_not_change_covariance(self):
        rng=torch.Generator().manual_seed(13)
        patterns=torch.rand((8,4,5),dtype=torch.float64,generator=rng)
        offset=torch.rand((4,5),dtype=torch.float64,generator=rng)
        value=torch.rand((3,4,5),dtype=torch.float64,generator=rng)
        torch.testing.assert_close(FiniteEnsembleCs(patterns).action(value),
                                   FiniteEnsembleCs(patterns+offset).action(value),rtol=1e-12,atol=1e-12)

    def test_cnn_offset_is_frozen_and_state_restores_exact_output(self):
        torch.set_num_threads(2)
        anchor = torch.rand(16, 16); anchor /= anchor.sum()
        model = FrozenOffsetResidual(anchor, 52, context=4, channels=(4, 8, 12))
        self.assertEqual(float(model().abs().max()), 0.)
        initial_offset = model.initial_output.clone()
        input_copy = model.fixed_input.clone()
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        gradient = torch.randn_like(anchor); gradient -= gradient.mean()
        (model() * gradient).sum().backward()
        self.assertGreater(checked_gradient_norm(list(model.parameters())), 0)
        with torch.no_grad():
            model.model.head.weight.add_(.02)
        self.assertGreater(float(model().abs().max()), 0.)
        torch.testing.assert_close(model.initial_output, initial_offset, rtol=0, atol=0)
        torch.testing.assert_close(model.fixed_input, input_copy, rtol=0, atol=0)
        model.load_state_dict(initial)
        self.assertEqual(float(model().abs().max()), 0.)

    def test_cnn_probe_chunk_gradient_accumulation(self):
        torch.set_num_threads(2)
        anchor = torch.rand(16, 16); anchor /= anchor.sum()
        model = FrozenOffsetResidual(anchor, 52, context=4, channels=(4, 8, 12))
        targets = torch.randn(5, 16, 16)
        ((model()[None] - targets).square().flatten(1).mean(1)).mean().backward()
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        for start in (0, 2, 4):
            selected = targets[start:start+2]
            loss = (model()[None] - selected).square().flatten(1).mean(1).mean()
            (loss * (len(selected)/len(targets))).backward()
        for p, value in zip(model.parameters(), expected):
            torch.testing.assert_close(p.grad, value, rtol=1e-4, atol=2e-6)

    def test_saved_first_adam_direction_replays_update(self):
        from tools.calibrate_cnn_initial_step import first_adam_direction
        torch.manual_seed(62)
        value = torch.nn.Parameter(torch.randn(4, 5, dtype=torch.float64))
        optimizer = torch.optim.Adam([value], lr=.001)
        (value * torch.randn_like(value)).sum().backward()
        before = value.detach().clone()
        optimizer.step()
        direction, lr = first_adam_direction(["value"], optimizer.state_dict())
        torch.testing.assert_close(value, before + lr*direction["value"], rtol=1e-13, atol=1e-14)
        optimizer.step()
        with self.assertRaises(ValueError):
            first_adam_direction(["value"], optimizer.state_dict())


if __name__=="__main__": unittest.main()
