import copy
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from tests.model_fixtures import tiny_inputs, tiny_model
from training.multivolume_trainer import (
    _configure_trainable_parameters,
    _optimizer,
    _resume_contract,
    _validate_config,
)
from utils.experiment_paths import next_experiment_path


ROOT = Path(__file__).resolve().parents[1]


class NoSetAblationTest(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(
            (ROOT / "configs/multivolume_n10_no_set.yaml").read_text()
        )

    def test_config_changes_only_set_and_experiment_identity(self):
        baseline = yaml.safe_load(
            (ROOT / "configs/multivolume_n10_no_mean.yaml").read_text()
        )
        candidate = copy.deepcopy(self.config)
        for value in (baseline, candidate):
            value["experiment"].pop("name")
            value["experiment"].pop("output_dir")
        self.assertTrue(baseline["ablation"].pop("use_set_branch"))
        self.assertFalse(candidate["ablation"].pop("use_set_branch"))
        self.assertEqual(baseline, candidate)

    def test_validation_accepts_both_modes_but_not_string_false(self):
        for enabled in (True, False):
            self.config["ablation"]["use_set_branch"] = enabled
            _validate_config(self.config)
        self.config["ablation"]["use_set_branch"] = "false"
        with self.assertRaises(ValueError):
            _validate_config(self.config)

    def test_resume_does_not_mix_ablation_modes(self):
        other = copy.deepcopy(self.config)
        other["ablation"]["use_set_branch"] = True
        self.assertNotEqual(_resume_contract(self.config), _resume_contract(other))

    def test_identical_initial_state_and_rng(self):
        torch.manual_seed(20260901)
        baseline = tiny_model(use_set_branch=True)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(20260901)
        candidate = tiny_model(use_set_branch=False)
        _configure_trainable_parameters(candidate)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)

    def test_freezing_optimizer_and_two_updates(self):
        model = tiny_model(use_set_branch=False)
        contract = _configure_trainable_parameters(model)
        expected = {
            name
            for name, _ in model.named_parameters()
            if name.startswith(
                ("set_encoder.", "fusion.set_projection.", "fusion.gate.")
            )
            or name == "fusion.raw_alpha"
        }
        self.assertEqual(set(contract["frozen_parameter_names"]), expected)
        optimizer = _optimizer(model, self.config)
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(
            optimized, {id(p) for p in model.parameters() if p.requires_grad}
        )
        inputs = tiny_inputs()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(*inputs)
            (output.reconstruction - inputs[0] * 0.6).square().mean().backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertIsNone(parameter.grad, name)
            optimizer.step()
        with torch.no_grad(), patch.object(
            model.set_encoder, "forward", side_effect=AssertionError("Set was called")
        ):
            output = model(*inputs)
            self.assertGreater(float(output.residual.abs().max()), 0)
            changed = list(inputs)
            changed[2] = torch.randn_like(changed[2]) * 100
            self.assertTrue(
                torch.equal(output.reconstruction, model(*changed).reconstruction)
            )

    def test_full_set_parameters_and_optimizer_are_unchanged(self):
        model = tiny_model(use_set_branch=True)
        contract = _configure_trainable_parameters(model)
        self.assertEqual(contract["frozen_parameter_names"], [])
        optimizer = _optimizer(model, self.config)
        self.assertEqual(
            [id(p) for p in optimizer.param_groups[0]["params"]],
            [id(p) for name, p in model.named_parameters() if name != "raw_beta"],
        )

    def test_no_set_optimizer_checkpoint_round_trip(self):
        original = tiny_model(use_set_branch=False)
        _configure_trainable_parameters(original)
        optimizer = _optimizer(original, self.config)
        inputs = tiny_inputs()

        def update(model, adam):
            adam.zero_grad(set_to_none=True)
            (model(*inputs).reconstruction - inputs[0] * 0.6).square().mean().backward()
            adam.step()

        update(original, optimizer)
        resumed = tiny_model(use_set_branch=False)
        _configure_trainable_parameters(resumed)
        resumed.load_state_dict(copy.deepcopy(original.state_dict()))
        resumed_optimizer = _optimizer(resumed, self.config)
        resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        update(original, optimizer)
        update(resumed, resumed_optimizer)
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)


class DatedExperimentPathTest(unittest.TestCase):
    def test_beijing_date_and_increment_without_overwriting(self):
        instant = datetime(2026, 9, 6, 17, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            first = next_experiment_path(Path(directory), "example", now=instant)
            self.assertEqual(first.name, "example_20260907_run01")
            first.mkdir()
            second = next_experiment_path(Path(directory), "example", now=instant)
            self.assertEqual(second.name, "example_20260907_run02")
            second.with_suffix(".launcher.log").touch()
            self.assertTrue(
                next_experiment_path(
                    Path(directory), "example", now=instant
                ).name.endswith("run03")
            )

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            next_experiment_path(Path("/tmp"), "../escape")
        with self.assertRaises(ValueError):
            next_experiment_path(Path("/tmp"), "example", now=datetime(2026, 9, 7))

    def test_dotted_names_keep_date_in_log_collision_check(self):
        instant = datetime(2026, 9, 7, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            first = next_experiment_path(Path(directory), "example.v1", now=instant)
            (first.parent / f"{first.name}.launcher.log").touch()
            second = next_experiment_path(Path(directory), "example.v1", now=instant)
            self.assertEqual(second.name, "example.v1_20260907_run02")


if __name__ == "__main__":
    unittest.main()
