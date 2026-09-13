import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import h5py
import tifffile
import torch

from models.configurable_anchor_lfm_net import ConfigurableAnchorLFMNet
from physics.lfm_operator import LFMOperator
from tests.model_fixtures import tiny_inputs
from tools.mixed_resolution_lfm import MixedResolutionLFM
from tools.v5_mixed_dataset import MixedSixOneOneScheduler, RealFieldDataset
from tools.v5_mixed_real_data import FIXED_SUBSET_01, subset_indices
from tools.run_v5_mixed_real_anchor_compare import config as v5_config
from tools.v5_mixed_real_anchor_experiment import cached_beta0


def tiny_model(checkpoint_segments: bool) -> ConfigurableAnchorLFMNet:
    return ConfigurableAnchorLFMNet(
        reconstruction_anchor="mean_rl3",
        var_channels=(4, 8, 12), mean_channels=(4, 8, 12),
        set_channels=(4, 8, 12), decoder_channels=(4, 8, 12),
        set_frame_chunk_size=2,
        activation_checkpoint_segments=checkpoint_segments,
    )


def empty_sparse_cache(root: Path, *, z: int = 2, height: int = 5, width: int = 7) -> Path:
    root.mkdir()
    rows, columns = height * width, z * height * width
    (root / "complete.json").write_text(json.dumps({
        "height": height, "width": width, "H_shape": [z, 2, 2, 3, 3],
    }))
    for prefix, size in (("forward", rows), ("transpose", columns)):
        np.save(root / f"{prefix}_indptr.npy", np.zeros(size + 1, dtype=np.int32))
        np.save(root / f"{prefix}_indices.npy", np.empty(0, dtype=np.int32))
        np.save(root / f"{prefix}_data.npy", np.empty(0, dtype=np.float32))
    return root


class V5RealSubsetTests(unittest.TestCase):
    def test_ten_subsets_partition_all_frames_and_preserve_first(self):
        values = subset_indices()
        self.assertEqual(values[0], list(FIXED_SUBSET_01))
        flat = [index for subset in values for index in subset]
        self.assertEqual(len(flat), 100)
        self.assertEqual(set(flat), set(range(1, 101)))
        self.assertEqual(values, subset_indices())

    def test_real_item_has_holdout_targets_but_never_fake_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "45/subset_01"
            frames_folder = root / "frames"
            folder.mkdir(parents=True)
            frames_folder.mkdir()
            frames = []
            selected_files = []
            for index in range(10):
                value = np.full((3, 4), (index + 1) / 20, dtype=np.float32)
                path = frames_folder / f"frame_{index + 1:03d}.tif"
                tifffile.imwrite(path, value)
                frames.append(value)
                selected_files.append(str(path))
            mean = np.mean(frames, axis=0, dtype=np.float64).astype(np.float32)
            for name, value in (
                ("mean", mean), ("variance", np.ones((3, 4), np.float32)),
                ("holdout_mean", np.full((3, 4), .2, np.float32)),
                ("holdout_variance", np.full((3, 4), .03, np.float32)),
            ):
                tifffile.imwrite(folder / f"{name}.tif", value)
            for name in ("mean_rl3", "taylor_rl3"):
                with h5py.File(folder / f"{name}.mat", "w") as handle:
                    handle.create_dataset("reconstruction_raw", data=np.ones((10, 4, 3), np.float32))
                    if name.startswith("taylor"):
                        handle.create_dataset("reconstruction_sqrt", data=np.ones((10, 4, 3), np.float32))
            manifest = {
                "field_id": "45", "subset_index": 1,
                "input_indices": list(range(1, 11)), "holdout_indices": list(range(11, 101)),
                "selected_files": selected_files, "z_um": list(range(10, 101, 10)),
                "mean_tiff": str(folder / "mean.tif"),
                "holdout_mean_tiff": str(folder / "holdout_mean.tif"),
                "holdout_variance_tiff": str(folder / "holdout_variance.tif"),
                "output_mat": {"mean": str(folder / "mean_rl3.mat"), "taylor": str(folder / "taylor_rl3.mat")},
            }
            (folder / "manifest.json").write_text(json.dumps(manifest))
            item = RealFieldDataset(root)[0]
            self.assertNotIn("ground_truth", item)
            self.assertFalse(item["has_ground_truth"])
            self.assertEqual(item["domain"], "real_train")
            self.assertEqual(tuple(item["measured_mean"].shape), (1, 3, 4))


class V5MixedSchedulerTests(unittest.TestCase):
    def test_every_batch_is_six_one_one_and_resume_is_exact(self):
        scheduler = MixedSixOneOneScheduler(130, 8, 20260901)
        for _ in range(17):
            batch = scheduler.next_batch()
            self.assertEqual(sum(index < 110 for index in batch), 6)
            self.assertEqual(sum(110 <= index < 120 for index in batch), 1)
            self.assertEqual(sum(index >= 120 for index in batch), 1)
        state = scheduler.state_dict()
        expected = [scheduler.next_batch() for _ in range(20)]
        resumed = MixedSixOneOneScheduler(130, 8, 20260901)
        resumed.load_state_dict(state)
        self.assertEqual(expected, [resumed.next_batch() for _ in range(20)])

    def test_two_configs_differ_only_in_identity_and_anchor_contract(self):
        taylor = v5_config("taylor_anchor_e3_mean100")
        mean = v5_config("mean_anchor_e3_mean100")
        for value in (taylor, mean):
            value["experiment"].pop("name")
            value["experiment"].pop("output_dir")
            value["model"].pop("reconstruction_anchor")
            value["v3_compare"].pop("kind")
            value["v5_mixed"].pop("kind")
            value["v5_mixed"].pop("beta_anchor")
        self.assertEqual(taylor, mean)

    def test_beta_cache_isolated_by_anchor_domain_and_shape(self):
        class SumOperator:
            def __call__(self, value):
                return value.sum(2)

        item = {
            "sample_id": "case", "subset_index": 1, "domain": "real_train",
            "f_var": torch.ones((1, 1, 2, 3, 4)),
            "g_mean": torch.full((1, 1, 2, 3, 4), 2.0),
            "input_mean": torch.ones((1, 1, 3, 4)),
        }
        cache = {}
        taylor = cached_beta0(cache, item, SumOperator(), v5_config("taylor_anchor_e3_mean100"))
        mean = cached_beta0(cache, item, SumOperator(), v5_config("mean_anchor_e3_mean100"))
        self.assertEqual(len(cache), 2)
        self.assertNotEqual(float(taylor), float(mean))


class SegmentCheckpointTests(unittest.TestCase):
    def test_checkpointed_and_plain_model_match_outputs_and_gradients(self):
        torch.manual_seed(20260901)
        plain = tiny_model(False)
        checkpointed = tiny_model(True)
        checkpointed.load_state_dict(plain.state_dict())
        with torch.no_grad():
            plain.decoder.head.weight.fill_(0.01)
            checkpointed.decoder.head.weight.fill_(0.01)
        inputs = tiny_inputs()
        plain.train(); checkpointed.train()
        left = plain(*inputs).reconstruction
        right = checkpointed(*inputs).reconstruction
        torch.testing.assert_close(left, right)
        left.square().mean().backward()
        right.square().mean().backward()
        for (left_name, left_parameter), (right_name, right_parameter) in zip(
            plain.named_parameters(), checkpointed.named_parameters()
        ):
            self.assertEqual(left_name, right_name)
            self.assertEqual(left_parameter.grad is None, right_parameter.grad is None)
            if left_parameter.grad is not None:
                torch.testing.assert_close(left_parameter.grad, right_parameter.grad, rtol=2e-5, atol=2e-6)


class RecomputedFullFieldOperatorTests(unittest.TestCase):
    def make_operator(self, folder: Path, h: torch.Tensor) -> MixedResolutionLFM:
        cache = empty_sparse_cache(folder)
        return MixedResolutionLFM(
            cache, torch.device("cpu"), full_h=h, phase_chunk_size=2, real_shape=(6, 8)
        )

    def test_forward_squared_and_gradients_match_reference(self):
        torch.manual_seed(7)
        h = torch.rand((2, 2, 2, 3, 3), dtype=torch.float64)
        value = torch.rand((1, 1, 2, 6, 8), dtype=torch.float64)
        with tempfile.TemporaryDirectory() as directory:
            mixed = self.make_operator(Path(directory) / "sparse", h)
            reference = LFMOperator(h, mode="optimized", phase_chunk_size=2)
            torch.testing.assert_close(mixed(value), reference(value), rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(
                mixed.forward_squared(value.square()),
                reference.forward_squared(value.square()), rtol=1e-10, atol=1e-10,
            )
            left = value.clone().requires_grad_(True)
            right = value.clone().requires_grad_(True)
            mixed(left).square().sum().backward()
            reference(right).square().sum().backward()
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-9, atol=1e-9)

    def test_detached_forward_is_reused_with_exact_adjoint_gradient(self):
        torch.manual_seed(8)
        h = torch.rand((2, 2, 2, 3, 3), dtype=torch.float64)
        value = torch.rand((1, 1, 2, 6, 8), dtype=torch.float64, requires_grad=True)
        with tempfile.TemporaryDirectory() as directory:
            mixed = self.make_operator(Path(directory) / "sparse", h)
            calls = 0
            original = mixed._full_project_plain

            def counted(*args, **kwargs):
                nonlocal calls
                calls += 1
                return original(*args, **kwargs)

            mixed._full_project_plain = counted
            detached = mixed(value.detach())
            attached = mixed(value)
            self.assertEqual(calls, 1)
            torch.testing.assert_close(detached, attached)
            attached.sum().backward()
            expected = LFMOperator(h, mode="optimized", phase_chunk_size=2)
            probe = value.detach().clone().requires_grad_(True)
            expected(probe).sum().backward()
            torch.testing.assert_close(value.grad, probe.grad, rtol=1e-9, atol=1e-9)

    def test_full_rank_can_omit_sparse_matrix_residency(self):
        torch.manual_seed(9)
        h = torch.rand((2, 2, 2, 3, 3), dtype=torch.float64)
        value = torch.rand((1, 1, 2, 6, 8), dtype=torch.float64)
        mixed = MixedResolutionLFM(
            "unused", torch.device("cpu"), full_h=h, phase_chunk_size=2,
            real_shape=(6, 8), load_sparse_simulation=False,
        )
        self.assertIsNone(mixed.simulation)
        torch.testing.assert_close(
            mixed(value), LFMOperator(h, mode="optimized", phase_chunk_size=2)(value),
            rtol=1e-10, atol=1e-10,
        )

    def test_inference_tensor_bypasses_versioned_forward_cache(self):
        torch.manual_seed(10)
        h = torch.rand((2, 2, 2, 3, 3), dtype=torch.float64)
        value = torch.rand((1, 1, 2, 6, 8), dtype=torch.float64)
        with tempfile.TemporaryDirectory() as directory:
            mixed = self.make_operator(Path(directory) / "sparse", h)
            reference = LFMOperator(h, mode="optimized", phase_chunk_size=2)
            with torch.inference_mode():
                actual = mixed(value.clone())
                expected = reference(value.clone())
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
            self.assertIsNone(mixed._linear_cache)


if __name__ == "__main__":
    unittest.main()
