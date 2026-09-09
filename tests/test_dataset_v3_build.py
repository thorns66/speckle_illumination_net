import json
from pathlib import Path
import tempfile
import unittest

from datasets.matlab_multivolume_dataset import load_dataset_index
from tools.run_dataset_v3 import gpu_ready, rewrite_json
from utils.dataset_splits import LEGACY_SPLITS, V3_SPLITS, expected_splits


class DatasetV3BuildTest(unittest.TestCase):
    def test_versioned_splits_keep_legacy_and_prevent_object_leakage(self):
        self.assertEqual(expected_splits(2), LEGACY_SPLITS)
        self.assertEqual(expected_splits(3), V3_SPLITS)
        self.assertIn("P07", expected_splits(2)["test"])
        self.assertIn("P09", expected_splits(2)["validation"])
        self.assertEqual(V3_SPLITS["train"], tuple(f"P{i:02d}" for i in range(1, 12)))
        all_ids = sum(V3_SPLITS.values(), ())
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertNotIn("T01", all_ids)
        self.assertEqual(V3_SPLITS["test"], ("T02", "T03", "T04"))
        for version in (1, 4, True, "3", 3.0):
            with self.assertRaises(ValueError):
                expected_splits(version)

    def test_metadata_rewrite_preserves_numbers_and_path_boundaries(self):
        source = {
            "sample_id": "T01",
            "split": "test",
            "seed": 123,
            "values": [1.0, 2.0],
            "cfg": {"sample_dir": "/old/T01", "output_root": "/old"},
            "preview": "/old/T01/previews/a.tif",
            "unrelated": "/older/T01",
        }
        result = rewrite_json(source, "T01", "P11", "train", ["/old"], "/new")
        self.assertEqual(result["sample_id"], "P11")
        self.assertEqual(result["split"], "train")
        self.assertEqual(result["cfg"]["sample_dir"], "/new/P11")
        self.assertEqual(result["preview"], "/new/P11/previews/a.tif")
        self.assertEqual(result["unrelated"], "/older/T01")
        self.assertEqual(result["values"], source["values"])
        self.assertEqual(result["seed"], 123)
        self.assertEqual(source["sample_id"], "T01")

    def test_gpu_gate_accepts_idle_existing_contexts_but_never_another_gpu(self):
        snapshot = {
            "physical_index": 0,
            "uuid": "gpu-zero",
            "name": "NVIDIA A40",
            "free_mib": 34000,
            "utilization_percent": 0,
            "compute_processes": [1, 2, 3, 4],
        }
        self.assertTrue(gpu_ready(snapshot, "gpu-zero"))
        self.assertFalse(gpu_ready({**snapshot, "free_mib": 20000}, "gpu-zero"))
        self.assertFalse(gpu_ready({**snapshot, "utilization_percent": 80}, "gpu-zero"))
        for change in ({"physical_index": 1}, {"uuid": "gpu-one"}, {"name": "V100"}):
            with self.assertRaises(RuntimeError):
                gpu_ready({**snapshot, **change}, "gpu-zero")

    def make_index(self, root: Path, version: int):
        rows = []
        for split, objects in expected_splits(version).items():
            for sample in objects:
                folder = root / sample
                (folder / "subsets").mkdir(parents=True)
                (folder / "sensor_frames").mkdir()
                (folder / "prepared.mat").touch()
                validation = {
                    "complete": True,
                    "sample_id": sample,
                    "frame_count": 100,
                    "subset_count": 10,
                }
                (folder / "validation_manifest.json").write_text(json.dumps(validation))
                for i in range(1, 11):
                    (folder / "subsets" / f"subset_{i:02d}.mat").touch()
                for i in range(1, 101):
                    (folder / "sensor_frames" / f"frame_{i:03d}.mat").touch()
                rows.append(
                    {
                        "sample_id": sample,
                        "object_group_id": sample,
                        "split": split,
                        "sample_dir": str(folder),
                    }
                )
        (root / "dataset_splits.json").write_text(
            json.dumps({"version": version, "dataset_complete": True, "samples": rows})
        )
        (root / "FINAL_DATASET_MANIFEST.json").write_text(
            json.dumps({"version": version, "complete": True, "samples": rows})
        )

    def test_reader_indexes_both_versions_without_changing_baseline(self):
        for version, count in ((2, 80), (3, 110)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_index(root, version)
                indexed, fingerprint = load_dataset_index(root)
                self.assertEqual(
                    [len(indexed[k]) for k in ("train", "validation", "test")],
                    [count, 30, 30],
                )
                self.assertEqual(len(fingerprint), 64)

    def test_reader_rejects_incomplete_mismatched_and_duplicate_v3(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_index(root, 3)
            path = root / "FINAL_DATASET_MANIFEST.json"
            original = json.loads(path.read_text())
            for change in (
                {"complete": False},
                {"version": 2},
                {"samples": original["samples"] + [original["samples"][0]]},
            ):
                path.write_text(json.dumps({**original, **change}))
                with self.assertRaises(ValueError):
                    load_dataset_index(root)
            path.write_text(json.dumps(original))
            (root / "T04" / "sensor_frames" / "frame_100.mat").unlink()
            with self.assertRaisesRegex(ValueError, "100-frame"):
                load_dataset_index(root)


if __name__ == "__main__":
    unittest.main()
