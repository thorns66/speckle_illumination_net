from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from datasets.matlab_multivolume_dataset import EXPECTED_SPLITS, load_dataset_index
from tools.run_dataset_v3_preview import (
    GPU_POLICY,
    NEW_OBJECTS,
    NEW_SPLITS,
    OLD_SPLITS,
    cpu_environment,
    matlab_string,
    split_plan,
    tree_signature,
    write_json,
)


class DatasetV3PreviewTest(unittest.TestCase):
    def test_exact_object_partition_and_origin_mapping(self):
        draft = split_plan(Path("/source"), Path("/new"))
        rows = draft["samples"]
        self.assertEqual(draft["counts"], {"train": 11, "validation": 3, "test": 3})
        self.assertEqual(len(rows), 17)
        self.assertEqual(len({r["sample_id"] for r in rows}), 17)
        self.assertEqual(len({r["object_group_id"] for r in rows}), 17)
        self.assertEqual(
            {r["sample_id"] for r in rows if r["source_dir"] is None}, set(NEW_OBJECTS)
        )
        by_id = {r["sample_id"]: r for r in rows}
        self.assertNotIn("T01", by_id)
        self.assertEqual(by_id["P11"]["source_sample_id"], "T01")
        self.assertEqual(by_id["P11"]["source_dir"], "/source/T01")
        self.assertEqual(by_id["P11"]["planned_sample_dir"], "/new/P11")
        self.assertEqual(by_id["P11"]["stage"], "reuse_planned_not_migrated")
        self.assertEqual(
            [by_id[x]["split"] for x in ("P07", "P09", "P11")], ["train"] * 3
        )
        self.assertEqual(by_id["V03"]["split"], "validation")
        self.assertFalse(draft["dataset_complete"])
        self.assertFalse(draft["morphology_approved"])
        self.assertFalse(draft["migration_executed"])

    def test_existing_authoritative_split_not_modified(self):
        self.assertEqual({k: list(v) for k, v in EXPECTED_SPLITS.items()}, OLD_SPLITS)
        self.assertEqual(NEW_SPLITS["test"], ["T02", "T03", "T04"])
        self.assertIn("T01", EXPECTED_SPLITS["test"])

    def test_preview_cannot_be_loaded_as_training_dataset(self):
        with tempfile.TemporaryDirectory(prefix="speckle_v3_fixture_") as temp:
            root = Path(temp)
            draft = split_plan(root / "source", root)
            write_json(root / "dataset_splits_preview.json", draft)
            with self.assertRaises(FileNotFoundError):
                load_dataset_index(root)
            # Even accidental renaming of the draft cannot make it trainable.
            write_json(root / "dataset_splits.json", draft)
            write_json(root / "FINAL_DATASET_MANIFEST.json", {"complete": False})
            with self.assertRaisesRegex(ValueError, "complete"):
                load_dataset_index(root)

    def test_cpu_visibility_overrides_inherited_gpu_selection(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5"}):
            env = cpu_environment(Path("/preview"))
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5")
            self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(GPU_POLICY["allowed_physical_gpu_indices"], [0])
        self.assertEqual(GPU_POLICY["required_gpu_model"], "A40")
        self.assertFalse(GPU_POLICY["allow_sharing"])
        self.assertFalse(GPU_POLICY["allow_fallback_gpu"])

    def test_tree_scan_is_read_only_and_detects_size_changes(self):
        with tempfile.TemporaryDirectory(prefix="speckle_v3_tree_") as temp:
            root = Path(temp)
            data = root / "sample.json"
            write_json(data, {"value": 1})
            first = tree_signature(root)
            self.assertEqual(tree_signature(root), first)
            self.assertEqual(json.loads(data.read_text())["value"], 1)
            write_json(data, {"value": 100000})
            self.assertNotEqual(tree_signature(root), first)

    def test_matlab_path_escaping(self):
        self.assertEqual(matlab_string("/a/it's"), "'/a/it''s'")

    def test_depth_coloring_uses_absolute_depth_and_global_alpha(self):
        from tools.report_dataset_v3_preview import depth_rgba

        volume = np.zeros((3, 3, 3), dtype=np.float32)
        volume[0, 0, 0] = 1
        volume[2, 2, 2] = 0.25
        before = volume.copy()
        color = depth_rgba(volume, np.array([10, 50, 100]))
        np.testing.assert_array_equal(volume, before)
        self.assertEqual(color[0, 0, 3], 1)
        self.assertEqual(color[2, 2, 3], 0.25)
        self.assertEqual(color[1, 1, 3], 0)
        self.assertFalse(np.array_equal(color[0, 0, :3], color[2, 2, :3]))


if __name__ == "__main__":
    unittest.main()
