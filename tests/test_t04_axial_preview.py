from pathlib import Path
import unittest

import numpy as np

from tools.report_t04_axial_preview import label, local_roi
from tools.run_t04_axial_preview import REVISION, replacement_plan, summarize_unittest


class T04AxialPreviewTest(unittest.TestCase):
    def test_local_projection_excludes_another_row(self):
        volume = np.zeros((10, 20, 20), dtype=np.float32)
        volume[2, 5, 5] = 1
        volume[4, 5, 5] = 0.5
        volume[8, 15, 5] = 8
        before = volume.copy()
        roi, xs, ys = local_roi(volume, {"roi_bounds_xy_um": [3, 3, 7, 7]}, 1)
        self.assertEqual(roi.shape, (10, 5, 5))
        np.testing.assert_array_equal(xs, [3, 4, 5, 6, 7])
        np.testing.assert_array_equal(ys, [3, 4, 5, 6, 7])
        profile = roi.sum(axis=(1, 2))
        self.assertEqual(profile[2], 1)
        self.assertEqual(profile[4], 0.5)
        self.assertEqual(profile[8], 0)
        np.testing.assert_array_equal(volume, before)

    def test_empty_or_nonvolumetric_roi_rejected(self):
        with self.assertRaises(ValueError):
            local_roi(np.zeros((2, 2)), {"roi_bounds_xy_um": [0, 0, 1, 1]}, 1)
        with self.assertRaises(ValueError):
            local_roi(np.zeros((2, 2, 2)), {"roi_bounds_xy_um": [20, 20, 30, 30]}, 1)

    def test_draft_replacement_preserves_history_and_gpu_restriction(self):
        record = replacement_plan(Path("/old"), Path("/new"), Path("/old_axial"))
        self.assertEqual(record["geometry_revision"], REVISION)
        self.assertEqual(record["new_truth_dir"], "/new/T04")
        self.assertEqual(record["rejected_mesh_preserved_at"], "/old/T04")
        self.assertEqual(record["previous_axial_preview_preserved_at"], "/old_axial")
        self.assertEqual(REVISION, "t04_axial_line_pairs_v2")
        self.assertEqual(
            record["retained_previews"], {"T03": "/old/T03", "V03": "/old/V03"}
        )
        self.assertEqual(record["allowed_physical_gpu_indices"], [0])
        self.assertEqual(record["required_gpu_model"], "A40")
        for key in (
            "dataset_complete",
            "morphology_approved",
            "migration_executed",
            "allow_sharing",
            "allow_fallback_gpu",
        ):
            self.assertFalse(record[key])

    def test_unittest_summary_discloses_skips(self):
        text = "test_a ... ok\ntest_b ... skipped 'fixture missing'\nRan 2 tests in 0.1s\n\nOK (skipped=1)\n"
        summary = summarize_unittest(text)
        self.assertEqual(
            (summary["total"], summary["passed"], summary["skipped"]), (2, 1, 1)
        )
        self.assertIn("fixture missing", summary["skipped_test_records"][0])
        self.assertEqual(summarize_unittest("Ran 3 tests in 1.0s\nOK\n")["passed"], 3)

    def test_incomplete_failed_or_inconsistent_logs_rejected(self):
        for text in (
            "test running",
            "Ran 1 test in 1s\nFAILED (failures=1)\n",
            "Ran 2 tests in 1s\nOK (skipped=1)\n",
        ):
            with self.assertRaises(ValueError):
                summarize_unittest(text)

    def test_labels_show_physical_depth_and_single_control(self):
        self.assertIn("20/40", label({"region_id": "A1", "z_um": [20, 40]}))
        self.assertIn("center dz=20", label({"region_id": "A2", "z_um": [20, 40]}))
        self.assertIn("single z=30", label({"region_id": "D1", "z_um": 30}))

    def test_ten_micron_label_and_adjacent_profile_are_not_interpolated(self):
        self.assertIn("center dz=10", label({"region_id": "A1", "z_um": [20, 30]}))
        volume = np.zeros((10, 20, 20), dtype=np.float32)
        volume[1, 5, 5] = 1
        volume[2, 5, 5] = 0.5
        roi, _, _ = local_roi(volume, {"roi_bounds_xy_um": [3, 3, 7, 7]}, 1)
        profile = roi.sum(axis=(1, 2))
        np.testing.assert_array_equal(profile, [0, 1, 0.5, 0, 0, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
