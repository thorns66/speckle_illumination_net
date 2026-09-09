import unittest
from unittest.mock import patch

from tools.report_no_set_ablation import macro, paired_rows, verdict
from tools.run_no_set_ablation import idle_gpus, normalized_config


def row(sample, subset, error, step=20):
    return {
        "sample_id": sample,
        "subset_index": subset,
        "step": step,
        "split": "test",
        "gt_axial_w1_um": error,
        "gt_scale_aligned_nrmse": error / 10,
    }


class NoSetReportingTest(unittest.TestCase):
    def test_object_macro_does_not_count_subsets_as_objects(self):
        rows = [row("A", 1, 1), row("A", 2, 3), row("B", 1, 10)]
        self.assertEqual(macro(rows)["gt_axial_w1_um"], 6)

    def test_paired_differences_match_by_keys_not_order(self):
        left = [row("A", 1, 2), row("B", 1, 4)]
        right = [row("B", 1, 3, 40), row("A", 1, 5, 40)]
        rows = paired_rows(left, right, "test")
        self.assertEqual(rows[0]["gt_axial_w1_um_delta_no_set_minus_with_set"], 3)
        self.assertEqual(rows[1]["gt_axial_w1_um_delta_no_set_minus_with_set"], -1)
        self.assertEqual(rows[0]["with_set_step"], 20)
        self.assertEqual(rows[0]["no_set_step"], 40)

    def test_missing_and_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            paired_rows([row("A", 1, 1)], [row("B", 1, 2)], "test")
        with self.assertRaises(ValueError):
            paired_rows([row("A", 1, 1)] * 2, [row("A", 1, 2)], "test")

    def test_directional_and_mixed_verdicts(self):
        left = [row(sample, 1, 1) for sample in "ABC"]
        right = [row(sample, 1, 2) for sample in "ABC"]
        self.assertEqual(verdict(left, right)["objects_both_metrics_favor_with_set"], 3)
        self.assertIn("正向作用", verdict(left, right)["label"])
        self.assertIn("支持无 Set", verdict(right, left)["label"])
        self.assertIn("混合收益", verdict(left, left)["label"])
        for item in right:
            item["gt_scale_aligned_nrmse"] = 0.01
        self.assertIn("混合收益", verdict(left, right)["label"])

    def test_config_normalization_does_not_mutate_input(self):
        config = {
            "experiment": {"name": "A", "output_dir": "old", "seed": 1},
            "data": {},
            "ablation": {"use_set_branch": True},
        }
        result = normalized_config(config)
        self.assertEqual(result["data"]["var_feature_representation"], "sqrt")
        self.assertEqual(config["data"], {})
        self.assertTrue(config["ablation"]["use_set_branch"])

    @patch("tools.run_no_set_ablation.subprocess.run")
    @patch("tools.run_no_set_ablation._gpu_inventory")
    def test_idle_selection_excludes_low_memory_compute_processes(
        self, inventory, query
    ):
        inventory.return_value = [
            {
                "index": 0,
                "uuid": "busy-process",
                "memory_used_mib": 100,
                "utilization_percent": 0,
            },
            {
                "index": 1,
                "uuid": "idle",
                "memory_used_mib": 16,
                "utilization_percent": 0,
            },
            {
                "index": 2,
                "uuid": "busy-memory",
                "memory_used_mib": 2048,
                "utilization_percent": 0,
            },
            {
                "index": 3,
                "uuid": "busy-util",
                "memory_used_mib": 16,
                "utilization_percent": 50,
            },
        ]
        query.return_value.stdout = "busy-process, 123\n"
        self.assertEqual(idle_gpus()[0], [1])


if __name__ == "__main__":
    unittest.main()
