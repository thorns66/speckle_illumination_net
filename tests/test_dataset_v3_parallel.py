from pathlib import Path
import tempfile
import unittest

from tools.dataset_v3_parallel_common import missing_jobs, partition_jobs, validate_gpu


class DatasetV3ParallelTest(unittest.TestCase):
    def test_balanced_disjoint_full_coverage(self):
        jobs = [{"kind": "frame", "index": i} for i in range(65, 101)] + [
            {"kind": "subset", "index": i} for i in range(1, 11)
        ]
        shards = partition_jobs(jobs)
        flattened = [j for shard in shards for j in shard]
        self.assertCountEqual(flattened, jobs)
        self.assertEqual(len({(j["kind"], j["index"]) for j in flattened}), len(jobs))
        loads = [
            sum(2 if j["kind"] == "subset" else 1 for j in shard) for shard in shards
        ]
        self.assertLessEqual(max(loads) - min(loads), 1)
        self.assertEqual(len([s for s in shards if s]), 6)

    def test_invalid_or_duplicate_jobs_rejected(self):
        for jobs in (
            [{"kind": "frame", "index": 101}],
            [{"kind": "subset", "index": 0}],
            [{"kind": "frame", "index": True}],
            [{"kind": "train", "index": 1}],
            [{"kind": "frame", "index": 1}] * 2,
        ):
            with self.assertRaises(ValueError):
                partition_jobs(jobs)
        with self.assertRaises(ValueError):
            partition_jobs([], [0, 1, 2])

    def test_completed_files_are_never_assigned(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "recon_frames").mkdir()
            (root / "subsets").mkdir()
            (root / "recon_frames/frame_001.mat").touch()
            (root / "subsets/subset_03.mat").touch()
            jobs = missing_jobs(root, "reconstruct")
            self.assertEqual(len(jobs), 108)
            self.assertNotIn({"kind": "frame", "index": 1}, jobs)
            self.assertNotIn({"kind": "subset", "index": 3}, jobs)
            self.assertEqual(len(missing_jobs(root, "sensor")), 100)

    def test_only_gpu0_may_share_existing_contexts(self):
        base = {
            "physical_index": 0,
            "uuid": "a",
            "name": "NVIDIA A40",
            "free_mib": 34000,
            "utilization_percent": 0,
            "compute_processes": ["existing"],
        }
        self.assertTrue(validate_gpu(base, base))
        card = {**base, "physical_index": 1}
        self.assertFalse(validate_gpu(card, card))
        self.assertTrue(validate_gpu({**card, "compute_processes": []}, card))
        self.assertFalse(validate_gpu({**base, "free_mib": 10000}, base))
        with self.assertRaises(ValueError):
            validate_gpu({**base, "uuid": "b"}, base)


if __name__ == "__main__":
    unittest.main()
