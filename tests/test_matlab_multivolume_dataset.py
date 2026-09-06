import unittest
from pathlib import Path

import numpy as np

from datasets.matlab_multivolume_dataset import (
    MatlabMultiVolumeDataset,
    load_inference_input,
)


ROOT = Path(__file__).resolve().parents[1] / "data" / "matlab_cells_pilot_v2_r04"


@unittest.skipUnless(ROOT.is_dir(), "unified MATLAB dataset is not present")
class MatlabMultiVolumeDatasetTest(unittest.TestCase):
    def test_authoritative_object_splits_and_item_counts(self):
        expected = {
            "train": (80, {"P01", "P02", "P03", "P04", "P05", "P06", "P08", "P10"}),
            "validation": (30, {"P09", "V01", "V02"}),
            "test": (30, {"P07", "T01", "T02"}),
        }
        fingerprints = set()
        for split, (count, objects) in expected.items():
            dataset = MatlabMultiVolumeDataset(ROOT, split)
            self.assertEqual(len(dataset), count)
            self.assertEqual({key.sample_id for key in dataset.keys}, objects)
            fingerprints.add(dataset.dataset_fingerprint)
        self.assertEqual(len(fingerprints), 1)

    def test_p07_is_test_and_p10_is_train_despite_stale_embedded_metadata(self):
        train = MatlabMultiVolumeDataset(ROOT, "train")
        test = MatlabMultiVolumeDataset(ROOT, "test")
        self.assertIn("P10", {key.sample_id for key in train.keys})
        self.assertNotIn("P07", {key.sample_id for key in train.keys})
        self.assertIn("P07", {key.sample_id for key in test.keys})
        self.assertNotIn("P10", {key.sample_id for key in test.keys})

    def test_input_shapes_and_exact_ten_frame_centering(self):
        item = MatlabMultiVolumeDataset(ROOT, "test")[0]
        self.assertEqual(tuple(item["f_var"].shape), (1, 10, 260, 260))
        self.assertEqual(tuple(item["g_mean"].shape), (1, 10, 260, 260))
        self.assertEqual(tuple(item["residual_frames"].shape), (10, 1, 260, 260))
        self.assertEqual(tuple(item["measured_variance"].shape), (1, 260, 260))
        np.testing.assert_allclose(
            item["residual_frames"].numpy().mean(axis=0), 0.0, atol=2e-7
        )

    def test_training_path_never_loads_ground_truth(self):
        train_item = MatlabMultiVolumeDataset(ROOT, "train")[0]
        validation_item = MatlabMultiVolumeDataset(ROOT, "validation")[0]
        self.assertNotIn("ground_truth", train_item)
        self.assertIn("ground_truth", validation_item)
        with self.assertRaises(ValueError):
            MatlabMultiVolumeDataset(ROOT, "train", include_ground_truth=True)

    def test_inference_loader_does_not_return_target_or_ground_truth(self):
        item = load_inference_input(ROOT / "P07", 1)
        self.assertEqual(
            set(item),
            {
                "sample_id",
                "subset_index",
                "split",
                "input_indices",
                "z_values_um",
                "f_var",
                "g_mean",
                "input_mean",
                "residual_frames",
            },
        )
        self.assertNotIn("measured_variance", item)
        self.assertNotIn("ground_truth", item)


if __name__ == "__main__":
    unittest.main()
