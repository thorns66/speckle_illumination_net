from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.spinach_real_reprocess import FIXED_INDICES, source_files, statistics


class RealSpinachPreparationTests(unittest.TestCase):
    def make_files(self, folder):
        for index in range(100):
            (folder / f"Z50_image_{index}({4 + index % 2}).bmp").touch()

    def test_numeric_order_and_suffix_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.make_files(folder)
            files = source_files(folder, "50")
            self.assertEqual(files[0].name, "Z50_image_0(4).bmp")
            self.assertEqual(files[9].name, "Z50_image_9(5).bmp")
            self.assertEqual(files[-1].name, "Z50_image_99(5).bmp")

    def test_duplicate_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.make_files(folder)
            (folder / "Z50_image_0(7).bmp").touch()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                source_files(folder, "50")

    def test_incomplete_frames_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "Z50_image_0(4).bmp").touch()
            with self.assertRaisesRegex(ValueError, "exactly"):
                source_files(Path(directory), "50")

    def test_wrong_field_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "Z45_image_0(4).bmp").touch()
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                source_files(Path(directory), "50")

    def test_fixed_input_and_holdout_are_disjoint(self):
        selected = set(FIXED_INDICES)
        holdout = set(range(1, 101)) - selected
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(holdout), 90)
        self.assertFalse(selected & holdout)
        self.assertEqual(selected | holdout, set(range(1, 101)))

    def test_statistics_keep_common_scale_and_sample_variance(self):
        frames = np.asarray([[[.1, .2]], [[.3, .6]], [[.9, .8]]], dtype=np.float32)
        mean, variance = statistics(frames, [1, 2])
        expected = frames[:2].astype(np.float64)
        np.testing.assert_array_equal(mean, expected.mean(0).astype(np.float32))
        np.testing.assert_array_equal(variance, expected.var(0, ddof=1).astype(np.float32))
        self.assertAlmostEqual(float(mean[0, 0]), .2, places=6)
        self.assertAlmostEqual(float(variance[0, 0]), .02, places=6)
        self.assertEqual(mean.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
