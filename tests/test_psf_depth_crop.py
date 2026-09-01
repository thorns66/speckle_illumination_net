import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from physics.psf_loader import inspect_psf, load_psf


Z_REQUESTED = list(range(10, 101, 10))


def _canonical_kernel() -> np.ndarray:
    values = np.zeros((13, 3, 3, 5, 7), dtype=np.float32)
    for z in range(13):
        for aa in range(3):
            for bb in range(3):
                values[z, aa, bb] = z * 100 + aa * 10 + bb
    return values


class PSFDepthCropTest(unittest.TestCase):
    def test_hdf5_v73_depth_and_axis_order(self):
        canonical = _canonical_kernel()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "psf_v73.mat"
            with h5py.File(path, "w") as handle:
                # MATLAB [row,col,aa,bb,z] is persisted with reversed axes.
                handle.create_dataset("H", data=canonical.transpose(0, 2, 1, 4, 3))
                handle.create_dataset("Ht", data=(canonical + 1).transpose(0, 2, 1, 4, 3))
                handle.create_dataset("x3objspace", data=(np.arange(10, 131, 10) * 1e-6)[:, None])
                handle.create_dataset("Nnum", data=np.array([[3.0]]))
                handle.create_dataset("CAindex", data=np.vstack([np.ones(13), np.full(13, 5)]))
            result = load_psf(path, Z_REQUESTED, load_h=True, load_ht=True)
        np.testing.assert_array_equal(result.metadata.selected_indices, np.arange(10))
        np.testing.assert_allclose(result.metadata.selected_z_um, Z_REQUESTED, atol=1e-8)
        np.testing.assert_array_equal(result.H, canonical[:10])
        np.testing.assert_array_equal(result.Ht, canonical[:10] + 1)
        self.assertEqual(result.H.shape, (10, 3, 3, 5, 7))

    def test_classic_mat_depth_and_axis_order(self):
        canonical = _canonical_kernel()
        matlab_h = canonical.transpose(3, 4, 1, 2, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "psf_classic.mat"
            savemat(
                path,
                {
                    "H": matlab_h,
                    "Ht": matlab_h + 1,
                    "x3objspace": np.arange(10, 131, 10, dtype=np.float64)[:, None] * 1e-6,
                    "Nnum": np.array([[3.0]]),
                    "CAindex": np.column_stack([np.ones(13), np.full(13, 5)]),
                },
            )
            result = load_psf(path, Z_REQUESTED, load_h=True, load_ht=True)
        np.testing.assert_array_equal(result.H, canonical[:10])
        np.testing.assert_array_equal(result.Ht, canonical[:10] + 1)

    def test_missing_depth_fails_instead_of_nearest_neighbor(self):
        canonical = _canonical_kernel()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "psf.mat"
            savemat(
                path,
                {
                    "H": canonical.transpose(3, 4, 1, 2, 0),
                    "x3objspace": np.arange(10, 131, 10)[:, None] * 1e-6,
                },
            )
            with self.assertRaisesRegex(ValueError, "15"):
                inspect_psf(path, [10, 15])

    def test_hdf5_requested_order_is_preserved(self):
        canonical = _canonical_kernel()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "psf_v73.mat"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("H", data=canonical.transpose(0, 2, 1, 4, 3))
                handle.create_dataset("x3objspace", data=(np.arange(10, 131, 10) * 1e-6)[:, None])
            result = load_psf(path, [100, 10], load_h=True)
        np.testing.assert_array_equal(result.metadata.selected_indices, [9, 0])
        np.testing.assert_array_equal(result.H, canonical[[9, 0]])


if __name__ == "__main__":
    unittest.main()
