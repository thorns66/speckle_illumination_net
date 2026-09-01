from __future__ import annotations

import argparse
import json

from physics.psf_loader import inspect_psf


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LF-PSF metadata without loading dense kernels")
    parser.add_argument("path")
    parser.add_argument("--z", type=float, nargs="+", default=list(range(10, 101, 10)))
    args = parser.parse_args()
    metadata = inspect_psf(args.path, args.z)
    print(
        json.dumps(
            {
                "path": str(metadata.path),
                "format": metadata.mat_format,
                "variables_matlab_shape": metadata.variable_shapes_matlab,
                "z_all_um": metadata.z_all_um.tolist(),
                "selected_indices_zero_based": metadata.selected_indices.tolist(),
                "selected_z_um": metadata.selected_z_um.tolist(),
                "phase_period": metadata.phase_period,
                "selected_caindex_matlab_one_based": (
                    metadata.caindex.tolist() if metadata.caindex is not None else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
