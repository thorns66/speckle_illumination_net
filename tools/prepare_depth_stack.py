from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import tifffile


def _frame_number(path: Path) -> int:
    match = re.search(r"_(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Cannot read frame number from {path.name}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack numbered depth frames into a TIFF stack")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    patterns = (f"img_detph{args.depth}_*.tif", f"img_depth{args.depth}_*.tif")
    files = sorted(
        {path for pattern in patterns for path in args.input_dir.glob(pattern)},
        key=_frame_number,
    )
    numbers = [_frame_number(path) for path in files]
    expected = list(range(1, args.frames + 1))
    if numbers != expected:
        missing = sorted(set(expected) - set(numbers))
        extra = sorted(set(numbers) - set(expected))
        raise ValueError(f"Expected frames 1..{args.frames}; missing={missing}, extra={extra}")

    reference_shape = None
    reference_dtype = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(args.output, bigtiff=False) as writer:
        for path in files:
            frame = tifffile.imread(path)
            if frame.ndim != 2:
                raise ValueError(f"Expected a single grayscale page in {path}")
            if reference_shape is None:
                reference_shape = frame.shape
                reference_dtype = frame.dtype
            if frame.shape != reference_shape or frame.dtype != reference_dtype:
                raise ValueError(
                    f"Frame {path.name} has shape/dtype {frame.shape}/{frame.dtype}; "
                    f"expected {reference_shape}/{reference_dtype}"
                )
            writer.write(np.ascontiguousarray(frame), photometric="minisblack")

    with tifffile.TiffFile(args.output) as tif:
        if len(tif.pages) != args.frames:
            raise RuntimeError(f"Output contains {len(tif.pages)} pages, expected {args.frames}")
    print(f"Wrote {args.frames} frames to {args.output}")


if __name__ == "__main__":
    main()
