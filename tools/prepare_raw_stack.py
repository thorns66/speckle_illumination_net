from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import tifffile


def _natural_key(path: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine ordered grayscale TIFF frames without rescaling their dtype"
    )
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-frames", type=int, default=100)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob), key=_natural_key)
    if len(paths) != args.expected_frames:
        raise ValueError(
            f"Expected {args.expected_frames} frames, found {len(paths)} for "
            f"{args.input_glob!r}"
        )
    frames = [tifffile.imread(path) for path in paths]
    first = np.asarray(frames[0])
    if first.ndim != 2:
        raise ValueError(f"Expected a grayscale frame, got {first.shape}")
    for path, frame in zip(paths, frames):
        value = np.asarray(frame)
        if value.shape != first.shape or value.dtype != first.dtype:
            raise ValueError(
                f"Frame {path} has shape/dtype {value.shape}/{value.dtype}; "
                f"expected {first.shape}/{first.dtype}"
            )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(output, bigtiff=False) as writer:
        for frame in frames:
            writer.write(
                np.asarray(frame),
                photometric="minisblack",
                contiguous=False,
            )
    with tifffile.TiffFile(output) as tif:
        if len(tif.pages) != args.expected_frames:
            raise RuntimeError(
                f"Wrote {len(tif.pages)} pages instead of {args.expected_frames}"
            )
    print(
        f"wrote {output} frames={len(frames)} shape={first.shape} dtype={first.dtype}"
    )


if __name__ == "__main__":
    main()
