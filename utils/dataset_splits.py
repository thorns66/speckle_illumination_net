"""Explicit dataset revisions; never reinterpret historical experiments."""

LEGACY_SPLITS = {
    "train": ("P01", "P02", "P03", "P04", "P05", "P06", "P08", "P10"),
    "validation": ("P09", "V01", "V02"),
    "test": ("P07", "T01", "T02"),
}
V3_SPLITS = {
    "train": tuple(f"P{i:02d}" for i in range(1, 12)),
    "validation": ("V01", "V02", "V03"),
    "test": ("T02", "T03", "T04"),
}


def expected_splits(version: int) -> dict[str, tuple[str, ...]]:
    if type(version) is not int or version not in (2, 3):
        raise ValueError(f"Unsupported dataset split version: {version!r}")
    return dict(LEGACY_SPLITS if version == 2 else V3_SPLITS)
