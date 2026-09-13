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
V4_SPLITS = {
    "train": tuple(f"P{i:02d}" for i in range(1, 13)),
    "validation": V3_SPLITS["validation"],
    "test": V3_SPLITS["test"],
}


def expected_splits(version: int) -> dict[str, tuple[str, ...]]:
    if type(version) is not int or version not in (2, 3, 4):
        raise ValueError(f"Unsupported dataset split version: {version!r}")
    if version == 2:
        return dict(LEGACY_SPLITS)
    return dict(V3_SPLITS if version == 3 else V4_SPLITS)
