from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from priority_validation_analysis import (
    axial_blur,
    axial_shift,
    axial_targets,
    find_peaks,
    line_definitions,
    match_points,
    point_targets,
)
from priority_validation_common import load_config, scenes


def test_standard_protocol_has_exact_scene_and_target_counts() -> None:
    _, config = load_config()
    assert len(scenes(config)) == 11
    assert len(point_targets(config, 20)) == 36
    assert len(axial_targets(config)) == 16
    assert len(line_definitions(config)) == 24


def test_controlled_axial_errors_preserve_or_shift_mass_without_wrap() -> None:
    truth = np.zeros((10, 5, 4), dtype=np.float32)
    truth[3, 2, 2] = 1
    once = axial_blur(truth, 1)
    twice = axial_blur(truth, 2)
    assert np.isclose(once.sum(), truth.sum())
    assert np.isclose(twice.sum(), truth.sum())
    assert np.count_nonzero(once.sum(axis=(1, 2))) == 3
    assert np.count_nonzero(twice.sum(axis=(1, 2))) == 5
    assert np.argmax(axial_shift(truth, 1).sum(axis=(1, 2))) == 4
    assert np.argmax(axial_shift(truth, -1).sum(axis=(1, 2))) == 2
    edge = np.zeros_like(truth); edge[0, 2, 2] = 1
    assert axial_shift(edge, -1).sum() == 0


def test_peak_matching_is_one_to_one_and_rejects_wrong_depth() -> None:
    volume = np.zeros((10, 32, 32), dtype=np.float32)
    volume[3, 10, 10] = 1
    volume[3, 10, 14] = 0.7
    volume[7, 20, 20] = 0.9
    peaks = find_peaks(volume, 0.1)
    targets = [
        {"x_um": 10, "y_um": 10, "z_um": 40},
        {"x_um": 14, "y_um": 10, "z_um": 40},
        {"x_um": 20, "y_um": 20, "z_um": 40},
    ]
    matches, used = match_points(targets, peaks, pitch=1, xy_limit=2, z_limit=1)
    assert matches[0]["matched"] and matches[1]["matched"]
    assert not matches[2]["matched"]
    assert len(used) == 2
