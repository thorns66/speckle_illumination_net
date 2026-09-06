from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from physics.covariance_sketch import covariance_psd_report, load_preprocess_cs
from utils.io import load_tiff_stack


def _unit_norm(values: np.ndarray) -> np.ndarray:
    flattened = values.reshape(values.shape[0], -1)
    norms = np.linalg.norm(flattened, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("Probe generation produced a zero probe")
    return (flattened / norms).astype(np.float32, copy=False)


def _make_probe_families(
    count: int,
    shape: tuple[int, int],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    height, width = shape
    families: dict[str, np.ndarray] = {}
    raw = np.random.default_rng(seed).choice(
        (-1.0, 1.0), size=(count, height, width)
    ).astype(np.float32)
    families["global_rademacher"] = _unit_norm(raw)

    for sigma in (2.0, 4.0):
        blurred = np.stack(
            [gaussian_filter(probe, sigma=sigma, mode="reflect") for probe in raw],
            axis=0,
        )
        blurred -= blurred.mean(axis=(1, 2), keepdims=True)
        families[f"lowpass_sigma{sigma:g}"] = _unit_norm(blurred)

    local = np.zeros_like(raw)
    window = min(96, height, width)
    rng = np.random.default_rng(seed + 1)
    for index in range(count):
        row = int(rng.integers(0, height - window + 1))
        col = int(rng.integers(0, width - window + 1))
        local[index, row : row + window, col : col + window] = raw[
            index, row : row + window, col : col + window
        ]
    families[f"local_{window}x{window}"] = _unit_norm(local)
    return families


def _covariance_actions(
    frames: np.ndarray,
    probes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flattened = frames.reshape(frames.shape[0], -1).astype(np.float32, copy=False)
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    weights = centered @ probes.T
    full = (weights.T @ centered) / float(frames.shape[0] - 1)
    diagonal = centered.var(axis=0, ddof=1)[None] * probes
    return full, diagonal


def _row_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = np.finfo(np.float32).eps
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    cosine = np.einsum("ij,ij->i", first, second) / np.maximum(
        first_norm * second_norm, eps
    )
    first_centered = first - first.mean(axis=1, keepdims=True)
    second_centered = second - second.mean(axis=1, keepdims=True)
    pearson = np.einsum("ij,ij->i", first_centered, second_centered) / np.maximum(
        np.linalg.norm(first_centered, axis=1)
        * np.linalg.norm(second_centered, axis=1),
        eps,
    )
    relative_difference = np.linalg.norm(first - second, axis=1) / np.maximum(
        np.sqrt(0.5 * (first_norm**2 + second_norm**2)), eps
    )
    return cosine, pearson, relative_difference


def _summarize_variant(
    family: str,
    shrinkage: float,
    cosine: np.ndarray,
    pearson: np.ndarray,
    relative_difference: np.ndarray,
    norms: np.ndarray,
    off_diagonal_cosine: np.ndarray,
    off_diagonal_fraction: np.ndarray,
) -> dict[str, object]:
    # Arrays are [split, probe], except norms [2*split, probe].
    per_probe_reliability = np.maximum(
        np.median(cosine, axis=0), np.median(pearson, axis=0)
    )
    norm_mean = norms.mean(axis=0)
    norm_cv = norms.std(axis=0, ddof=1) / np.maximum(
        norm_mean, np.finfo(np.float32).eps
    )
    median_reliability = max(float(np.median(cosine)), float(np.median(pearson)))
    reliable_count = int(np.count_nonzero(per_probe_reliability >= 0.5))
    median_norm_cv = float(np.median(norm_cv))
    median_off_diagonal_cosine = float(np.median(off_diagonal_cosine))
    median_off_diagonal_fraction = float(np.median(off_diagonal_fraction))
    total_action_gate = (
        median_reliability >= 0.5
        and reliable_count >= 12
        and median_norm_cv < 0.2
    )
    # A shrinkage setting is not considered informative if it passes only by
    # becoming a copy of the diagonal estimator. The incremental off-diagonal
    # action must itself be reproducible.
    informative_off_diagonal_gate = (
        median_off_diagonal_cosine >= 0.2
        and median_off_diagonal_fraction >= 0.1
    )
    return {
        "family": family,
        "shrinkage_to_diagonal": shrinkage,
        "cosine_median": float(np.median(cosine)),
        "cosine_q10": float(np.quantile(cosine, 0.1)),
        "cosine_q90": float(np.quantile(cosine, 0.9)),
        "pearson_median": float(np.median(pearson)),
        "relative_difference_median": float(np.median(relative_difference)),
        "norm_cv_median": median_norm_cv,
        "reliable_probes_ge_0p5": reliable_count,
        "probe_count": int(cosine.shape[1]),
        "off_diagonal_cosine_median": median_off_diagonal_cosine,
        "off_diagonal_fraction_median": median_off_diagonal_fraction,
        "passes_total_action_gate": total_action_gate,
        "passes_informative_off_diagonal_gate": informative_off_diagonal_gate,
        "passes_stage_a": total_action_gate and informative_off_diagonal_gate,
    }


def _evaluate(
    frames: np.ndarray,
    families: dict[str, np.ndarray],
    *,
    split_count: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    if frames.shape[0] < 4:
        raise ValueError("At least four frames are required for split-half validation")
    if frames.shape[0] % 2:
        frames = frames[:-1]
    half = frames.shape[0] // 2
    names = list(families)
    probes_per_family = next(iter(families.values())).shape[0]
    all_probes = np.concatenate([families[name] for name in names], axis=0)
    records: dict[tuple[str, float], dict[str, list[np.ndarray]]] = {}
    shrinkages = (0.0, 0.25, 0.5, 0.75)
    for name in names:
        for shrinkage in shrinkages:
            records[(name, shrinkage)] = {
                "cosine": [],
                "pearson": [],
                "relative": [],
                "norm_a": [],
                "norm_b": [],
                "off_cosine": [],
                "off_fraction": [],
            }

    diagonal_correlations: list[float] = []
    rng = np.random.default_rng(seed)
    for split_index in range(split_count):
        permutation = rng.permutation(frames.shape[0])
        first = frames[permutation[:half]]
        second = frames[permutation[half : 2 * half]]
        full_a, diagonal_a = _covariance_actions(first, all_probes)
        full_b, diagonal_b = _covariance_actions(second, all_probes)
        variance_a = first.var(axis=0, ddof=1, dtype=np.float64).reshape(-1)
        variance_b = second.var(axis=0, ddof=1, dtype=np.float64).reshape(-1)
        diagonal_correlations.append(float(np.corrcoef(variance_a, variance_b)[0, 1]))

        for family_index, name in enumerate(names):
            start = family_index * probes_per_family
            stop = start + probes_per_family
            family_full_a, family_full_b = full_a[start:stop], full_b[start:stop]
            family_diag_a, family_diag_b = diagonal_a[start:stop], diagonal_b[start:stop]
            off_a = family_full_a - family_diag_a
            off_b = family_full_b - family_diag_b
            off_cosine, _, _ = _row_similarity(off_a, off_b)
            for shrinkage in shrinkages:
                action_a = (1.0 - shrinkage) * family_full_a + shrinkage * family_diag_a
                action_b = (1.0 - shrinkage) * family_full_b + shrinkage * family_diag_b
                cosine, pearson, relative = _row_similarity(action_a, action_b)
                denominator = 0.5 * (
                    np.linalg.norm(action_a, axis=1)
                    + np.linalg.norm(action_b, axis=1)
                )
                off_fraction = 0.5 * (
                    np.linalg.norm((1.0 - shrinkage) * off_a, axis=1)
                    + np.linalg.norm((1.0 - shrinkage) * off_b, axis=1)
                ) / np.maximum(denominator, np.finfo(np.float32).eps)
                record = records[(name, shrinkage)]
                record["cosine"].append(cosine)
                record["pearson"].append(pearson)
                record["relative"].append(relative)
                record["norm_a"].append(np.linalg.norm(action_a, axis=1))
                record["norm_b"].append(np.linalg.norm(action_b, axis=1))
                record["off_cosine"].append(off_cosine)
                record["off_fraction"].append(off_fraction)
        print(f"completed split {split_index + 1}/{split_count}", flush=True)

    summaries: list[dict[str, object]] = []
    for (name, shrinkage), record in records.items():
        norms = np.concatenate(
            [np.stack(record["norm_a"]), np.stack(record["norm_b"])], axis=0
        )
        summaries.append(
            _summarize_variant(
                name,
                shrinkage,
                np.stack(record["cosine"]),
                np.stack(record["pearson"]),
                np.stack(record["relative"]),
                norms,
                np.stack(record["off_cosine"]),
                np.stack(record["off_fraction"]),
            )
        )
    diagonal_summary = {
        "correlation_median": float(np.median(diagonal_correlations)),
        "correlation_q10": float(np.quantile(diagonal_correlations, 0.1)),
        "correlation_q90": float(np.quantile(diagonal_correlations, 0.9)),
    }
    return summaries, diagonal_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-A split-half reliability gate for covariance-vector probes"
    )
    parser.add_argument(
        "--raw",
        default="data/分辨率图仿真/prepared/depth50_raw_100.tif",
    )
    parser.add_argument(
        "--cs",
        default=(
            "/workspace/xyx/.codex/attachments/"
            "2b2936ce-408b-4e55-8025-f006f69f19e5/psf_NA0.04479.tif"
        ),
    )
    parser.add_argument(
        "--output-dir", default="outputs/covariance_sketch_validation/stage_a"
    )
    parser.add_argument("--num-probes", type=int, default=16)
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    if args.num_probes < 1 or args.num_splits < 1:
        raise ValueError("Probe and split counts must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_tiff_stack(args.raw, intensity_mode="matlab_im2double")
    kernel, cs_preprocess = load_preprocess_cs(args.cs)
    cs_psd = covariance_psd_report(kernel, tuple(int(value) for value in frames.shape[-2:]))
    families = _make_probe_families(
        args.num_probes,
        tuple(int(value) for value in frames.shape[-2:]),
        seed=args.seed,
    )
    summaries, diagonal_summary = _evaluate(
        frames,
        families,
        split_count=args.num_splits,
        seed=args.seed + 1000,
    )
    summaries.sort(
        key=lambda row: (
            not bool(row["passes_stage_a"]),
            -max(float(row["cosine_median"]), float(row["pearson_median"])),
        )
    )
    result = {
        "raw_path": str(Path(args.raw).expanduser().resolve()),
        "frame_count": int(frames.shape[0]),
        "sensor_shape": [int(value) for value in frames.shape[-2:]],
        "probe_count": int(args.num_probes),
        "split_count": int(args.num_splits),
        "seed": int(args.seed),
        "cs_path": str(Path(args.cs).expanduser().resolve()),
        "cs_preprocess": cs_preprocess.to_dict(),
        "cs_psd": cs_psd,
        "diagonal_variance_split_half": diagonal_summary,
        "variants": summaries,
        "stage_a_passed": any(bool(row["passes_stage_a"]) for row in summaries),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    with (output_dir / "probe_reliability.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["stage_a_passed"]:
        print(
            "STAGE_A_BLOCKED: no statistically reliable, informative covariance-action target",
            flush=True,
        )
        raise SystemExit(2)
    print("STAGE_A_PASSED: depth scanning is authorized", flush=True)


if __name__ == "__main__":
    main()
