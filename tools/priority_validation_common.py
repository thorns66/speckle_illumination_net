from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "priority_validation_20260907.yaml"


@dataclass(frozen=True)
class Scene:
    scene_id: str
    family: str
    depth_um: int | None


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["acquisition"]["input_indices_one_based"] != list(range(1, 11)):
        raise ValueError("Priority validation requires frames 1--10 as network input")
    if config["acquisition"]["holdout_indices_one_based"] != list(range(11, 101)):
        raise ValueError("Priority validation requires frames 11--100 as holdout")
    if len(config["acquisition"]["repeat_seeds"]) != 3:
        raise ValueError("The standard protocol requires exactly three repeats")
    if config["evaluation"]["primary_threshold"] not in config["evaluation"]["thresholds"]:
        raise ValueError("Primary threshold must be included in thresholds")
    return config_path, config


def resolve_from_repo(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def scenes(config: dict[str, Any]) -> list[Scene]:
    result = [
        Scene(f"points_z{int(depth):03d}", "points", int(depth))
        for depth in config["scenes"]["point_depths_um"]
    ]
    result.extend(
        Scene(f"lines_z{int(depth):03d}", "lines", int(depth))
        for depth in config["scenes"]["line_depths_um"]
    )
    result.append(Scene(str(config["scenes"]["axial_pair_scene"]), "axial_pairs", None))
    if len(result) != 11 or len({scene.scene_id for scene in result}) != 11:
        raise ValueError("The standard protocol must define eleven unique scenes")
    return result


def task_id(scene: Scene, repeat: int) -> str:
    return f"{scene.scene_id}_r{repeat:02d}"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_inventory() -> dict[int, dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result: dict[int, dict[str, Any]] = {}
    for line in output.strip().splitlines():
        index, uuid, name, used, total, utilization = [part.strip() for part in line.split(",")]
        result[int(index)] = {
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
            "utilization_percent": int(utilization),
        }
    return result


def select_gpus(
    requested: str,
    config: dict[str, Any],
    *,
    allow_busy: bool = False,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    inventory = gpu_inventory()
    if requested == "all":
        candidates = sorted(inventory)
    else:
        candidates = [int(item) for item in requested.split(",")]
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("--gpus must contain one or more unique physical indices")
    minimum_free = float(config["runtime"]["min_free_gib"]) * 1024
    selected = []
    for index in candidates:
        if index not in inventory:
            raise ValueError(f"GPU {index} is unavailable")
        item = inventory[index]
        free = int(item["memory_total_mib"]) - int(item["memory_used_mib"])
        idle = int(item["memory_used_mib"]) < 1024 and int(item["utilization_percent"]) <= 10
        if free < minimum_free:
            continue
        if idle or allow_busy:
            selected.append(index)
    if not selected:
        raise RuntimeError("No requested GPU satisfies the free-memory and occupancy policy")
    return selected, inventory


def source_manifest(paths: Iterable[Path]) -> dict[str, str]:
    unique = sorted({Path(path).resolve() for path in paths}, key=str)
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")
    return {str(path): sha256(path) for path in unique}


def check_source_manifest(manifest: dict[str, str]) -> list[str]:
    return [path for path, digest in manifest.items() if not Path(path).is_file() or sha256(path) != digest]

