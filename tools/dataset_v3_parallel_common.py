"""Pure job partitioning and explicit physical-GPU authorization helpers."""
from pathlib import Path

GPU_INDICES = tuple(range(6))


def partition_jobs(jobs: list[dict], slots=GPU_INDICES) -> list[list[dict]]:
    if tuple(slots) != GPU_INDICES:
        raise ValueError("This authorization is specifically physical GPUs 0..5")
    owners = set()
    result = [[] for _ in slots]
    loads = [0] * len(slots)
    for job in sorted(jobs, key=lambda j: (-int(j["kind"] == "subset"), j["index"])):
        kind, index = job["kind"], job["index"]
        maximum = {"sensor": 100, "frame": 100, "subset": 10}.get(kind)
        if maximum is None or type(index) is not int or not 1 <= index <= maximum:
            raise ValueError(f"Invalid job {job}")
        key = (kind, index)
        if key in owners:
            raise ValueError(f"Duplicate output ownership: {key}")
        owners.add(key)
        slot = min(range(len(slots)), key=lambda i: (loads[i], i))
        result[slot].append(dict(job))
        loads[slot] += 2 if kind == "subset" else 1
    return result


def missing_jobs(folder: Path, stage: str) -> list[dict]:
    if stage == "sensor":
        definitions = [("sensor", "sensor_frames", "frame_%03d.mat", 100)]
    elif stage == "reconstruct":
        definitions = [
            ("frame", "recon_frames", "frame_%03d.mat", 100),
            ("subset", "subsets", "subset_%02d.mat", 10),
        ]
    else:
        raise ValueError(stage)
    return [
        {"kind": kind, "index": index}
        for kind, subdir, pattern, count in definitions
        for index in range(1, count + 1)
        if not (folder / subdir / (pattern % index)).exists()
    ]


def validate_gpu(snapshot: dict, expected: dict) -> bool:
    index = snapshot["physical_index"]
    if (
        index not in GPU_INDICES
        or index != expected["physical_index"]
        or snapshot["uuid"] != expected["uuid"]
        or "A40" not in snapshot["name"]
    ):
        raise ValueError("GPU identity mismatch; no fallback allowed")
    # Sharing exemption is only for physical 0, previously explicitly allowed.
    if index != 0 and snapshot["compute_processes"]:
        return False
    return snapshot["free_mib"] >= 24 * 1024 and snapshot["utilization_percent"] <= 5
