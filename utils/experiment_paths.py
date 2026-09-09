"""Dated, non-overwriting experiment output names (Asia/Shanghai)."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def next_experiment_path(root: Path, name: str, *, now: datetime | None = None) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Experiment name must be a single path component")
    instant = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    if instant.tzinfo is None:
        raise ValueError("An explicit datetime must be timezone-aware")
    day = instant.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    run = 1
    while True:
        candidate = Path(root) / f"{name}_{day}_run{run:02d}"
        log_path = candidate.parent / f"{candidate.name}.launcher.log"
        if not any(
            path.exists() or path.is_symlink() for path in (candidate, log_path)
        ):
            return candidate
        run += 1
