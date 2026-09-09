"""Robust entry point for the dated spinach-root exploratory run."""
from __future__ import annotations

import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
existing = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = str(ROOT) if not existing else str(ROOT) + os.pathsep + existing
runpy.run_path(str(ROOT / "tools" / "run_spinach_root_exploratory_v2.py"), run_name="__main__")

