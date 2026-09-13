"""Compatibility entry point for the V4 P12 anchor comparison supervisor."""
from __future__ import annotations

import hashlib

from tools import three_way_experiment as old


if not hasattr(old, "sha256_bytes"):
    old.sha256_bytes = lambda value: hashlib.sha256(value).hexdigest()

from tools.run_v4_p12_anchor_compare import main


if __name__ == "__main__":
    main()
