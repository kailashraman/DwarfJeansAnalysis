"""Shared helpers for verifying staged data folders under data/<bibkey>/."""

from __future__ import annotations

import hashlib
from pathlib import Path


def verify_checksums(folder: Path) -> None:
    """Verify every file listed in `folder/checksums.sha256`. Raise on mismatch."""
    chk = folder / "checksums.sha256"
    if not chk.exists():
        raise FileNotFoundError(
            f"Missing {chk}. Stage the folder per data_sources.md "
            "(copy + PROVENANCE.md + checksums.sha256) before re-running."
        )
    for line in chk.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")  # `sha256sum -b` prefix
        path = folder / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"SHA256 mismatch for {path}: expected {expected}, got {actual}. "
                "Re-stage from the upstream source."
            )
