"""Shared runtime paths for GUI logs and diagnostic captures."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_ROOT = REPO_ROOT / "data" / "logs"
RUNS_DIR = LOG_ROOT / "runs"
DIAGNOSTICS_DIR = LOG_ROOT / "diagnostics"


def ensure_log_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def new_run_dir(prefix: str) -> Path:
    """Create a collision-proof UTC-stamped run bundle directory."""
    ensure_log_dirs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = RUNS_DIR / f"{stamp}_{prefix}"
    suffix = 1
    while path.exists():
        path = RUNS_DIR / f"{stamp}_{prefix}_{suffix}"
        suffix += 1
    path.mkdir(parents=False)
    return path


ensure_log_dirs()
