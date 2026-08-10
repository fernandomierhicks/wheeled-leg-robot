"""Small checked wrapper around the GUI's existing robot_ctl transport."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[5]
ROBOT_CTL = ROOT / "software" / "gui" / "tools" / "robot_ctl.py"


def request(*arguments: object) -> dict:
    command = [sys.executable, str(ROBOT_CTL), *(str(value) for value in arguments)]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise RuntimeError(f"robot_ctl returned invalid JSON: {detail}") from exc
    if completed.returncode or not response.get("ok"):
        raise RuntimeError(response.get("error") or json.dumps(response))
    return response
