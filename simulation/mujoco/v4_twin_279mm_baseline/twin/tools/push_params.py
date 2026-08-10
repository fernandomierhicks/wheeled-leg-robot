"""Diff and optionally push a strictly allowlisted controller-gain snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..firmware_control import GAIN_ALLOWLIST
from ..params_control import PARAMS_BY_NAME
from ._robot_ctl import request
from .param_snapshot import load_snapshot, snapshot_from_live


def planned_changes(snapshot: dict[str, float], live: dict[str, float]) -> list[dict]:
    refused = sorted(
        name for name, value in snapshot.items()
        if name not in GAIN_ALLOWLIST
        and (name not in live or abs(live[name] - value) > max(1e-9, abs(value) * 1e-7))
    )
    if refused:
        raise ValueError(
            "snapshot contains non-allowlisted parameters: " + ", ".join(refused)
        )
    changes = []
    for name in sorted(set(snapshot) & GAIN_ALLOWLIST):
        definition = PARAMS_BY_NAME[name]
        if "readonly" in definition.flags or "command" in definition.flags:
            raise ValueError(f"{name} is not a pushable persistent gain")
        old = live.get(name)
        new = snapshot[name]
        if old is None or abs(old - new) > max(1e-9, abs(new) * 1e-7):
            changes.append({"name": name, "id": definition.id,
                            "old": old, "new": new})
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--apply", action="store_true",
                        help="perform writes; without this flag the command is a dry run")
    args = parser.parse_args()
    desired = load_snapshot(args.snapshot)
    response = request("param_list")
    live_snapshot = snapshot_from_live(response.get("params") or response.get("parameters") or [])
    live = {item["name"]: item["value"] for item in live_snapshot.values()}
    changes = planned_changes(desired, live)
    result = {"ok": True, "dry_run": not args.apply, "changes": changes}
    if args.apply:
        for change in changes:
            request("param_set", change["name"], change["new"])
        result["applied"] = len(changes)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
