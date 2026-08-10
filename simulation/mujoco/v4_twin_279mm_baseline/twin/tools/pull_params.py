"""Pull live Teensy parameters through the GUI and save a twin snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._robot_ctl import request
from .param_snapshot import snapshot_from_live, write_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination JSON snapshot")
    args = parser.parse_args()
    response = request("param_list")
    items = response.get("params") or response.get("parameters")
    if not isinstance(items, list):
        raise RuntimeError("GUI param_list response did not contain a params list")
    snapshot = snapshot_from_live(items, source="gui-live-robot")
    write_snapshot(args.output, snapshot)
    print(json.dumps({"ok": True, "output": str(args.output),
                      "count": len(snapshot)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
