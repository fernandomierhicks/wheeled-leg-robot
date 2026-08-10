"""Dry-run by default hardware executor for the shared scenario JSON format."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from ..params_control import PARAMS_BY_NAME
from ..scenario import Scenario
from ._robot_ctl import request


def _actions(scenario: Scenario) -> list[dict]:
    actions = []
    for name, value in scenario.initial_params.items():
        definition = PARAMS_BY_NAME[name]
        if "readonly" not in definition.flags and "command" not in definition.flags:
            actions.append({"time_s": 0.0, "kind": "param_set", "name": name, "value": value})
    actions.append({"time_s": 0.0, "kind": "log_start",
                    "duration_ms": math.ceil((scenario.duration_s + 1.0) * 1000)})
    for event in scenario.events:
        command = dict(event.params)
        v = command.pop("v_cmd_ms", None)
        omega = command.pop("omega_cmd_rds", None)
        for name, value in command.items():
            definition = PARAMS_BY_NAME[name]
            if "readonly" in definition.flags:
                raise ValueError(f"scenario tries to write readonly {name}")
            actions.append({"time_s": event.time_s, "kind": "param_set",
                            "name": name, "value": value})
        if v is not None or omega is not None:
            actions.append({"time_s": event.time_s, "kind": "motion_set",
                            "v": 0.0 if v is None else v,
                            "omega": 0.0 if omega is None else omega})
    actions.append({"time_s": scenario.duration_s, "kind": "motion_release"})
    actions.append({"time_s": scenario.duration_s, "kind": "log_stop"})
    return sorted(actions, key=lambda item: item["time_s"])


def execute(actions: list[dict]) -> None:
    telemetry = request("telem")
    sample = telemetry.get("telem") or telemetry.get("telemetry") or telemetry
    state = sample.get("robot_state_name") or sample.get("state")
    if state not in ("RUNNING", 3, "3"):
        raise RuntimeError(f"robot must already be RUNNING; reported state={state!r}")
    if int(sample.get("fault_code", 0)):
        raise RuntimeError(f"robot has active fault {sample.get('fault_code')}")
    start = time.monotonic()
    try:
        for action in actions:
            delay = start + float(action["time_s"]) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            kind = action["kind"]
            if kind == "param_set":
                request(kind, action["name"], action["value"])
            elif kind == "motion_set":
                request(kind, action["v"], action["omega"])
            elif kind == "log_start":
                request(kind, action["duration_ms"])
            else:
                request(kind)
    finally:
        # Best effort neutral command even if a timed step or log command fails.
        try:
            request("motion_release")
        except RuntimeError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--execute", action="store_true",
                        help="run on the connected robot (default only prints the plan)")
    parser.add_argument("--acknowledge-motion-risk", action="store_true")
    args = parser.parse_args()
    scenario = Scenario.load(args.scenario)
    actions = _actions(scenario)
    if args.execute:
        if not args.acknowledge_motion_risk:
            raise SystemExit("--execute also requires --acknowledge-motion-risk")
        execute(actions)
    print(json.dumps({"ok": True, "dry_run": not args.execute,
                      "scenario": scenario.name, "actions": actions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
