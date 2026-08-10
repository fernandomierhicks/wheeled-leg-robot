"""Generate the twin's control-parameter namespace from firmware schema.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCHEMA = ROOT / "firmware" / "robot_teensy" / "protocol" / "schema.json"
OUTPUT = Path(__file__).with_name("params_control.py")


def render() -> str:
    raw = SCHEMA.read_bytes()
    schema = json.loads(raw)
    params = sorted(schema["parameters"], key=lambda item: item["id"])
    records = {
        p["name"]: {
            "id": p["id"],
            "symbol": p["symbol"],
            "group": p["group"],
            "default": p["default"],
            "min": p["min"],
            "max": p["max"],
            "flags": p["flags"],
            "description": p["description"],
        }
        for p in params
    }
    literal = repr(records)
    digest = hashlib.sha256(raw).hexdigest()
    return f'''"""Generated from firmware/robot_teensy/protocol/schema.json. DO NOT EDIT."""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_SHA256 = {digest!r}


@dataclass(frozen=True)
class ParamDef:
    id: int
    symbol: str
    group: str
    default: float
    min: float
    max: float
    flags: frozenset[str]
    description: str


_RAW = {literal}
PARAMS_BY_NAME = {{
    name: ParamDef(
        id=item["id"], symbol=item["symbol"], group=item["group"],
        default=float(item["default"]), min=float(item["min"]),
        max=float(item["max"]), flags=frozenset(item["flags"]),
        description=item["description"],
    )
    for name, item in _RAW.items()
}}
PARAMS_BY_ID = {{definition.id: definition for definition in PARAMS_BY_NAME.values()}}


def default_values() -> dict[str, float]:
    return {{name: definition.default for name, definition in PARAMS_BY_NAME.items()}}


def validate_values(values: dict[str, float], *, require_known: bool = True) -> None:
    for name, value in values.items():
        definition = PARAMS_BY_NAME.get(name)
        if definition is None:
            if require_known:
                raise KeyError(f"unknown firmware parameter: {{name}}")
            continue
        number = float(value)
        if not definition.min <= number <= definition.max:
            raise ValueError(
                f"{{name}}={{number}} outside schema bounds "
                f"[{{definition.min}}, {{definition.max}}]"
            )
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != expected:
            print(f"stale generated artifact: {OUTPUT.relative_to(ROOT)}")
            return 1
        return 0
    OUTPUT.write_text(expected, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
