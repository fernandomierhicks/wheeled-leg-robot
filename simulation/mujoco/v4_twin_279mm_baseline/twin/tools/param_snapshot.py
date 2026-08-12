"""Validated interchange format shared by parameter pull/push and the twin."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from ..params_control import PARAMS_BY_ID, PARAMS_BY_NAME, validate_values


def _normalise_wire_value(name: str, value: float) -> float:
    """Clamp only float32-sized endpoint round-off from robot/GUI exports.

    Parameter values cross a float32 wire/storage boundary before the GUI writes
    JSON.  A value configured at a schema endpoint can therefore come back a few
    ulps outside that endpoint (for example ``-1.4000001515`` for a ``-1.4``
    minimum).  Keep rejecting real out-of-range values, but do not reject a
    snapshot produced by the project's own GUI.
    """
    definition = PARAMS_BY_NAME[name]
    tolerance = 1e-6 * max(1.0, abs(definition.min), abs(definition.max))
    if value < definition.min and math.isclose(
            value, definition.min, rel_tol=0.0, abs_tol=tolerance):
        return definition.min
    if value > definition.max and math.isclose(
            value, definition.max, rel_tol=0.0, abs_tol=tolerance):
        return definition.max
    return value


def snapshot_from_live(items: list[dict], source: str = "robot") -> dict:
    del source  # existing GUI interchange format deliberately has no metadata
    parameters: dict[str, dict] = {}
    for item in items:
        raw_id = item["id"]
        param_id = int(raw_id, 0) if isinstance(raw_id, str) else int(raw_id)
        definition = PARAMS_BY_ID.get(param_id)
        if definition is None:
            raise ValueError(f"robot reported unknown parameter ID 0x{param_id:04X}")
        name = next(name for name, value in PARAMS_BY_NAME.items() if value.id == param_id)
        reported_name = str(item.get("name", ""))
        if reported_name and reported_name not in (name, definition.symbol):
            # Older firmware can report the C symbol; an unrelated name means
            # firmware/schema drift and must not silently poison a snapshot.
            raise ValueError(
                f"ID 0x{param_id:04X} is {definition.symbol} in schema, "
                f"robot reported {reported_name!r}"
            )
        value = float(item["value"])
        validate_values({name: value})
        parameters[f"0x{param_id:04X}"] = {"name": name, "value": value}
    return parameters


def load_snapshot(path: Path, *, require_schema_match: bool = True) -> dict[str, float]:
    del require_schema_match  # IDs + names are the schema-drift guard in this format
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: snapshot must be an object")
    values: dict[str, float] = {}
    for id_text, item in data.items():
        try:
            param_id = int(str(id_text), 0)
        except ValueError as exc:
            raise ValueError(f"{path}: invalid parameter ID {id_text!r}") from exc
        definition = PARAMS_BY_ID.get(param_id)
        if definition is None:
            raise ValueError(f"{path}: unknown parameter ID {id_text}")
        name = str(item["name"])
        expected = next(key for key, value in PARAMS_BY_NAME.items() if value.id == param_id)
        if name != expected:
            raise ValueError(f"{path}: {id_text} is {expected!r}, not {name!r}")
        values[name] = _normalise_wire_value(name, float(item["value"]))
    validate_values(values)
    return values


def write_snapshot(path: Path, snapshot: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
