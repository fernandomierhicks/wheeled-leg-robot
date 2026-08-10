"""Shared JSON scenario definitions used by both twin and hardware runners."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .params_control import validate_values


@dataclass(frozen=True)
class ScenarioEvent:
    time_s: float
    params: dict[str, float]


@dataclass(frozen=True)
class Scenario:
    name: str
    duration_s: float
    stage: str
    initial_params: dict[str, float]
    initial_state: dict[str, float]
    events: tuple[ScenarioEvent, ...]

    @classmethod
    def load(cls, path: Path) -> "Scenario":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        initial_params = {k: float(v) for k, v in data.get("initial_params", {}).items()}
        validate_values(initial_params)
        events = []
        previous = -1.0
        for raw in data.get("events", []):
            time_s = float(raw["time_s"])
            if time_s < previous:
                raise ValueError("scenario events must be sorted by time_s")
            if not 0.0 <= time_s <= float(data["duration_s"]):
                raise ValueError(f"event time {time_s} outside scenario duration")
            params = {k: float(v) for k, v in raw.get("params", {}).items()}
            validate_values(params)
            events.append(ScenarioEvent(time_s=time_s, params=params))
            previous = time_s
        return cls(
            name=str(data["name"]), duration_s=float(data["duration_s"]),
            stage=str(data.get("stage", "lqr")), initial_params=initial_params,
            initial_state={k: float(v) for k, v in data.get("initial_state", {}).items()},
            events=tuple(events),
        )
