"""Read LOGnnnn.PARAMS files and align parameter values to WLOG samples.

The sidecar and WLOG use the same uint32 ``micros()`` clock.  A sidecar starts
with one DUMP row per non-command parameter and then records CHANGE rows while
the log is active.  This module deliberately has no Qt dependency so both the
GUI and offline analysis tools can reuse it.
"""

from __future__ import annotations

import csv
import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ParamEvent:
    t_micros: int
    event: str
    param_id: int
    name: str
    value: float


class ParamSidecar:
    def __init__(self, path: Path, events: list[ParamEvent]):
        self.path = Path(path)
        self.events = tuple(events)
        by_name: dict[str, list[ParamEvent]] = {}
        for event in events:
            by_name.setdefault(event.name, []).append(event)
        self._by_name = by_name

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._by_name)

    def initial_value(self, name: str) -> float | None:
        events = self._by_name.get(name)
        return events[0].value if events else None

    def series(self, name: str, sample_t_micros: np.ndarray) -> np.ndarray | None:
        """Return a zero-order-held value at each WLOG sample.

        The first event is treated as the initial value.  Subsequent events are
        aligned using signed uint32 clock deltas, including across micros()
        rollover.
        """
        events = self._by_name.get(name)
        if not events:
            return None

        sample_t_micros = np.asarray(sample_t_micros, dtype=np.uint32)
        if sample_t_micros.size == 0:
            return np.asarray([], dtype=np.float64)

        values = np.full(sample_t_micros.size, events[0].value, dtype=np.float64)
        start = int(sample_t_micros[0])
        sample_elapsed_us = np.diff(sample_t_micros).astype(np.uint32).astype(np.uint64)
        sample_elapsed_us = np.concatenate((np.asarray([0], dtype=np.uint64),
                                            np.cumsum(sample_elapsed_us, dtype=np.uint64)))

        for event in events[1:]:
            # Map uint32 to the closest signed delta from the first WLOG tick.
            delta = ((int(event.t_micros) - start + (1 << 31)) % (1 << 32)) - (1 << 31)
            index = 0 if delta <= 0 else int(np.searchsorted(sample_elapsed_us, delta, side="left"))
            if index < values.size:
                values[index:] = event.value
        return values


def find_matching_sidecar(wlog_path: Path) -> Path | None:
    """Find the case-insensitive same-stem .PARAMS companion, if present."""
    wlog_path = Path(wlog_path)
    direct = wlog_path.with_suffix(".PARAMS")
    if direct.exists():
        return direct
    stem = wlog_path.stem.casefold()
    try:
        return next(
            path for path in wlog_path.parent.iterdir()
            if path.is_file() and path.stem.casefold() == stem
            and path.suffix.casefold() == ".params"
        )
    except (OSError, StopIteration):
        return None


def load_param_sidecar(path: Path) -> ParamSidecar:
    path = Path(path)
    events: list[ParamEvent] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = csv.reader(line for line in handle if not line.lstrip().startswith("#"))
        for line_number, row in enumerate(rows, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 5:
                raise ValueError(f"{path}: line {line_number}: expected 5 columns, got {len(row)}")
            try:
                events.append(ParamEvent(
                    t_micros=int(row[0]), event=row[1].strip().upper(),
                    param_id=int(row[2]), name=row[3].strip(), value=float(row[4]),
                ))
            except ValueError as exc:
                raise ValueError(f"{path}: line {line_number}: invalid parameter event") from exc
    if not events:
        raise ValueError(f"{path}: no parameter events")
    return ParamSidecar(path, events)


def load_matching_sidecar(wlog_path: Path) -> ParamSidecar | None:
    path = find_matching_sidecar(wlog_path)
    return load_param_sidecar(path) if path is not None else None


def load_host_param_sidecar(path: Path) -> ParamSidecar | None:
    """Build a parameter timeline from PARAM_REPORT records in a host log.

    Host packets do not carry firmware micros() directly, so each report is
    aligned to the most recent captured TELEM timestamp on the monotonic host
    clock. HostLogger requests a complete dump immediately after capture starts.
    """
    path = Path(path)
    anchors: list[tuple[int, int]] = []
    reports: list[tuple[int, dict]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        lines = handle.readlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith(("\n", "\r")):
                continue
            raise ValueError(f"{path}: line {index + 1}: malformed JSON") from exc
        if not isinstance(record, dict):
            continue
        host_ns = record.get("host_monotonic_ns")
        if host_ns is None:
            continue
        if record.get("ptype") == 0x01 and "timestamp_ms" in record:
            t_micros = (int(record["timestamp_ms"]) * 1000) & 0xFFFFFFFF
            anchors.append((int(host_ns), t_micros))
        elif record.get("ptype") == 0x06 and all(
                key in record for key in ("param_id", "param_name", "param_value")):
            reports.append((int(host_ns), record))

    if not reports or not anchors:
        return None
    anchors.sort(key=lambda item: item[0])
    anchor_host = [item[0] for item in anchors]
    last_values: dict[str, float] = {}
    events: list[ParamEvent] = []
    for host_ns, record in reports:
        anchor_index = max(0, bisect_right(anchor_host, host_ns) - 1)
        name = str(record["param_name"])
        value = float(record["param_value"])
        previous = last_values.get(name)
        if previous is not None and value == previous:
            # The GUI may poll a parameter while capture is active. Repeated
            # reports are observations, not changes, and would otherwise
            # clutter the analyzer's parameter-change timeline.
            continue
        events.append(ParamEvent(
            t_micros=anchors[anchor_index][1],
            event="CHANGE" if previous is not None else "DUMP",
            param_id=int(record["param_id"]),
            name=name, value=value,
        ))
        last_values[name] = value
    return ParamSidecar(path, events)


def active_profile_series(sidecar: ParamSidecar | None, suffix: str,
                          active_profile: np.ndarray,
                          sample_t_micros: np.ndarray) -> np.ndarray | None:
    """Select profile1/2/3_<suffix> using per-sample active_profile telemetry."""
    if sidecar is None:
        return None
    profile_values = []
    for profile_number in (1, 2, 3):
        values = sidecar.series(f"profile{profile_number}_{suffix}", sample_t_micros)
        if values is None:
            return None
        profile_values.append(values)
    profiles = np.clip(np.asarray(active_profile, dtype=np.int64), 0, 2)
    stacked = np.vstack(profile_values)
    return stacked[profiles, np.arange(profiles.size)]
