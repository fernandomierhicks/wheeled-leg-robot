"""Pure jump segmentation and landing inference for decoded robot logs.

The live firmware reports LANDING/HANDOFF directly. Older V12 captures stop at
phase 3 (the former JP_DONE), so this module also reproduces the firmware's
multi-sample IMU detector offline. Keeping this Qt-free makes the result usable
by the Log Analyzer, tests, and future command-line reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_JUMPING = 7

PHASE_CROUCH = 0
PHASE_EXTEND = 1
PHASE_RETRACT = 2
PHASE_LANDING = 3
PHASE_HANDOFF = 4

CURRENT_PHASE_NAMES = {
    PHASE_CROUCH: "CROUCH",
    PHASE_EXTEND: "EXTEND",
    PHASE_RETRACT: "RETRACT",
    PHASE_LANDING: "LANDING",
    PHASE_HANDOFF: "HANDOFF",
}
LEGACY_PHASE_NAMES = {
    PHASE_CROUCH: "CROUCH",
    PHASE_EXTEND: "EXTEND",
    PHASE_RETRACT: "RETRACT",
    PHASE_LANDING: "LEGACY HOLD",
}

DEFAULT_AIRBORNE_ACCEL_Z = -3.0
DEFAULT_LANDING_ACCEL_Z = 1.5
DEFAULT_GYRO_IMPULSE = 2.5
DEFAULT_MIN_AIR_S = 0.16
GYRO_WINDOW_S = 0.012
GYRO_MIN_EVENTS = 2
GYRO_EVENT_EPS = 0.01


@dataclass(frozen=True)
class LandingInference:
    index: int | None
    source: str
    airborne_seen: bool
    gyro_score: float = 0.0


@dataclass(frozen=True)
class JumpEpisode:
    number: int
    i0: int
    i1: int
    phase_entries: tuple[tuple[int, int], ...]
    landing_index: int | None
    landing_source: str
    airborne_seen: bool
    modern_phases: bool

    def first_phase_index(self, phase: int) -> int | None:
        return next((index for value, index in self.phase_entries if value == phase), None)


def _contiguous_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1))
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size - 1)
    return list(zip(starts, ends))


def _phase_entries(phases: np.ndarray, i0: int, i1: int) -> tuple[tuple[int, int], ...]:
    entries = [(int(phases[i0]), i0)]
    changes = np.flatnonzero(np.diff(phases[i0:i1 + 1])) + i0 + 1
    entries.extend((int(phases[index]), int(index)) for index in changes)
    return tuple(entries)


def infer_landing(
    t_s: np.ndarray,
    fields: dict[str, np.ndarray],
    start_index: int,
    end_index: int,
    *,
    airborne_accel_z: float = DEFAULT_AIRBORNE_ACCEL_Z,
    landing_accel_z: float = DEFAULT_LANDING_ACCEL_Z,
    gyro_impulse: float = DEFAULT_GYRO_IMPULSE,
    min_air_s: float = DEFAULT_MIN_AIR_S,
) -> LandingInference:
    """Infer first touchdown after RETRACT using the live detector's evidence.

    Acceleration has no explicit freshness bit in telemetry, so a vector change
    is treated as a newly received BNO086 report. The gyro path accumulates at
    least two non-trivial 3-axis vector changes in the preceding 12 ms, which
    rejects an isolated corrupt sample and is robust to held 500 Hz log rows.
    """
    t_s = np.asarray(t_s, dtype=np.float64)
    if t_s.size == 0 or start_index < 0 or end_index >= t_s.size or start_index > end_index:
        return LandingInference(None, "not detected", False)

    required = (
        "accel_x_ms2", "accel_y_ms2", "accel_z_ms2",
        "roll_rate_rads", "pitch_rate_rads", "yaw_rate_rads",
    )
    if any(name not in fields for name in required):
        return LandingInference(None, "IMU series unavailable", False)

    accel = np.column_stack([np.asarray(fields[name], dtype=np.float64) for name in required[:3]])
    rates = np.column_stack([np.asarray(fields[name], dtype=np.float64) for name in required[3:]])
    accel_changed = np.zeros(t_s.size, dtype=bool)
    gyro_delta = np.zeros(t_s.size, dtype=np.float64)
    if t_s.size > 1:
        accel_changed[1:] = np.any(np.abs(np.diff(accel, axis=0)) > 1e-6, axis=1)
        gyro_delta[1:] = np.linalg.norm(np.diff(rates, axis=0), axis=1)

    airborne_seen = False
    gyro_events: list[tuple[float, float]] = []
    eligible_at = float(t_s[start_index]) + max(0.0, float(min_air_s))

    for index in range(start_index, end_index + 1):
        now = float(t_s[index])
        if accel_changed[index]:
            az = float(accel[index, 2])
            if az <= airborne_accel_z:
                airborne_seen = True
            accel_contact = airborne_seen and az >= landing_accel_z
        else:
            accel_contact = False

        delta = float(gyro_delta[index])
        if delta > GYRO_EVENT_EPS:
            gyro_events.append((now, delta))
        cutoff = now - GYRO_WINDOW_S
        while gyro_events and gyro_events[0][0] < cutoff:
            gyro_events.pop(0)

        score = sum(value for _time, value in gyro_events)
        gyro_contact = len(gyro_events) >= GYRO_MIN_EVENTS and score >= gyro_impulse
        if now >= eligible_at and (accel_contact or gyro_contact):
            if accel_contact and gyro_contact:
                source = "accel + gyro"
            elif accel_contact:
                source = "accel rebound"
            else:
                source = "gyro impulse"
            return LandingInference(index, source, airborne_seen, score)

    return LandingInference(None, "not detected", airborne_seen)


def analyze_jumps(
    t_s: np.ndarray,
    fields: dict[str, np.ndarray],
    *,
    modern_phases: bool | None = None,
    airborne_accel_z: float = DEFAULT_AIRBORNE_ACCEL_Z,
    landing_accel_z: float = DEFAULT_LANDING_ACCEL_Z,
    gyro_impulse: float = DEFAULT_GYRO_IMPULSE,
    min_air_s: float = DEFAULT_MIN_AIR_S,
) -> list[JumpEpisode]:
    """Return every contiguous JUMPING episode and its contact evidence."""
    t_s = np.asarray(t_s, dtype=np.float64)
    if "robot_state" not in fields or "jump_state" not in fields:
        return []
    robot_state = np.asarray(fields["robot_state"], dtype=np.int64)
    phases = np.asarray(fields["jump_state"], dtype=np.int64)
    if robot_state.size != t_s.size or phases.size != t_s.size:
        return []

    episodes: list[JumpEpisode] = []
    for number, (i0, i1) in enumerate(_contiguous_intervals(robot_state == STATE_JUMPING), 1):
        entries = _phase_entries(phases, i0, i1)
        episode_modern = (
            bool(modern_phases) if modern_phases is not None
            else any(phase == PHASE_HANDOFF for phase, _index in entries)
        )
        landing_index = None
        landing_source = "not detected"
        airborne_seen = False

        if episode_modern:
            landing_index = next(
                (index for phase, index in entries if phase == PHASE_LANDING), None)
            landing_source = "live firmware phase" if landing_index is not None else "not detected"
        else:
            retract_index = next(
                (index for phase, index in entries if phase == PHASE_RETRACT), None)
            if retract_index is not None:
                inferred = infer_landing(
                    t_s, fields, retract_index, i1,
                    airborne_accel_z=airborne_accel_z,
                    landing_accel_z=landing_accel_z,
                    gyro_impulse=gyro_impulse,
                    min_air_s=min_air_s,
                )
                landing_index = inferred.index
                landing_source = inferred.source
                airborne_seen = inferred.airborne_seen

        episodes.append(JumpEpisode(
            number=number, i0=i0, i1=i1, phase_entries=entries,
            landing_index=landing_index, landing_source=landing_source,
            airborne_seen=airborne_seen, modern_phases=episode_modern,
        ))
    return episodes


def jump_focus_mask(t_s: np.ndarray, episodes: list[JumpEpisode],
                    before_s: float = 0.25, after_s: float = 0.75) -> np.ndarray:
    """Mask jump episodes plus a small pre-launch/post-recovery context."""
    t_s = np.asarray(t_s, dtype=np.float64)
    mask = np.zeros(t_s.size, dtype=bool)
    epsilon = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(t_s))) if t_s.size else 1.0) * 8.0
    for episode in episodes:
        lo = float(t_s[episode.i0]) - max(0.0, before_s)
        hi = float(t_s[episode.i1]) + max(0.0, after_s)
        mask |= (t_s >= lo - epsilon) & (t_s <= hi + epsilon)
    return mask


def phase_name(phase: int, modern: bool) -> str:
    names = CURRENT_PHASE_NAMES if modern else LEGACY_PHASE_NAMES
    return names.get(int(phase), f"PHASE {int(phase)}")
