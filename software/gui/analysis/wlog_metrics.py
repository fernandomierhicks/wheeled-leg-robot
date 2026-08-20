"""wlog_metrics.py — shared hardware-log metrics (LQR / vel-PI / yaw-PI).

Pure Python/numpy, no Qt import — usable headless from tools/analyze_hw_run.py
and from inside the GUI process (tabs/log_analyzer_tab.py) alike, so a number
Fernando sees on screen in the Log Analyzer tab is the exact number that drove
Claude's accept/reject decision, not a re-derivation of it (see tuning.md §4).

Decodes a .wlog with its own header+record loop (mirrors tools/wlog_to_csv.py)
rather than tabs.log_playback.LogReader, so this module stays Qt-free — same
decode logic either way (tabs.telem_format.decode_telem_full), just without
pulling in LogReader's QObject base.
"""

import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # software/gui/
from tabs.telem_format import TELEM_VERSION, decode_telem_full  # noqa: E402

# WlogHeader (32 bytes, packed) — shared/comm_protocol.h
_FMT_HEADER = "<8sBBHHI14s"
_T_MICROS_SIZE = 4  # uint32_t t_micros — first field of LogRecord

STAGES = ("lqr", "vel_pi", "yaw_pi")

# Settle-time detector — same shape as the sim's (simulation/mujoco/master_sim/
# params.py: settle_deg=2.0, settle_window_s=0.5), re-picked here if real
# sensor noise makes it too tight/loose for hardware.
SETTLE_DEG = 2.0
SETTLE_WINDOW_S = 0.5

# Health flag bits — shared/comm_protocol.h HEALTH_FLAG_*
_HEALTH_FRACTION_FLAGS = {
    "vel_pi_sat_frac":       1 << 7,   # HEALTH_VEL_PI_SAT
    "yaw_pi_sat_frac":       1 << 8,   # HEALTH_YAW_PI_SAT
    "wm_l_vel_limited_frac": 1 << 9,   # HEALTH_WM_L_VEL_LIMITED
    "wm_r_vel_limited_frac": 1 << 10,  # HEALTH_WM_R_VEL_LIMITED
    "loop_overrun_frac":     1 << 11,  # HEALTH_LOOP_OVERRUN
}

FAULT_NONE = 0x00

# Safety-maxima defaults. wheel_vel_limit mirrors PARAM_WHEEL_VEL_LIMIT_TURNS_S's
# firmware compile-time default (param_registry.cpp) — pass the live value to
# evaluate()/check_safety() if it's been changed on the device. The hip ceiling
# is a real overtorque limit, not a "should be idle" epsilon — hips are
# free-running and holding/dynamic torque is expected. AK45-10 ratings: 2.5 N·m
# rated continuous, 7 N·m mechanical peak (the MIT protocol encodes ±8). 4 N·m
# sits between them and is the same physical level the old 10 "A" limit meant
# before the MIT torque scale was corrected — a peak-excursion ceiling. It says
# nothing about *continuous* duty: compare the RMS against the 2.5 N·m rating
# for that, since a run can stay under 4 N·m and still cook the motor.
DEFAULT_WHEEL_VEL_LIMIT_TURNS_S = 3.0
HIP_TORQUE_LIMIT_NM = 4.0

REJECT_PENALTY = 200.0  # matches sim's FALL_PENALTY / W_FALL magnitude

# Fields pulled into per-run numpy arrays — everything the plot flavors (§4c)
# and the metrics below need. Add here before consuming a new telemetry field.
_SCALAR_FIELDS = [
    "timestamp_ms", "pitch_rad", "pitch_rate_rads", "roll_rad", "roll_rate_rads",
    "yaw_rad", "yaw_rate_rads", "accel_x_ms2", "accel_y_ms2", "accel_z_ms2",
    "wheel_vel_avg", "wm_l_vel_turns_s", "wm_r_vel_turns_s",
    "hip_l_torque_nm", "hip_r_torque_nm", "hip_l_vel_rads", "hip_r_vel_rads",
    "whl_tau_l", "whl_tau_r",
    "hip_l_pos_rad", "hip_r_pos_rad", "hip_l_cmd_pos_rad", "hip_r_cmd_pos_rad",
    "theta_ref", "v_ref", "omega_cmd_rds", "tau_sym", "tau_yaw", "pitch_trim_rad",
    "health_flags", "fault_code", "robot_state", "jump_state", "standup_state",
    "loop_count", "active_profile",
]

# gain_sched_alpha is present in V10+ telemetry. It remains optional here so
# older captures decode as None/"N/A" instead of failing.
_ALPHA_KEY = "gain_sched_alpha"

# ── TELEM_VERSION 11 back-compatibility ──────────────────────────────────────
# V11 and V12 have byte-identical layouts; only the two hip feedback fields
# changed meaning. They were decoded as "current in amps" over a ±20 full scale
# when the AK45-10 MIT reply is really shaft torque over ±8 N·m, so a V11
# capture holds the same physical quantity at 20/8 = 2.5× the true value.
# Rescale and rename it rather than rejecting the file, so runs captured before
# the fix stay comparable with runs captured after it.
_COMPAT_TELEM_VERSION = 11
_V11_TORQUE_SCALE = 8.0 / 20.0
_V11_RENAMES = {
    "hip_l_current_a": "hip_l_torque_nm",
    "hip_r_current_a": "hip_r_torque_nm",
}


def _telem_version_supported(version: int) -> bool:
    return version in (TELEM_VERSION, _COMPAT_TELEM_VERSION)


def _upgrade_v11_record(record: dict) -> dict:
    """Convert a V11 record's hip fields in place to V12 names and units.

    Two shapes reach this: a `.wlog` record, already decoded by the current
    decoder so it carries the V12 *names* with V11 *values*; and a host `.jsonl`
    record, written verbatim by an older GUI so it still carries the V11 names.
    """
    for old, new in _V11_RENAMES.items():
        if old in record:
            record[new] = record.pop(old) * _V11_TORQUE_SCALE
        elif new in record:
            record[new] = record[new] * _V11_TORQUE_SCALE
    return record


@dataclass
class DecodedRun:
    path: Path
    telem_version: int
    sample_rate_hz: int
    count: int
    t_micros: np.ndarray        # raw uint32 firmware clock, shared with .PARAMS
    t_s: np.ndarray             # seconds since first record (from t_micros)
    fields: dict                # field name -> np.ndarray, see _SCALAR_FIELDS
    has_gain_sched_alpha: bool
    source_kind: str = "wlog"


def decode_wlog(path) -> DecodedRun:
    """Decode a .wlog file into per-field numpy arrays."""
    path = Path(path)
    with open(path, "rb") as f:
        header = f.read(32)
        if len(header) < 32:
            raise ValueError(f"{path}: too short for WlogHeader")
        magic, _fmt_ver, telem_ver, record_size, sample_hz, _start_ms, _reserved = \
            struct.unpack(_FMT_HEADER, header)
        magic = magic.split(b"\x00", 1)[0].decode("ascii", errors="replace")
        if magic != "WLRLOG":
            raise ValueError(f"{path}: bad magic {magic!r} — not a .wlog file")
        if not _telem_version_supported(telem_ver):
            raise ValueError(
                f"{path}: captured with TELEM_VERSION {telem_ver}, this decoder is "
                f"TELEM_VERSION {TELEM_VERSION} — recapture after reflashing, "
                "fields would otherwise be misread"
            )

        telem_size = record_size - _T_MICROS_SIZE
        t_micros_list = []
        per_field = {name: [] for name in _SCALAR_FIELDS}
        alpha_vals = []
        has_alpha = None
        n = 0
        while True:
            rec = f.read(record_size)
            if len(rec) < record_size:
                break
            t_micros = struct.unpack_from("<I", rec, 0)[0]
            telem = decode_telem_full(rec[_T_MICROS_SIZE:_T_MICROS_SIZE + telem_size])
            if telem_ver == _COMPAT_TELEM_VERSION:
                # Layout is identical, so the V12 decoder reads a V11 record
                # correctly; only the hip fields' units need converting.
                _upgrade_v11_record(telem)
            if has_alpha is None:
                has_alpha = _ALPHA_KEY in telem
            t_micros_list.append(t_micros)
            for name in _SCALAR_FIELDS:
                per_field[name].append(telem[name])
            if has_alpha:
                alpha_vals.append(telem[_ALPHA_KEY])
            n += 1

    if n == 0:
        raise ValueError(f"{path}: no records decoded")

    t_micros_arr = np.asarray(t_micros_list, dtype=np.uint32)
    # Accumulate uint32 deltas so logs remain monotonic across a micros() wrap.
    if t_micros_arr.size > 1:
        deltas_us = np.diff(t_micros_arr).astype(np.uint32).astype(np.float64)
        elapsed_us = np.concatenate(([0.0], np.cumsum(deltas_us)))
    else:
        elapsed_us = np.zeros(t_micros_arr.size, dtype=np.float64)
    t_s = elapsed_us / 1e6

    fields = {name: np.asarray(vals, dtype=np.float64) for name, vals in per_field.items()}
    if has_alpha:
        fields[_ALPHA_KEY] = np.asarray(alpha_vals, dtype=np.float64)

    return DecodedRun(
        path=path, telem_version=telem_ver, sample_rate_hz=sample_hz, count=n,
        t_micros=t_micros_arr, t_s=t_s, fields=fields,
        has_gain_sched_alpha=bool(has_alpha),
    )


def _host_json_records(path: Path):
    """Yield JSON objects while tolerating a crash-truncated final line."""
    with path.open("r", encoding="utf-8-sig") as handle:
        lines = handle.readlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            is_partial_tail = index == len(lines) - 1 and not line.endswith(("\n", "\r"))
            if is_partial_tail:
                continue
            raise ValueError(f"{path}: line {index + 1}: malformed JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}: line {index + 1}: expected a JSON object")
        yield index + 1, record


def decode_hostlog(path) -> DecodedRun:
    """Decode a GUI host JSONL capture into the shared run representation."""
    path = Path(path)
    per_field = {name: [] for name in _SCALAR_FIELDS}
    alpha_vals = []
    timestamp_ms_raw = []
    elapsed_ms = []
    previous_timestamp = None
    elapsed = 0
    telem_version = None
    has_alpha = None
    previous_identity = None

    for line_number, record in _host_json_records(path):
        if record.get("ptype") != 0x01:
            continue
        version = record.get("version")
        if version is None:
            raise ValueError(f"{path}: line {line_number}: TELEM record has no version")
        if not _telem_version_supported(int(version)):
            raise ValueError(
                f"{path}: captured with TELEM_VERSION {version}, this decoder is "
                f"TELEM_VERSION {TELEM_VERSION}"
            )
        if telem_version is None:
            telem_version = int(version)
        elif int(version) != telem_version:
            raise ValueError(f"{path}: line {line_number}: mixed TELEM versions")
        if int(version) == _COMPAT_TELEM_VERSION:
            _upgrade_v11_record(record)

        try:
            timestamp = int(record["timestamp_ms"]) & 0xFFFFFFFF
            identity = (timestamp, int(record.get("loop_count", -1)))
            if identity == previous_identity:
                continue
            if previous_timestamp is not None:
                delta = (timestamp - previous_timestamp) & 0xFFFFFFFF
                if delta == 0:
                    continue
                if delta >= 0x80000000:
                    raise ValueError(
                        f"{path}: line {line_number}: firmware time moved backward; "
                        "the capture may contain a source switch or reordered telemetry"
                    )
                elapsed += delta
            for name in _SCALAR_FIELDS:
                per_field[name].append(record[name])
        except KeyError as exc:
            raise ValueError(
                f"{path}: line {line_number}: TELEM record missing field {exc.args[0]!r}"
            ) from exc

        if has_alpha is None:
            has_alpha = _ALPHA_KEY in record
        if has_alpha:
            if _ALPHA_KEY not in record:
                raise ValueError(f"{path}: line {line_number}: inconsistent gain schedule field")
            alpha_vals.append(record[_ALPHA_KEY])
        timestamp_ms_raw.append(timestamp)
        elapsed_ms.append(elapsed)
        previous_timestamp = timestamp
        previous_identity = identity

    count = len(timestamp_ms_raw)
    if count == 0:
        raise ValueError(f"{path}: no TELEM records decoded")

    timestamp_arr = np.asarray(timestamp_ms_raw, dtype=np.uint64)
    # timestamp_ms and micros() share the firmware uptime epoch. The derived
    # value has millisecond precision but preserves the uint32 clock domain
    # expected by parameter-sidecar alignment.
    t_micros = ((timestamp_arr * 1000) & 0xFFFFFFFF).astype(np.uint32)
    t_s = np.asarray(elapsed_ms, dtype=np.float64) / 1000.0
    positive_dt = np.diff(t_s)
    positive_dt = positive_dt[positive_dt > 0]
    sample_rate_hz = max(1, int(round(1.0 / np.median(positive_dt)))) if positive_dt.size else 1

    fields = {name: np.asarray(values, dtype=np.float64) for name, values in per_field.items()}
    if has_alpha:
        fields[_ALPHA_KEY] = np.asarray(alpha_vals, dtype=np.float64)
    return DecodedRun(
        path=path,
        telem_version=int(telem_version),
        sample_rate_hz=sample_rate_hz,
        count=count,
        t_micros=t_micros,
        t_s=t_s,
        fields=fields,
        has_gain_sched_alpha=bool(has_alpha),
        source_kind="host",
    )


def decode_run(path) -> DecodedRun:
    """Decode any supported controller-run log by extension."""
    path = Path(path)
    if path.suffix.casefold() == ".jsonl":
        return decode_hostlog(path)
    if path.suffix.casefold() == ".wlog":
        return decode_wlog(path)
    raise ValueError(f"{path}: supported log types are .WLOG and .jsonl")


def _settle_time(t: np.ndarray, err_deg: np.ndarray, settle_deg: float,
                  settle_window_s: float, duration_s: float) -> float:
    """First timestamp after which err_deg stays below settle_deg for at
    least settle_window_s continuously. Falls back to full duration (never
    settled) — same convention as the sim's sim_loop.py."""
    under = err_deg < settle_deg
    streak_start = None
    for i in range(len(t)):
        if under[i]:
            if streak_start is None:
                streak_start = t[i]
            elif t[i] - streak_start >= settle_window_s:
                return float(streak_start)
        else:
            streak_start = None
    return float(duration_s)


def _oscillation_check(t: np.ndarray, pitch_rad: np.ndarray, settle_time_s: float,
                        sample_rate_hz: int) -> tuple[float, float | None]:
    """Zero-crossing rate + dominant FFT frequency of pitch after the initial
    transient — a damping sanity check real hardware needs that the sim
    doesn't (no actuator/sensor noise there to distinguish from genuine
    oscillation)."""
    mask = t >= settle_time_s
    seg_t = t[mask]
    seg = pitch_rad[mask]
    if seg.size < 4 or seg_t[-1] <= seg_t[0]:
        return 0.0, None
    seg_t, unique_indices = np.unique(seg_t, return_index=True)
    seg = seg[unique_indices]
    dt = 1.0 / sample_rate_hz
    uniform_t = np.arange(seg_t[0], seg_t[-1] + 0.5 * dt, dt)
    if uniform_t.size < 4:
        return 0.0, None
    uniform_seg = np.interp(uniform_t, seg_t, seg)
    uniform_seg = uniform_seg - np.mean(uniform_seg)
    signs = np.sign(uniform_seg)
    signs[signs == 0] = 1
    crossings = int(np.sum(signs[:-1] != signs[1:]))
    span_s = float(uniform_t[-1] - uniform_t[0])
    zero_cross_hz = crossings / span_s if span_s > 0 else 0.0

    freqs = np.fft.rfftfreq(uniform_seg.size, d=dt)
    mag = np.abs(np.fft.rfft(uniform_seg))
    mag[0] = 0.0  # drop DC bin
    dominant_freq_hz = float(freqs[int(np.argmax(mag))]) if mag.size > 1 else None
    return zero_cross_hz, dominant_freq_hz


def compute_metrics(run: DecodedRun) -> dict:
    """Stage-agnostic metrics computed from one decoded run. Independent of
    which stage (LQR/vel-PI/yaw-PI) the trial was — the caller picks which
    numbers matter via the fitness_* functions below."""
    f = run.fields
    t = run.t_s
    duration_s = float(t[-1]) if run.count > 0 else 0.0

    def integral(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        if values.size <= 1 or duration_s <= 0:
            return float(values[0] / run.sample_rate_hz) if values.size else 0.0
        return float(np.trapezoid(values, t))

    def time_mean(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        if values.size <= 1 or duration_s <= 0:
            return float(values[0]) if values.size else 0.0
        return integral(values) / duration_s

    pitch_rad = f["pitch_rad"]
    pitch_rate_rads = f["pitch_rate_rads"]
    theta_ref = f["theta_ref"]
    pitch_trim_rad = f["pitch_trim_rad"]

    # Must match control_loop.cpp's LQR x0 exactly: the regulated quantity is
    # pitch - theta_ref - pitch_trim, not pitch - theta_ref. pitch_trim
    # (lqr_pitch_trim_ret/ext, alpha-scheduled) offsets the pitch target so zero
    # velocity holds at the true balance lean when the CG isn't over the axle,
    # and it is NOT small: the measured retracted trim is about -8 deg. Omitting
    # it put a near-constant bias into every error metric below, which swamped
    # the differences between candidate gain sets these metrics exist to rank.
    # Hardware exposes no equivalent of the sim's separate pitch_ff term, so
    # this is the whole setpoint (formulas otherwise ported from sim_loop.py).
    pitch_err_rad = pitch_rad - theta_ref - pitch_trim_rad
    pitch_err_deg = np.degrees(np.abs(pitch_err_rad))

    rms_pitch_deg = float(np.sqrt(time_mean(pitch_err_deg ** 2)))
    ise_pitch = integral(pitch_err_rad ** 2)
    ise_pitch_rate = integral(pitch_rate_rads ** 2)
    rms_pitch_rate_dps = float(np.sqrt(time_mean(np.degrees(pitch_rate_rads) ** 2)))
    max_pitch_deg = float(np.max(pitch_err_deg))

    settle_time_s = _settle_time(t, pitch_err_deg, SETTLE_DEG, SETTLE_WINDOW_S, duration_s)
    zero_cross_hz, dominant_freq_hz = _oscillation_check(t, pitch_rad, settle_time_s, run.sample_rate_hz)

    health = f["health_flags"].astype(np.int64)
    health_fractions = {
        name: time_mean(((health & bit) != 0).astype(np.float64))
        for name, bit in _HEALTH_FRACTION_FLAGS.items()
    }

    fault_code = f["fault_code"].astype(np.int64)
    fault_fired = bool(np.any(fault_code != FAULT_NONE))
    faults_seen = sorted(int(c) for c in np.unique(fault_code) if c != FAULT_NONE)

    max_hip_l_torque_nm = float(np.max(np.abs(f["hip_l_torque_nm"])))
    max_hip_r_torque_nm = float(np.max(np.abs(f["hip_r_torque_nm"])))
    max_wm_l_vel_turns_s = float(np.max(np.abs(f["wm_l_vel_turns_s"])))
    max_wm_r_vel_turns_s = float(np.max(np.abs(f["wm_r_vel_turns_s"])))

    vel_track_rms_ms = float(np.sqrt(time_mean((f["wheel_vel_avg"] - f["v_ref"]) ** 2)))
    yaw_track_rms_rads = float(np.sqrt(time_mean((f["yaw_rate_rads"] - f["omega_cmd_rds"]) ** 2)))

    if run.has_gain_sched_alpha:
        alpha = f[_ALPHA_KEY]
        gain_sched_alpha_range = {
            "min": float(np.min(alpha)), "max": float(np.max(alpha)), "mean": float(np.mean(alpha)),
        }
    else:
        gain_sched_alpha_range = None  # not in telemetry yet — §1a pending

    return {
        "duration_s":          round(duration_s, 3),
        "n_samples":           run.count,
        "rms_pitch_deg":       round(rms_pitch_deg, 4),
        "ise_pitch":           round(ise_pitch, 6),
        "ise_pitch_rate":      round(ise_pitch_rate, 6),
        "rms_pitch_rate_dps":  round(rms_pitch_rate_dps, 4),
        "max_pitch_deg":       round(max_pitch_deg, 4),
        "settle_time_s":       round(settle_time_s, 3),
        "zero_cross_hz":       round(zero_cross_hz, 3),
        "dominant_freq_hz":    round(dominant_freq_hz, 3) if dominant_freq_hz is not None else None,
        "vel_track_rms_ms":    round(vel_track_rms_ms, 4),
        "yaw_track_rms_rads":  round(yaw_track_rms_rads, 4),
        "health_fractions":    {k: round(v, 4) for k, v in health_fractions.items()},
        "max_hip_l_torque_nm": round(max_hip_l_torque_nm, 4),
        "max_hip_r_torque_nm": round(max_hip_r_torque_nm, 4),
        "max_wm_l_vel_turns_s": round(max_wm_l_vel_turns_s, 4),
        "max_wm_r_vel_turns_s": round(max_wm_r_vel_turns_s, 4),
        "fault_fired":         fault_fired,
        "faults_seen":         faults_seen,
        "gain_sched_alpha_range": gain_sched_alpha_range,
        "sd_overflow_status":  None,  # not recoverable from a .wlog alone — check
                                       # LOG_INFO_STATUS at log_stop time (live only)
        "fell":                fault_fired,  # hardware analogue of the sim's "fell"
    }


def check_safety(m: dict, wheel_vel_limit_turns_s: float = DEFAULT_WHEEL_VEL_LIMIT_TURNS_S
                  ) -> tuple[bool, list[str]]:
    """Any violation here forces the trial to a rejection regardless of the
    weighted fitness score (tuning.md §4a) — a candidate that saturates or
    trips a fault is never adopted even if its raw tracking numbers look good."""
    reasons = []
    if m["fault_fired"]:
        reasons.append(f"fault fired mid-run: codes {m['faults_seen']}")
    if m["max_hip_l_torque_nm"] > HIP_TORQUE_LIMIT_NM:
        reasons.append(
            f"hip_l_torque_nm max {m['max_hip_l_torque_nm']:.3f} N·m > {HIP_TORQUE_LIMIT_NM} N·m limit")
    if m["max_hip_r_torque_nm"] > HIP_TORQUE_LIMIT_NM:
        reasons.append(
            f"hip_r_torque_nm max {m['max_hip_r_torque_nm']:.3f} N·m > {HIP_TORQUE_LIMIT_NM} N·m limit")
    if m["max_wm_l_vel_turns_s"] > wheel_vel_limit_turns_s:
        reasons.append(
            f"wm_l_vel_turns_s max {m['max_wm_l_vel_turns_s']:.3f} > limit {wheel_vel_limit_turns_s}")
    if m["max_wm_r_vel_turns_s"] > wheel_vel_limit_turns_s:
        reasons.append(
            f"wm_r_vel_turns_s max {m['max_wm_r_vel_turns_s']:.3f} > limit {wheel_vel_limit_turns_s}")
    return (len(reasons) == 0), reasons


# ── Fitness — ported 1:1 from the matching sim scenario objective ────────────
# (simulation/mujoco/master_sim/scenarios/s01_lqr_pitch_step.py, s04_vel_pi_staircase.py,
# s06_yaw_pi_turn.py): same weights/refs, applied to hardware metrics instead of sim ones.

_LQR_REF_ISE_PITCH, _LQR_REF_ISE_PITCH_RATE, _LQR_REF_SETTLE_S = 0.005, 0.02, 2.0
_LQR_W_PITCH, _LQR_W_PITCH_RATE, _LQR_W_SETTLE, _LQR_W_FELL = 0.50, 0.50, 0.0, 200.0


def fitness_lqr(m: dict) -> tuple[float, dict]:
    fell = m["fault_fired"]
    bd = {
        "pitch (ISE)":      _LQR_W_PITCH      * m["ise_pitch"]      / _LQR_REF_ISE_PITCH,
        "pitch_rate (ISE)": _LQR_W_PITCH_RATE * m["ise_pitch_rate"] / _LQR_REF_ISE_PITCH_RATE,
        "settle":           _LQR_W_SETTLE     * m["settle_time_s"]  / _LQR_REF_SETTLE_S,
        "FELL":             _LQR_W_FELL if fell else 0.0,
    }
    return sum(bd.values()), bd


_VEL_PI_REF_VEL_MS, _VEL_PI_REF_PITCH_RATE_DPS = 0.25, 15.0
_VEL_PI_W_VEL, _VEL_PI_W_RATE, _VEL_PI_W_FELL = 0.75, 0.25, 200.0


def fitness_vel_pi(m: dict) -> tuple[float, dict]:
    fell = m["fault_fired"]
    bd = {
        "velocity":   _VEL_PI_W_VEL  * m["vel_track_rms_ms"]   / _VEL_PI_REF_VEL_MS,
        "pitch_rate": _VEL_PI_W_RATE * m["rms_pitch_rate_dps"] / _VEL_PI_REF_PITCH_RATE_DPS,
        "FELL":       _VEL_PI_W_FELL if fell else 0.0,
    }
    return sum(bd.values()), bd


_YAW_PI_W_YAW_ERR, _YAW_PI_W_RMS, _YAW_PI_W_FELL = 3.0, 1.0, 200.0


def fitness_yaw_pi(m: dict) -> tuple[float, dict]:
    fell = m["fault_fired"]
    bd = {
        "yaw":   _YAW_PI_W_YAW_ERR * m["yaw_track_rms_rads"],
        "pitch": 0.1 * _YAW_PI_W_RMS * m["rms_pitch_deg"],
        "FELL":  _YAW_PI_W_FELL if fell else 0.0,
    }
    return sum(bd.values()), bd


FITNESS_FUNCS = {"lqr": fitness_lqr, "vel_pi": fitness_vel_pi, "yaw_pi": fitness_yaw_pi}


def evaluate(path, stage: str, wheel_vel_limit_turns_s: float = DEFAULT_WHEEL_VEL_LIMIT_TURNS_S) -> dict:
    """Decode + compute_metrics + safety check + stage fitness, in one call —
    what tools/analyze_hw_run.py and tabs/log_analyzer_tab.py both build on."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    run = decode_run(path)
    metrics = compute_metrics(run)
    safety_ok, safety_reasons = check_safety(metrics, wheel_vel_limit_turns_s)
    fitness_val, breakdown = FITNESS_FUNCS[stage](metrics)
    if not safety_ok:
        fitness_val += REJECT_PENALTY
    return {
        "path":            str(run.path),
        "stage":           stage,
        "telem_version":   run.telem_version,
        "sample_rate_hz":  run.sample_rate_hz,
        "n_samples":       run.count,
        "metrics":         metrics,
        "safety_ok":       safety_ok,
        "safety_reasons":  safety_reasons,
        "fitness":         round(fitness_val, 4),
        "fitness_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
    }
