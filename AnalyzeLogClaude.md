# AnalyzeLogClaude.md — analysing robot logs (AI context)

Everything an AI session needs to open a run log, get trustworthy numbers out of
it, and not draw a wrong conclusion. Scoped to *offline analysis* — flashing,
protocol changes, and control theory live in `README.md`.

> Keep this file current: if you find something here that is stale while
> analysing a log, fix it in the same change.

---

## 1. Where the logs are

Run bundles live under the repo-root, git-ignored tree:

```
data/logs/runs/<UTC timestamp>_<kind>/
    manifest.json          # capture_schema, source_kind, started_utc, telem_version
    host.jsonl             # kind = HOST  — GUI-side capture
    LOG####.WLOG           # kind = SD_LOG#### — on-robot SD capture
    LOG####.PARAMS         # param sidecar for that SD run
```

Two capture paths, **not** interchangeable:

| | `host.jsonl` (HOST) | `.WLOG` (SD_LOG) |
|---|---|---|
| Written by | GUI `HostLogger` (`software/gui/tabs/host_logger.py`) | Teensy, straight to SD |
| Rate | **50 Hz** (the telemetry rate) | up to **500 Hz** (header field `sample_hz`) |
| Contains | TELEM + firmware `comm_log()` lines + CALIB + WIFI_DIAG | TELEM only |
| Loses data if | link drops / GUI stalls | SD overflow |
| Use it for | anything needing log text, or when no SD run exists | anything needing >50 Hz detail |

If a question is about fast transients (torque spikes, loop timing, glitch
filters), prefer a `.WLOG`. For duty-cycle / RMS / percentile questions the 50 Hz
HOST capture is fine and there are far more of them.

**But a `.WLOG` can never contain `RUNNING`.** `main.cpp` stops the SD logger on
entry to any state outside `{STARTUP, STANDBY, ESTOP, CMD_REJECT}`, so the 500 Hz
path is unavailable for exactly the state balance tuning cares about. Anything
about balance behaviour has to come from a 50 Hz HOST capture, and transients
faster than ~25 Hz are simply not observable today. Treat the preference above as
applying to bench work in permitted states only. Lifting this is the single
biggest unlock for log-driven tuning — see §7.

`host.jsonl` is one JSON object per line. `ptype == 0x01` is a TELEM record;
other ptypes are LOG (`type_name == "LOG"`, field `log_msg`), CALIB, WIFI_DIAG.

---

## 2. How to decode — always use the shared module

**Do not write a new decoder.** `software/gui/analysis/wlog_metrics.py` is
Qt-free, handles both formats, and is the same code the GUI's Log Analyzer tab
uses, so a number you compute matches what the operator sees on screen.

```python
import sys
sys.path.insert(0, "software/gui")          # repo-relative
from analysis.wlog_metrics import decode_run, compute_metrics, check_safety

run = decode_run("data/logs/runs/<bundle>/host.jsonl")   # or .../LOG0007.WLOG
run.t_s          # np.float64 seconds since first record, wrap-safe
run.fields       # dict: name -> np.ndarray, see §3
run.sample_rate_hz
run.telem_version
metrics = compute_metrics(run)
```

`decode_run()` dispatches on extension. It raises `ValueError` on a
`TELEM_VERSION` it can't read rather than silently misreading bytes.

Ready-made CLI for the standard controller-tuning metrics:

```
python software/gui/tools/analyze_hw_run.py <path> --stage {lqr,vel_pi,yaw_pi}
```
Exit 0 = safe, 2 = safety-maxima violation, 1 = couldn't analyze.

**Live GUI route** (only when the GUI is already running, and only if you need
its plots): `python software/gui/tools/robot_ctl.py analyzer_load <path>`, then
`ui_snapshot tab/log-analyzer`. For pure number-crunching the Python route above
is faster and has no GUI dependency.

---

## 3. Telemetry fields worth knowing

Full wire layout: `shared/comm_protocol.h` (`TelemetryPayload`) and
`software/gui/tabs/telem_format.py`. `_SCALAR_FIELDS` in `wlog_metrics.py` lists
what gets pulled into arrays — **add a field there before trying to use it.**

| Field | Unit | Notes |
|---|---|---|
| `robot_state` | enum | filter on this first — see §4.1 |
| `fault_code` | enum | non-zero only in ESTOP |
| `health_flags` | bitfield | see table below |
| `pitch_rad`, `pitch_rate_rads` | rad, rad/s | +X forward, +Y left, +Z up |
| `roll_rad`, `yaw_rad`, `*_rate_rads` | rad, rad/s | |
| `theta_ref` | rad | pitch target from the velocity PI |
| `pitch_trim_rad` | rad | balance-point offset, α-scheduled, **not small** (≈ −0.14 rad retracted). Part of the control target — see §4.8 |
| `whl_tau_l`, `whl_tau_r` | **N·m** | final *commanded* wheel torque — see §4.2 |
| `tau_sym`, `tau_yaw` | N·m | LQR symmetric / yaw-PI differential parts of the above |
| `wm_l_vel_turns_s`, `wm_r_vel_turns_s` | turns/s | ×2π×0.075 m → m/s |
| `wheel_vel_avg`, `v_ref` | m/s | |
| `omega_cmd_rds` | rad/s | yaw rate command |
| `hip_l_torque_nm`, `hip_r_torque_nm` | **N·m** | AK45 MIT reply, motor-side (see §4.3) |
| `hip_*_pos_rad`, `hip_*_cmd_pos_rad` | rad | firmware frame (L/R already unflipped) |
| `active_profile` | 1..3 | CH9 speed profile — sets the torque clamp |
| `loop_count`, `timestamp_ms` | — | reset to small values ⇒ reboot |

Health-flag bits (`comm_protocol.h`):

| Bit | Name | Meaning |
|---|---|---|
| 0,1 | `HIP_L/R_OK` | hip CAN feedback fresh |
| 2,3 | `WM_L/R_OK` | wheel encoder feedback fresh |
| 4 | `HIP_LIMITS_VALID` | calibration valid |
| 5 | `IMU_NOMINAL` | |
| 6 | `LQR_ACTIVE` | lqr_enable && RUNNING |
| 7 | `VEL_PI_SAT` | `theta_ref` clamped this tick |
| 8 | `YAW_PI_SAT` | `tau_yaw` clamped this tick |
| 9,10 | `WM_L/R_VEL_LIMITED` | soft velocity governor clamping |
| 11 | `LOOP_OVERRUN` | control loop over its 2 ms budget in the last second |

States: 0 STARTUP, 1 CALIBRATION, 2 STANDBY, 3 RUNNING, 4 ESTOP, 5 MANUAL,
6 CMD_REJECT, 7 JUMPING, 8 STANDING_UP, 9 DISARMING.
Fault codes: see the table in `README.md`.

---

## 4. Interpretation traps

These are the ones that have actually produced wrong answers.

### 4.1 Filter by `robot_state` before computing anything

Most bundles are mostly STANDBY or MANUAL bench time. Torque, pitch and velocity
statistics computed over a whole file are meaningless. For balance questions use
`robot_state == 3` (RUNNING) only; a bundle with < ~10 s of RUNNING is not
evidence of anything.

### 4.2 `whl_tau_*` is a *command*, and it is clamped

It is what the control loop asked the ODrive for, not a measured shaft torque
(there is no wheel torque/current feedback in telemetry at all). Two consequences:

- **Check saturation before concluding anything about demand.** `|tau_sym|` is
  clamped to `lqr_torque_limit`, which is READONLY and slewed from the active
  CH9 profile's `profileN_torque_lim` (compile-time defaults **0.1 / 0.2 /
  0.3 N·m** — very low). A peak that lands exactly on a round number is the
  clamp, not the physics. Recover the active clamp from the run itself as
  `max(|tau_sym|)` and report the fraction of samples at ≥ 98 % of it. If that
  fraction is small (< ~1 %), the demand numbers are genuine; if it is large,
  the log only tells you what the operator allowed.
- `whl_tau_l/r = tau_sym ± tau_yaw`. It is 0 when the LQR is disabled, and
  during CALIBRATION the field is reused as the hip ramp target.

### 4.3 Hip torque scale and `TELEM_VERSION` 11 vs 12

V11 and V12 are byte-identical; only the two hip fields changed meaning
(`hip_*_current_a` "amps" → `hip_*_torque_nm`, ×8/20). `wlog_metrics.py` rescales
V11 on load, so decoded runs are directly comparable — **but only through that
module.** Any hand-rolled decode of a V11 file reports hip torque 2.5× too high.
Reported hip torque is motor-side (from q-axis current); true output torque is
10–20 % lower after planetary losses.

### 4.4 HOST captures are 50 Hz, and the GUI drops packets elsewhere

`HostLogger` sits on the uncoalesced stream, so a `host.jsonl` is a true 50 Hz
record. The GUI's `TelemetryBus.packet` signal is coalesced to 20 Hz — never
measure rates from anything downstream of it. Duplicate/stalled records are
already suppressed by `decode_hostlog()` on `(timestamp_ms, loop_count)`.

### 4.5 Encoder glitches are real

The ODrive encoder feed has produced single-sample jumps of several turns/s
(bench log `20260728T053232`). `wheel_vel_glitch_filter()` catches most, but a
lone extreme `wm_*_vel_turns_s` sample in an older log is probably corruption,
not physics. Sanity-check against the free-spin ceiling (~1300 turns/s²) and
against `wheel_vel_avg` before treating an outlier as real.

### 4.6 `gain_sched_alpha` is not in telemetry yet

Leg-height scheduling drives the LQR gains, trim, barrier and hip feedforward,
but α itself is not logged. `wlog_metrics.py` treats it as optional
(`has_gain_sched_alpha`). Infer it from `hip_*_pos_rad` if you need it, and say
that you did.

### 4.7 `.PARAMS` sidecar

SD runs carry the param values at capture time —
`software/gui/analysis/param_sidecar.py` aligns them to `run.t_micros`. HOST
captures have no sidecar; if a param value matters, get it from the run's own
telemetry (`active_profile`, `max(|tau_sym|)`) or say it is unknown.

### 4.8 Pitch error is trim-relative — and older numbers are not

The regulated quantity is the LQR's `x0` in `control_loop.cpp`:

```text
pitch_error = pitch_rad - theta_ref - pitch_trim_rad
```

All three terms. `compute_metrics()` has done this since **2026-08-03**; before
that it computed `pitch_rad - theta_ref` and silently dropped the trim.

That is not a rounding difference. The measured retracted balance point is
≈ −0.14 rad (−8°), so a robot holding its balance point *perfectly* scored
≈ 8° RMS pitch error — a near-constant offset far larger than the differences
between candidate gain sets these metrics exist to rank. Every affected metric
(`rms_pitch_deg`, `ise_pitch`, `max_pitch_deg`, `settle_time_s`, and any score
derived from them) was biased, and comparisons between runs at *different* leg
heights were biased by different amounts, because the trim is α-scheduled.

Consequences when reading older material:

- **Discard pitch-error numbers quoted in any analysis written before
  2026-08-03**, including tuning-session records under
  `software/gui/analysis/tuning_session.py`. Re-run `compute_metrics()` on the
  original bundle instead — the logs themselves are fine, only the derived
  numbers were wrong.
- Gain rankings from that period are not automatically wrong, but they were
  made against a distorted objective; treat them as unverified.
- The GUI Log Analyzer's *plots* were always correct
  (`log_analyzer_tab.py` plotted `theta_ref + pitch_trim_rad`). Only the
  computed metrics were affected, which is why the discrepancy went unnoticed.

---

## 5. Physical constants for sanity checks

| Quantity | Value | Source |
|---|---|---|
| Total robot mass | ~3.06 kg (2783 g + 10 %) | `components/COMPONENTS.md` |
| Wheel radius | 0.075 m (Ø150 mm) | |
| Wheel motor | Maytech MTO5065-70-HA-C, 450 g ea, Kt 0.1364 N·m/A, 6.82 N·m peak @ 50 A, ω₀ 175.9 rad/s @ 24 V | |
| Wheel controller | ODESC 3.6 dual, 160 g | |
| Hip motor | CubeMars AK45-10, 260 g ea, 10:1, 2.5 N·m rated / 7 N·m peak, Kt 1.27 N·m/A at output | |
| Battery | 6S LiPo, treat V_nom as **24.0 V** | |
| Control rate | 500 Hz | `config.h` |

Handy per-wheel torque identities (M = total mass, r = 0.075 m):
`τ = M·a·r/2` (level accel) · `τ = M·g·sinθ·r/2` (slope hold) ·
`τ = M·g·tanθ·r/2` (catching a lean of θ).

**Known inconsistency:** `simulation/mujoco/master_sim/params.py` uses
`m_wheel = 0.270 kg` for the whole wheel assembly, while `COMPONENTS.md` gives
450 g motor + 70 g wheel = 520 g per side. The BOM masses are labelled
best-estimate and have not been verified on a scale. Flag which one you used.

---

## 6. Minimal working example

```python
import sys; sys.path.insert(0, "software/gui")
import numpy as np
from analysis.wlog_metrics import decode_run

run = decode_run("data/logs/runs/20260802T174130_628413Z_HOST/host.jsonl")
f   = run.fields
m   = f["robot_state"].astype(int) == 3            # RUNNING only  (§4.1)

tau   = np.concatenate([f["whl_tau_l"][m], f["whl_tau_r"][m]])
clamp = np.abs(f["tau_sym"][m]).max()              # active limit  (§4.2)
sat   = np.mean(np.abs(f["tau_sym"][m]) >= 0.98 * clamp)

print(f"RUNNING {m.sum()/run.sample_rate_hz:.1f} s   clamp {clamp:.3f} N.m   "
      f"saturated {sat*100:.2f} % of samples")
print(f"|tau| p50 {np.percentile(abs(tau),50):.4f}  p99 {np.percentile(abs(tau),99):.4f}  "
      f"max {abs(tau).max():.4f} N.m   RMS {np.sqrt((tau**2).mean()):.4f}")
```

---

## 7. Changes that affect how a log reads

Newest first. Only entries that change the meaning of a field, a metric, or what
a log can tell you — not every firmware change.

### 2026-08-03 — reliability-audit fixes

Acted on the verified subset of `ChatGPTfixes.md`. Analysis-relevant results:

| Change | Effect on analysis |
|---|---|
| **Pitch error now subtracts `pitch_trim_rad`** | See §4.8. Invalidates pitch-error metrics computed before this date; re-run them. |
| **`FAULT_JUMP_TIMEOUT` (`0x0F`) added** | New `fault_code` value. `JUMPING` now ESTOPs if the sequence overruns its phase budget instead of quietly returning to `RUNNING`, so a jump that used to end silently mid-sequence is now visible in the log as a fault. |
| **`JUMPING` duration is no longer a flat 3 s** | Time in state 7 is now phase-driven (`JP_DONE` + 300 ms settle) rather than a fixed timer, so JUMPING episode lengths are not comparable across this boundary. |
| **CAN TX deferrals counted on both buses** | A sustained TX stall now clears `HIP_L/R_OK` / `WM_L/R_OK` (health-flag bits 0–3) and forces IDLE. A run ending with those bits dropping may be a *bus* failure, not a feedback failure — the distinguishing evidence is the `comm_log` line `CAN2/CAN3 TX stalled` (HOST captures only; `.WLOG` has no log text). Counters themselves are **not** in telemetry yet. |
| **`wm_*.ok` now also requires a fresh ODrive heartbeat in `CLOSED_LOOP`** | Bits 2/3 mean more than "encoder fresh" from this date on. If the firmware logged `no ODrive heartbeat seen` at boot, the added check is inert and the bits mean exactly what they used to. |
| **`alpha_force_ret_en` no longer persistent** | It now always boots to 0. In logs *before* this date a forced-retracted gain schedule could silently be active across a power cycle, so an unexplained retracted-anchor response in an older run is worth suspecting. Still not directly observable — see §4.6. |
| **`COMM_TYPE_ESP32_STATUS` `0x16` → `0x17`** | Wire-level only; no telemetry field changed and `TELEM_VERSION` is unchanged at 12. Affects live capture with mismatched Teensy/ESP32 firmware (heartbeat down, `esp32_link_ok` false), not decoding of existing files. |

Not done, and still true as written elsewhere in this file: SD logging is still
stopped on entry to any energetic state, so **there are still no `.WLOG`
captures of `RUNNING`** — §1's advice to prefer a `.WLOG` for fast transients
only applies to states that permit recording. This is the main open limitation
on log-driven tuning.
