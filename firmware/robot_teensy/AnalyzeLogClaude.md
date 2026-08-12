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

**A `.WLOG` can contain `RUNNING` again as of 2026-08-09.** `main.cpp` used to
stop the SD logger on entry to any state outside `{STARTUP, STANDBY, ESTOP,
CMD_REJECT}`, which made the 500 Hz path unavailable for exactly the state
balance tuning cares about. That auto-stop is gone; recording now continues
through RUNNING/JUMPING/STANDING_UP.

A log must still be **started** in STANDBY/ESTOP — opening one preallocates and
blocks the loop ~96 ms — so the workflow is: start the log, then arm. Closing is
still deferred until a non-energetic state (`sd_logger_finalize_service()`).

**Any capture dated before 2026-08-09 is still STANDBY-only**, so balance
behaviour in those bundles has to come from a 50 Hz HOST capture, where
transients faster than ~25 Hz are not observable.

`host.jsonl` is one JSON object per line. `ptype == 0x01` is a TELEM record;
other ptypes are LOG (`type_name == "LOG"`, field `log_msg`), PARAM_REPORT,
CALIB, WIFI_DIAG. HostLogger requests a parameter dump when capture starts;
`load_host_param_sidecar()` in `analysis/param_sidecar.py` reconstructs a
parameter timeline from the PARAM_REPORT records that actually arrived.

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

**For a multi-leg-height run, use `analysis/leg_height_sweep.py` instead** — see
§4.10. Whole-file statistics average the heights together and answer nothing.

```python
from analysis.leg_height_sweep import plateau_report, fit_trim_schedule
rows = plateau_report(run, torque_limit_nm=..., rate_lim=..., theta_max_bwd=...)
```

The same numbers are on screen under the Log Analyzer tab's **Leg-height sweep**
view, which reads the clamps from the `.PARAMS` sidecar itself.

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
| `wm_l_vel_turns_s`, `wm_r_vel_turns_s` | turns/s | ×2π×`WHEEL_R` → m/s. `WHEEL_R` = **0.056 m** from 2026-08-07; 0.075 m before — see §4.9 |
| `wheel_vel_avg` | m/s | computed on-robot with the `WHEEL_R` of the day (§4.9) |
| `v_ref` | m/s | the `v_cmd_ms` param, not radius-derived |
| `omega_cmd_rds` | rad/s | yaw rate command |
| `hip_l_torque_nm`, `hip_r_torque_nm` | **N·m** | AK45 MIT reply, motor-side (see §4.3) |
| `hip_*_pos_rad`, `hip_*_cmd_pos_rad` | rad | firmware frame (L/R already unflipped) |
| `active_profile` | 0..2 | CH9 speed profile index (0 = profile1) — sets the torque clamp |
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

### 4.6 `gain_sched_alpha` *is* in telemetry (V10+)

Leg-height scheduling drives the LQR gains, trim, barrier and hip feedforward.
α is logged — `telem_format.py` unpacks `gain_sched_alpha` from the payload and
`wlog_metrics.py` exposes it in `run.fields`. It is still guarded as optional
(`run.has_gain_sched_alpha`) for pre-V10 files, so check that flag before using
it rather than assuming the key exists; only fall back to inferring α from
`hip_*_pos_rad` when the flag is false, and say that you did.

Note α = 0 means *fully retracted*, and it is also the value used whenever hip
calibration is invalid (`control_loop.cpp` defaults to the retracted anchor) or
`alpha_force_ret_en=1`. A run that is α ≡ 0 throughout is therefore either
genuinely retracted, uncalibrated, or deliberately pinned. Cross-check
`HIP_LIMITS_VALID` (health bit 4), and compare measured/commanded hip position:
valid calibration plus substantial hip travel while α stays zero is direct
evidence that the override is pinned.

### 4.7 `.PARAMS` sidecar

SD runs carry the param values at capture time —
`software/gui/analysis/param_sidecar.py` aligns them to `run.t_micros`. HOST
captures can carry PARAM_REPORT records; use `load_host_param_sidecar()` to
align the reports to telemetry. A HOST dump can be incomplete if packets were
lost, so only treat parameters present in the reconstructed timeline as known.
For a missing value, infer it from a telemetry identity only when the fit is
unambiguous (and say that you inferred it), otherwise report it as unknown.

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

### 4.9 Every speed in m/s changed meaning on 2026-08-07

The wheel went Ø150 mm → **Ø112 mm**, so `WHEEL_R` in `control_loop.cpp` went
0.075 → **0.056 m**. `wheel_vel_avg` is computed on the robot as
`turns/s × 2π × WHEEL_R`, so the *same physical speed* reads **0.747×** in a log
captured after the change as it did before.

- **`wm_*_vel_turns_s` is the invariant.** It is straight off the encoder and
  means the same thing on both sides of the boundary. Compare in turns/s, or
  re-derive m/s yourself with the radius that run was captured on.
- **`v_ref` did not rescale** — it is the `v_cmd_ms` param passed through. So the
  same `v_ref` now demands a *higher* wheel speed, and the velocity-PI error
  `wheel_vel_avg − v_ref` is not comparable across the boundary either.
- Velocity-loop and LQR gains tuned before this date are stale for the same
  reason (`CLAUDE.md`). Do not rank a pre- and a post-change run against each
  other on any m/s-derived metric.
- The torque identities in §5 scale with `r` too — a given acceleration now
  needs 0.747× the wheel torque it did on the Ø150 wheel.

Nothing in the wire format changed, so `TELEM_VERSION` is still 12 and there is
**no field that tells you which radius a run used.** Date the bundle: anything
under `data/logs/runs/` timestamped before `20260807` is a Ø150 capture.

### 4.10 A multi-height run must be split by α before anything is computed

Everything worth asking about a leg-height sweep is a per-height question, and
the plant genuinely changes across the stroke (§4.6). Averaging heights together
produces numbers that describe no configuration the robot was ever in.

`software/gui/analysis/leg_height_sweep.py` does the split — `alpha_plateaus()`
for the segmentation, `plateau_report()` for the per-height table. Use it rather
than re-deriving; the GUI's **Leg-height sweep** view and
`tools/balance_trim_sweep.py` both call it, so screen and CLI agree.

Traps this exists to stop:

- **"Stationary" is not "at one leg height."** A sweep that changes height
  without touching the sticks is stationary end to end, so splitting on stick
  input alone yields one huge stretch straddling every height. That is exactly
  what `balance_trim_sweep.py` did to the 2026-08-09 four-height run: one 43.5 s
  window, failed equilibrium, entire sweep reported nothing. Fixed 2026-08-09 by
  intersecting with the α plateaus.
- **α ≈ 0 does not mean the leg is parked.** α is clipped at 0 below the
  calibration backoff, so the whole arm-in transient — 15.5° of hip travel while
  `hip_running_ramp_s` ramps stiffness in — logs as a flat α ≈ 0. Detecting
  plateaus by *drift* (half-mean difference) rather than peak-to-peak separates
  it from the settled retracted stance that follows; a peak-to-peak test merges
  the two and contaminates the retracted balance point with the transient.
- **The equilibrium gate needs both halves.** `tau_sym ≈ 0` alone will certify
  an oscillating window, because a limit cycle averages its torque to nearly
  zero while the robot still creeps. Require the wheel-drift gate too. Plateaus
  that fail are reported with `equilibrium=False`, not dropped — "it never
  settled at this height" is the finding when a run went unstable up-stroke.
- **`trim_ext` from a sweep that stopped short of α = 1 is an extrapolation.**
  `fit_trim_schedule()` fits the firmware's own `scheduled_pitch_trim()` basis,
  which includes a quadratic term and therefore extrapolates hard. It sets
  `extrapolated` whenever the sweep topped out below α = 0.9. Do not type an
  `_ext` anchor into params off an interior-only sweep.

### 4.11 Band-split the pitch error before blaming a loop

The band a disturbance lives in names the loop that owns it, and an RMS figure
hides this completely. `band_split()` / `band_rms()` in the same module use the
boundaries the loops actually separate at: **0.3–1.5 Hz is the velocity PI,
1.5–4 Hz the LQR pitch loop, >10 Hz leg/hip structure.** On the 2026-08-09 run,
total pitch-error RMS grew 2.4° → 4.8° across the stroke and *all* of the growth
was in the velocity-PI band while the LQR band stayed flat at ~0.7° — which
rules out the inner loop and the hips as the cause before any gain is touched.

`rate_limit_duty()` covers the other thing RMS cannot see: how much of the time
`vel_pi_rate_lim` is slewing `theta_ref`. A rate limiter inside a feedback loop
is a describing-function limit-cycle source. `mean_run_s` is what matters —
runs of a few ms are the limiter smoothing encoder noise (what it is for), runs
approaching half an oscillation period mean `theta_ref` has become a triangle
wave and the limiter now sets the loop's phase.

---

### 4.12 Direct MuJoCo replay is windowed and controller-locked

`simulation/mujoco/v4_twin_279mm_baseline/twin/tools/replay_wlog.py` now runs
the real MuJoCo model, not the older analytical surrogate. Use `--mode both` to
produce open-loop plant and closed-loop controller scores. It consumes every
RUNNING command in consecutive reset windows (0.1 s open, 2.0 s closed) because
an inverted pendulum open-loop rollout cannot remain meaningful after tiny
initial-state differences grow.

The sidecar's control values are fixed evidence and are never fit. WLOG does not
contain raw `radio_hip_cmd`; it contains the post-rate-limit MIT hip setpoints,
so replay seeds the hip slew state from those setpoints instead of applying the
rate limiter twice. Hip positions in WLOG are in the calibrated switch-zero
frame; MuJoCo adds the +28 degree retract-stop offset to enter the CAD frame.

The reported XY score is wheel-speed/yaw dead reckoning, not global-position
ground truth. Use pitch, pitch rate, wheel speed, yaw rate, hip position, and
torque residuals to identify plant error.

## 5. Physical constants for sanity checks

| Quantity | Value | Source |
|---|---|---|
| Total robot mass | ~3.06 kg (2783 g + 10 %) | `components/COMPONENTS.md` — best-estimate, not weighed; not re-estimated for the smaller v4 wheel |
| Wheel radius | **0.056 m (Ø112 mm)** since 2026-08-07; 0.075 m (Ø150 mm) before — §4.9 | `control_loop.cpp` `WHEEL_R` |
| Wheel motor | Maytech MTO5065-70-HA-C, 450 g ea, Kt 0.1364 N·m/A, 6.82 N·m peak @ 50 A, ω₀ 175.9 rad/s @ 24 V | Still the fitted motor. `NewWheelMotor.md` argues it is ~26× oversized, but no replacement is installed — size sanity checks against these numbers |
| Wheel controller | ODESC 3.6 dual, 160 g | |
| Hip motor | CubeMars AK45-10, 260 g ea, 10:1, 2.5 N·m rated / 7 N·m peak, Kt 1.27 N·m/A at output | |
| Battery | 6S LiPo, treat V_nom as **24.0 V** | |
| Control rate | 500 Hz | `config.h` |

Handy per-wheel torque identities (M = total mass, r = wheel radius — **0.056 m**
for a post-2026-08-07 run, 0.075 m before):
`τ = M·a·r/2` (level accel) · `τ = M·g·sinθ·r/2` (slope hold) ·
`τ = M·g·tanθ·r/2` (catching a lean of θ).

**Known inconsistencies with the sim.** `simulation/mujoco/master_sim/params.py`
disagrees with the hardware on two counts, both still open:

- `wheel_r = 0.075` — the **pre-v4** radius. The sim has not been updated for the
  Ø112 wheel, so no m/s quantity from it is comparable to a current log (§4.9).
- `m_wheel = 0.270 kg` for the whole wheel assembly, while `COMPONENTS.md` gives
  450 g motor + 70 g wheel = 520 g per side.

The BOM masses are labelled best-estimate and have not been verified on a scale.
Flag which set you used whenever you compare sim to hardware.

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

### 2026-08-10 — as-built mass inventory

`M_BODY` changed from the stale 1.638 kg prior to **2.562 kg**: measured/modelled
3.518 kg driving mass minus two measured 0.478 kg wheel assemblies. This changes
FF2 gravity compensation only when `ff2_alpha > 0`; the 2026-08-10 default
export and reference run both have FF2 disabled.

### 2026-08-07 — v4 leg geometry and the Ø112 wheel

| Change | Effect on analysis |
|---|---|
| **`WHEEL_R` 0.075 → 0.056 m** (Ø150 → Ø112 wheel) | See §4.9. `wheel_vel_avg` for a given physical speed reads 0.747× what it used to; `wm_*_vel_turns_s` is unaffected. No telemetry field records which radius was used — go by the bundle's date. |
| **Velocity-loop and LQR gains are stale across this line** | The plant's torque-to-accel gain changed with `r`. Gain rankings from pre-2026-08-07 runs do not carry over. |
| **v4 link lengths, new hard stops `Q_RET` +28° / `Q_EXT` −57°** | `hip_*_pos_rad` still means the same firmware-frame angle, but the α it implies is different, so the §4.6 trick of inferring `gain_sched_alpha` from hip position needs the v4 geometry. `L_EFF_RET`/`L_EFF_EXT` were recomputed; `M_BODY` was still stale at this date and was updated on 2026-08-10. |
| **Wire format unchanged** | `TELEM_VERSION` still 12; `wlog_metrics.py` decodes both sides identically. This is a *meaning* change, not a format change — nothing will raise on load. |

### 2026-08-03 — reliability-audit fixes

Acted on the verified subset of `ChatGPTfixes.md`. Analysis-relevant results:

| Change | Effect on analysis |
|---|---|
| **Pitch error now subtracts `pitch_trim_rad`** | See §4.8. Invalidates pitch-error metrics computed before this date; re-run them. |
| **`FAULT_JUMP_TIMEOUT` (`0x0F`) added** | New `fault_code` value. `JUMPING` now ESTOPs if the sequence overruns its phase budget instead of quietly returning to `RUNNING`, so a jump that used to end silently mid-sequence is now visible in the log as a fault. |
| **`JUMPING` duration is no longer a flat 3 s** | Time in state 7 is now phase-driven (`JP_DONE` + 300 ms settle) rather than a fixed timer, so JUMPING episode lengths are not comparable across this boundary. |
| **CAN TX deferrals counted on both buses** | A sustained TX stall now clears `HIP_L/R_OK` / `WM_L/R_OK` (health-flag bits 0–3) and forces IDLE. A run ending with those bits dropping may be a *bus* failure, not a feedback failure — the distinguishing evidence is the `comm_log` line `CAN2/CAN3 TX stalled` (HOST captures only; `.WLOG` has no log text). Counters themselves are **not** in telemetry yet. |
| **`wm_*.ok` now also requires a fresh ODrive heartbeat in `CLOSED_LOOP`** | Bits 2/3 mean more than "encoder fresh" from this date on. If the firmware logged `no ODrive heartbeat seen` at boot, the added check is inert and the bits mean exactly what they used to. |
| **`alpha_force_ret_en` became non-persistent** | At this point it booted to 0. This was temporary: the 2026-08-09 change below made it persistent again. For captures in this interval, a forced-retracted schedule could not survive a reboot. See §4.6 for how to identify a pinned run. |
| **`COMM_TYPE_ESP32_STATUS` `0x16` → `0x17`** | Wire-level only; no telemetry field changed and `TELEM_VERSION` is unchanged at 12. Affects live capture with mismatched Teensy/ESP32 firmware (heartbeat down, `esp32_link_ok` false), not decoding of existing files. |

### 2026-08-09 — leg-height sweep analysis

| Change | Effect on analysis |
|---|---|
| **`analysis/leg_height_sweep.py` added** | Plateau segmentation, per-height metrics, band splitting, rate-limiter duty, and trim-schedule fitting, shared by the GUI tab and the CLI. See §4.10 / §4.11. |
| **Log Analyzer gained a "Leg-height sweep" view** | Per-plateau balance point with the equilibrium gate shown, pitch error, and `dθref/dt` against `vel_pi_rate_lim`. Clamps are read from the `.PARAMS` sidecar; without one the saturation percentages read 0 rather than being guessed. |
| **`balance_trim_sweep.py` now splits by α plateau** | It previously split only on stick input, so any multi-height run collapsed into one window and reported nothing. Re-run any earlier sweep that came back empty — the log was probably fine. |

### 2026-08-09 — SD logging survives RUNNING

| Change | Effect on analysis |
|---|---|
| **The "stopping before energetic state" auto-stop was removed** | `.WLOG` captures can now cover `RUNNING`/`JUMPING`/`STANDING_UP` at 500 Hz. Fast transients (torque spikes, the ~13 Hz hip/leg mode, glitch filters) are observable for the first time. Bundles dated before this are still STANDBY-only. |
| **Starting a log is still gated to STANDBY/ESTOP** | Workflow is start-then-arm; a log cannot be opened mid-run (preAllocate blocks ~96 ms). C6 on the radio drives this when `live_tune_multi_en = 0`. |
| **Ring-buffer overflow is now the failure mode to watch** | 32 KB buffer, 251 B/record at 500 Hz = ~261 ms of card stall absorbed before samples drop. Overflow logs `"SDLogger: ring buffer overflow, samples dropped"` — check for it before trusting sample continuity in a long RUNNING capture. |
| **`alpha_force_ret_en` is persistent again** | A tuning session can now stay pinned to the retracted gain/trim anchors across a reboot. Telemetry still reports α = 0 while pinned, even if the calibrated legs move through the stroke. Check hip motion as described in §4.6; firmware also emits a boot warning in HOST logs when the override is restored. |
