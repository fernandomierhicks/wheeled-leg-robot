# robot_teensy firmware

Two-microcontroller architecture for a wheeled-leg balancing robot.

> **AI maintenance note:** If you find anything here that is stale while
> working in this tree, update this README in the same change.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Teensy 4.1  (500 Hz control loop)                              │
│                                                                  │
│  control_loop.cpp  ← LQR + vel/yaw PI + feedforward            │
│  state_machine.cpp ← 8-state FSM                                │
│  hip_motors.cpp    ← AK45-10 MIT Cheetah (CAN2)                │
│  wheel_motors.cpp  ← ODrive/ODESC (CAN3)                       │
│  IMU.cpp           ← BNO086 (SPI)                               │
│  main.cpp          ← scheduler, radio, telemetry, LED, buzzer   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ UART 4 Mbaud (CommLink framed)
                               │ telemetry @ 50 Hz (split TELEM_A + TELEM_B)
                               │ commands ← GUI / ESP32
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  ESP32  (display, telemetry bridge, obstacle sensing)           │
│                                                                  │
│  WiFi UDP leased unicast → Python GUI (software/gui/)            │
│  WiFi TCP server        ← GUI commands/results                   │
│  TFT display (SPI)   ← paginated flight/drive/comms pages       │
│  Neopixel strip      ← state colour + animations                │
│  VL53L1X ToF ×4      → obstacle distances forwarded to Teensy   │
│  USB serial (CP2102) → PC passthrough when WiFi unavailable     │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             Python GUI              Browser / raw serial
         software/gui/main.py
```

**Network exposure (accepted risk — home lab only):** a GUI first broadcasts
`WLR_CLAIM_V1 <token>` on UDP `:5007`. The ESP32 grants one 3.5-second session
lease and sends combined 247-byte telemetry datagrams by unicast to that
client on UDP `:5005`; renewals arrive every 0.8 seconds. A competing token is
reported busy and a second TCP `:5006` connection is closed without evicting
the owner. TCP carries commands, results, parameters, and log transfers but
does not duplicate telemetry. The token prevents accidental client collision,
not hostile access or spoofing: there is no authentication or encryption, so
do not operate the robot on an untrusted network.

**Key buses on Teensy:**
- CAN2 @ 1 Mbps → AK45-10 hip motors (MIT Cheetah protocol)
- CAN3 @ 1 Mbps → ODrive wheel motors
- SPI0 → BNO086 IMU (CS=D10, INT=D9, RST=D6)
- Serial2/3 → AK45 UART encoder readback
- Serial4 RX → FlySky iBUS RC receiver
- Serial5 ↔ ESP32 (CommLink UART)

## Directory layout

```
firmware/robot_teensy/
├── shared/
│   ├── comm_protocol.h   ← packet types, TelemetryPayload, fault codes (single source of truth)
│   └── CommLink/         ← framed protocol implementation used by both MCUs
├── teensy/               ← Teensy 4.1 PlatformIO project
│   ├── src/
│   │   ├── config.h          ← all pin and bus constants
│   │   ├── robot_state.h     ← RobotState struct + RobotStateEnum
│   │   ├── state_machine.cpp ← 8-state FSM (see state_machine.md)
│   │   ├── control_loop.cpp  ← LQR + vel PI + yaw PI + feedforward
│   │   ├── live_tune.h       ← generic radio-knob live param tuning (slot table in main.cpp)
│   │   └── main.cpp          ← loop(), radio, telemetry, LED, buzzer
│   └── lib/
│       ├── AK45Uart/         ← AK45-10 UART encoder readback
│       ├── HipMotors/        ← AK45-10 MIT Cheetah CAN driver (CAN2)
│       ├── WheelMotors/      ← ODrive CAN driver (CAN3)
│       ├── IMU/              ← BNO086 SPI driver
│       ├── LED/              ← Non-blocking RGB LED
│       ├── Buzzer/           ← Non-blocking passive buzzer
│       ├── ParamRegistry/    ← Runtime param table (GUI-tunable, 500 Hz safe)
│       └── Calibration/      ← Hip retract-switch calibration FSM
└── esp32/                ← ESP32 PlatformIO project
    └── src/main.cpp      ← all ESP32 logic (display, WiFi, Neopixel, ToF)
```

## GUI control API (preferred for agents and headless inspection)

The running Python GUI exposes a local-only command server on
`127.0.0.1:8765`. It starts automatically with the GUI and is controlled with:

```text
python software/gui/tools/robot_ctl.py <command> [args...]
```

Use this API whenever a task asks to inspect or control something "in the
GUI." Prefer it over Windows mouse/keyboard automation and over writing a
standalone client that reimplements the robot protocol. Start with
`capabilities`, `health`, or `service_status`; use `service_start` if the GUI
is not running.

The API includes dedicated robot commands (`telem`, `param_get`, `param_set`,
`set_mode`, logging, connection control, and firmware flashing) plus a generic
Qt operator bridge:

- `tab_select <title>` selects a GUI tab.
- `ui_manifest` enumerates widget IDs and supported actions.
- `ui_snapshot [query]` returns widget values and text without screen scraping.
- `ui_invoke <id> <action> [json_value]` invokes a real Qt widget action.
- `ui_screenshot` captures the GUI only when visual plot inspection is useful.

For an already-loaded Log Analyzer run, use
`ui_snapshot tab/log-analyzer` to read its source, duration, metrics, limits,
warnings, selected view, and available view choices. Change the analyzer view
through the combo box returned by that snapshot using
`ui_invoke <combo-id> select_text '"Vel-PI"'`, then snapshot the same group
again. To load a specific `.wlog` or host `jsonl` first, use
`analyzer_load <path>`. Full command and safety details are in
`software/gui/CLAUDE.md` under **Remote control / automation** and in
`software/gui/tools/robot_ctl.py`.

Select the **Jumping** analyzer view for a phase-aligned launch and recovery
report. It focuses the plots around every `JUMPING` episode and shows phase and
touchdown markers alongside the effective forward command/nudge, body attitude,
all three IMU rates, hip position/speed/torque, and wheel speed/torque authority.
Current logs use the firmware's live `LANDING` phase. Acceleration plots remain
for older captures but read zero with the integrated-only production IMU; older
V12 captures infer touchdown from their historical acceleration/gyro evidence.

## Flashing (PlatformIO / GUI)

```
cd teensy && pio run -e teensy41 -t upload    # or esp32/ -e esp32dev
```

Or remotely, via the running GUI's command server (see `software/gui/CLAUDE.md`
→ "Remote control / automation"): `python software/gui/tools/robot_ctl.py
firmware_flash teensy`. Either way, PlatformIO builds first, then needs the
Teensy in its HalfKay bootloader to actually upload — normally triggered by a
soft-reset touch over the serial port, but that can fail to land (e.g. the
GUI already holds the port open in monitor mode) and `teensy_loader_cli` then
sits waiting for the bootloader indefinitely instead of erroring out.

**Expect to flash twice.** Symptom of the stuck first attempt: the build
succeeds but the command returns `ok: false` with a truncated/empty output
tail, and a `teensy_loader_cli` process is left running (check with
`Get-Process teensy_loader_cli` on Windows). A second `firmware_flash` call
while that's still alive fails fast with `"teensy flash is already
running"` — kill the stuck process first (`Stop-Process -Id <pid> -Force`),
then retry. The retry upload usually succeeds outright; if not, a physical
tap of the Teensy's PROGRAM button while the loader is waiting forces it into
bootloader mode. After a successful flash, confirm the new firmware is
actually running (not just that the loader exited 0) via `telem`'s
`timestamp_ms`/`loop_count` resetting to a small value — a stale-but-still-
responding board can look deceptively like a success.

## Generated protocol and durable parameters

The only human-edited source for shared state, fault, command, group, and
parameter definitions is `protocol/schema.json`. Run
`python protocol/generate_protocol.py` after editing it, and use `--check` in
tests/CI to detect stale generated artifacts. The generator writes the C++
IDs/tables, the pure-Python GUI module, documentation, and frozen vectors.
Generated files are checked in so embedded builds do not require Python.

### Digital-twin parameter and plant-identification hooks

`simulation/mujoco/v4_twin_279mm_baseline/twin/params_control.py` is generated
from the same schema, so firmware and twin use one parameter namespace. Run its
generator with `--check` alongside the protocol check when the schema changes.
The GUI Params tab is also generated from this schema and therefore exposes new
parameters without a hand-written widget.

The control group includes five non-persistent identification parameters:
`plant_id_en`, `plant_id_amp`, `plant_id_f0`, `plant_id_f1`, and
`plant_id_dur`. When enabled in RUNNING/JUMPING, `control_loop.cpp` adds a
linear-frequency symmetric torque chirp after the normal LQR/barrier and before
the existing torque clamps and wheel governors. It auto-clears at the requested
duration and on every controlled-state exit. These are test hooks, not normal
driving settings; use them only with the robot restrained and a reviewed test
procedure. The complete offline/hardware workflow is documented in the twin's
`README.md` and `HARDWARE_TEST_HANDOFF.md`.

Persistent Teensy parameters use two generation-numbered, CRC32-protected
LittleFS slots. A save is verified before it becomes current, the previous
valid slot remains available after interruption or corruption, and legacy
`params.bin` data is migrated on first boot. Recovery and migration are
reported in the robot log.

## Key constants (`teensy/src/config.h`)

| Symbol | Value | Meaning |
|---|---|---|
| `CONTROL_HZ` | 500 | Teensy main loop rate |
| `ESP32_BAUD` | 4 000 000 | Teensy↔ESP32 UART baud |
| `CAN_BAUD` | 1 000 000 | Both CAN buses |
| `CAN_INTER_FRAME_US` | 100 | Gap inserted between back-to-back CAN TX frames (was 500 — an 8-byte frame at 1 Mbps takes ~110–130 µs bit-stuffed, so 500 was ~4× over; under re-characterization) |
| `ODESC_NODE_L/R` | 0 / 1 | Wheel motor ODrive node IDs |
| `AK45_ID_L/R` | 11 / 12 | Hip motor CAN IDs |

## Coordinate system

+X forward, +Y left, +Z up (matches MuJoCo world frame).

## Leg geometry (v4 — 2026-08-07)

**The linkage was redesigned. Every link length changed, the tibia became a
dogleg, and the wheel shrank from Ø150 to Ø112.** Mechanical baseline drawing:
`components/2N_10mm_279mm.pdf`; the machine-readable form of the same geometry
is `simulation/2d/fourbar_optimizer_gui/presets/2N_10mm_279mm.json` (the two
agree to 0.01 mm). All lengths in the A-at-origin frame (A = hip motor output
shaft), +X forward, +Z up.

| Link | Old (18 mm as-built) | **v4** |
|---|---|---|
| femur A→C | 173.78 mm | **187.58 mm** |
| coupler F→E | 150.81 mm | **169.54 mm** |
| stub C→E | 35.13 mm | **39.01 mm** |
| tibia \|C→W\| | 129.39 mm (straight) | **185.91 mm (dogleg)** |
| F relative to A | (−58.87, +18.00) mm | **(−36.42, +37.54) mm**, \|AF\| = 52.30 mm |
| wheel | Ø150 (r = 75 mm) | **Ø112 (r = 56 mm)** |

The tibia is no longer straight. The drawing states the kink as
**`C_offset` = 5.28 mm** — the knee C's perpendicular offset from the E–W line,
away from the hip motor — with `|EC|` = 39.01, `|CW|` = 185.91, `|EW|` =
224.49 mm. The preset stores the same kink in the tibia's local frame instead
(W at 183.41 mm along the C→E axis, 30.35 mm perpendicular; a 9.4° bend at C).
The two agree — Heron on the 39.01/185.91/224.49 triangle gives 5.26 mm. Note
`|C→W|` ≠ the preset's `L_tibia` field because of this.

**Hard stops (CAD): `Q_RET` = +28°, `Q_EXT` = −57°, with 0° = femur
horizontal — 85° of stop-to-stop travel.** Positive q retracts (wheel up,
body low). The extended stop sits essentially on the 4-bar singularity
(kr = 0.939 against the solver's 0.95 cutoff), so there is no travel to be
recovered past it — the stop is protecting a real limit, not an arbitrary one.

Derived, in `teensy/src/control_loop.cpp`:

| Constant | Value | Meaning |
|---|---|---|
| `L_EFF_RET` | 0.098915 | \|W_z\| body-centre frame at **α = 0, q = +23°** |
| `L_EFF_EXT` | 0.363396 | \|W_z\| body-centre frame at **α = 1, q = −57°** |
| `WHEEL_R` | 0.056 | wheel radius [m] |

**The α endpoints are not the hard stops.** α is the fraction of the
*calibrated* span, and `define_limits()` puts its zero one `calib_backoff_rad`
in from the retract switch — so α = 0 is **q = +23°**, not the +28° stop
(α = 1 and the −57° stop do coincide). Evaluating the retracted end at the stop
gives 0.086777, which is **12.3% low**. Same trap `AngleRetractedExt.md` flags
for the hip scale: stance angles are not α endpoints. Both constants assume
`calib_backoff_rad` is at its 0.0872665 default.

`l_eff` is the body origin's height above the wheel axle, *not* CG-to-axle —
the same definition the previous 0.183117/0.295390 pair used (recomputing the
old geometry with this definition reproduces those two to 5 µm), so the LQR
gains keep meaning what they meant. Ride height (A above ground) runs
119.3 → 395.9 mm.

**Two stroke figures, both correct — don't mix them.** The drawing's headline
**279.95 mm** is the wheel's vertical travel over the optimizer's evaluated
band, which runs slightly past both stops (−57.64° … +28.65° = 86.29°). Over
the **hard stops** at −57° … +28° (85.0°) the travel is **276.62 mm**. Use
276.62 for anything the firmware can actually command; 279.95 is a
design-envelope number.

**One geometry input is still unverified and is flagged in the source:**

- `A_Z = −23.5 mm` (hip axis below body centre) is inherited from baseline-1
  and has not been re-measured on the v4 box. `L_EFF_RET`/`L_EFF_EXT` shift
  1:1 with it.

`M_BODY = 2.562 kg` now comes from the 2026-08-09 scale inventory: 3.518 kg
driving mass minus two 0.478 kg wheel assemblies. Only FF2 depends on it, and
the current `Default gains.json` keeps FF2 disabled.

**`WHEEL_R` also scales reported speed.** `wheel_vel_avg_ms` is
`turns/s × 2π × WHEEL_R`, so the same physical speed now reads 0.747× what it
did. Everything tuned in m/s — `vel_pi_*`, `v_cmd_ms`, `radio_vel_max`,
`profileN_vel_max` — is off by that factor until re-checked.

## Control algorithm (`teensy/src/control_loop.cpp`)

LQR on 3-state linearised inverted pendulum: `[pitch−θ_ref−trim, pitch_rate, wheel_vel_avg−v_ref]`.  
Gains are scheduled with leg height (α ∈ [0,1], retracted→extended).  
Balance-point trim offsets the pitch target so zero velocity holds at the true
balance lean when the CG isn't over the axle. Its height schedule is
`trim(α) = trim_ret + α(trim_ext − trim_ret) + lqr_trim_curve·α(1−α)`;
the curve coefficient leaves both endpoint trims unchanged, and zero preserves
the old linear schedule. The GUI groups all three parameters under Hip →
Pitch Trim vs Leg Height.

**Backward soft-limit barrier** — authority is asymmetric: the leg linkage
hits the ground on a hard backward lean well before forward pitch runs out of
room. Rather than an asymmetric clamp on `lqr_torque_limit` (which would also
cap the *positive* torque used to recover from that same backward lean —
tightening the wrong direction), the LQR gets one extra term, added right
after `tau_sym`: once pitch swings past a backward threshold
(`lqr_barrier_th_ret/ext`, α-scheduled like everything else — default ~12°,
between the vel-PI's saturated lean and the pitch-watchdog trip angle so it
bites first), `lqr_barrier_k` times the overshoot is subtracted from
`tau_sym` (sign convention: this drives more backward-recovery torque, same
direction `k_pitch` already pushes on a backward pitch error). Continuous at
the boundary and exactly zero inside it, so `lqr_barrier_k=0` (its default)
leaves today's tuning byte-identical; tune it up on the bench once the
threshold is confirmed to sit inside `pitch_wd_bwd_ret/ext`.  
Outer loops: velocity PI (sets θ_ref), yaw PI (differential torque). The
velocity PI uses directional conditional-integration anti-windup: its integral
freezes when an update would push the requested lean farther past the active
asymmetric clamp, but remains free to unwind back out of saturation.
Feedforward: FF1 cancels hip reaction torque; FF2 adds gravity compensation.

**Code-level backward target invariant** — a post-mortem on a fall found the
velocity PI's configured `theta_max_bwd_ret/ext` clamp, combined with a
negative (backward-leaning) `lqr_pitch_trim_ret/ext`, could ask for an
absolute pitch target past the pitch watchdog's own trip angle — the velocity
loop was, in effect, allowed to request a lean the watchdog would then fault
on. `control_safety.h`'s `safe_backward_theta_limit()` now caps the
*effective* backward clamp every tick, independent of what's persisted:
`theta_max_bwd <= pitch_watchdog_bwd + pitch_trim − margin`, with a fixed
`margin` of 3° (`BACKWARD_TARGET_MARGIN_RAD` in `control_loop.cpp`, not a
runtime param — a code-level floor, not a tuning knob). This tighter of the
two limits also feeds `g_state.theta_max_bwd` and the vel-PI anti-windup, so
health/saturation reporting reflects what's actually enforced. A degenerate
trim/watchdog pair that would leave zero backward authority clamps to 0
rather than honoring a configured limit past the watchdog.

**Direct velocity-term guard** — at the same fall, the direct `lqr_k_vel`
term (the `x2 = wheel_vel − v_ref` contribution to `tau_sym`) was found
opposing/cancelling much of the pitch and pitch-rate recovery torque right as
the pitch watchdog was about to trip on a full-reverse command.
`backward_velocity_term_guard()` now fades *only* the velocity term, and only
when it opposes recovery, once pitch passes the (already α-scheduled)
backward barrier threshold — linearly to zero at the backward pitch
watchdog angle. It leaves the term untouched inside the barrier and
untouched whenever it's already assisting recovery, so `pitch_term`/
`rate_term` keep full authority to arrest the fall near the mechanical
boundary while the tuned response elsewhere is unaffected.

**RUNNING hip-height rate limit** — the raw radio hip-height target (CH3,
`radio_hip_cmd`, normalized `t∈[0,1]`) is slewed by `hip_cmd_rate_lim`
[normalized extension/s] *before* `hip_cmd_to_setpoints()` converts it to
motor positions, so a snapped hip stick can't step the legs straight to a new
height — the default `0.2` gives a 5 s full-stroke transition and cuts down
reaction-current impulses and linkage chatter. `0` disables limiting. Only
applies to normal RUNNING hip-height commands: calibration, MANUAL, the ESTOP
hold ramp, JUMPING's own crouch/extend phases, and STANDING_UP's CROUCH/STIFFEN
ramps all set hip setpoints directly and bypass it. STANDING_UP's RECOVER phase goes
through `controlLoop_run()` but holds the hip override, which writes the
rate-limiter's shadow directly instead of slewing — that is what lets the
RUNNING handoff resume normal CH3 slewing from the pinned height with no step.
The disarm ramp
(RUNNING → STANDBY) slews from the last *filtered* command, not the raw radio
target, so releasing the hips doesn't also snap the target underneath the
ramp.

**Calibration only ever seeks the retract switch.** It finds the switch, zeroes
there, backs off `calib_backoff_rad`, and stops. The extended end of the span
comes from `calib_range_*_rad` — a property of the linkage, not something
calibration goes out and measures. Full phase order in
`Calibration/calibration.cpp`:

```
RELEASE_SWITCH? -> SEEK_SWITCH -> WAIT_ZERO_SYNC -> BACKOFF -> RAMPDOWN -> DONE
```

The leg's total commanded excursion is therefore `calib_backoff_rad` (5 deg by
default) either side of the switch. It never travels toward extension.

> **Removed 2026-08-08: the `SEEK_EXTENDED` / `RETURN_HOME` phases.** An earlier
> version drove the leg from the switch all the way out to the extended hard
> stop and read the over-torque trip as a measurement of the span, then walked
> it back. It was removed unrun: **the robot is unsupported and not balancing
> during CALIBRATION**, so swinging a leg through its full ~85 deg stroke there
> is a large unforced disturbance — to learn a number the mechanism already
> fixes and the CAD already gives. If the retract switch trips δ before the
> +28 deg stop, the usable span is `85 deg − δ`; set `calib_range_*_rad` to that
> rather than having calibration discover it by pushing into the stop.
>
> A consequence worth knowing: `calib_move_trq_lim` is now purely a *fault*
> threshold in every phase. There is no longer any phase where exceeding it is
> the intended outcome, so it should be set as low as the backoff move allows.

Two consequences of the α rescale, unchanged in kind from the old geometry:

- `gain_sched_alpha` now spans the real mechanism, so the extended anchor of
  every α-scheduled pair (`lqr_*_ext`, `hip_running_tff_ext`,
  `lqr_pitch_trim_ext`, `lqr_barrier_th_ext`) is reachable for the first time.
  Those anchors were tuned against a schedule that saturated near 0.825 —
  **re-check them, do not assume they transfer.**
- Any hip angle read from telemetry is normalised against this span.

The AK45 position scale itself is fine: 1 firmware-rad = 1 rad at the femur, no
reduction between the output shaft and pivot A.

### Right-leg-only calibration (bench)

`calibration_start()` already supports it: with `hip_l_enable = 0` the left axis
is forced straight to `CAL_DONE` and logs `Calib L: skipped (hip_l_enable=0)`,
and `calibration_done()` ignores it. So `hip_r_enable = 1`, `hip_l_enable = 0`,
both `wheel_*_enable = 0` runs the real seek/zero/backoff sequence on the right
hip alone, with the robot on a stand.

**One catch:** `measured_hip_alpha()` requires `hm_limits_L.valid &&
hm_limits_R.valid`. A skipped left axis never reaches `define_limits()`, so
`hm_limits_L.valid` stays false and `gain_sched_alpha` reports invalid (0) for
the whole session. Fine for calibrating and for reading raw hip angles, but
**gain scheduling is not exercised by a single-leg run** — don't read anything
into α, or into any α-scheduled value, from one.

To measure the switch-to-stop offset δ: after calibration completes, jog the
right hip toward retraction in `MANUAL` until it meets the retract hard stop and
read `hip_r_pos_rad`. Firmware-zero is the switch, so δ is that reading, and the
corrected `calib_range_r_rad = 1.48353 − δ`.

**Hip feedforward is α-scheduled** (`hip_running_tff_ret`/`_ext`, same α as
the LQR gains). The hip hold is a pure MIT impedance with no integrator, so
its steady-state sag is `(holding_torque − tff)/kp` — and because the 4-bar's
mechanical advantage changes with leg height, the holding torque does too
(measured 2026-07-28: ~4.0 N·m of hip torque at α≈0.05 versus ~2.7 N·m at
α≈0.145, i.e. *more* torque needed retracted — those readings were logged as
"10 A" and "6.8 A" before the MIT scale fix below). A single constant `tff`
therefore can only null the sag at one height; the two anchors let it track.
Both default to 0, so an untuned robot behaves exactly as before. Measure the
hold error at several α plateaus and fit the anchors from that rather than
extrapolating from one — and note `hip_running_kp` is bypassed entirely while
the roll controller is active (`hip_roll_kp` replaces it), so tune the anchors
in whichever mode you actually fly.

**AK45-10 MIT scale factors are per motor model** (`hip_motors.cpp`) — position,
Kp and Kd are common to the AK family, but **speed and torque are not**. The
AK45-10 is ±20 rad/s and ±8 N·m (§5.3 of the driver manual, v1.0.18 — whose
changelog specifically notes "Corrected the AK45-10 motor parameters"). This
driver previously used another model's ±65 / ±18, plus an invented ±20 "current"
range for the reply field, with three consequences, all silent:

1. The reply's third field is **shaft torque in N·m, not amps** — the manual's
   byte table says "current" but its own decoder scales it by the model's
   *torque* range. It was reported 20/8 = 2.5× too large, which is why hip
   telemetry used to plateau near "10 A" (really 4.0 N·m).
2. Every commanded feedforward torque was packed over ±18 and decoded by the
   motor over ±8, so only 8/18 = 44 % of it was delivered.
3. Reported hip velocity was 65/20 = 3.25× too large.

Both were confirmed against bench log `20260728T053232` before the change:
reconstructing the MIT impedance law from telemetry fit the reported torque at
0.400 = 8/20, and position-derivative versus reported velocity fit
0.308 = 20/65.

Telemetry renamed `hip_l/r_current_a` → `hip_l/r_torque_nm` (`TELEM_VERSION`
12 — same byte layout as 11, bumped so a stale ESP32/GUI rejects rather than
mislabels). Params tuned against the old scale are converted **once, on load**
by the param-store v2→v3 migration (`param_registry.cpp`): `hip_running_tff_*`
and `jump_torque_max` ×8/18, `calib_*_trq_lim` ×8/20, `jump_omega_max` ×20/65.
The migration runs before the min/max clamp, since several of those params also
got tighter bounds in the same change. `ff1_kt_hip` is now 1.0 — FF1's input is
already torque, so there is no Kt to apply. The GUI's log analyzer still reads
`TELEM_VERSION` 11 captures, rescaling them on load so old runs stay comparable.

**Validated against a physical standard (2026-08-02).** Hip torque telemetry
was checked end-to-end with a lever arm and a scale — the only way to catch
this class of bug, for the reason in the last paragraph below. Setup: hip L in
`MANUAL`, `kp = kd = 0` so the commanded `tff` is the entire torque, 100 mm
lever (1 N·m = 1020 g), log `20260802T022757`.

| Commanded `tff` | Old-scale firmware would give | This firmware should give | Measured |
|---|---|---|---|
| 0.3 N·m | 136 g | 306 g | **300 g** |
| 0.5 N·m | 227 g | 510 g | **400 g** |

Command → telemetry agreement is exact: `tff = 0.500` read back as
`hip_l_torque_nm` = 0.5025 N·m mean (sd 0.021, n = 250, ratio **1.005**), i.e.
inside the 0.0039 N·m quantisation step of the 12-bit field.

Scope of the claim — the scale was **hand-held**, so its repeatability is only
about ±20 % (the two points come out at 98 % and 78 % of prediction; ~22° of
unnoticed lever tilt accounts for the low one on its own). That is ample to
confirm the corrected scaling and to exclude the old one by a wide margin, and
**not** enough to resolve gearbox efficiency, which is expected to be a 10–20 %
effect in the same direction. Treat reported hip torque as motor-side: it is
derived from q-axis current and cannot see planetary losses, so true output
torque is somewhat lower. Quantifying that needs a clamped, horizontal,
bench-mounted rig.

**Electrical cross-check (2026-08-02): holding 3 N·m draws 0.4 A from the 24 V
supply** (9.6 W). At stall there is no mechanical output, so essentially all of
that is winding copper loss — which makes it an independent check on the torque
scale, arrived at without a lever or a scale.

Predicted from the datasheet constants alone: Kt 0.127 N·m/A motor-side × 10:1
= 1.27 N·m/A at the output, so 3 N·m needs 2.36 A of phase current; the motor
constant km = 0.0858 N·m/√W implies phase-to-phase R = (0.127/0.0858)² = 2.19 Ω;
copper loss = 1.5·I²·R_phase = 0.75·I²·R_pp = **9.2 W = 0.38 A at 24 V**.
Measured 0.4 A — **5 % agreement**, from a completely different direction than
the lever test.

Extrapolating the same relation (P ∝ τ²) gives the thermal picture, and it is
the reason hip holding torque is worth watching:

| Hold torque | Phase current | Copper loss | 24 V draw | Winding rise |
|---|---|---|---|---|
| 2.5 N·m (rated continuous) | 1.97 A | 6.4 W | 0.27 A | ~45 °C |
| **3.0 N·m (measured)** | 2.36 A | 9.2 W | **0.40 A** | ~64 °C |
| 4.07 N·m (log hold B) | 3.21 A | 16.9 W | 0.70 A | ~118 °C |
| 4.67 N·m (log peak) | 3.68 A | 22.2 W | 0.93 A | ~156 °C |

That 2.5 N·m rated continuous corresponds to a ~45 °C rise is self-consistent
with how CubeMars would set the rating, which is a good sign the model is
sound. It also means the 4.07 N·m holds seen in `20260728T053232` would settle
at roughly 143 °C winding — **past the 130 °C Class B limit** — if sustained.
They were not sustained (11.9 s, far under the ~30 s winding time constant), so
nothing was damaged, but it is not a pose to park in.

Two caveats on that table. The temperature column uses R_th = 7 °C/W
(`R_th_wc 1.0 + R_th_ca 6.0` from `simulation/mujoco/*/params.py`), which is an
estimate and has never been measured — treat the °C values as indicative, the W
column as solid. And it is not recorded whether the 0.4 A was total bus current
or the increase on applying torque; if it includes quiescent electronics draw,
the motor's true share is lower and the 5 % agreement is partly luck. Worth
pinning down with a before/after reading.

Note why a firmware-only round trip could never have caught the original bug:
the TX and RX errors partially cancelled. Commanding 0.5 N·m packed over ±18
delivered 8/18 of it, and the reply decoded over ±20 reported that back as
0.556 — an 11 % discrepancy that reads as noise, while the actual shaft torque
was 2.25× too small. Only an external physical reference separates the two.

**Wheel-velocity plausibility filter and debounced runaway watchdog** — the
ODrive encoder feed can develop escalating single-sample corruption (bench
log 2026-07-28: jumps up to 6.9 turns/s between consecutive 50 Hz samples,
against a free-spin physical ceiling near 1300 turns/s², rising from ~0 % to
14.6 % of samples over the last minute of a run). One such spike tripped
`FAULT_WHEEL_RUNAWAY` while the robot was otherwise balancing normally.
Two independent mitigations, both defence-in-depth — the real fix is
electrical:
1. `wheel_motors_poll()` validates each new encoder sample against the last
   accepted one via `wheel_vel_glitch_filter()` (`teensy/src/wheel_safety.h`),
   allowing `wm_vel_slew_max` [turns/s²] of change scaled by the time since
   the last accepted sample (so a CAN gap widens the window rather than
   rejecting the sample that ends it). Rejected samples hold the previous
   value and bump `vel_glitch_count`. It **fails open after 3 consecutive
   rejections**, so a genuine runaway can never be masked. `0` disables it.
   Filtering happens at source, so both runaway checks (RUNNING and
   STANDING_UP) and the balance loop's velocity term all get clean data.
2. The RUNNING runaway watchdog is now **debounced** (`WHEEL_RUNAWAY_MS`,
   50 ms) like the pitch/roll watchdogs, instead of ESTOPping on a single
   over-limit sample. Much shorter than their 200 ms because a genuinely
   runaway wheel covers ground fast.

`wheel_runaway_en` (default 1) switches that watchdog off entirely, for bench
work that legitimately spins the wheels past `2 × wm_vel_limit` — free-spin
torque tests, encoder-scaling checks. With it off the per-tick velocity
governor still clamps commands, but nothing backs it up and a real runaway
falls to the pitch watchdog alone. It is **persistent**, so a disable survives
a power cycle; `setup()` logs an unconditional `WARN` on any boot that comes up
with it cleared, which is what replaces the boot-to-safe guarantee.

**Jump phase machine and overrun fault** — `JUMPING` is
`CROUCH → EXTEND → RETRACT → LANDING → HANDOFF → RUNNING`. The former flat
3000 ms state timer and blind `JP_DONE` settle have been removed. CROUCH and
RETRACT durations follow their live angle/speed settings, EXTEND has its own
cap, landing must be detected before `jump_land_timeout`, and HANDOFF must
capture before `jmp_handoff_timeout`. A complete budget derived from those
settings plus `JUMP_OVERRUN_MARGIN_S` remains as a final scheduling guard.
Any phase timeout raises `FAULT_JUMP_TIMEOUT`; it never silently hands an
unfinished jump to RUNNING. With `jump_enable=0` or invalid hip calibration,
the request remains a no-op rather than a fault.

`JUMPING` is reached through GUI/API `SET_MODE(JUMPING)` or the CH6 rising edge
in SIMPLE live-tune mode. Both remain gated by persistent `jump_enable`, whose
default is 0. The two hardware jumps in `LOG0015.WLOG` are the current reference
captures.

**`jump_effort` is persistent and defaults to 1.0.** EXTEND commands
`jump_torque_max × jump_effort`. `jump_torque_max` remains the reviewed ceiling;
`jump_effort` is the scale factor. It is deliberately not named for a vertical
speed because extension is still open loop. The independent `jump_enable=0`
default is the boot-safe arming gate.

**Note the jump is a threshold, not a proportion.** During EXTEND `kp = 0`, so
the commanded `tff` is the *entire* hip torque — `hip_running_tff` is not
applied. Holding the robot up at a crouched pose already takes roughly
3.5–4 N·m per hip, so anything below that collapses the leg instead of
extending it and there is no jump at all, at any effort. `jump_torque_max`
defaults to **7.0**, the AK45-10 peak, which puts the useful band of
`jump_effort` at roughly **0.6–1.0**. A lower ceiling does not give a gentler
jump; it gives no jump.

**Phases are specified as angle + speed, not duration.** `jump_crouch_time` and
`jump_ramp_up` are retired (IDs `0x0417`/`0x0418` are dead — the param store
skips unknown IDs on load, so an old store is fine). A duration says nothing
about where the leg ends up or how fast it is moving when it gets there, and the
crouch was worse than merely unintuitive: its *distance* was not a parameter at
all — it always went to the calibrated retracted endpoint — so the same
`jump_crouch_time` meant a completely different crouch speed depending on the
ride height you launched from. Two jumps from different heights were two
different manoeuvres, and none of them were repeatable.

| Parameter | Unit | What it sets |
|---|---|---|
| `jump_crouch_angle` | rad (GUI: deg) | CROUCH target, as hip extension from the retract switch |
| `jump_crouch_speed` | rad/s (GUI: deg/s) | **peak** hip speed getting there |
| `jump_extend_angle` | rad (GUI: deg) | how far EXTEND pushes, stated forwards |
| `jump_retract_angle` | rad (GUI: deg) | the landing pose; **negative = return to the pre-jump pose** |
| `jump_retract_speed` | rad/s (GUI: deg/s) | **peak** hip speed of the tuck back |
| `jump_retract_torque` | N·m | commanded hip-feedback torque ceiling during RETRACT |
| `jump_torque_rate` | N·m/s | EXTEND torque onset rate |
| `jump_nudge_fwd_vel` | m/s | forward offset added to the pilot's live velocity command |
| `jump_nudge_fwd_dur` | s | how long that offset is active immediately before EXTEND |

At EXTEND→RETRACT, firmware carries the measured hip position and velocity into
a 15 ms smooth braking blend before reversing toward the landing pose. The
blend shortens automatically near the calibrated extended limit. During all of
RETRACT, `jump_retract_torque` (default 7 N·m) scales `jump_kp`/`jump_kd`
together to cap the predicted feedback request; it is not a motor current limit,
so impact or external back-driving can still report more torque.

Phase durations are now *derived*: `1.875 × travel / peak_speed`, the 1.875
being the quintic minimum-jerk profile's peak-to-mean rate ratio (asserted in
`test_control_math.cpp`). CROUCH and RETRACT use `standup_min_jerk_position()`
/ `standup_min_jerk_rate()` from `standup_safety.h` — the same helpers
STANDING_UP's CROUCH uses — instead of the linear ramps they had, so both ends
of both ramps now start and stop with zero commanded velocity and acceleration.
Speeds are specified as **peak** because that is what the motor and the balance
disturbance actually see; average would understate both by nearly 2×.

`jump_torque_rate` replaces `jump_ramp_up` for the same class of reason: a fixed
softstart *duration* means a harder push ramps in proportionally faster, so the
most violent jumps got the sharpest torque step. A rate makes onset time scale
with the torque being commanded.

The forward nudge is active only during the final `jump_nudge_fwd_dur` seconds
of CROUCH. The effective request is always `stick velocity + jump_nudge_fwd_vel`;
the nudge never replaces the operator command. A zero velocity or duration
disables it. Defaults are 0.15 m/s and 0.10 s.

**Live landing detection is gyro-only.** Starting at RETRACT, it sums changes
in the full 3-axis gyro vector over a fixed 12 ms window.
`jump_land_gyro_imp` must be exceeded by at least two fresh IMU reports, so a
held 400 Hz value spanning multiple 500 Hz control ticks and a lone corrupted
sample cannot trigger it. The detector is blanked for `jump_land_min_air` after
RETRACT begins, rejecting the launch impulse. Defaults are a 2.5 rad/s gyro
threshold, 0.16 s blanking, and 1.0 s timeout. `jump_air_accel_z` and
`jump_land_accel_z` remain reserved for parameter-file/protocol compatibility
but have no effect.
The 0.16 s blanking window leaves about 50 ms before the earliest reference
touchdown and rejects early-tuck rotation, which can otherwise look like contact
while both wheels are airborne. The gyro detector
found both reference contacts 0.560/0.570 s after the jump request and
0.224/0.210 s after RETRACT began.

The detector consumes only fresh gyro report timestamps. This matters because
the production BNO086 report arrives at 400 Hz while the control loop runs at
500 Hz: held values are ignored rather than counted as extra impact evidence.

**LANDING and HANDOFF use the normal running controller with scoped recovery
authority.** LANDING is an explicit telemetry phase at contact. HANDOFF keeps
the same velocity, yaw, roll, hip, scheduled LQR, barrier, feedforward, and
wheel-governor code as RUNNING, while applying only these temporary overrides:

| Parameter | Default | HANDOFF effect |
|---|---:|---|
| `jmp_handoff_kp_mul` | 1.5 | multiplies scheduled LQR pitch gain |
| `jmp_handoff_kr_mul` | 1.5 | multiplies scheduled pitch-rate gain |
| `jmp_handoff_kv_mul` | 1.0 | multiplies direct wheel-velocity gain |
| `jmp_handoff_torque` | 0.6 N·m | temporary symmetric/per-wheel torque limit; 0 inherits RUNNING |
| `jmp_handoff_vel_lim` | 10 turns/s | temporary wheel governor/runaway baseline; 0 inherits RUNNING |
| `jmp_handoff_pitch` | 0.05 rad | trim-relative pitch-error capture band |
| `jmp_handoff_rate` | 1.0 rad/s | pitch-rate capture band |
| `jmp_handoff_hold_s` | 0.15 s | continuous time in the full capture band |
| `jmp_handoff_timeout` | 1.5 s | recovery deadline before `FAULT_JUMP_TIMEOUT` |

Both wheels must also return inside the normal `wm_vel_limit` before capture.
At landing the hip command limiter is seeded from measured pose, so CH3 slews
back in without a position step. The controller state is intentionally carried
across HANDOFF→RUNNING; only the temporary authority overrides are removed.

**The ground-tuned wheel loop remains active through CROUCH/EXTEND/RETRACT.**
Zero wheel torque there is unsafe when a weak attempt never leaves the floor:
at the crouched effective length the open-loop inverted-pendulum time constant
is about 0.1 s. The detector now supplies the timing required for a future
airborne reaction-wheel controller, but the present firmware does not switch to
one; free-spinning wheel velocity is still interpreted by the ground LQR as
forward velocity.

**Expect a small hop, and don't trust the old numbers.** At 3.518 kg total
(`params.py`), an average leg Jacobian of `dz/dq ≈ 0.19 m/rad`
((`L_EFF_EXT − L_EFF_RET`)/80°) and a *measured* ~4 N·m static hip hold at
α ≈ 0.05, gravity alone consumes most of the AK45-10's 7 N·m peak at the crouch.
That leaves roughly 3 N·m per hip net, ~9 m/s² of body acceleration, and the
`jump_omega_max` taper collapsing torque past ~10 rad/s — so **5–10 cm**, in
line with the twin's recorded ~70 mm (`flip_analysis.py`). The 283 mm design
target there and the 282.65 mm optimizer result are **pre-v4 and invalid**
(`COMPONENTS.md`). Thermally the push is a non-issue: 0.15 s at 7 N·m is ~7 J of
copper loss against a ~30 s winding time constant. It is the *crouch hold* that
sits above the 2.5 N·m continuous rating, not the jump.

Bring-up ladder, one step per log, reviewing `jump_state`, `hip_l/r_torque_nm`,
`hip_l/r_pos_rad`, `gain_sched_alpha`, `wheel_vel_avg_ms` and `pitch` between
each. Suspended sign check first at `jump_effort` 0.05 — during
`jump_state == 1`, `gain_sched_alpha` must **increase**. Then, given the
threshold behaviour above, the ground ladder is 0.5 → 0.6 → 0.7 → 0.8 → 0.9 →
1.0 against the 7.0 N·m ceiling; expect nothing at all below ~0.6. The most
likely first failure is `FAULT_PITCH_WATCHDOG` on landing — flight is
~0.2–0.3 s against the watchdog's 200 ms debounce. Measure that before deciding
whether to mask anything.

`JUMP_MELODY` is non-blocking; depending on landing and recovery time it may
continue after RUNNING resumes.

**CAN TX is no longer write-and-hope** — `FlexCAN_T4::write()` returns 1 for
"placed in a hardware mailbox" and -1 for "all mailboxes busy, queued to the
bounded software TX queue", which drops silently once full. Both drivers
discarded that return entirely, so a saturated bus lost torque setpoints and
mode changes invisibly. `hip_motors.cpp` and `wheel_motors.cpp` now count
deferrals; a run of 200 consecutive (~200 ms of a bus that has stopped
draining) latches a TX stall, logs once, and clears `hm_*.ok` / `wm_*.ok` — a
motor you cannot command is not one you can balance on. A single deferral is
normal under burst load and never faults. The wheel latch clears with
`wheel_motors_clear_errors()`; the hip latch is REBOOT-severity like the other
hip feedback faults. Lifetime counts are available via
`{hip,wheel}_motors_tx_defer_count()` but are **not** in telemetry yet.

**Wheel readiness confirms the ODrive axis state** — `wm_*.ok` previously meant
only "encoder feedback is fresh". An axis that dropped out of
`CLOSED_LOOP_CONTROL` keeps streaming perfectly good encoder estimates while
ignoring every torque command, which read as a healthy wheel right up until the
robot fell over. `axis_confirmed()` now also requires a fresh heartbeat
(`WM_HB_TIMEOUT_MS` 500) reporting `AXIS_CLOSED_LOOP` whenever `wm_mode` isn't
IDLE. Two deliberate escape hatches so a diagnostic can't brick the robot: the
check is inert until a heartbeat has actually been seen (`hb_ever_heard` — if
ODrive heartbeat transmission is off, behaviour is unchanged plus a one-time
warning), and it is suspended for `WM_MODE_SETTLE_MS` (500 ms) after a mode
change so the ODrive has time to reach closed loop. **Confirm on the bench that
heartbeats are actually arriving** — if the one-time "no ODrive heartbeat seen"
warning appears, this check and the pre-existing `wm_*.error` check are both
inert.

**Param store never self-formats** — `param_init()` used to call
`s_fs.format()` on a mount failure, erasing both CRC-protected generations from
inside the init path of the recovery design meant to protect them. The
framework's own `LittleFS_Program::begin()` already does mount → format → mount,
so a virgin chip is still handled on first boot; the removed retry could only
ever fire after that had *also* failed. Mount failure now means "run on
compiled defaults and say so".

**Roll controller (active suspension)** — off by default (`roll_ctrl_en`). In
RUNNING and the post-contact jump LANDING/HANDOFF, a PI-D loop on roll
angle/rate (`roll_kp`, `roll_ki`, `roll_kd`) produces a
differential hip position offset (`+offset` one leg, `−offset` the other,
clamped to `roll_offset_max` and to calibrated hip travel) that levels/leans the
body about +X. The setpoint comes from radio **CH1**, scaled by the active
profile's `radio_roll_max` (per-profile `profileN_roll_max`, selected by CH9) and
slew-limited (`roll_rate_lim`) so a snapped stick doesn't step-perturb pitch.
While active the hips are held with a soft, backdrivable impedance
(`hip_roll_kp`/`hip_roll_kd` replace `hip_running_kp`/`kd`; `hip_running_tff`
still carries the static leg load). A steady roll shifts the CG laterally toward
the low wheel and wheels make no lateral force, so the setpoint is hard-clamped
and a roll watchdog (`roll_watchdog_en`, `roll_watchdog_limit`) ESTOPs with
`FAULT_ROLL_WATCHDOG` if `|roll|` exceeds the limit for > 200 ms. FF1 uses the
hip-current *sum*, so pure differential roll motion cancels in it.

The **integral term** (`roll_ki`, clamp `roll_int_max`) exists because
differential-hip-offset → roll-angle is essentially a static gain, not an
integrator, so a P-only law *must* leave steady-state error against any
standing asymmetry — measured −1.9° mean roll, 98 % one-sided, on the
2026-07-28 run. Defaults to **0** (disabled), which reproduces the previous PD
response exactly. Anti-windup reuses the velocity PI's conditional-integration
helper (`velocity_pi_integral_step()`, symmetric limits here): the integral
freezes only when it would push an already-saturated offset further past
`roll_offset_max`, and always stays free to unwind. It is cleared whenever
roll control is disabled and on every RUNNING entry, so re-enabling can't snap
the legs apart from a stale value.

Tune it knowing what the integrator physically *is*: a persistent differential
leg offset, i.e. a permanent lateral CG shift toward the low wheel — the exact
quantity `roll_watchdog_limit` guards. That is why `roll_int_max` is a separate,
deliberately small clamp (default 0.1 rad·s) rather than relying on
`roll_offset_max` alone. It also masks rather than fixes a mechanical
asymmetry, so it is worth confirming the bias isn't a real CG/leg-length
problem first. There is no integrator state in telemetry, but the total
commanded offset is observable as `(hip_l_cmd_pos_rad − hip_r_cmd_pos_rad)/2`.

**Symmetric travel-headroom clamp.** The effective offset limit each tick is
`min(roll_offset_max, min(t, 1−t) × span)`, applied *before* the differential
is split onto the two legs. At normalized ride height `t` a leg has `t × span`
of retract headroom and `(1−t) × span` of extend headroom, and a differential
spends one of each whichever way it leans — so `min(t, 1−t) × span` is the
largest magnitude *both* legs can honour. The per-leg clamps against
`hm_limits_*` are still there but are now only a backstop; previously they were
the whole story, which meant one leg could saturate while the other didn't,
silently turning a symmetric offset into an asymmetric one and shifting mean
ride height mid-correction. `span` is the smaller of the two calibrated spans,
so a mismatched pair is governed by the tighter leg. The anti-windup is given
this same effective limit rather than `roll_offset_max`, so the integral can't
wind up against travel headroom the clamp is about to remove. This is what
makes a larger `roll_offset_max` safe at low ride height — the binding
constraint becomes headroom, automatically.

## Live parameter tuning (`teensy/src/live_tune.h`)

Lets an operator feel a gain/limit's effect live on the bench, via two radio
knobs (CH7, CH8), instead of editing the Params tab blind and re-arming to see
it. CH7/CH8 each drive one slot of whichever *gain group* is currently
selected by the CH5/CH6 switch combination — 3 groups of 2 slots, `LIVE_TUNE_SLOTS`
in `main.cpp`:

```cpp
static const LiveTuneSlot LIVE_TUNE_SLOTS[] = {
    // Group 0: CH5 down, CH6 up -- LQR pitch/rate (retracted)
    { 0, 7, PARAM_LIVE_TUNE_CH7_VAL, PARAM_LQR_K_PITCH_RET, -0.1f,  -0.5f },
    { 0, 8, PARAM_LIVE_TUNE_CH8_VAL, PARAM_LQR_K_RATE_RET,  -0.01f, -0.5f },
    // Group 1: CH5 up, CH6 down -- vel_pi KP/KI
    { 1, 7, PARAM_LIVE_TUNE_CH7_VAL, PARAM_VEL_PI_KP, 0.05f, 1.0f  },
    { 1, 8, PARAM_LIVE_TUNE_CH8_VAL, PARAM_VEL_PI_KI, 0.02f, 0.5f  },
    // Group 2: CH5 down, CH6 down -- roll KP/KD
    { 2, 7, PARAM_LIVE_TUNE_CH7_VAL, PARAM_ROLL_KP, 0.3f,  4.0f  },
    { 2, 8, PARAM_LIVE_TUNE_CH8_VAL, PARAM_ROLL_KD, 0.02f, 0.5f  },
};
```

**Gated behind `live_tune_multi_en = 1` (LEGACY).** In the default SIMPLE mode
CH5 is the SD-log switch and CH6 is the jump trigger, so no switch is free to
select a group and the knobs are inert. In LEGACY mode CH5/CH6 give up both of
those functions, which is why it is bench-only.

Active while `RUNNING` with CH5+CH6 selecting a group (CH5 down + CH6 up ->
group 0, CH5 up + CH6 down -> group 1, both down -> group 2, both up -> no
group / tuning inactive). Two safety properties, independent of which params
are currently wired:

- **Pickup, not snap.** A knob does nothing on entering a group until it's
  swept through that slot's target *current* value; only then does it "pick
  up" and start tracking 1:1. Prevents a gain jumping instantly to wherever
  the knob physically happens to be sitting when the group is selected.
  Resets every time live-tune mode is exited or the group changes (CH5/CH6
  combination changes, or leaving `RUNNING`) — re-entering always requires
  re-sweeping.
- **Explicit latch, nothing persists by accident.** While picked up, a slot's
  live shadow value (`live_tune_ch7_val`/`live_tune_ch8_val`, telemetry-visible)
  is what the control loop actually uses — real, felt effect on balance — but
  the real underlying param is untouched until `PARAM_LIVE_TUNE_LATCH`
  (`live_tune_latch`) is written `1`. One-shot: firmware commits every
  currently-picked-up slot and resets the flag. Un-picked-up slots are
  skipped, not latched at a stale value.

**Repointing a knob at a different param** for a future bench session is a
one-line edit to `LIVE_TUNE_SLOTS` + reflash. The only other requirement: the
target's read site in `control_loop.cpp` must go through
`live_tune_value(PARAM_X)` instead of a bare `param_get(PARAM_X)` — that's
what makes the override actually take effect.

Full step-by-step operator procedure and the CH5/CH6/CH7/CH8 radio table: see
"Live parameter tuning" in `radio_channels.md`.

## Radio rescue combo — clear ESTOP / reboot (`teensy/src/main.cpp`, `radio_update()`)

A transmitter-only escape hatch for when the GUI isn't connected: **both sticks
jammed into opposite corners** — CH3 and CH2 full up (`> 1990`), CH1 and CH4
full down (`< 1010`), debounced 3 ticks (~6 ms @ 500 Hz).

| Event | Effect | Cue |
|---|---|---|
| Rising edge, in `ESTOP` | `stateMachine_request_reset()` → `STARTUP`. Clears `fault_code` regardless of severity and re-runs the startup checks. | A5⇄E6 siren + white LED flash |
| Rising edge, in `STANDBY` | Nothing to clear — beep only, and the 3 s countdown starts. | same siren |
| Held 3 s | Full MCU reset (`SCB_AIRCR`), the same path as `CMD_ID_REBOOT`: hips out of MIT, wheels IDLE, flush, reset. | E6→A5→D5→G4 fall |

Notes:

- **Armed only in `STANDBY`/`ESTOP`** — never with torque live. The *hold timer*
  additionally survives `STARTUP`, because that is where the reset lands: a fault
  that can't really be cleared re-faults straight back to `ESTOP`, and dropping
  the timer in that gap would make the 3 s fallback unreachable in the one case
  it exists for. `STARTUP` is torque-free and already an accepted
  `CMD_ID_REBOOT` state (`cmd_allowed()`), so this widens nothing.
- Releasing the combo (or leaving those states) re-arms it, so a second attempt
  starts a fresh 3 s countdown rather than resuming one.
- Every term is gated on `g_ibus.alive()`: `IBus::channel()` returns `0` on
  signal loss, so without that gate a dead radio would satisfy both stick-low
  tests for free.
- Like `CMD_ID_REBOOT`, the reboot does **not** finalize an in-progress SD log —
  a recording started in `STANDBY`/`ESTOP` will be left unclosed.

## Motor direction / sign conventions

Every place a sign flip or direction constant is applied between a "positive means X physically" intent and the raw motor command. Reference for bench-testing motor direction (`ControllersTest.md` Phase 1) and for anyone chasing a motor spinning the wrong way.

### All sign-affecting entities

| # | Name | Type | Location | Current value |
|---|---|---|---|---|
| 1 | Wheel L TX (no flip — reference side) | hardcoded | `teensy/lib/WheelMotors/wheel_motors.cpp` (`L_hw = L`) | identity (+1) |
| 2 | Wheel R TX flip | hardcoded | `wheel_motors.cpp` (`R_hw = -R`) | −1, unconditional |
| 3 | Wheel R RX flip (encoder feedback) | hardcoded | `wheel_motors.cpp` (`pos = -pos; vel = -vel;`) | −1, unconditional |
| 4 | Yaw→per-wheel torque split | hardcoded (structural) | `teensy/src/control_loop.cpp` (wheel-torque output) | `tau_L = tau_sym − tau_yaw`, `tau_R = tau_sym + tau_yaw` |
| 5 | Hip L TX flip (commanded pos/vel/torque) | hardcoded | `teensy/lib/HipMotors/hip_motors.cpp` (`pack_and_send`) | −1, unconditional, applied when `id == AK45_ID_L` |
| 6 | Hip L RX flip (pos/vel/torque feedback) | hardcoded | `hip_motors.cpp` (`rx_callback`) | −1, unconditional, applied when `msg.id == AK45_ID_L` |
| 7 | Hip L/R retract-switch seek direction | compile-time hardware constants | `teensy/src/config.h` (`CALIB_L_SEEK_DIR`, `CALIB_R_SEEK_DIR`) | L `+1.0`, R `+1.0`; each points toward its retract switch in the normalized firmware frame |
| 8 | Hip normalized-command mapping | hardcoded (structural) | `teensy/lib/HipMotors/hip_motors.cpp` (`hip_cmd_to_setpoints`) | `t∈[0,1]`: 0 = switch-zero backoff limit, 1 = configured extended limit; endpoint selection follows each axis's seek direction |
| 9 | GUI hip jog slider → raw degrees | GUI-side, not firmware | `software/gui/tabs/hip_motors.py:493-501` | slider low end → `lo_deg` (≈ `min_rad`), slider high end → `hi_deg` (≈ `max_rad`) — **raw degrees, not the normalized `t`**; now the same physical sense on both sides since #5/#6 unify the frame |

### Per-motor: what positive should do

| Motor | Applicable entities | Positive command means |
|---|---|---|
| **wheel_left** | #1 (no flip), #4 (`−tau_yaw` term) | Positive torque/velocity → wheel drives robot **forward (+X)**. This is the reference side; should stay unflipped. |
| **wheel_right** | #2, #3 (−1 flip both ways), #4 (`+tau_yaw` term) | Positive torque/velocity, in the *firmware-frame* value (same value the control loop and GUI use, before the internal CAN flip) → wheel also drives robot **forward (+X)** — same convention as left, because the −1 flip compensates for the physically mirrored mounting. Flip should stay in place; do not "fix" it by changing sign elsewhere. |
| **hip_right** | #7 (`dir_R=+1`), #8 | Reference side, no CAN-level flip. Raw MIT position: **more negative = leg extends**, near `max_rad` (≈0) = **retracted**. Increasing position retracts the leg. Via GUI jog slider (#9): dragging toward the **low** end extends, toward the **high** end retracts. |
| **hip_left** | #5, #6 (−1 flip both ways), #7 (`dir_L=+1`, same as R), #8 | Positive command/position, in the *firmware-frame* value (same value calibration, jump FSM, GUI, and radio all use, before the internal CAN flip) → behaves identically to hip_right: more negative = extend, increasing = retract. The −1 flip at the CAN boundary compensates for the physically mirrored mounting, so nothing above `hip_motors.cpp` needs to know left and right are wired differently. **Requires recalibration** after this fix — the previous calibration's zero point and limits were established in the old (unflipped, backwards) frame. |



Full FSM diagram: `teensy/state_machine.md`.

| Value | Name | Description |
|---|---|---|
| 0 | `STATE_STARTUP` | Boot checks: waits for IMU NOMINAL + hip CAN heartbeats |
| 1 | `STATE_CALIBRATION` | Hip retract-switch homing — only from STANDBY; re-entering the calibration stick combo after a radio start cancels through DISARMING |
| 2 | `STATE_STANDBY` | Idle, motors energised but zero torque |
| 3 | `STATE_RUNNING` | Active balancing — LQR + vel/yaw PI — requires calibration valid |
| 4 | `STATE_ESTOP` | Fault latch — see fault table below |
| 5 | `STATE_MANUAL` | GUI direct control (MIT frames); watchdog 500 ms |
| 6 | `STATE_CMD_REJECT` | ~1 s transient: buzzer + red blink, auto-returns to prior state |
| 7 | `STATE_JUMPING` | Launch sequence from RUNNING (~0.9 s at default phase params); auto-returns to RUNNING. Triggered by GUI/API `SET_MODE` or the CH6 rising edge in SIMPLE live-tune mode |
| 8 | `STATE_STANDING_UP` | Arm-time settling window — crouch to a fixed leg height, stiffen the hips, balance with the pitch watchdog masked, then RUNNING |
| 9 | `STATE_DISARMING` | Normal active-state or radio-calibration exit: wheel IDLE immediately, hip torque ramps safely to zero, then STANDBY |

**Standing-up mode (`STATE_STANDING_UP`)**: entered on arm only when `standup_enable=1` and signed pitch is within `[standup_pitch_min, standup_pitch_max]` (negative = backward, positive = forward; checked once at arm time only); with `standup_enable=0` (default) arming goes straight to `RUNNING`, byte-identical to before this state existed. It exists because the pitch watchdog is checked from the very first RUNNING tick, so arming from any appreciable lean ESTOPs before the balance loop gets a chance to pull the robot in.

**It is not a separate controller.** Two phases: **CROUCH** moves the hips to the calibrated retracted endpoint over `standup_crouch_time` with a quintic minimum-jerk S-curve, at the same `hip_running_kp/kd` RUNNING uses and with wheels at zero. The S-curve starts and ends with zero commanded velocity and acceleration, removing the former linear ramp's instantaneous velocity steps. That endpoint is exactly `calib_backoff_rad` away from each retract switch (15° when that parameter was set to 15° before calibration); there is no separate stand-up hip-height parameter. After the ramp, the hips must remain within 2° of both targets and below 0.2 rad/s for 100 ms before continuing. Failure to settle within 2 s faults `FAULT_STANDUP_FAILED`, so elapsed trajectory time alone can never enable wheel balance. Passing the gate explicitly completes the normal hip-gain ramp, preserving full `hip_running_kp/kd` stiffness when **RECOVER** calls `controlLoop_run()` — the ordinary balance LQR, velocity PI, yaw PI, feedforward, per-wheel soft governor and wheel-runaway watchdog — with *only* the pitch watchdog masked (gated on `STATE_STANDING_UP`, so it can't be left suppressed by accident). The hips stay pinned at the calibrated retracted endpoint via the control-loop hip override, so CH3 can't walk them off that pose mid-catch. `standup_ret_gains=1` (default) additionally pins the gain-schedule alpha to the matching retracted anchor for the duration. Changing `calib_backoff_rad` after calibration requires recalibrating before arming, because calibration builds the endpoint stored in the hip limits.

Handoff to `RUNNING` happens once `|pitch − lqr_pitch_trim_ret| < standup_cap_pitch` and `|pitch_rate| < standup_cap_rate` hold continuously for `standup_cap_hold`. Keep `standup_cap_pitch` well inside `pitch_wd_bwd_ret − |lqr_pitch_trim_ret|`: at its former 0.12 default a capture at the backward edge of the band handed off ~0.002 rad from the backward watchdog trip angle, so a successful catch was immediately followed by `FAULT_PITCH_WATCHDOG`. The handoff deliberately does *not* call `controlLoop_reset()` — the loop is already running and settled, and resetting would step both torque and leg position exactly when the watchdog goes live. A separate pitch range (`standup_div_fwd/bwd`, debounced 50 ms) aborts a catch diverging mid-attempt; it is the only pitch bound in effect while the watchdog is masked, hence `standup_div_bwd` is tighter than `standup_div_fwd`. Full spec: `standing_up.md`.

### Canonical state colour table

One source of truth for `state → RGB`; used by Teensy LED, ESP32 Neopixel base hue, TFT banner, and GUI header.

| State | R | G | B | Appearance |
|---|---|---|---|---|
| `STATE_STARTUP` | 255 | 255 | 255 | white |
| `STATE_CALIBRATION` | 0 | 120 | 255 | blue |
| `STATE_STANDBY` | 255 | 180 | 0 | amber |
| `STATE_RUNNING` | 0 | 230 | 80 | green |
| `STATE_ESTOP` | 255 | 40 | 40 | red |
| `STATE_MANUAL` | 0 | 200 | 255 | cyan |
| `STATE_CMD_REJECT` | 255 | 120 | 0 | orange |
| `STATE_JUMPING` | 200 | 0 | 255 | magenta |
| `STATE_STANDING_UP` | 255 | 60 | 0 | red-orange, fast strobe |
| `STATE_DISARMING` | 255 | 180 | 0 | amber blink during normal torque ramp-down |

## Fault codes (`shared/comm_protocol.h`)

Set in `g_state.fault_code` before entering `STATE_ESTOP`. Non-zero only while in ESTOP.

| Value | Name | Cause | Severity |
|---|---|---|---|
| `0x00` | `FAULT_NONE` | No fault | — |
| `0x01` | `FAULT_IMU_ERROR` | IMU ERROR during startup | REBOOT |
| `0x02` | `FAULT_HIP_INIT_TIMEOUT` | No CAN reply from hip motors within 2 s of boot | REBOOT |
| `0x03` | `FAULT_HIP_FEEDBACK_LOST` | Hip CAN feedback timed out (> 20 ms) during operation | REBOOT |
| `0x04` | `FAULT_HIP_LARGE_POS_CMD` | Hip position jump exceeded `MAX_HIP_DELTA_RAD` | GUI_FIX |
| `0x05` | `FAULT_CALIBRATION_TIMEOUT` | Retract-switch homing safety check failed | REPOSITION |
| `0x06` | `FAULT_HUMAN_ESTOP` | ESTOP requested by GUI or radio | SOFT |
| `0x07` | *(reserved)* | Was `FAULT_PARAM_OUT_OF_BOUNDS` — removed; out-of-range param writes always clamp | — |
| `0x08` | `FAULT_PITCH_WATCHDOG` | pitch outside `[-pitch_wd_bwd, +pitch_wd_fwd]` (asymmetric, gain-scheduled ret/ext by leg height) for > 200 ms. Not checked in `STANDING_UP` | REPOSITION |
| `0x09` | `FAULT_WHEEL_RUNAWAY` | Wheel velocity exceeded 2× soft governor limit for > 50 ms. Requires `wheel_runaway_en=1` | SOFT |
| `0x0A` | `FAULT_IMU_LOST` | IMU left NOMINAL while RUNNING/JUMPING (silence or heavy packet loss) | REBOOT |
| `0x0B` | `FAULT_WHEEL_FEEDBACK_LOST` | Wheel encoder timeout or ODrive error during operation | REBOOT |
| `0x0C` | `FAULT_WHEEL_INIT_TIMEOUT` | No CAN reply from wheel motors within 2 s of boot | REBOOT |
| `0x0D` | `FAULT_STANDUP_FAILED` | Standup denied (pitch out of recoverable range) or exhausted retries/diverged | REPOSITION |
| `0x0E` | `FAULT_ROLL_WATCHDOG` | `|roll| > roll_watchdog_limit` for > 200 ms (lateral tip guard) | REPOSITION |
| `0x0F` | `FAULT_JUMP_TIMEOUT` | Landing/handoff timed out, or JUMPING overran its computed live phase budget | REPOSITION |

**Severity tiers:** SOFT → ESTOP→STANDBY directly; REPOSITION → reposition robot then reset; GUI_FIX → fix param in GUI then reset; REBOOT → power-cycle required.

Mirror `_FAULT_NAMES` / `_FAULT_DESCRIPTIONS` in `software/gui/tabs/telem_format.py` and `fault_description()` in `esp32/src/main.cpp` when adding/changing codes.

## Telemetry and param pipeline

```
Teensy main.cpp  send_telemetry()
    │  packs RobotState + sensor data into TelemetryPayload (247 bytes, TELEM_VERSION 12,
    │  see comm_protocol.h) — includes ESP32<->Teensy link-supervision fields (esp32_link_ok,
    │  esp32_status_age_ms, uart_rx_drops, uart_seq_gaps)
    │  splits into TELEM_A (118 bytes, offset 0) + TELEM_B (129 bytes, offset 118)
    │  sends both framed packets via CommLink at 50 Hz
    ▼
ESP32 on_teensy_packet()  (core 1, inside g_teensy.update()'s parse loop)
    │  version-checks TELEM_VERSION (mismatch → logged, packet dropped)
    │  copies fields into volatile g_telem_* globals
    │  enqueues the raw frame for uplink_task — never sends inline here (a blocking
    │  USB/TCP/UDP send in this parse loop was the root cause of intermittent
    │  NO TEENSY under WiFi load; see git log for "ESP32 Phase 1")
    ▼
ESP32 uplink_task  (core 0 — the only writer to Serial/TCP/UDP)
    │  drains control/result, ACK-paced bulk-log, and lossy telemetry queues
    │  in that order; every queue operation is bounded
    │  uses one CommLink sequence generator per physical output
    │  uses non-blocking TCP sends and disconnects a stalled peer after 250 ms
    forwards over USB serial (CP2102) and WiFi UDP/TCP as appropriate

Each ESP32 parser pass is byte-budgeted (`UART_PARSE_BUDGET_BYTES` and
`HOST_PARSE_BUDGET_BYTES`). Serial2 has one core-1 reader/writer, USB has one
core-1 reader and the sole core-0 writer, and each network stream has one
reader and one writer. No UART callback performs network or USB I/O.

For WiFi, the ESP32 reassembles each adjacent `TELEM_A`/`TELEM_B` pair and
sends one `COMM_TYPE_TELEM_FULL_WIFI` datagram at 50 Hz. UDP liveness and the
TCP command connection are supervised independently: losing telemetry marks
only the WiFi source stale, while a still-readable TCP connection remains
available for recovery. The GUI renews discovery/session ownership separately
and will rediscover an address after restart or DHCP change.

`COMM_TYPE_ESP32_STATUS` moved from `0x16` to `0x17` on 2026-08-02: it shared
`0x16` with `COMM_TYPE_COMMAND_RESULT`, distinguished by direction alone. Since
the ESP32 relays GUI frames to the Teensy verbatim, any `0x16` arriving from the
host side would have been accepted as a status heartbeat. **Flash Teensy and
ESP32 together** — a skewed pair reports `esp32_link_ok = false` (already
supervised) but the heartbeat stays down until both sides match.

Independently, the ESP32 sends its own COMM_TYPE_ESP32_STATUS heartbeat to the
Teensy at 5 Hz (ESP32<->Teensy link supervision, telemetry-only), and its own
WIFI_DIAG (WifiDiagPayload V2, 38 bytes) to the GUI at 5 Hz — both keep flowing
even if the other side of the link goes quiet, so the GUI can tell "ESP32
alive, Teensy silent" apart from "everything down".

Python GUI  flash_monitor.py  PacketDecoder._parse()
    │  decodes with _FMT_TELEM_A / _FMT_TELEM_B (struct.calcsize asserted at import)
    │  emits dict via TelemetryBus.instance().packet signal
    ▼
GUI tabs  (imu_tab, raw_data_tab, robot_visualizer_tab, …)
    subscribe to TelemetryBus and render

Params (GUI → Teensy):
    GUI sends CMD_ID_PARAM_SET frames → CommLink → param_registry.cpp
    Teensy replies with PARAM_REPORT packets (min/max/flags/name per param)
    GUI renders controls from PARAM_REPORT — no hardcoded layout needed
    Normal GUI commands use CMD_PAYLOAD_V2: uint32 request_id + the existing
    command bytes. Teensy replies with COMMAND_RESULT (accepted/applied or a
    structured rejection reason), and ReliableCommand also verifies the
    operation-specific telemetry effect. V1 remains accepted during migration.
```

**Propagation checklist** when adding/removing telemetry fields: see the `PROPAGATION CHECKLIST` comment in `shared/comm_protocol.h`.

## Stress-test & fault-injection instrumentation

Built during the Phase 7-9 WiFi reboot-loop / reliability investigation (see
`UARTplat.md` at the repo root for the full narrative — root causes found,
fixes applied, and verification history). Kept in permanently, not stripped
from builds: gated by explicit commands, inert by default (zero behavior
change unless a test harness deliberately arms them). Use this instead of
writing new ad-hoc test scripts — it already speaks the real wire protocol
through the real Teensy/ESP32/GUI stack.

### Deliberate frame corruption (`CMD_ID_TEST_INJECT_CORRUPT`, `0x14`)

Asks the Teensy or ESP32 to deliberately damage its own next N outgoing
`TELEM_A` frames, to verify a receiver's `CommLink` parser actually detects
and drops a bad frame rather than accepting garbage. Payload:
`uint8_t count, uint8_t target, uint8_t mode` (`target`/`mode` default to
0/1 for backward compat with older callers).

| Field | Values | Meaning |
|---|---|---|
| `count` | 1-255 | how many upcoming `TELEM_A` sends to damage |
| `target` | 0 = UART | Teensy's next `count` sends to the ESP32 over Serial5 — forwarded through by the ESP32 like any command |
| | 1 = WiFi | ESP32's next `count` UDP datagrams to the GUI — intercepted by the ESP32 itself (`forward_to_teensy()`), never reaches the Teensy |
| `mode` | 1 = CRC | flips the CRC-8 byte (checksum-compare path) |
| | 2 = END | flips the END byte (`PS_END` bad-byte path) |
| | 3 = length | claims an oversized on-wire length (length-guard/resync path, `CommLink.cpp` Fix 2) |

Implemented via `CommLink::send()`'s `corrupt_mode_for_test` parameter
(`shared/CommLink/CommLink.{h,cpp}`) — default `0` = no corruption, so every
real call site is unaffected. Armed/consumed by `g_test_corrupt_remaining` /
`g_test_corrupt_mode` (Teensy, `teensy/src/main.cpp`) and
`g_test_corrupt_wifi_remaining` / `g_test_corrupt_wifi_mode` (ESP32,
`esp32/src/main.cpp`).

GUI sender: `comm_commands.send_test_inject_corrupt(count, target, mode)`
(`software/gui/tabs/comm_commands.py`). Ground truth for whether it worked:
watch `WIFI_DIAG.wifi_uart_crc_drops` for `target=0`, or this GUI's own
`link_crc_drops` (`PacketDecoder`, "wifi" transport) for `target=1` — note
mode 2 (END byte) frames are correctly rejected but don't increment either
counter (a GUI-side observability gap, not a detection failure — see
`command_corruption_probes` below for a mode-agnostic way to confirm this).

### Command-frame corruption, receive side (`build_frame_corrupted()`)

The corruption above only exercises the *send* side (Teensy/ESP32 sending a
deliberately bad `TELEM_A`). `comm_commands.build_frame_corrupted(payload,
mode)` builds a COMMAND frame with the same 3 corruption modes, for testing
the *receive* side of the identical `CommLink` parser instead — i.e.
confirming a malformed frame from the GUI can't get misdispatched as a
command. Send with `comm_commands.send_frame(...)` like any other frame.

### GUI automation harness (`main.py --automation <scenario.json>`)

```
python software/gui/main.py --automation path/to/scenario.json
```

Forces WiFi as the active transport, drives load through the real senders
(`comm_commands.py`, `WifiTransport`) for a fixed duration, and writes a
report to `software/gui/logs/<name>_report.json` before exiting — no GUI
interaction needed. Implemented in `software/gui/tabs/automation_runner.py`
(`AutomationRunner`); full field docs are in its module docstring.

| Scenario field | Type | Meaning |
|---|---|---|
| `name` | str | used in the default report filename |
| `duration_s` | float | timed run length, starting once WiFi telemetry is first seen |
| `param_dump_period_s` | float | interval between `PARAM_GET(0xFFFF)` dumps (load generator; default 5) |
| `tcp_churn_period_s` | float | if set, force-close/reopen the GUI's own WiFi TCP command socket on this interval |
| `bootstrap_timeout_s` | float | max wait for first WiFi telemetry packet before starting anyway (default 30) |
| `snapshot_period_s` | float | interval for periodic counter snapshots (default 30) |
| `corrupt_injections` | list | `{t_s, target: 0\|1, count, mode: 1\|2\|3}` — see corruption command above |
| `rogue_tcp_connects` | list | `{t_s, hold_ms}` — opens a second, independent raw TCP connection to the ESP32's command port alongside the GUI's own, to verify the accept-swap logic (`loop()`, `esp32/src/main.cpp`) survives an interloper client cleanly and the GUI's own connection self-heals afterward |
| `command_corruption_probes` | list | `{t_s, mode: 1\|2\|3, param_id}` — sends a corrupted `PARAM_GET`, confirms no reply leaked, then sends a valid follow-up and confirms the parser resynced (receive-side guard + recovery, mode-agnostic) |
| `report_path` | str | output path, relative to `software/gui/`; default `logs/<name>_report.json` |

Report highlights: `pass`/`fail_reasons` (note — **expect `pass: false`
whenever `corrupt_injections` is used**; the injected CRC/seq-gap counters
are exactly what the pass/fail check watches, so a "failing" report from a
corruption-injection scenario is normal, not a regression), `reboot_count`/
`reboot_events` (from `wifi_esp_uptime_ms` decreasing between packets),
`telemetry.actual_hz`/`inter_arrival_jitter`, per-direction UART counters,
`link_down_events`, `wifi_link_quality`, `injection_results`,
`rogue_connect_results`, `command_probe_results`.

Past example scenarios (kept for reference) and their reports:
`software/gui/logs/wifi_*_report.json`.

### Arm / state-machine stress test (`trigger_running_test.py`, `stress_test_arm.py`)

Built to bench-verify the arm state machine (`state_machine.cpp`) without
needing hands on the physical RC transmitter — e.g. after touching
`req_running()`/`req_calibration()` or the radio arm path in
`main.cpp radio_update()`. Standalone (pyserial only, no Qt — same pattern as
`tools/trigger_log_test.py`), in `software/gui/tools/`.

**`CMD_ID_SET_MODE(STATE_RUNNING)` is now a real command**, not just a radio
trigger — previously `on_command()`'s `SET_MODE` handler had no `RUNNING`
branch at all, so arming was only reachable via the physical CH10 switch
(`radio_update()`). It's now routed through the identical `req_running()`
gate the radio path uses (same IMU/calibration/motor-enable checks), so
there's no separate/weaker software arm path.

**`PARAM_RUNNING_WHEEL_BYPASS_EN`** (`0x0429`, `run_wheel_bypass_en`) — lets
`req_running()` arm with `wheel_l/r_enable` off, for a pure
command/state-machine smoke test with zero real torque anywhere when
combined with `hip_l/r_enable` also off (bypassed via the existing
`PARAM_CALIB_BYPASS_EN`). Independent of `PARAM_CALIB_BYPASS_EN`, which only
ever covered the hip check — that asymmetry was the bug this param fixes. It is
persisted; boot logs a warning whenever it is restored enabled.

**GUI No Motors preset** — the firmware does not infer a special operating mode
from all four motor-enable values. The GUI explicitly writes and read-verifies
the four motor enables, both arm bypasses, `standup_enable`, and the
pitch/roll/runaway watchdog enables. A live motor-enable `1 -> 0` write sends
MIT-exit to that AK45 or IDLE to that ODrive before its polling is skipped; this
is the direct effect of the motor-enable parameter, not an implicit safety
override. **Full Robot** restores the affected safety parameters; reboot after
returning to a motor-enabled configuration so boot-time peripheral
initialization is rerun before real operation.

**Software disarm is explicit.** `SET_MODE(STANDBY)` from `RUNNING`,
`JUMPING`, or `STANDING_UP` enters `STATE_DISARMING`. Fault/ESTOP guards have
higher priority, wheel output is idled immediately, jump/stand-up sequencing
is cancelled, and the hip command ramps to zero before STANDBY is reported.
Re-arm/manual/calibration requests are blocked during this interval.

**Radio disarm interlock fires regardless of arm source.**
`radio_update()`'s disarm check is level-based and covers every energetic state:
```cpp
bool armed = alive && (ch10 > 1990);
if (!armed && (g_state.state == STATE_RUNNING ||
               g_state.state == STATE_JUMPING ||
               g_state.state == STATE_STANDING_UP)) {
    stateMachine_disarm_running();
}
```
`alive` (iBus signal) is false whenever no RC receiver is connected, so
`armed` is always false — meaning a software-triggered `RUNNING` gets
sent to `DISARMING` on the next ~2 ms tick unless a live receiver also has
CH10 physically held up. This is intentional (RUNNING should only *persist*
with a live radio link corroborating "armed", regardless of entry path) —
confirmed and left as-is rather than "fixed". Both tools below account for
this: they check for the `-> RUNNING (armed)` log line (the `on_running()`
entry action — a guaranteed one-shot event) rather than expecting the
`TELEM_A` `robot_state` field to still read `RUNNING` by the time they check,
since the armed state can be shorter than one telemetry period.

#### `trigger_running_test.py` — single arm + confirm

```
python trigger_running_test.py [port]
```

Sends `CMD_ID_SET_MODE(STATE_RUNNING)` once and reports whether the state
machine actually armed, via `comm_log` lines and live `TELEM_A` decoding
(`tabs.telem_format.decode_telem_a`, Qt-free, imported directly). Prints the
pre-arm state first and warns if it isn't `STANDBY` (the request will be a
silent no-op — `stateMachine_request_running()`'s own `STANDBY`-only latch).

#### `stress_test_arm.py` — multi-round stress test

```
python stress_test_arm.py [--port COM12] [--rounds 5]
```

Per round: a fuzzed out-of-range `SET_MODE` target (expect safe no-op, no
crash), a `CALIBRATION` request (expect denial while hips are disabled), an
arm into `RUNNING` (expect the `-> RUNNING (armed)` log line, then recovery
to `STANDBY`), and explicit recovery before the next round. Prints a
pass/fail summary table at the end and exits non-zero on any failure.

**Safety gate, checked live before anything else runs:** reads
`hip_l/r_enable`, `wheel_l/r_enable`, `imu_enable`, `calib_bypass_en`, and
`run_wheel_bypass_en` via `CMD_ID_PARAM_GET` and aborts if any motor-enable
param is actually on (RUNNING would command real torque), or if
`imu_enable`/`calib_bypass_en` are off (test wouldn't be meaningful/would be
denied outright). `run_wheel_bypass_en` is the one param it will flip on
itself if needed — safe because it's non-persistent — and it flips it back
off during cleanup regardless of pass/fail.

**Out of scope — no software RC channel injection.** Neither tool can
simulate physical transmitter input. `PARAM_IBUS_CH*` and the radio-derived
params (`radio_vel_max`, `radio_yaw_max`, `active_profile`, ...) are all
`PARAM_FLAG_READONLY` — firmware-written mirrors of the real iBus receiver,
with no command-side injection point. Profile switching (CH9) in particular
has no non-radio trigger at all right now. "Radio commands" in these tools
means the same *state transitions* the radio triggers (`CALIBRATION`,
`RUNNING`), issued instead through `CMD_ID_SET_MODE`.

### ESP-IDF task watchdog (diagnostic, not invoked manually)

`esp_task_wdt_init()`/`esp_task_wdt_add(uplink_task_handle)` in
`esp32/src/main.cpp` — kept permanently as a safety net, not a test you run.
This originally caught `uplink_task` hanging inside a blocking
`WiFiClient::write()` call. Production TCP output now uses `MSG_DONTWAIT` and
disconnects a non-reading client after 250 ms; the watchdog remains a final
safety net for any unrelated future task stall.

## Each driver has its own README

See `teensy/lib/<DriverName>/README.md` for wiring, API, and gotchas.
