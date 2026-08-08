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

**Two inputs are still unverified and both are flagged in the source:**

- `A_Z = −23.5 mm` (hip axis below body centre) is inherited from baseline-1
  and has not been re-measured on the v4 box. `L_EFF_RET`/`L_EFF_EXT` shift
  1:1 with it.
- `M_BODY = 1.638 kg` was not re-derived for the longer v4 links. Only FF2
  depends on it.

**`WHEEL_R` also scales reported speed.** `wheel_vel_avg_ms` is
`turns/s × 2π × WHEEL_R`, so the same physical speed now reads 0.747× what it
did. Everything tuned in m/s — `vel_pi_*`, `v_cmd_ms`, `radio_vel_max`,
`profileN_vel_max` — is off by that factor until re-checked.

## Control algorithm (`teensy/src/control_loop.cpp`)

LQR on 3-state linearised inverted pendulum: `[pitch−θ_ref−trim, pitch_rate, wheel_vel_avg−v_ref]`.  
Gains are scheduled with leg height (α ∈ [0,1], retracted→extended).  
Balance-point trim (`lqr_pitch_trim_ret/ext`, same α schedule) offsets the pitch
target so zero velocity holds at the true balance lean when the CG isn't over the
axle. Edit via the Params tab, or repoint a live-tune slot at it for a future
bench session (see below).  

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
hold ramp, JUMPING's own crouch/extend phases, and STANDING_UP's catch
sequencing all set hip setpoints directly and bypass it. The disarm ramp
(RUNNING → STANDBY) slews from the last *filtered* command, not the raw radio
target, so releasing the hips doesn't also snap the target underneath the
ramp.

**The calibrated hip span is measured at both ends.** Calibration finds the
retract switch, zeroes there, backs off `calib_backoff_rad`, then *drives to
the extended hard stop and measures it* rather than trusting a configured
number. Full phase order in `Calibration/calibration.cpp`:

```
RELEASE_SWITCH? → SEEK_SWITCH → WAIT_ZERO_SYNC → BACKOFF
                → SEEK_EXTENDED → RETURN_HOME → RAMPDOWN → DONE
```

`SEEK_EXTENDED` reuses the existing torque-trip machinery, with one deliberate
inversion: in the seek phases an over-torque trip means *something jammed* and
faults the axis, but here it means *we reached the stop* and is the
measurement. That is what `command_motion()`'s `stop_found` out-parameter
selects. The software limit is then backed off the measured stop by
`calib_backoff_rad`, mirroring the retract end, so `t = 1` never commands the
leg onto the stop itself.

**`calib_range_*_rad` is now a safety ceiling, not the answer.** If no stop is
found within it, calibration logs a `WARN` and falls back to exactly the old
configured-range behaviour — so a missing, soft, or mis-tuned stop can never
let the leg run away, it just degrades to the previous assumption and says so.
Watch for that warning: it means `calib_move_trq_lim` is too high, or the stop
isn't where it should be.

`RETURN_HOME` walks the leg back to the retract-backoff rest pose under the
same speed and torque limits before `RAMPDOWN`. Without it the leg would still
be at full extension when `RAMPDOWN` starts commanding `home_target` — a
near-full-span position step under a decaying `kp`.

For reference, the v4 CAD stops predict what the search should find:

```
Q_RET +28°, Q_EXT −57°  →  85° stop to stop = 1.48353 rad
  t = 0  →  q = +23°   (backed off the retract stop)
  t = 1  →  q ≈ −52°   (backed off the measured extended stop)
```

`calib_range_l_rad`/`calib_range_r_rad` default to **1.48353** as that ceiling.

> **Not yet bench-validated.** `SEEK_EXTENDED` deliberately drives the leg into
> a hard stop, and has only been built and unit-tested — never run on hardware.
> It is bounded three ways (torque trip, the `calib_range` travel ceiling, and
> `calib_seek_timeout`), but watch the first run with a hand near the estop, and
> confirm the measured stop lands near the 85° the CAD predicts. If the switch
> trips δ before the retract stop, the measured span comes out `85° − δ` — which
> is now handled automatically rather than being an assumption to verify.

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

**Jump phase budget and overrun fault** — `JUMPING` used to exit on a flat
3000 ms timer regardless of what the phase machine was doing. That was safe
only by arithmetic coincidence: the three phase budgets at their schema maxima
(`jump_crouch_time` 1.0 + `jump_ext_timeout` 1.0 + retract 0.2 = 2.2 s) happened
to fit inside 3 s, so raising one bound or adding a phase would have silently
started handing off to `RUNNING` mid-extension with the hips still under a
torque command. The two concerns are now separate: `jump_done()` requires the
phase machine to actually reach `JP_DONE` and hold the nominal pose stiffly for
`JUMP_SETTLE_MS` (300 ms), while `s_jump_deadline_ms` is derived at entry from
the same live params it guards (+ `JUMP_OVERRUN_MARGIN_S` 0.5 s) and is a pure
safety net — overrunning it raises `FAULT_JUMP_TIMEOUT` instead of quietly
succeeding. Its transition is registered *before* `jump_done` so an overrun can
never lose the race. With `jump_enable=0` (the default) or invalid calibration
limits, the sequence retires straight to `JP_DONE` and exits normally, exactly
as before — an unarmed jump is a no-op, not a fault.

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
RUNNING only, a PI-D loop on roll angle/rate (`roll_kp`, `roll_ki`, `roll_kd`) produces a
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
| 1 | `STATE_CALIBRATION` | Hip retract-switch homing — only from STANDBY; lowering CH5 after a radio start cancels through DISARMING |
| 2 | `STATE_STANDBY` | Idle, motors energised but zero torque |
| 3 | `STATE_RUNNING` | Active balancing — LQR + vel/yaw PI — requires calibration valid |
| 4 | `STATE_ESTOP` | Fault latch — see fault table below |
| 5 | `STATE_MANUAL` | GUI direct control (MIT frames); watchdog 500 ms |
| 6 | `STATE_CMD_REJECT` | ~1 s transient: buzzer + red blink, auto-returns to prior state |
| 7 | `STATE_JUMPING` | ~3 s jump sequence from RUNNING; auto-returns to RUNNING |
| 8 | `STATE_STANDING_UP` | Arm-time recovery from a fallen pose — retract legs, energetic wheel push, then RUNNING |
| 9 | `STATE_DISARMING` | Normal active-state or radio-calibration exit: wheel IDLE immediately, hip torque ramps safely to zero, then STANDBY |

**Standing-up mode (`STATE_STANDING_UP`)**: entered on arm only when `standup_enable=1` and pitch is within the recoverable range (`standup_pitch_fwd/bwd`, checked once at arm time only); with `standup_enable=0` (default) arming goes straight to `RUNNING`, byte-identical to before this state existed. Two phases: **CROUCH** ramps the hips to the retracted pose and holds them there rigidly for the rest of the sequence, at the same `hip_running_kp/kd` RUNNING uses (not a separate standup-only gain) so the eventual handoff is a hip *position* step only, not also a stiffness step — hips move once, to a fixed pose, and never actively right the robot. **RECOVER** does the actual catch entirely with wheel torque: a saturated P/D law on the trim-relative pitch error and pitch-rate (`tau = K_pitch*(pitch−lqr_pitch_trim_ret) + K_rate*pitch_rate`, same sign convention as the small-angle LQR, and the same trim RUNNING's LQR regulates to since legs are pinned retracted throughout) pushes the wheelbase back under the CG until pitch settles in-band around that trim, then hands off to `RUNNING` for the tuned LQR to take over. A separate, looser pitch range (`standup_div_fwd/bwd`, debounced 50 ms) aborts a catch that's diverging mid-attempt — deliberately not the same params as the arm-time gate, so a saturated catch has overshoot budget instead of self-tripping the instant it starts. Full spec: `standing_up.md`.

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
| `0x08` | `FAULT_PITCH_WATCHDOG` | pitch outside `[-pitch_wd_bwd, +pitch_wd_fwd]` (asymmetric, gain-scheduled ret/ext by leg height) for > 200 ms | REPOSITION |
| `0x09` | `FAULT_WHEEL_RUNAWAY` | Wheel velocity exceeded 2× soft governor limit | SOFT |
| `0x0A` | `FAULT_IMU_LOST` | IMU left NOMINAL while RUNNING/JUMPING (silence or heavy packet loss) | REBOOT |
| `0x0B` | `FAULT_WHEEL_FEEDBACK_LOST` | Wheel encoder timeout or ODrive error during operation | REBOOT |
| `0x0C` | `FAULT_WHEEL_INIT_TIMEOUT` | No CAN reply from wheel motors within 2 s of boot | REBOOT |
| `0x0D` | `FAULT_STANDUP_FAILED` | Standup denied (pitch out of recoverable range) or exhausted retries/diverged | REPOSITION |
| `0x0E` | `FAULT_ROLL_WATCHDOG` | `|roll| > roll_watchdog_limit` for > 200 ms (lateral tip guard) | REPOSITION |
| `0x0F` | `FAULT_JUMP_TIMEOUT` | JUMPING overran its computed phase budget without reaching `JP_DONE` | REPOSITION |

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
ever covered the hip check — that asymmetry was the bug this param fixes.
**Not persisted** — always boots to 0 (bypass off), unlike
`PARAM_CALIB_BYPASS_EN`, so it can't be left silently armed across a power
cycle.

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
