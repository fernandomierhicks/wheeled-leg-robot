# Digital twin of the v4 robot — `v4_twin_279mm_baseline`

## Context

Two MuJoCo sims exist (`simulation/mujoco/master_sim/`, `master_sim_jump/`), both
built around **baseline-1 geometry** (femur 173.78 mm, straight tibia, Ø150 wheel)
and a controller that is *not* the one the Teensy runs. The robot has since been
rebuilt to v4 (every link length changed, the tibia became a dogleg, wheel Ø112),
and the firmware control law has grown terms the sim has never had: α-scheduled
direct gains, a backward barrier, a velocity-term guard, a code-level backward
target cap, directional anti-windup, FF1/FF2, a roll PI-D with travel-headroom
clamping.

So today, sim and robot disagree in three independent ways at once — geometry,
control law, and plant parameters — which makes it impossible to attribute any
mismatch. The goal is a sim you can trust for two specific things:

1. **Find the stability boundary in sim, port it, and see the same boundary on the robot.**
2. **Optimize gains in sim and have them transfer without re-tuning.**

Both depend on the *margins* matching, not just the trajectories. Margins are set
almost entirely by four quantities the sim currently guesses: **body inertia about
the wheel axle**, **total loop delay**, **wheel torque scale/bandwidth**, and
**effective pendulum length vs leg height**. Those four drive the test program below.

Decisions taken (2026-08-08): new package alongside the existing two (no renames);
firmware law hand-ported to Python **plus** an automated equivalence test against a
host-compiled build of the real `control_loop.cpp`; v1 scope is balance + drive +
yaw (roll and jump modelled, not hardware-validated yet); all four bench rigs
available.

---

## Part 1 — The new package

New directory `simulation/mujoco/v4_twin_279mm_baseline/`. `master_sim/` and
`master_sim_jump/` are left byte-identical — the 4-bar geometry optimizer still
lives in the latter and still works.

Seed it by copying `master_sim_jump/`, then make these changes. Everything here is
a correction, not a preference:

### 1.1 Geometry — v4, with a dogleg tibia

`master_sim_jump/physics.py:66 solve_ik()` computes `W = C − L_t·(sin α, cos α)`:
a **straight** tibia. The v4 tibia is kinked. Add a `w_perp` field and change the
W computation to place W in the tibia's local frame:

```
W_x = C_x − L_t·sin(α) + w_perp·cos(α)
W_z = C_z − L_t·cos(α) − w_perp·sin(α)      (sign fixed by the unit test in 1.3)
```

Values come from `simulation/2d/fourbar_optimizer_gui/presets/2N_10mm_279mm.json`,
which agrees with the drawing `components/2N_10mm_279mm.pdf` to 0.01 mm:

| Field | Value [m] | Note |
|---|---|---|
| `L_femur` | 0.187577 | A→C |
| `L_stub` | 0.039013 | C→E |
| `L_tibia` | 0.183411 | **local-frame** C→W axial component, *not* \|C→W\| = 0.185910 |
| `w_perp` | −0.030354 | new field; the kink |
| `Lc` | 0.169536 | F→E |
| `F_X`, `F_Z` | −0.036425, +0.037537 | **relative to A** |
| `wheel_r` | 0.056 | was 0.075 |
| `A_Z` | −0.0235 | **unverified** — measured by T0.3 |

`RobotGeometry.F_Z` in `params.py` is a **body-centre** coordinate, the A-relative
one is what the drawing gives. `F_Z_body = A_Z + 0.037537`. Per CLAUDE.md this
exact confusion once cost a whole robot build — the twin stores the A-relative pair
and derives the body-centre value, never the reverse.

`leg_y` (0.1430 m, Y-offset of the leg plane) has not been re-checked for the v4
box — measure it with calipers when doing T0.1.

### 1.2 Hip angle convention and α

Firmware: `0° = femur horizontal, positive retracts`, hard stops `Q_RET = +28°`,
`Q_EXT = −57°`. Sim `q_hip`: `C_x = −L_f·cos(q)`, more negative = extended, current
values `−0.379 / −1.219`. **These are different frames.** Derive the affine map
once, put it in `physics.py` as a documented function, and never inline it.

α is *not* `(q − Q_RET)/(Q_EXT − Q_RET)` as `controllers/lqr.py:148` has it. Firmware
α is the fraction of the **calibrated** span, whose zero sits `calib_backoff_rad`
(0.0872665 = 5°) inside the retract switch:

```
α = 0  ->  q = +23°   (NOT the +28° stop)
α = 1  ->  q = −57°   (== the stop)
```

Replicate that, including the backoff, or every α-scheduled gain lands at the wrong
leg height.

### 1.3 The geometry acceptance test (do this before anything else)

`tests/test_geometry_matches_firmware.py` must reproduce the firmware's own derived
constants from the twin's IK, to the digit:

```
l_eff(α=0) == 0.098915    # L_EFF_RET, control_loop.cpp
l_eff(α=1) == 0.363396    # L_EFF_EXT
vertical wheel stroke over the hard stops == 0.27662 m
ride height (A above ground) == 0.1193 -> 0.3959 m
```

where `l_eff = |W_z|` in the **body-centre** frame (body origin above the axle, not
CG-to-axle). If these don't reproduce, the geometry, the frame, or the α map is
wrong — and there is no point running a single dynamic test until they do. This one
file is the cheapest, highest-value artifact in the whole plan.

### 1.4 Control law — port `control_loop.cpp`, keyed on firmware param names

New `firmware_control.py`, structured as a 1:1 transliteration of
`firmware/robot_teensy/teensy/src/control_loop.cpp` + `control_safety.h`. Its only
control input is a dict keyed by **firmware parameter names**. Delete
`controllers/lqr.py`'s Riccati path from the twin's control path entirely (keep it
in a `design/` module — it's still the right tool for *proposing* seed gains, just
not for *simulating* what the robot does).

Terms that must be ported, all currently absent from the sim:

- α-interpolated direct gains `lqr_k_pitch_ret/ext`, `lqr_k_rate_ret/ext`, `lqr_k_vel`
- α-interpolated `lqr_pitch_trim_ret/ext` (balance-point trim), and `l_eff` interpolation
- backward soft-limit barrier: `lqr_barrier_k`, `lqr_barrier_th_ret/ext`
- `backward_velocity_term_guard()` — fades only the velocity term, only when opposing recovery
- `safe_backward_theta_limit()` — 3° `BACKWARD_TARGET_MARGIN_RAD`, a code-level floor
- vel-PI with asymmetric `theta_max_fwd/bwd_ret/ext` and **directional** conditional-integration anti-windup, `vel_pi_kff`, `vel_pi_rate_lim`
- yaw PI, `yaw_pi_torque_max`, split `tau_L = tau_sym − tau_yaw`, `tau_R = tau_sym + tau_yaw`
- FF1 (hip reaction, from the hip torque **sum**) and FF2 (`M_BODY·g·l_eff·sin(pitch)`)
- `hip_cmd_rate_lim` slew on the hip height target; hip arm-in ramp `hip_running_ramp_s`
- hip MIT impedance with α-scheduled `hip_running_tff_ret/ext`, and `hip_roll_kp/kd` replacing `hip_running_kp/kd` when roll control is active
- pitch / roll / wheel-runaway watchdogs with their debounce windows, as sim abort conditions
- `wheel_vel_glitch_filter()` (`wm_vel_slew_max`) on the encoder feed
- clamps: `lqr_torque_limit`, `wm_vel_limit`

`M_BODY = 2.562 kg` now matches the 2026-08-09 mass inventory: 3.518 kg driving
mass minus two 0.478 kg wheel assemblies.

### 1.5 The equivalence test (`simulation/sil/` — currently an empty directory)

`simulation/sil/` exists and is empty; this is what goes in it.

1. A PlatformIO `native` environment building `control_loop.cpp` against stub
   `param_get` / `g_state` / `hip_motors` / `wheel_motors` / `millis()` shims.
2. A golden-vector CSV: a few thousand `(pitch, pitch_rate, wheel_vel, hip_L, hip_R,
   v_ref, omega_cmd, α, params…)` rows, spanning both α ends, both sides of the
   barrier threshold, saturated and unsaturated, integral wound both ways.
3. `tests/test_control_equivalence.py` runs both and asserts `|tau_py − tau_cpp| < 1e-5`
   on `tau_sym`, `tau_yaw`, `theta_ref`, and both hip setpoints.

This is what stops the twin from silently drifting the next time `control_loop.cpp`
is edited. Wire it into the same `--check` habit `protocol/generate_protocol.py`
already uses.

### 1.6 Actuator and sensor models — the corrections needed

| Item | Today in `master_sim_jump/params.py` | Should be | Source |
|---|---|---|---|
| Hip torque limit | `torque_limit = 7.0` | **±8.0 N·m** | AK45-10 §5.3 (driver manual v1.0.18) |
| Hip speed limit | `omega_max = 18.85` | **±20 rad/s** | same |
| Hip winding R | `R_eff = 0.22 Ω` | **≈2.19 Ω phase-to-phase** | firmware README's `(0.127/0.0858)²`; the 0.22 is off by 10× and makes every thermal/current number wrong |
| Hip `Kt_output` | derived from KV75 → 1.27 | 1.27 N·m/A ✓ | agrees, keep |
| Hip quantisation | none | 12-bit torque (0.0039 N·m step), 16-bit position | MIT protocol |
| Wheel `Kt` | 9.55/70 = 0.1364 | same, **but never physically verified** | T2.1 |
| Wheel radius | 0.075 | **0.056** | v4 |
| `R_th_wc/R_th_ca` | 1.0 / 6.0 | keep, but mark **estimated, never measured** | firmware README says so explicitly |
| Sensor/actuator delay | 0.002 / 0.001 guessed | **measured** | T3.1 |
| IMU noise | BNO086 bench numbers | re-take **on the assembled robot with motors energised** | T3.2 |

**Every speed tuned in m/s is stale by 0.747×** because of the Ø150→Ø112 change —
`vel_pi_*`, `v_cmd_ms`, `radio_vel_max`, `profileN_vel_max`. The twin should not
inherit any of them; it pulls live values from the robot (§3.2).

Critical files: `v4_twin_279mm_baseline/{params.py, physics.py, sim_loop.py,
firmware_control.py}`, `simulation/sil/`.

---

## Part 2 — Plant characterization: the test-by-test walkthrough

This is the part that decides whether the twin is real. Tests are ordered so each
feeds the next; **do not skip forward** — a delay measurement on a robot whose
inertia is wrong tells you nothing.

Everything is logged with the existing SD logger at **500 Hz** (`WLOG_SAMPLE_HZ`),
with its `.PARAMS` sidecar, and analysed with `software/gui/analysis/wlog_metrics.py`
+ `param_sidecar.py` — no new logging path. Drive the robot with
`software/gui/tools/robot_ctl.py` (`param_set`, `set_mode`, logging commands), not
ad-hoc scripts.

Format per test: **what it identifies → rig → procedure → sim acceptance check.**

### Tier 0 — Static (no motion, robot on the bench)

**T0.1 — Mass inventory.** → `m_box`, `m_wheel`, `m_femur/tibia/coupler`, `motor_mass`, `M_BODY`.
Scale. Weigh the whole robot; remove one wheel assembly and weigh it; weigh a spare
link set if you have one, else trust CAD for the links only. Also caliper `leg_y`.
*Accept:* twin's computed `m_b` (excl. wheels) within 3% of `total − 2·m_wheel`.
This also sets firmware `M_BODY`; only FF2 depends on it.

**2026-08-09 result: T0.1 complete.** Scale inventory gives 3.242 kg without
the 0.276 kg battery and 3.518 kg driving mass. Each wheel assembly is 0.478 kg,
so the measured/accounted body excluding wheels is 2.562 kg. Known component
masses plus the retained 260 g-per-AK45 catalog values leave a 0.653 kg
unweighed residual; MuJoCo carries it explicitly on the link bodies so total
mass closes exactly. Its true placement remains a T0.2 CG question, not a mass
uncertainty. Full ledger: `components/COMPONENTS.md`.

**T0.2 — CG location vs leg height.** → CG x/z, and therefore `lqr_pitch_trim_ret/ext`.
Two scales. `MANUAL` mode, hips jogged to a fixed α, robot resting on both wheels
plus one known bracing point at a measured horizontal distance; the weight split
gives CG x. Repeat at **α = 0, 0.25, 0.5, 0.75, 1**. Then lay the robot on its side
on the two scales to get CG z the same way.
*Accept:* CG-to-axle horizontal offset at α≈0 comes out near **25 mm ahead of the
axle**, matching the already-measured −8° retracted balance point
(`lqr_pitch_trim_ret ≈ −0.14 rad`). If it doesn't, one of the two is wrong and this
must be resolved before Tier 1. Twin then predicts `lqr_pitch_trim_ext` rather than
you tuning it blind.

**T0.3 — Ride height vs α, and `A_Z`.** → validates the whole 4-bar chain + pins `A_Z`.
Ruler. `MANUAL`, jog to α = 0, 0.25, 0.5, 0.75, 1; measure the hip shaft (A) height
above ground at each.
*Accept:* twin's IK ride-height curve within **3 mm** at every point, endpoints near
**119.3 → 395.9 mm**. `A_Z` is the single free parameter here — fit it, and it stops
being "inherited from baseline-1 and never re-measured". This also retires the
firmware README's standing caveat that `L_EFF_RET/EXT` shift 1:1 with `A_Z`.

**T0.4 — Switch-to-stop offset δ.** → `calib_range_l/r_rad`, hence α, hence everything α-scheduled.
Procedure is already written out in the firmware README: calibrate, then jog toward
retraction in `MANUAL` until the retract hard stop, read `hip_r_pos_rad`. δ is that
reading; `calib_range_r_rad = 1.48353 − δ`. Do both legs (single-leg runs leave
`gain_sched_alpha` invalid — see the README's caveat; run both for this).
*Accept:* twin uses the measured span, not 85°.

### Tier 1 — Passive dynamics (the highest-value tier)

**T1.1 — Axle pendulum swing.** → **body inertia `I_axle`** and bearing damping. *The most important test in this plan.*
Suspend from wheel axle. Hips held rigid in `MANUAL` at fixed α, wheels IDLE, robot
hanging free on a rod through/under the axle. Start SD logging, displace ~10–15°,
release, let it ring down for 10+ s. Repeat at **α = 0, 0.5, 1**, three swings each.
Extract period T and log-decrement from `pitch_rad`.
`I_axle = m·g·d·T²/(4π²)`, with `d` = axle-to-CG from T0.2; damping from the decrement.
*Why it matters:* `controllers/lqr.py:31` currently hardcodes `I_b = m_b·l_eff²` —
a **point mass at the pendulum tip**. A real robot with a wide body box and legs is
nowhere near that; the error goes straight into the natural frequency, which *is*
the stability margin. Expect the sim to be meaningfully wrong here today.
*Accept:* twin's free-swing period within **5%** at all three α, with wheels locked.

**T1.2 — Same swing, wheels free.** → wheel/rotor rotational inertia's contribution.
Identical to T1.1 but with the wheels free to spin (ODrive IDLE, no brake). The
period shift between T1.1 and T1.2 isolates the reflected wheel inertia.
*Accept:* twin reproduces the period *difference* within 10%.

**T1.3 — Wheel free spin-up / spin-down.** → wheel+rotor inertia `J_w`, viscous + Coulomb friction, and a first `Kt` sanity check.
Robot on a stand, wheels off the ground. `MANUAL`, single wheel, torque step —
**start at 0.1 N·m, not 1** — hold 2 s, then command zero and log the full coast-down.
Repeat at 0.1 / 0.2 / 0.4 N·m. Slope of the spin-up gives `τ/J_w`; the coast-down
curve separates viscous (exponential tail) from Coulomb (linear tail) friction.
*Accept:* twin's spin-up slope within 8%, coast-down time constant within 15%.
Cross-check against the known free-spin physical ceiling of ~1300 turns/s².

### Tier 2 — Actuators

**T2.1 — Wheel torque against a physical standard.** → the wheel `Kt` end-to-end. **Currently unverified, and every LQR gain is denominated in wheel N·m.**
Lever + scale (the same 100 mm rig from the hip validation, log `20260802T022757`;
1 N·m = 1020 g at 100 mm). Wheel blocked by the lever onto the scale, `MANUAL`,
command 0.1 / 0.2 / 0.3 / 0.5 N·m, read the scale each time.
*Why it matters:* the hip scale bug (packed over ±18, decoded over ±8) was invisible
to every firmware-only round trip because the TX and RX errors partially cancelled —
only an external physical reference caught it. The wheel path has never had that
treatment, and it is the path the balance loop actually uses.
*Accept:* measured within **±20%** of command across the range and **linear** — the
hand-held scale's own repeatability is ~±20%, which is ample to exclude a 2.5×
scaling error but not to resolve efficiency. Set the twin's `Kt` from the fitted slope.

**T2.2 — Wheel torque bandwidth and actuator delay.** → `actuator_delay_s` + any first-order torque lag.
Wheel free-spinning on the stand. Command a torque square wave (±0.15 N·m, 2 Hz,
then 5 Hz, then 10 Hz) and log at 500 Hz. Differentiate `wm_l_vel_turns_s` to get
achieved torque; cross-correlate against the commanded `whl_tau_l`.
*Accept:* twin's actuator delay set from the measured lag (replacing the guessed
1 ms), and its torque step response matches the measured rise.

**T2.3 — Hip impedance step.** → AK45 inner-loop bandwidth, gearbox backlash/stiction.
Leg unloaded on the stand. `MANUAL`, known `kp`/`kd`, step the hip position setpoint
by 0.05 rad; log `hip_*_pos_rad`, `hip_*_cmd_pos_rad`, `hip_*_torque_nm`.
*Accept:* twin's hip position response matches overshoot and settling within 15%.
Backlash shows up as a dead zone on setpoint reversal — record its width; if it's
significant it belongs in the model.

**T2.4 — Hip holding torque vs α.** → `hip_running_tff_ret/ext`, and a direct check on the 4-bar mechanical advantage.
No extra rig. Robot standing on its wheels (supported), `MANUAL` or `STANDBY` with
hips holding, at α = 0, 0.25, 0.5, 0.75, 1. Log `hip_l/r_torque_nm` at each plateau,
30 s each.
*Prior data:* ~4.0 N·m at α≈0.05 vs ~2.7 N·m at α≈0.145 (2026-07-28) — *more* torque
retracted. The firmware README explicitly asks for exactly this multi-α sweep rather
than extrapolating from one point.
*Accept:* twin predicts the holding-torque-vs-α curve within 10%. This validates the
entire linkage model under load — arguably a better geometry check than T0.3, because
it's sensitive to the *derivative* of the kinematics. Fit `hip_running_tff_ret/ext`
from the two endpoints and push them to the robot.
*Watch:* 4.07 N·m sustained sits past the Class B winding limit at steady state
(~143 °C). Keep each plateau short; these are 30 s samples, well inside the ~30 s
winding time constant, but don't park there.

**T2.5 — Bus current vs hip torque.** → independent confirmation of the torque scale, no rig at all.
Read supply current at 0 N·m (quiescent) and at 1 / 2 / 3 N·m hold. The README's
3 N·m → 0.4 A datapoint agrees with the datasheet-derived 0.38 A to 5%, but **it is
not recorded whether that 0.4 A was total bus current or the delta**. Take both this
time. `P ∝ τ²` should hold.
*Accept:* twin's electrical model reproduces the current curve within 15%.

### Tier 3 — Sensing and timing

**T3.1 — Total loop delay.** → `sensor_delay_s + actuator_delay_s`, the number that most directly sets how far gains can go before instability.
Safety stand. Robot balancing (or held at fixed pitch with wheels loaded), inject a
small torque dither and cross-correlate commanded `tau_sym` against measured
`pitch_rate_rads` at 500 Hz. The phase slope vs frequency gives the transport delay
directly. Subtract T2.2's actuator half to isolate the sensor half.
*Needs a small firmware hook* — see §2.6 below.
*Accept:* twin's total delay set from the measurement, replacing the guessed 3 ms.
**Expect this to move the stability boundary more than any other single parameter.**

**T3.2 — IMU noise, on the assembled robot, motors energised.** → `NoiseParams`.
30 s stationary in `STANDBY`, logged at 500 Hz, twice: motors de-energised, then
energised. The existing numbers (pitch 0.0101°, gyro 0.121°/s) were taken on a bench
BNO086, not next to two running motor controllers.
*Accept:* twin's noise std matches per channel. Also re-check the roll channel.
*Known hazard:* the gyro stream can freeze (`pitch_rate` exactly 0.000) while the
rotation vector keeps flowing, with `IMU_NOMINAL` still set — only a power cycle
clears it. Check for stuck-zero runs before trusting any noise fit.

**T3.3 — Control-loop jitter.** → the twin's tick model.
No rig. Take the `t_micros` deltas straight out of any existing 500 Hz `.wlog` and
histogram them.
*Accept:* twin injects the measured jitter distribution rather than a perfect 2 ms tick.

### Tier 4 — Closed loop: the tests that actually answer the user's two questions

**T4.1 — Ultimate gain and period (`K_u`, `T_u`).** → **the headline sim-vs-real acceptance criterion.**
Safety stand/tether. Velocity PI off (`vel_pi_en = 0`), yaw off, α held fixed,
`lqr_torque_limit` set low. Use the **existing live-tune** path: CH5 down + CH6 up
selects group 0, CH7 drives `lqr_k_pitch_ret`. Sweep the knob up until sustained
oscillation appears; record the gain (`live_tune_ch7_val`, telemetry-visible) and
the oscillation frequency from the 500 Hz log. Repeat at **α = 0** and **α = 1**
(the extended anchor is reachable for the first time on v4 — the firmware README
warns those anchors were tuned against a schedule saturating at 0.825, so they
transfer to nothing).
Do the same for `lqr_k_rate_ret` and, with the velocity loop re-enabled, `vel_pi_kp`.
Live-tune's pickup-not-snap behaviour makes this safe: a knob does nothing until
swept through the current value. Do **not** latch (`live_tune_latch`) unless you
want the value kept.
*Accept:* the twin predicts `K_u` within **15%** and `T_u` within **10%**, at both
α ends. **If this passes, sim-to-real gain transfer works. If it fails, go back to
T1.1 (inertia) and T3.1 (delay) — those two dominate it.**

**T4.2 — Pitch step response, no hardware disturbance needed.**
`enable_sim_pitch` / `sim_pitch_rad` **already exist as firmware params** — they
inject a fake pitch offset into the control input, i.e. a perfectly repeatable step
disturbance without touching the robot. Step 3° and 5°, both signs, at α = 0 and 1.
*Accept:* twin's `tau_sym`, `pitch`, and wheel-velocity traces overlay the hardware
log within 15% RMS. Run the identical injection in the twin — same param, same value.

**T4.3 — Velocity step.** `v_cmd_ms` 0 → 0.3 → 0 → −0.3 m/s with vel-PI on.
*Accept:* rise time, overshoot, and steady-state error within 20%; the anti-windup
behaviour (integral freezing at the asymmetric clamp) visibly matches.
*Note:* all m/s-denominated params are 0.747× stale post-Ø112 — expect to re-fit,
and treat any pre-v4 velocity tuning as void.

**T4.4 — Yaw rate step.** `omega_cmd_rds` steps ±1 rad/s.
*Accept:* yaw rate rise and steady-state within 20% — this is where wheel-ground
friction and yaw inertia show up. Sets the twin's ground friction coefficient.

**T4.5 — Rolling resistance / coast-down.** Drive to 0.5 m/s, cut to `v_cmd_ms = 0`
with the balance loop holding, and log the decay.
*Accept:* twin's coast distance within 20%. Feeds ground rolling friction, which is
otherwise a pure guess.

### 2.6 — The one firmware change the test program needs

T3.1 (and optionally a chirp version of T4.1) needs an excitation source that the
existing params don't provide. Add to `protocol/schema.json`, `GROUP_CONTROL`:

| Param | Range | Meaning |
|---|---|---|
| `plant_id_en` | 0/1 | default **0**, non-persistent |
| `plant_id_amp` | 0–0.3 N·m | dither amplitude added to `tau_sym` |
| `plant_id_f0`, `plant_id_f1` | 0.2–40 Hz | chirp start/end |
| `plant_id_dur` | 0.5–20 s | sweep length, then auto-disarms |

Injected in `control_loop.cpp` **after** `tau_sym` is formed and **before** the
`lqr_torque_limit` clamp, so the clamp still governs. Auto-clears at `plant_id_dur`
and on any state exit — the same "inert by default, armed only deliberately" pattern
the existing `CMD_ID_TEST_INJECT_CORRUPT` harness uses. Then regenerate with
`python protocol/generate_protocol.py` and flash both MCUs together.

Everything else in Tiers 0–4 uses params and modes that already exist.

---

## Part 3 — Twin infrastructure

### 3.1 One parameter namespace, generated from the firmware schema

`firmware/robot_teensy/protocol/schema.json` is already the single source of truth
for all 139 params (name, default, min, max, flags, description). The twin's control
params are **generated from it**, not retyped:

```
twin/params_control.py   <- generated from schema.json (names, defaults, bounds)
twin/params_plant.py     <- hand-maintained: masses, inertias, friction,
                            delays, noise — the things the tests identify
```

The split is the point: **control params come from the firmware; plant params come
from the bench.** A `--check` mode (same convention as `generate_protocol.py --check`)
fails CI when a param the twin's control law reads has been renamed or re-bounded.

Tuning `lqr_k_pitch_ret` in the sim then means editing the exact same name you'd
edit in the GUI's Params tab.

### 3.2 Params flow both directions

`software/gui/parameter_exports/Default gains.json` already defines an interchange
format (`{"0x0421": {"name": "lqr_k_pitch_ret", "value": -0.3}}`). Reuse it verbatim.

- `twin/tools/pull_params.py` — dumps live robot params via `robot_ctl.py param_get`
  into that format, loads them into the twin. *"Simulate exactly what's on the robot
  right now."*
- `twin/tools/push_params.py` — pushes twin gains to the robot via `robot_ctl.py
  param_set`, with a **dry-run diff by default**, an allowlist restricted to control
  gains, and a hard refusal on anything outside schema min/max.

### 3.3 The twin writes real `.wlog` files

Have the twin emit `TelemetryPayload` records in the **existing** `.wlog` format
(`WLOG_FORMAT_V1`, 500 Hz, `LogRecord` = 251 bytes) plus a matching `.PARAMS`
sidecar. Consequences, all free:

- The GUI's **Log Analyzer opens sim runs**, with the same Overview / LQR / Vel-PI /
  Yaw-PI / Hip / Torque / Gain-schedule views.
- `wlog_metrics.py` computes **the same fitness numbers** for sim and hardware runs —
  so an optimizer's objective in sim is literally the number you read off a hardware
  log, not a re-derivation of it.
- `tools/wlog_to_csv.py`, `analyze_hw_run.py` work on sim output unchanged.

This is the cheapest high-leverage feature in the plan and it should land early.

### 3.4 Log replay — the fidelity meter

`twin/tools/replay_wlog.py`, in two modes:

- **Closed-loop replay** — load a hardware `.wlog` + sidecar, set the twin to those
  exact params (`param_sidecar.py` already aligns param changes to sample times),
  drive it with recorded `v_ref`, `omega_cmd_rds`, and the logged post-rate-limit
  hip setpoints, and overlay recorded vs simulated pitch / pitch_rate / wheel_vel /
  tau_sym. Raw `radio_hip_cmd` is not in WLOG, so replay seeds the controller slew
  state from the observed MIT setpoint instead of rate-limiting it a second time.
- **Open-loop torque replay** — feed the twin the recorded `whl_tau_l/r` and hip
  setpoints as raw inputs. This mode **isolates plant error from controller error**:
  if open-loop matches and closed-loop doesn't, the bug is in the control port, not
  the physics. That separation is worth building both modes for.

Output a per-channel NRMSE **twin-fidelity score**. Keep a set of reference hardware
logs checked in and regress the score — the twin's accuracy becomes a tracked number
that can't quietly rot.

**2026-08-10 result:** implemented as direct MuJoCo multiple-shooting replay, not
the earlier low-order analytical surrogate. Consecutive 0.1 s open-loop and 2.0 s
closed-loop reset windows consume 100% of the latest RUNNING command history.
Closed-loop RMSE is 2.11° pitch, 0.136 m/s wheel speed, 0.221 rad/s yaw rate, and
0.102 m local dead-reckoned XY. Controller values are SHA-256 locked to the
current export and cannot be fit; only plant parameters may vary. Machine-readable
results are in `robot_match_replay.json`.

### 3.5 GUI ↔ sim

Two levels, in order:

1. **Now (free, via 3.3):** the GUI's Log Analyzer already consumes sim runs. Sim and
   hardware sit side by side in the same tool with the same metrics.
2. **Later:** have the twin speak the wire protocol — answer `WLR_CLAIM_V1` on UDP
   `:5007`, emit `COMM_TYPE_TELEM_FULL_WIFI` datagrams at 50 Hz on `:5005`, accept
   commands and `PARAM_SET`/`PARAM_REPORT` on TCP `:5006`. The GUI then connects to
   the sim **as if it were the robot** — every tab, every plot, the Params tab, the
   visualizer, live tuning, all work unchanged, with no GUI changes at all. Packing
   reuses `tabs/telem_format.py` and `tabs/generated_protocol.py`.

Level 2 turns the GUI into a single cockpit for both, and is the natural home for
"tweak a param in sim, watch the result" — which is exactly what was asked for. It's
maybe two days of work and depends on nothing else, so it can slot in whenever.

### 3.6 Shared scenario definitions

A scenario becomes a JSON time-series of setpoints and param changes
(`v_cmd_ms`, `omega_cmd_rds`, `radio_hip_cmd`, `enable_sim_pitch`, …). The twin's
scenario runner executes it; a hardware runner executes the same file through
`robot_ctl.py`. One definition, two executions, directly comparable logs — this is
what makes T4.2/T4.3/T4.4 one command instead of a manual procedure.

### 3.7 Optimizer changes

The existing `(1+λ)-ES` machinery (`optimizer/es_engine.py`, `optimize_integrated.py`,
`ProgressUI`, `run_log.py`) carries over. Two changes:

- **Search over firmware gains** (`lqr_k_pitch_ret/ext`, `lqr_k_rate_*`, `lqr_k_vel`,
  `vel_pi_*`, `yaw_pi_*`), inside schema min/max — not Q/R weights. A gain vector the
  optimizer emits is then directly pushable by §3.2.
- **Optimize against a plant ensemble, not the nominal plant.** Evaluate each
  candidate over ~5 perturbed plants (±20% body inertia, ±1 ms delay, ±15% wheel
  torque, ±10% CG offset) and score the worst case. This is the standard fix for
  gains that are optimal in sim and marginal on hardware, it's cheap here, and it
  directly serves goal #2. Even a perfectly matched twin only matches at one
  operating point — robustness is what makes the transfer survive a battery sag or
  a warm motor.

`software/gui/analysis/tuning_session.py` already implements the hardware-side
serialized version of this same ES loop (the deleted `tuning.md` §5) — the twin
should feed its output *into* that, not duplicate it.

**2026-08-11 result:** implemented in `optimizer/robust_mujoco.py` against the
matched MuJoCo plant and firmware-equivalent controller. The staged search tunes
27 LQR/velocity/yaw/hip/roll/feedforward parameters while keeping every plant,
trim, watchdog, torque, and command-limit parameter fixed. The final candidate
passed 49/49 combinations of seven scenarios and seven plant variations; robust
fitness improved from 2.6854 to 1.3070. A separate LQR scale sweep reproduces
the upper-gain boundary: the optimized set passes at 1.0x but fails the
pitch-rate/saturation guards at 1.5x. Full evidence and transfer caveats are in
`simulation/mujoco/v4_twin_279mm_baseline/GAIN_OPTIMIZATION_FINDINGS.md`.

---

## Part 4 — How this delivers the two stated goals

**"Find max gains in sim, port to robot, see the same behaviour."**
Gain margin is set by delay (T3.1), inertia (T1.1/T1.2), torque scale and bandwidth
(T2.1/T2.2), and `l_eff` vs α (T0.3 + the §1.3 constants test). T4.1 measures the
boundary directly as `K_u`/`T_u` and is the pass/fail gate. Anything else matching
while T4.1 fails means the twin is fitting trajectories, not dynamics.

**"Optimize gains in sim and know the robot behaves the same."**
Requires: the same control law (§1.4 + the §1.5 equivalence test), the same parameter
names and bounds (§3.1), the same fitness metric (§3.3 → `wlog_metrics.py`), a
push path (§3.2), and robustness margin (§3.7). Then a sim-optimized set goes to the
robot as a `param_set` batch and is validated by re-running the same scenario file
(§3.6) on hardware and diffing the two `.wlog`s (§3.4).

---

## Phasing

| Phase | Work | Gate |
|---|---|---|
| **0** | Create `v4_twin_279mm_baseline/`, v4 geometry + dogleg IK, α/backoff map, `.wlog` output (§3.3) | §1.3 geometry test reproduces `L_EFF_RET/EXT` and the 276.62 mm stroke |
| **1** | Port `control_loop.cpp` (§1.4), schema-generated params (§3.1), SIL equivalence harness (§1.5) | equivalence test green on the golden vectors |
| **2** | Bench: **T0.1–T0.4, T1.1–T1.3, T2.1–T2.5** | plant params replaced by measurements; T0.3 pins `A_Z`; T2.4 curve matches within 10% |
| **3** | Firmware `plant_id_*` hook (§2.6); **T3.1–T3.3** | delay and noise measured, not guessed |
| **4** | **T4.1–T4.5**; iterate the plant model until matched | `K_u` within 15%, `T_u` within 10%, both α ends |
| **5** | Replay/fidelity score (§3.4), guarded param flow (§3.2), and robust MuJoCo ensemble optimizer (§3.7) **complete offline** | the simulation-only gain candidate survives a restrained hardware run unchanged after plant-ID gates |
| **6** | Optional: GUI-as-sim-client (§3.5 level 2); then roll and jump validation | — |

Phase 2 onward is hands-on hardware; I'll walk through each test one at a time —
setup, the exact `robot_ctl.py` calls, what to watch on the log, and the twin
comparison — rather than handing over the whole list at once.

---

## Verification

- `pytest simulation/mujoco/v4_twin_279mm_baseline/tests/` — geometry-vs-firmware
  constants (§1.3), control equivalence (§1.5), schema param sync (§3.1).
- `python protocol/generate_protocol.py --check` after the §2.6 schema edit; flash
  Teensy and ESP32 **together** (a skewed pair reports `esp32_link_ok = false`).
- Headless scenario run + fitness, per the existing `master_sim_jump/README.md` recipe.
- `python v4_twin_279mm_baseline/launcher.py` for a visual check of every new geometry.
- Open a twin `.wlog` in the GUI Log Analyzer (`analyzer_load <path>` via
  `robot_ctl.py`) and confirm the views render and metrics compute.
- `twin/tools/replay_wlog.py <hardware.wlog>` on the reference logs; fidelity score
  must not regress.
- Per CLAUDE.md: compile/run after every change; nothing outside the new package,
  `simulation/sil/`, and the §2.6 schema addition gets touched.

## Open items carried forward (flagged, not silently assumed)

- `A_Z = −23.5 mm` — unverified until T0.3. `L_EFF_RET/EXT` shift 1:1 with it.
- The 0.653 kg whole-robot mass residual is closed in the twin but its true
  placement is unresolved; T0.2 replaces the current equivalent link-mass fit.
- Hip gearbox efficiency — expected 10–20%, reported hip torque is motor-side and
  cannot see planetary losses. Not resolvable with a hand-held scale; the twin
  carries it as an explicit unknown rather than folding it into `Kt`.
- `R_th_wc`/`R_th_ca` thermal resistances — estimates, never measured. The watts
  column is solid, the °C column is indicative.
- All m/s-denominated tuning is 0.747× stale post-Ø112 and is not inherited.
