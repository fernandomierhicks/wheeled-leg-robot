# Control Architecture — Wheeled-Leg Robot

**Reference robot:** Ascento (ETH Zurich, ICRA 2019) — same cascade architecture.

---

## Architecture Overview

### 4 Controllers — 2 LQRs + 2 PIDs

```
Outer loops (500 Hz, same rate as inner)    Inner loop (500 Hz)          Actuators
────────────────────────────────────────    ───────────────────          ─────────
[Velocity PI] ──θ_ref──▶                   [Balance LQR] ──τ_sym─▶     τ_L = τ_sym + τ_yaw + τ_ff
[Yaw PI]      ──τ_yaw──▶────────────────────────────────────────────▶   τ_R = τ_sym − τ_yaw + τ_ff
                                            State: [φ−θ_ref, φ̇, v−v_ref]

Independent (500 Hz)
──────────────────────
[Hip Position Servo]  ──────────────────────────────────────────────▶  τ_hip (S1–S7: stiff PD)
[Hip Impedance + Roll]──────────────────────────────────────────────▶  τ_hip (S8: terrain compliance)

Feedforward (500 Hz, applied after LQR + hip outputs)
──────────────────────
[FF1 Hip reaction]    ── τ_hip_total, l_eff ───▶  τ_ff1 = −α₁·τ_hip·(r/l_eff)  ──▶  added to τ_L, τ_R
[FF2 Gravity comp]    ── pitch, m_b, l_eff ────▶  τ_ff2 = α₂·m·g·l·sin(φ)      ──▶  added to τ_L, τ_R
```

| # | Controller | Type | State / Input | Output | Rate |
|---|------------|------|--------------|--------|------|
| 1 | Balance LQR | LQR 3-state | [pitch−θ_ref, pitch_rate, wheel_vel_avg−v_ref] | τ_sym | 500 Hz |
| 2 | Hip control (S1–S7) | Position servo PD | q_hip_tgt − q_hip, dq_tgt − dq_hip | τ_hip | 500 Hz |
| 2 | Hip control (S8) | Impedance + Roll PD | q_hip − q_nom ± δq_roll, dq_hip | τ_hip | 500 Hz |
| 3 | Velocity PI | PI | v_desired − v_measured | θ_ref (lean cmd) | 500 Hz |
| 4 | Yaw PI | PI | ω_yaw_desired − ω_yaw_measured | τ_yaw | 500 Hz |
| 5 | FF1 Hip reaction cancel | Feedforward | τ_hip_total, l_eff | τ_wheel additive | 500 Hz |
| 5 | FF2 Gravity compensation | Feedforward | pitch, m_b, l_eff | τ_wheel additive | 500 Hz |

---

## Controller Details

### 1. Balance LQR (inner, 500 Hz)

**State vector:**
```
x = [pitch − θ_ref, pitch_rate, wheel_vel_avg − v_ref]   ← 3-state
```

`wheel_pos` is NOT in the state. Position hold in BALANCE mode is the outer loop's job (θ_ref from VelocityPI). This keeps the LQR compatible with intentional forward motion in DRIVE mode.

**LQR formulation:**
```
u = −K @ x
K = [K_pitch, K_pitch_rate, K_vel]
Q = diag([Q_pitch, Q_pitch_rate, Q_vel])
R = scalar
```

**Important — `v_ref` coupling:** The LQR third state is `(wheel_vel − v_ref)`, where `v_ref = v_desired / WHEEL_R`. This means the LQR does not penalise wheel velocity when the robot is intentionally moving at the commanded speed. Both the headless simulation and the replay must pass `v_ref = v_desired / WHEEL_R` — passing `v_ref=0.0` during drive incorrectly penalises the wheel velocity and fights the PI's lean command.

**Optimized gains (scenario 2_leg_height_gain_sched, 5-min / 1896 evals, 2026-03-18) — CURRENT BASELINE:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Q_PITCH | 0.014168 | soft pitch weight — ~10× lower than S1 |
| Q_PITCH_RATE | 0.033720 | dominant damping term — slightly higher than S1 |
| Q_VEL | 0.000250 | minimal velocity correction — ~18× lower than S1 |
| R | 28.734420 | moderate torque cost — ~3× higher than S1 (softer) |
| fitness | 0.017938 | RMS pitch 1.242°, survived 12 s (3 full leg cycles) |

**Prior S1 gains (retained for reference — do not use as baseline):**
| Parameter | Value | Scenario |
|-----------|-------|---------|
| Q_PITCH | 0.138282 | `1_LQR_pitch_step`, 5-min / 4448 evals, fitness=0.003267 |
| Q_PITCH_RATE | 0.023379 | RMS pitch 0.926°, survived 5 s |
| Q_VEL | 0.004591 | — |
| R | 9.998298 | — |

**Character of the S4 gains:** Softer overall than S1 — lower Q_PITCH, higher R. The optimizer
settled on a solution that tolerates more pitch excursion in exchange for lower torque, which is
sensible for a 12s scenario where the leg-induced equilibrium shifts are the primary stimulus.
Q_PITCH_RATE is slightly higher than S1, preserving damping. Steady-state pitch offset is **not
corrected by the LQR** — that is intentional and handled by the outer Velocity PI loop (see
§"Why We Do NOT Use LQI" below).

**Gain Scheduling (Ascento-style):**

The 4-bar stroke (~62°) changes effective pendulum length l_eff by ~50 mm.
Gains are precomputed at 3 operating points and interpolated online:

```python
K_table = {
    'retracted': compute_lqr_gain(q_hip=Q_RET),   # legs short
    'nominal':   compute_lqr_gain(q_hip=Q_NOM),   # default stance
    'extended':  compute_lqr_gain(q_hip=Q_EXT),   # legs long
}

# Online at 500 Hz:
alpha = (q_hip - Q_RET) / (Q_EXT - Q_RET)
K = (1 - alpha) * K_table['retracted'] + alpha * K_table['extended']
```

l_eff is computed via `physics.get_equilibrium_pitch(p, q_hip)` at each table entry.

---

### 2. Velocity PI (outer, 500 Hz)

Converts velocity command into a lean angle setpoint for the LQR:

```python
vel_est_ms = wheel_vel * WHEEL_R          # +wheel_vel = forward body motion
v_err = v_desired − vel_est_ms
θ_ref = Kp_v * v_err + Ki_v * integral(v_err)
θ_ref = clamp(θ_ref, −0.26 rad, +0.26 rad)   # ±15° hard limit
```

**BALANCE mode:** `v_desired = 0` → PI integrates any drift into a restoring lean.
**DRIVE mode:** `v_desired = slider / joystick command` → PI drives to target speed.

**Sign convention:** positive `wheel_vel` = forward body motion = positive `vel_est_ms`.
Positive `θ_ref` = lean forward = drive forward.

**Integrator reset:** On direction reversal (sign of `v_desired` flips), the PI integrator is
reset to zero to prevent windup carryover. Implemented in `_run_sim_loop` and replay.

**Optimized gains (S5 `5_VEL_PI_leg_cycling`, 5-min / 1776 evals, 2026-03-18):**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Kp_v | 0.251209 | rad/(m/s) — ~½ of prior combined_PI value; softer to absorb bump hits |
| Ki_v | 0.011405 | rad/m — similar to prior |
| fitness | 2.0635 | S5: vel_rms=0.502 m/s, pitch=5.59°, survived 13 s |
| THETA_REF_RATE_LIMIT | 2.0 rad/s | rate-limits lean command to suppress pitch-rate spikes on step changes |

**Manual sandbox tuning (2026-03-18) — CURRENT BASELINE:**
Optimizer-tuned gains felt too conservative in practice — robot accelerated sluggishly in the
sandbox even on flat ground. KP_V was doubled and rate limit raised 2.5× for snappier response.
KI_V left unchanged (integrator behaviour was fine).
| Parameter | Value | Notes |
|-----------|-------|-------|
| Kp_v | 0.502418 | rad/(m/s) — 2× optimizer value; matches pre-S5 combined_PI baseline |
| Ki_v | 0.011405 | rad/m — unchanged |
| THETA_REF_RATE_LIMIT | 5.0 rad/s | 2.5× optimizer value — lean ramp ~52 ms vs 130 ms |

**Prior combined_PI baseline (retained for reference):**
| Parameter | Value | Scenario |
|-----------|-------|---------|
| Kp_v | 0.502932 | combined_PI, 30-min / 5488 evals, fitness=0.61 |
| Ki_v | 0.012678 | — |

**Flag:** `USE_VELOCITY_PI` in `scenarios.py` — set `False` to disable outer loop for LQR-only tests.

---

### 3. Yaw PI (outer, 500 Hz)

Produces differential torque for turning. Blends with drive.

```python
yaw_err = ω_yaw_desired − ω_yaw_measured   # from BNO086 gyro Z
τ_yaw = Kp_yaw * yaw_err + Ki_yaw * integral(yaw_err)
τ_yaw = clamp(τ_yaw, −0.5 N·m, +0.5 N·m)

τ_L = τ_sym + τ_yaw
τ_R = τ_sym − τ_yaw
```

See Phase 3 and latency Step 4 for baselined gains.

---

### 4. Hip Control (independent, 500 Hz)

Decoupled from balance. Two modes selected per scenario:

#### S1–S7: Position Servo PD

Stiff PD that reliably tracks the commanded hip angle (fixed `Q_NOM` or sinusoidal cycling profile). No compliance — hip reaches target reliably regardless of disturbances.

```python
τ_hip = KP * (q_tgt − q_hip) + KD * (dq_tgt − dq_hip)
τ_hip = clamp(τ_hip, ±HIP_IMPEDANCE_TORQUE_LIMIT)
```

For static targets (`dq_tgt = 0`). For cycling scenarios (S2, S5), `dq_tgt` is the analytic derivative of the sine profile (feed-forward velocity term).

| Param | Value | Notes |
|-------|-------|-------|
| HIP_POSITION_KP | 50.0 N·m/rad | ω_n ≈ 31 rad/s >> 1.6 rad/s leg cycle |
| HIP_POSITION_KD | 3.0 N·m·s/rad | Near-critical damping |
| HIP_IMPEDANCE_TORQUE_LIMIT | 5.0 N·m | Per-scenario limits TBD |

#### S8 only: Impedance + Roll Leveling

Compliant spring-damper with differential roll correction for terrain. Optimised by `optimize_suspension.py`.

```python
# Roll leveling — differential hip target
roll_true = atan2(2*(w*x + y*z), 1 - 2*(x²+y²))   # from free joint quaternion
roll_meas = roll_true + N(0, ROLL_NOISE_STD_RAD)     # BNO086 noise (0.05°)
roll_rate = data.qvel[d_root + 3]                    # ωx world-frame

δq = K_ROLL * roll_meas + D_ROLL * roll_rate
q_nom_L = clamp(q_tgt + δq, HIP_SAFE_MIN, HIP_SAFE_MAX)
q_nom_R = clamp(q_tgt − δq, HIP_SAFE_MIN, HIP_SAFE_MAX)

# Spring-damper per leg
τ_hip = clamp(-(K_s*(q_hip − q_nom) + B_s*dq_hip), ±HIP_IMPEDANCE_TORQUE_LIMIT)
```

**Baselined gains (LQR_Control_optimization — zero-latency):**
| Param | Value | Notes |
|-------|-------|-------|
| K_s | 16.000 N·m/rad | Stiff spring |
| B_s | 0.799 N·m·s/rad | Low damping — spring-dominant |
| K_ROLL | 4.000 rad/rad | Roll correction |
| D_ROLL | 1.000 rad·s/rad | Roll rate damping |

**Baselined gains (latency_sensitivity — 5 ms sensor + 2.5 ms actuator delay):**
| Param | Value | Notes |
|-------|-------|-------|
| K_s | 2.040 N·m/rad | ↓8× vs zero-latency — stiff spring oscillates with delayed feedback |
| B_s | 4.179 N·m·s/rad | ↑5× vs zero-latency — high damping compensates for soft spring |
| K_ROLL | 3.912 rad/rad | Roll correction (similar to zero-latency) |
| D_ROLL | 0.916 rad·s/rad | Roll rate damping |

Roll leveling is **orthogonal** to pitch/yaw — δq has zero effect on average hip height.
Robot drives at Q_NOM (30% of stroke) to maximise suspension travel in both directions.
`HIP_SAFE_MIN = Q_EXT + 10° = −1.257 rad`, `HIP_SAFE_MAX = Q_RET − 10° = −0.526 rad`.

---

### 5. Feedforward Terms (500 Hz)

Feedforward terms run every control tick in `sim_loop.py`, after the LQR + hip torque outputs are written to `data.ctrl` but before the battery model step. Both share a single IK call to avoid redundant computation. Each has a tunable alpha gain (0 = disabled, 1 = full) and a live UI toggle checkbox.

#### FF1 — Hip Reaction Torque Cancellation

Every hip torque command creates an equal-and-opposite reaction torque on the body. The LQR sees this as a pitch disturbance and corrects it, but only after the disturbance has already occurred (one control loop late at minimum). FF1 cancels this in the same tick.

```python
tau_hip_total = ctrl[hip_L] + ctrl[hip_R]        # total hip torque this tick
l_eff         = abs(IK(hip_q_avg)['W_z'])         # current pendulum length
tau_ff1       = -ff1_alpha * tau_hip_total * (r_wheel / l_eff)
ctrl[wheel_L] += tau_ff1 / 2
ctrl[wheel_R] += tau_ff1 / 2
```

| Param | Location | Default | Notes |
|-------|----------|---------|-------|
| `ff1_alpha` | `params.gains.feedforward.ff1_alpha` | 0.0 | 0 = disabled, 1 = full cancellation |
| UI toggle | "Hip FF" checkbox | ON | Live enable/disable in visualizer |

#### FF2 — Gravity Compensation Torque on Wheels

When the robot leans, gravity exerts a pendulum torque `m_b·g·l_eff·sin(pitch)`. The LQR handles this through feedback, but during large lean angles (velocity tracking, jump recovery) the feedback is always chasing the gravitational load. FF2 adds this torque directly to the wheels.

```python
l_eff   = abs(IK(hip_q_avg)['W_z'])               # shared with FF1
tau_ff2 = ff2_alpha * m_b * g * l_eff * sin(pitch)
ctrl[wheel_L] += tau_ff2 / 2
ctrl[wheel_R] += tau_ff2 / 2
```

Where `m_b` is body mass excluding wheels (same definition as the LQR linearized model in `lqr.py`).

| Param | Location | Default | Notes |
|-------|----------|---------|-------|
| `ff2_alpha` | `params.gains.feedforward.ff2_alpha` | 0.0 | 0 = disabled, 1 = full gravity comp |
| UI toggle | "Grav FF" checkbox | ON | Live enable/disable in visualizer |

**Caution:** The LQR already has an implicit gravity term in its linearized `alpha` coefficient (`m_b·g·l_eff / denom`). Setting `ff2_alpha = 1.0` may double-count gravity at small angles where the LQR linearization is accurate. The term is most beneficial at large lean angles where `sin(pitch) ≈ pitch` breaks down. Sweep `ff2_alpha` from 0 to 1 to find the optimal value.

#### Execution order in `sim_loop.py`

```
LQR τ_sym  ──▶  Yaw τ_yaw  ──▶  Actuator delay buf  ──▶  motor_taper → ctrl[wheel]
                                                              │
Hip control ──────────────────────────────────────────────▶  ctrl[hip]
                                                              │
Shared IK(hip_q_avg) ──────────────────────────────────────  l_eff
    ├── FF1: ctrl[wheel] += -α₁·τ_hip_total·(r/l_eff) / 2
    └── FF2: ctrl[wheel] += α₂·m_b·g·l_eff·sin(pitch) / 2
                                                              │
Battery step ◀─────────────────────────────────────────────  final ctrl[]
```

---

## Optimizer Scenarios

All scenarios run at 500 Hz controller rate (2 kHz physics, CTRL_STEPS=4). Noise model active by default (BNO086 pitch/rate noise). All scenarios use `_run_sim_loop` as the shared simulation backend — S2 and S3 run byte-for-byte identical physics.

### Scenario Summary

| S# | Name | Controller under test | How it tests | Key metric | Duration |
|:--:|-----------------------|--------------|------------|----------|----------|
| 1 | `1_LQR_pitch_step` | Balance LQR | Fixed legs at Q_NOM; +5° pitch step + 2 force kicks; VelocityPI OFF | ISE pitch (rad²·s) | 5 s |
| 2 | `2_leg_height_gain_sched` | Balance LQR + gain scheduler | Legs sinusoidally cycling full stroke; ±4 N kicks; VelocityPI OFF | ISE pitch (rad²·s) | 12 s |
| 3 | `3_VEL_PI_disturbance` | Velocity PI | Fixed legs; v_desired=0 (position hold); ±1 N impulse kicks at t=2/3 s | Wheel-travel RMS [m/s] | 12 s |
| 4 | `4_VEL_PI_staircase` | Velocity PI | Fixed legs; 4-step speed profile 0→0.3→0.6→−0.3→0 m/s; no disturbances | Velocity-tracking RMS [m/s] | 13 s |
| 5 | `5_VEL_PI_leg_cycling` | Velocity PI (with cycling legs) | Legs cycling + staircase velocity + ±1 N kicks + 5 one-sided bumps; full stress test | Velocity-tracking + pitch RMS combined | 13 s |
| 6 | `6_YAW_PI_turn` | Yaw PI | Fixed legs; v_desired=0; ω_desired=1 rad/s; no disturbances | Yaw-rate-tracking RMS [rad/s] | 8 s |
| 7 | `7_DRIVE_TURN` | Yaw PI + Velocity PI together | Fixed legs; v=0.3 m/s + ω=0.5 rad/s simultaneous; no disturbances | Combined velocity + yaw RMS | 8 s |
| 8 | `8_terrain_compliance` | Leg Impedance + Roll Leveling | v=1.0 m/s; 1 one-sided bump (5 cm left at x=1.2 m); optimised hip impedance + roll leveling | Peak roll angle [deg] | 12 s |

**Optimizer targets:** LQR → S2 · VelocityPI → S5 · YawPI → S6 · Suspension (impedance+roll) → S8

---

### Scenario 1 — `1_LQR_pitch_step` *(superseded by S4 for LQR tuning)*

**Purpose:** Isolate and tune the Balance LQR inner loop. VelocityPI is OFF. Legs held at Q_NOM — leg height variation is not tested.

**Initial condition:** Equilibrium pitch + 5° perturbation (PITCH_STEP_RAD).

**Controller under test:** Balance LQR only. `θ_ref = 0`, `v_ref = 0`.

**Disturbances:**
- +4N at t=2.0s for 0.2s (forward kick)
- −4N at t=3.0s for 0.2s (backward kick)

**Duration:** 5s

**Metric / Fitness:** ISE (integral of squared pitch error) in rad²·s — lower is better.
```
fitness = W_FALL × fell  +  Σ (pitch_err²) × dt
```

**What is being optimised:** Q_PITCH, Q_PITCH_RATE, Q_VEL, R — the LQR cost weights.

**Pass criterion:** Survives full 5s, RMS pitch < 2°, no wheel liftoff.

**Optimizer:** `optimize_lqr.py` — (1+8)-ES over the 4 Q/R parameters.

**Last best:** Q=[0.138, 0.0234, 0.00459], R=9.998, fitness=0.003267 (4448 evals, 2026-03-18).

> **Note:** S4 (`2_leg_height_gain_sched`) supersedes S1 for LQR Q/R optimisation.
> S4 includes the same disturbances but additionally exercises the gain scheduler across
> the full leg stroke — gains tuned on S4 are strictly more general than S1-tuned gains.
> `optimize_lqr.py` is now pointed at S4.

---

### Scenario 4 — `2_leg_height_gain_sched` *(current LQR optimizer target)*

**Purpose:** Tune the Balance LQR Q/R weights while the gain scheduler is fully exercised. The legs cycle sinusoidally through the operating range, changing the effective pendulum length continuously. This makes S4 a strict superset of S1: any Q/R set that handles S4 also handles fixed-height balance.

**Initial condition:** Equilibrium pitch at Q_NOM. No pitch perturbation — the leg motion itself is the stimulus.

**Controller under test:** Balance LQR only (gain-scheduled). `θ_ref = 0`, `v_ref = 0`. VelocityPI OFF.

**Leg cycle:**
```
q_hip(t) = center + amp × sin(2π t / 4.0)

LEG_CYCLE_Q_RET = Q_RET − 5°  ≈ −25.1°   (less crouched than full-retract to avoid 4-bar instability)
Q_EXT           = −82.0°                   (fully extended)
center          ≈ −53.6°,  amp ≈ −28.5°
Period = 4 s  →  3 full cycles in 12 s
```

**Disturbances** (same as S1):
- +4N at t=2.0s for 0.2s (fires mid first cycle, legs near mid-stroke)
- −4N at t=3.0s for 0.2s

**Duration:** 12s (3 full leg cycles)

**Metric / Fitness:**
```
fitness = ISE_pitch  +  W_PITCH_RATE × ISE_pitch_rate  +  W_FALL × fell
```
Pitch error is measured against the **live** `pitch_ff(q_hip)` equilibrium, not a fixed Q_NOM reference — so the score purely reflects how well the LQR tracks the changing equilibrium as leg height varies.

**What is being optimised:** Q_PITCH, Q_PITCH_RATE, Q_VEL, R — the LQR cost weights.

**Optimizer:** `optimize_lqr.py` — (1+8)-ES, seeded from S1 best.

**Seed (from S1):** Q=[0.138282, 0.023379, 0.004591], R=9.998298.

**Status:** ✅ Baselined 2026-03-18 — 237 gens, 1896 evals, 5 min. Best: Q=[0.014168, 0.033720, 0.000250], R=28.734, fitness=0.017938, RMS pitch 1.242°, survived 12 s.

---

### Scenario 2 — `3_VEL_PI_disturbance`

**Purpose:** Tune the Velocity PI outer loop for position hold quality under external disturbances. The LQR Q/R are fixed at their S1-optimised values.

**Initial condition:** Equilibrium pitch, no perturbation.

**Controller under test:** VelocityPI (outer) + Balance LQR (inner). `v_desired = 0` throughout — position hold only.

**Disturbances:**
- +1N at t=2.0s for 0.2s (lighter than S1 — tests PI hold, not LQR recovery)
- −1N at t=3.0s for 0.2s

**Duration:** 6s (gives PI 2.8s to correct after the last kick at t=3.2s)

**Metric / Fitness:**
```
fitness = (wheel_travel_m + W_FALL × fell) / duration   [units: m/s]
```
Absolute wheel travel measures how far the robot drifts — lower = better position hold.
Normalised by duration so it is on the same scale as S3's `vel_track_rms_ms`.

**What is being optimised:** KP_V, KI_V — the VelocityPI gains.

**Pass criterion:** Survives 6s with < 2m total wheel travel.

**Optimizer:** `optimize_vel_pi.py` — (1+8)-ES over KP_V/KI_V.

**Current best:** KP_V=0.503, KI_V=0.0127, fitness=0.61 (combined_PI, 5488 evals, 2026-03-18).

---

### Scenario 3 — `4_VEL_PI_staircase`

**Purpose:** Tune the Velocity PI for velocity tracking across a four-step speed profile including direction reversal. Complements S2 (which tests position hold) — together they ensure the PI works in both static and dynamic modes.

**Initial condition:** Equilibrium pitch, no perturbation.

**Controller under test:** VelocityPI (outer) + Balance LQR (inner).

**Disturbances:** None.

**Velocity profile:**
```
t =  0.0 –  1.0 s  →  0.0 m/s  (settle; excluded from VEL_ERR metric)
t =  1.0 –  3.0 s  → +0.3 m/s  gentle forward
t =  3.0 –  5.0 s  → +0.6 m/s  step up
t =  5.0 –  7.0 s  → +1.0 m/s  high-speed forward
t =  7.0 –  9.0 s  → −0.5 m/s  direction reversal — integrator reset fires
t =  9.0 – 11.0 s  → −1.0 m/s  high-speed backward
t = 11.0 – 13.0 s  →  0.0 m/s  return to stop
```

**Duration:** 13s

**Metric / Fitness:**
```
fitness = W_VEL_ERR × vel_track_rms_ms  +  0.1 × W_RMS × rms_pitch_deg  +  W_FALL × fell
```
`vel_track_rms_ms` = RMS of `(v_commanded − v_measured)` from t=1.0s onward, in m/s.
Pitch penalty is secondary (0.1×) — exists to prevent the PI from solving tracking by leaning aggressively.

**What is being optimised:** KP_V, KI_V — the VelocityPI gains (shared with S2).

**Pass criterion:** vel_track_rms_ms < 0.15 m/s, survives full 8.5s.

**Optimizer:** `optimize_vel_pi.py` with `combined_PI` scenario (0.5 × S2 + 0.5 × S3) to jointly optimise for both position hold and velocity tracking.

**Status:** ✅ Baselined 2026-03-18 via combined_PI (5488 evals, 30 min).

---

### Scenario 6 — `6_YAW_PI_turn`

**Replay profile (latency_sensitivity/replay.py) — 2026-03-20:**
The optimizer runs a plain constant-ω step. The replay uses a richer multi-phase auto-profile driven by accumulated yaw angle:

| Phase | Trigger | ω_desired | v_desired |
|-------|---------|-----------|-----------|
| Settle | t < 1 s | 0 | 0 |
| 1 — 1 rotation fwd | until +2π accumulated | +YAW_TURN_RATE | 0 |
| 2 — 2 rotations rev | until −4π from phase entry | −2 × YAW_TURN_RATE | 0 |
| 3 — 2 rotations fwd | until +4π from phase entry | +2 × YAW_TURN_RATE | 0 |
| 4 — done | — | 0 | 0 |

**Bug fixed (2026-03-20):** `turn_active` was referenced in `replay()` but only defined inside `_plot_process()` — caused `NameError` on any S6 load. Fixed by adding `turn_active = [False]` to the `replay()` state variables.

---

### Scenario 7 — `7_DRIVE_TURN`

**Replay profile (latency_sensitivity/replay.py) — 2026-03-20:**
The optimizer uses constant v=0.3 m/s + ω=0.5 rad/s. The replay uses a time-based multi-phase profile:

| Phase | Time window | v_desired | ω_desired |
|-------|-------------|-----------|-----------|
| 1 — baseline | t < 5 s | 0.3 m/s | +0.5 rad/s |
| 2 — reverse + faster | 5–10 s | 0.5 m/s | −1.0 rad/s |
| 3 — forward + high yaw | 10 s+ | 0.7 m/s | +1.5 rad/s |

**Bug fixed (2026-03-20):** S7 fell through to the generic `else` branch in replay which never set `_omega_desired`, so the robot drove straight. Fixed by adding an explicit `elif scenario == "7_DRIVE_TURN":` block that sets both `_v_desired` and `_omega_desired`.

---

### Scenario 5 — `5_VEL_PI_leg_cycling` *(current VelocityPI optimizer target)*

**Purpose:** Tune the Velocity PI under the hardest realistic conditions: legs cycling continuously while the robot simultaneously tracks a velocity staircase and absorbs disturbance kicks. S5 is a strict superset of both S2 (disturbances) and S3 (staircase) — gains that work here are the most transfer-robust.

**Initial condition:** Equilibrium pitch at Q_NOM, `v_desired = 0`.

**Controller under test:** VelocityPI (outer) + Balance LQR (inner, gain-scheduled). LQR Q/R fixed at S4 baseline.

**Velocity profile:** Identical to S3 staircase:
```
t =  0.0 –  1.0 s  →  0.0 m/s  (settle)
t =  1.0 –  3.0 s  → +0.3 m/s  gentle forward
t =  3.0 –  5.0 s  → +0.6 m/s  step up
t =  5.0 –  7.0 s  → +1.0 m/s  high-speed forward
t =  7.0 –  9.0 s  → −0.5 m/s  direction reversal
t =  9.0 – 11.0 s  → −1.0 m/s  high-speed backward
t = 11.0 – 13.0 s  →  0.0 m/s  return to stop
```

**Disturbances:** S2 kicks: +1N at t=2s for 0.2s, −1N at t=3s for 0.2s (fire during the +0.3m/s phase, adding velocity AND pitch disturbance simultaneously).

**Leg motion:** Sinusoidal Q_RET ↔ Q_EXT, period = 4s (same as S4) — 3+ full cycles during the 13s run. Changing leg height continuously shifts pitch_ff, challenging the VelocityPI + LQR cascade throughout.

**Duration:** 13s

**Metric / Fitness:**
```
fitness = W_VEL_ERR × vel_track_rms_ms  +  0.1 × W_RMS × rms_pitch_deg  +  W_FALL × fell
```
Same form as S3 — velocity tracking is primary, pitch penalty is secondary.

**What is being optimised:** KP_V, KI_V — the VelocityPI gains.

**Optimizer:** `optimize_vel_pi.py` (ACTIVE_SCENARIO updated to S5).

**Status:** ✅ Baselined 2026-03-18 — 222 gens, 1776 evals, 5 min. Best: KP_V=0.251209, KI_V=0.011405, fitness=2.0635, vel_rms=0.502 m/s, pitch=5.59°, survived 13 s.

---

### Combined PI Scenario — `combined_PI`

Used by the optimizer to jointly tune KP_V/KI_V for both behaviours:

```
fitness = 0.5 × S2_fitness  +  0.5 × S3_fitness
```

Both sub-fitnesses are normalised to m/s units before weighting. This prevents either scenario from dominating and ensures the PI works well across position hold and velocity tracking.

---

## Shared Simulation Infrastructure

All scenarios from S2 onwards use `_run_sim_loop` in `scenarios.py` as the single shared physics backend. This guarantees:

- Identical PI controller instantiation (same `VelocityPI` class, same gains)
- Correct `v_ref = v_target_ms / WHEEL_R` passed to the LQR at every step
- Consistent PI integrator reset on direction reversal
- Identical noise model, sensor read order, and hip controller (position servo for S1–S7, impedance+roll for S8)
- Same liftoff detection and settle-time tracking

The `dist_fn(t) -> float` callable interface allows each scenario to specify an arbitrary disturbance profile without duplicating the sim loop.

---

## Simulation Tooling Notes (2026-03-20)

### Log file layout
CSV and `.log` files moved to `latency_sensitivity/logs/`. Naming convention: `S{scenario_name}.csv` (e.g. `S6_YAW_PI_turn.csv`). Archive files prefixed `archive_S{N}_…`. `run_log.py` updated — `get_scenario_csv_path()` now writes to `logs/S{name}.csv`.

### Lingering Python processes
After closing the MuJoCo viewer the matplotlib subprocess could remain alive. Fixed in both `latency_sensitivity/replay.py` and `LQR_Control_optimization/replay.py` by adding `plot_proc.terminate()` after the 2s `join()` timeout.

### S8 bump layout change
`S8_BUMPS` reduced from 5 alternating one-sided bumps to a single 5 cm left bump at x=1.2 m. Reason: the robot veers off-axis at the first bump and never reaches subsequent ones, making the multi-bump layout redundant for baseline visualization. The optimizer still uses whatever `S8_BUMPS` is defined in `sim_config.py`.

---

## Mode State Machine

```
         ┌─────────────────────────────────────────────┐
         ▼                                             │
    [BALANCE] ──drive_cmd──▶ [DRIVE] ──turn──▶ [TURN] │
         ▲                      ▲                      │
         └──────── stop ─────────┘◀──── stop ──────────┘

    [BALANCE] ──jump──▶ [JUMP] ──apex──▶ [FLYING]
         ▲                                    │
         │                               contact│
    [SETTLED] ◀── stable 500ms ── [LANDING] ◀──┘
```

| Mode | Balance LQR | v_desired | Entry Trigger |
|------|-------------|-----------|---------------|
| BALANCE | Normal | 0 (position hold) | default / stop |
| DRIVE | Normal + v_ref feedforward | slider/joystick | drive command |
| TURN | Normal + v_ref | slider/joystick | turn command |
| JUMP | Active, stiff legs | 0 | jump command |
| FLYING | K_pitch × 2 | 0 | contact force < 0.1 N for 50ms |
| LANDING | K_pitch × 2, soft legs | 0 | IMU Z accel > 1.5g |
| SETTLED | Normal | 0 | |pitch| < 2° for 500ms |

---

## Implementation Phases

### Phase 1 — 3-State LQR + Gain Scheduling ✅ COMPLETE & S4-BASELINED

**Files:** `lqr_design.py`, `scenarios.py`, `optimize_lqr.py`, `replay.py`

- LQR solver with gain scheduling at 3 leg positions (retracted/nominal/extended)
- (1+8)-ES optimizer over Q/R weights
- S4 baseline: Q=[0.014168, 0.033720, 0.000250], R=28.734, fitness=0.017938 (1896 evals, 2026-03-18)
- RMS pitch 1.242°, survived full 12 s (3 leg cycles + disturbances), softer/more damped than S1
- S1 reference gains (fixed-height only): Q=[0.138, 0.0234, 0.00459], R=9.998, fitness=0.003267

---

### Phase 2 — Velocity PI + Balance Position Hold ✅ COMPLETE & BASELINED

**Files:** `scenarios.py`, `optimize_vel_pi.py`, `replay.py`, `sim_config.py`

- `VelocityPI` class: velocity error → lean angle command (θ_ref) for LQR
- Dedicated optimizer `optimize_vel_pi.py` — (1+8)-ES over KP_V/KI_V with LQR Q/R fixed
- S2 `3_VEL_PI_disturbance`: equilibrium start, ±1N kicks, 6s, metric = wheel travel
- S3 `4_VEL_PI_staircase`: 4-step velocity profile, 8.5s, metric = vel_track_rms_ms
- `combined_PI` scenario: 50/50 weighted fitness of S2+S3
- S2 and S3 share `_run_sim_loop` — identical physics, consistent `v_ref` to LQR
- Best (combined_PI): KP_V=0.536, KI_V=0.028, fitness=1.056 (824 evals, 2026-03-18)
- `replay.py`: 4-panel telemetry (pitch, torque, pitch_rate, velocity with cmd overlay), v_desired slider, Drive ON/OFF toggle, staircase auto-scheduled for S3

> **Limitation:** VelocityPI is a *velocity* controller — it commands lean angle proportional to velocity error. It does NOT guarantee the robot stays at a fixed world-frame position; a constant disturbance will cause slow drift that the integrator only partially corrects. **Future work (Phase 2.5 or above):** add a Position PI outer-outer loop (odometry → position error → v_desired command) above VelocityPI so the robot can be commanded to hold or reach an absolute position. Until then, position control is not available — only velocity setpoint tracking.

---

### Phase 3 — Yaw PI + Turn Mode ✅ COMPLETE & BASELINED

**Files:** `scenarios.py`, `optimize_yaw_pi.py`, `replay.py`, `sim_config.py`, `sandbox_fastchart.py`

- `YawPI` class: yaw rate error → τ_yaw, ±0.5 N·m clamp, integrator anti-windup 0.5 N·m·s
- Differential mixing: `τ_L = τ_sym − τ_yaw`, `τ_R = τ_sym + τ_yaw` (sign verified experimentally — positive ω = CCW = left turn)
- World-frame yaw rate: `data.qvel[dof_free + 5]` — no pitch contamination (free joint qvel is always world-frame)
- Controller orthogonality confirmed: differential τ_yaw cancels in wheel average — LQR/VelocityPI are blind to it
- S6 `6_YAW_PI_turn`: pure 360° CCW turn at 1 rad/s, 8 s — yaw_rms=0.045 rad/s (4.5% error), pitch=3.3°
- S7 `7_DRIVE_TURN`: v=0.3 m/s + ω=0.5 rad/s simultaneously — pitch=3.2°, no cross-coupling instability
- Initial visual gains KP_YAW=0.3, KI_YAW=0.05 were a starting point; optimizer confirmed much higher gains needed
- `replay.py`: 4th panel changed to Yaw Rate with cmd overlay; ω_desired slider; Turn ON/OFF button; differential torque applied
- `sandbox_fastchart.py`: interactive free-drive arena — 28 obstacles (1–8 cm), v/ω/hip sliders + gamepad support; pyqtgraph 5×2 telemetry panel (60 Hz, 15 s window)

**Baselined gains (sim_config.py — optimizer result, 55752 evals / 75 min):**
| Param | Value |
|-------|-------|
| KP_YAW | 2.272 |
| KI_YAW | 1.125 |
| YAW_PI_TORQUE_MAX | 0.5 N·m |

---

### Phase 4 — Leg Suspension + Roll Leveling (S8 only) ✅ COMPLETE & BASELINED

- **S1–S7** hip control switched to **position servo PD** (KP=50, KD=3) — stiff, reliable tracking, no tunable compliance. `_run_sim_loop` uses `use_impedance=False` (default).
- **S8** (`8_terrain_compliance`) uses **impedance + roll leveling** via `use_impedance=True`. Per-leg spring-damper with differential δq = K_ROLL×roll + D_ROLL×roll_rate.
- `optimize_suspension.py`: (1+8)-ES over K_s, B_s, K_ROLL, D_ROLL; fitness = S8 (terrain compliance)
- **Zero-latency baseline**: K_s=16.0, B_s=0.799, K_ROLL=4.0, D_ROLL=1.0, fitness=4.09 (25016 evals, 2026-03-18)
- `replay.py` upgraded: 4×2 telemetry (8 panels) including Roll and Suspension Δq panels

---

### Phase 5 — Jump Recovery State Machine

| Step | Action | Pass Criterion |
|------|--------|---------------|
| 5.1 | Add `RobotMode` enum (BALANCE/DRIVE/TURN/JUMP/FLYING/LANDING/SETTLED) | — |
| 5.2 | FLYING detection: contact force < 0.1N for 50ms | Mode switches on 5cm drop |
| 5.3 | LANDING: K_s × 0.3 for 500ms, K_pitch × 2.0 | Land from 15cm, stable in 2s |
| 5.4 | JUMP: stiffen legs, extend hips, enter FLYING on liftoff | Jump height ≈ baseline1 (283mm) |

---

## Why We Do NOT Use LQI (LQR + Integrator on Pitch)

### Ascento precedent
Ascento (ETH Zurich, ICRA 2019) — the reference robot for this architecture — uses pure LQR for balance without integral action on pitch. The outer Velocity PI loop provides all necessary steady-state correction.

### The Velocity PI already IS the integrator
In BALANCE mode (`v_desired = 0`), the Velocity PI integrates wheel velocity error:
```
∫ v_err dt  →  θ_ref  →  LQR lean command
```
Any persistent pitch offset that causes the robot to drift forward is continuously integrated by the PI and corrected via the θ_ref lean command. This is **indirect integral action on position** — it achieves zero steady-state drift without touching the LQR.

### Why LQI would conflict
Adding a pitch integrator inside the LQR creates **two competing integrators** in the same loop:

```
LQI pitch integrator  →  wheel torque  →  velocity drift
Velocity PI integrator →  θ_ref        →  wheel torque
```

Both try to null pitch error simultaneously via different paths. This causes integrator wind-up interactions, potential instability, and makes tuning significantly harder.

### Why Scenario 1 shows steady-state offset
Scenario 1 runs with `USE_VELOCITY_PI = False` — the outer integrator is deliberately disabled to isolate LQR tuning. The residual ~0.74° pitch offset is **expected and correct** for a pure proportional LQR with no outer loop. It is NOT a sign the architecture is wrong.

### Conclusion
- **Do not add LQI.** The architecture is correct.
- The steady-state error visible in Scenario 1 headless tests is a testing artifact.
- The combined test (LQR + Velocity PI active) is the real measure of steady-state performance.

---

## Optimizing with Latencies

All work below is done in `simulation/mujoco/latency_sensitivity/` — a full copy of
`LQR_Control_optimization/` with 10-step sensor delay and 5-step actuator delay added
via ring buffers. The zero-latency folder is left untouched as a reference baseline.

**Why re-optimize:** gains tuned at zero latency assume ~60° phase margin. The real
robot has ~8.2 ms total wheel-path delay (~4 loop periods), which costs ~30° of phase
margin. Without re-tuning, the controller will be overconfident and likely oscillate.

**Dependency chain:** inner loops must be tuned before outer loops. Each optimizer
holds all inner-loop gains fixed at the previous step's result.

---

### Step 0 — LQR seed run on S1 ✅ DONE (2026-03-19, 5 min)

**Result:** Q=[0.003411, 0.000674, 0.000459], R=4.455, fitness=0.002386, rms=0.79°, survived 5s.
Used as SEED_WEIGHTS for Step 1.

---

### Step 1 — Balance LQR on S4 ✅ BASELINED (2026-03-19, 10 min)

**Purpose:** Fully re-tune the LQR Q/R weights for the delayed plant across the full
leg stroke. Seeded from Step 0 result.

**Scenario:** S4 `2_leg_height_gain_sched` — legs cycling full stroke + ±4N kicks, 12s.

**Params:** Q_PITCH, Q_PITCH_RATE, Q_VEL, R

**Result (10 min, 3302 evals, seeded from Step 0):**

| Param | Zero-latency | Delayed baseline |
|-------|-------------|------------------|
| Q_PITCH | 0.014168 | **0.063424** (+4.5×) |
| Q_PITCH_RATE | 0.033720 | **0.000219** (↓154×) |
| Q_VEL | 0.000250 | **0.000011** (↓23×) |
| R | 28.734 | **1.980** (↓14×) |
| fitness | 0.017938 | **0.044589** |
| rms_pitch | 1.24° | **1.77°** |
| survived | 12s | 12s ✅ |

**Character:** Q_PITCH↑ — robot must actively correct pitch using stale angle reading.
Q_PITCH_RATE/Q_VEL↓↓ — penalising delayed rates causes instability (kick info arrives late,
controller fights the motion it can no longer affect). R↓ — more torque is allowed because
the controller is now more selective about when to apply it.

Gains written to `latency_sensitivity/sim_config.py`.

---

### Step 2 — VelocityPI on S5 ✅ DONE (2026-03-19, 20 min)

**Purpose:** Re-tune the outer velocity loop. LQR fixed at Step 1 values.

**Scenario:** S5 `5_VEL_PI_leg_cycling` — staircase + ±1N kicks + leg cycling + 5 bumps, 13s.

**Result (9312 evals, 20 min):**

| Param | Zero-latency | Delayed baseline |
|-------|-------------|------------------|
| KP_V | 0.502418 | **0.375500** (↓25%) |
| KI_V | 0.011405 | **0.010810** (similar) |
| fitness | — | 6.508 |

**Character:** KP_V reduced — stale wheel velocity causes overshoot at high gain. KI_V largely unchanged. Note: prior stiff-suspension S5 result (KP_V≈0.013) was a suspension artefact — soft K_s masked PI response.

---

### Step 3 — Yaw PI on S6 ✅ DONE (2026-03-19, 20 min)

**Purpose:** Re-tune differential torque loop. Yaw rate measurement also delayed.

**Scenario:** S6 `6_YAW_PI_turn` — pure 360° CCW turn at 1 rad/s, 8s.

**Result (15288 evals, 20 min):**

| Param | Zero-latency | Delayed baseline |
|-------|-------------|------------------|
| KP_YAW | 2.272 | **2.192** (similar) |
| KI_YAW | 1.125 | **0.427** (↓62%) |
| fitness | 0.410 | 1.760 |

**Character:** KP_YAW largely unchanged (yaw inertia is high — phase margin less sensitive). KI_YAW↓↓ — integrator accumulates too much with stale yaw rate reading.

---

### Step 4 — Leg Impedance + Roll on S8 ✅ DONE (2026-03-19, 5216 evals)

**Purpose:** Re-tune suspension stiffness and roll leveling for delayed plant.

**Scenario:** S8 `8_terrain_compliance` — 1 bump (5 cm left at x=1.2 m), 1.0 m/s, 12s.

**Result:**

| Param | Zero-latency | Delayed baseline |
|-------|-------------|------------------|
| K_s | 16.000 N·m/rad | **2.040** (↓8×) |
| B_s | 0.799 N·m·s/rad | **4.179** (↑5×) |
| K_ROLL | 4.000 rad/rad | **3.912** (similar) |
| D_ROLL | 1.000 rad·s/rad | **0.916** (similar) |
| fitness | 4.09 | 7.498 |

**Character:** K_s↓↓ — stiff spring oscillates with delayed feedback. B_s↑↑ — heavy damping compensates. Roll leveling gains largely unchanged (hip torque delay rounds to 0 steps at 2 kHz).

---

### Validation after all steps

1. Run `python sandbox_fastchart.py` in `latency_sensitivity/` — robot should balance stably
   with the re-tuned gains. Expect slightly more pitch excursion than zero-latency
   sandbox but no sustained oscillation.
2. Run `python replay.py --baseline --scenario 5_VEL_PI_leg_cycling` — fitness will
   be higher than the zero-latency result (harder scenario) but robot must not fall.
3. Set `USE_LATENCY_MODEL = False` — replay fitness must match the original
   zero-latency results exactly, confirming the bypass works correctly.

---

### Latency tuning summary ✅ ALL STEPS COMPLETE (2026-03-19)

| Step | Actual run time | Scenario | Status |
|------|----------------|----------|--------|
| 0 | ~15 min | S1 (LQR seed) | ✅ Done |
| 1 | ~10 min | S2 (LQR full) | ✅ Done |
| 2 | ~20 min | S5 (VelocityPI) | ✅ Done |
| 3 | ~20 min | S6 (YawPI) | ✅ Done |
| 4 | ~5216 evals | S8 (Suspension) | ✅ Done |

---

## Tuning Summary

| Controller | Method | Parameters | Zero-latency baseline | Latency baseline |
|------------|--------|-----------|----------------------|-----------------|
| Balance LQR | (1+8)-ES on S2 | Q_pitch, Q_pitch_rate, Q_vel, R | ✅ Q=[0.0142, 0.0337, 0.00025], R=28.73 | ✅ Q=[0.0634, 0.000219, 0.000011], R=1.980 |
| Gain scheduling | Offline at 3 points | l_eff at Q_RET/NOM/EXT | ✅ Done | same |
| Hip servo (S1–S7) | Fixed (not optimised) | KP=50 N·m/rad, KD=3 N·m·s/rad | ✅ | ✅ |
| Velocity PI | (1+8)-ES on S5 | Kp_v, Ki_v | ✅ Kp=0.502, Ki=0.011 | ✅ Kp=0.376, Ki=0.011 |
| Yaw PI | (1+8)-ES on S6 | Kp_yaw, Ki_yaw | ✅ Kp=2.272, Ki=1.125 | ✅ Kp=2.192, Ki=0.427 |
| Hip Impedance + Roll (S8) | (1+8)-ES on S8 | K_s, B_s, K_ROLL, D_ROLL | ✅ K_s=16.0, B_s=0.799, K_ROLL=4.0, D_ROLL=1.0 | ✅ K_s=2.040, B_s=4.179, K_ROLL=3.912, D_ROLL=0.916 |
| Jump recovery | — | — | Not started | — |

---

## Knee Spring Analysis — Why It Does Not Help Jump Height

**Date:** 2026-03-27
**Conclusion:** Knee spring disabled (`use_knee_spring=False` default). Logic retained in code for future reference.

### What was tried

A torsional spring at the knee joint (`KneeSpringParams`, K=12 N·m/rad) was simulated to store energy during CROUCH and release it during EXTEND, aiming to increase jump height above the motor-only baseline (~283 mm).

### Why it fails — motor speed limit interaction

The AK45-10 hip motor's torque is speed-limited:
```
speed_limit = 1 − |ω_hip| / omega_max        (omega_max = 18.85 rad/s)
torque = max_torque × speed_limit
```

Without the spring, the motor runs at near-full torque throughout the 40° extension stroke and the hip arrives at Q_EXT at **17.8 rad/s** — just under `omega_max`. The motor is nearly optimal on its own.

The spring adds up to **10.8 N·m** (both legs) at the start of EXTEND on top of the motor's 14 N·m. This roughly doubles the initial torque, causing `omega_max` to be hit after only **25° of the 40° stroke**. Motor torque then drops to zero. The spring is also exhausted by that point (deflection → 0). For the remaining **15° of stroke**, neither motor nor spring provides meaningful force.

Energy comparison (per-leg estimates, I_leg ≈ 0.031 kg·m²):

| | KE at Q_EXT | Source |
|---|---|---|
| Without spring | 4.9 J/leg | Motor full torque, full stroke |
| Spring energy stored | +1.2 J/leg | Released in first 25° |
| Motor work lost (last 15°) | −1.8 J/leg | Motor at zero after omega_max hit |
| **Net with spring** | **~4.3 J/leg** | Slightly worse |

### B_spring (damping) was also wrong

The original `B_spring = 0.12 N·m·s/rad` was artificial numerical stabilisation, not physical. Real music wire (ASTM A228) has loss factor η ≈ 0.002:
```
B_real = η × K / ω_n = 0.002 × 12 / 20 ≈ 0.002 N·m·s/rad
```
This was corrected. Even with realistic damping the spring still slightly reduces jump height due to the speed-limit interaction above.

### When a spring could help

A knee spring would benefit jump height only if:
1. **Motor already hits `omega_max` well before Q_EXT without the spring** — so the spring fills a gap the motor cannot. This would require a lower `omega_max` (higher gear ratio) or longer stroke.
2. **Latch-and-release (catapult)**: spring pre-loaded slowly at motor stall torque, released by a mechanical latch. Decouples storage speed from release speed entirely.
3. **Series Elastic Actuator**: spring in series with the hip motor shaft; motor pre-winds spring at low speed, spring releases at whatever rate the mechanism demands.
4. **Lower K_spring (~6 N·m/rad)**: halving stiffness halves the initial over-torque, allowing the motor to stay in its efficient range longer. Small net gain (~1.2 J) is possible but not compelling.
