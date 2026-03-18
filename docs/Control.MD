# Control Architecture — Wheeled-Leg Robot

**Reference robot:** Ascento (ETH Zurich, ICRA 2019) — same cascade architecture.

---

## Architecture Overview

### 4 Controllers — 2 LQRs + 2 PIDs

```
Outer loops (500 Hz, same rate as inner)    Inner loop (500 Hz)          Actuators
────────────────────────────────────────    ───────────────────          ─────────
[Velocity PI] ──θ_ref──▶                   [Balance LQR] ──τ_sym─▶     τ_L = τ_sym + τ_yaw
[Yaw PI]      ──τ_yaw──▶────────────────────────────────────────────▶   τ_R = τ_sym − τ_yaw
                                            State: [φ−θ_ref, φ̇, v−v_ref]

Independent (500 Hz)
──────────────────────
[Leg Impedance PD] ──────────────────────────────────────────────────▶  τ_hip (both legs)
```

| # | Controller | Type | State / Input | Output | Rate |
|---|------------|------|--------------|--------|------|
| 1 | Balance LQR | LQR 3-state | [pitch−θ_ref, pitch_rate, wheel_vel_avg−v_ref] | τ_sym | 500 Hz |
| 2 | Leg Impedance | PD impedance | [q_hip−q_nom, dq_hip] | τ_hip | 500 Hz |
| 3 | Velocity PI | PI | v_desired − v_measured | θ_ref (lean cmd) | 500 Hz |
| 4 | Yaw PI | PI | ω_yaw_desired − ω_yaw_measured | τ_yaw | 500 Hz |

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

**Optimized gains (scenario 1_LQR_pitch_step, 5-min / 4448 evals, 2026-03-18) — CURRENT BASELINE:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Q_PITCH | 0.138282 | soft pitch weight |
| Q_PITCH_RATE | 0.023379 | dominant damping term |
| Q_VEL | 0.004591 | minimal velocity correction |
| R | 9.998298 | moderate torque cost |
| fitness | 0.003267 | RMS pitch 0.926°, survived 5 s |

**Character of these gains:** The optimizer settled on low Q_PITCH relative to Q_PITCH_RATE,
meaning the controller primarily damps pitch *rate* (resists angular velocity) rather than
aggressively correcting pitch angle. This produces smooth, low-torque recovery from the 5°
step perturbation. Steady-state pitch offset is **not corrected by the LQR** — that is
intentional and handled by the outer Velocity PI loop (see §"Why We Do NOT Use LQI" below).

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

**Optimized gains (combined_PI, 30-min / 5488 evals, 2026-03-18) — CURRENT BASELINE:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Kp_v | 0.502932 | rad/(m/s) |
| Ki_v | 0.012678 | rad/m — very low integrator; rate-limited theta_ref handles transients |
| fitness | 0.61 | on extended ±1 m/s S3 staircase |
| THETA_REF_RATE_LIMIT | 2.0 rad/s | rate-limits lean command to suppress pitch-rate spikes on step changes |

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

Starting gains: Kp_yaw = 0.3, Ki_yaw = 0.05. **Not yet implemented.**

---

### 4. Leg Impedance PD (independent, 500 Hz)

Decoupled from balance. Controls hip motor as a spring-damper:

```python
τ_hip = −K_s * (q_hip − q_nom) − B_s * dq_hip
```

Current simulation values (matched to baseline1):
- K_s = 8.0 N·m/rad
- B_s = 4.0 N·m·s/rad
- Torque limit: 2.0 N·m (HIP_IMPEDANCE_TORQUE_LIMIT) — keeps hips backdrivable

`q_nom` = Q_NOM (30% of stroke) — fixed in current sim, user-adjustable later.

---

## Optimizer Scenarios

All scenarios run at 500 Hz controller rate (2 kHz physics, CTRL_STEPS=4). Noise model active by default (BNO086 pitch/rate noise). All scenarios use `_run_sim_loop` as the shared simulation backend — S2 and S3 run byte-for-byte identical physics.

---

### Scenario 1 — `1_LQR_pitch_step`

**Purpose:** Isolate and tune the Balance LQR inner loop. VelocityPI is OFF — the outer loop must not interfere with pure inner-loop step-response characterisation.

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

**Current best:** Q=[0.138, 0.0234, 0.00459], R=9.998, fitness=0.003267 (4448 evals).

---

### Scenario 2 — `2_VEL_PI_disturbance`

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

### Scenario 3 — `3_VEL_PI_staircase`

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
- Identical noise model, sensor read order, and leg impedance controller
- Same liftoff detection and settle-time tracking

The `dist_fn(t) -> float` callable interface allows each scenario to specify an arbitrary disturbance profile without duplicating the sim loop.

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

### Phase 1 — 3-State LQR + Gain Scheduling ✅ COMPLETE

**Files:** `lqr_design.py`, `scenarios.py`, `optimize_lqr.py`, `replay.py`

- LQR solver with gain scheduling at 3 leg positions (retracted/nominal/extended)
- (1+8)-ES optimizer over Q/R weights
- Best: Q=[0.138, 0.0234, 0.00459], R=9.998, fitness=0.003267 (4448 evals, 2026-03-18)
- RMS pitch 0.926°, survived full 5 s, pitch-rate-dominated damping, no steady-state correction (outer PI handles that)

---

### Phase 2 — Velocity PI + Balance Position Hold ✅ COMPLETE & BASELINED

**Files:** `scenarios.py`, `optimize_vel_pi.py`, `replay.py`, `sim_config.py`

- `VelocityPI` class: velocity error → lean angle command (θ_ref) for LQR
- Dedicated optimizer `optimize_vel_pi.py` — (1+8)-ES over KP_V/KI_V with LQR Q/R fixed
- S2 `2_VEL_PI_disturbance`: equilibrium start, ±1N kicks, 6s, metric = wheel travel
- S3 `3_VEL_PI_staircase`: 4-step velocity profile, 8.5s, metric = vel_track_rms_ms
- `combined_PI` scenario: 50/50 weighted fitness of S2+S3
- S2 and S3 share `_run_sim_loop` — identical physics, consistent `v_ref` to LQR
- Best (S2 only so far): KP_V=0.499, KI_V=0.0585, wheel_travel=1.332m
- `replay.py`: 4-panel telemetry (pitch, torque, pitch_rate, velocity with cmd overlay), v_desired slider, Drive ON/OFF toggle, staircase auto-scheduled for S3

---

### Phase 3 — Yaw PI + Turn Mode

| Step | Action | Pass Criterion |
|------|--------|---------------|
| 3.1 | Add `YawPI` class, ±0.5 N·m clamp | Unit test: converges on ω step |
| 3.2 | Apply differential mixing τ_L/τ_R | — |
| 3.3 | TURN scenario: ω=1 rad/s for 6.28s | 360° with < 10cm drift |
| 3.4 | DRIVE+TURN scenario: v=0.3 + ω=0.5 | Curved path, pitch < 3° |

Tuning: manual Kp_yaw/Ki_yaw — start [0.3, 0.05]. Check cross-coupling (turning must not disturb pitch > 2°).

---

### Phase 4 — Leg Impedance Suspension

| Step | Action | Pass Criterion |
|------|--------|---------------|
| 4.1 | Add SUSPENSION_SOFTNESS param, multiple K_s/B_s presets | — |
| 4.2 | Terrain disturbance: reduce K_s 30% for 200ms on unexpected pitch | 5N impulse → Z deflect < 2cm |
| 4.3 | Tune HIP_IMPEDANCE_TORQUE_LIMIT (currently 2 N·m placeholder) | Balance holds while legs deflect |

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

## Tuning Summary

| Controller | Method | Parameters | Status |
|------------|--------|-----------|--------|
| Balance LQR | (1+8)-ES | Q_pitch, Q_pitch_rate, Q_vel, R | ✅ Done — Q=[0.138,0.0234,0.00459], R=9.998 |
| Gain scheduling | Offline at 3 points | l_eff at Q_RET/NOM/EXT | ✅ Done |
| Velocity PI (hold) | (1+8)-ES on S2 | Kp_v, Ki_v | ✅ Baselined — Kp=0.499, Ki=0.0585 |
| Velocity PI (tracking) | (1+8)-ES on combined_PI | Kp_v, Ki_v | ✅ Done — Kp=0.503, Ki=0.013, theta_ref rate-limited 2 rad/s |
| Yaw PI | Manual visual | Kp_yaw, Ki_yaw | Not started |
| Leg impedance | Critical damping formula | K_s, B_s | Placeholder values only |
| Jump recovery | 1D search | Landing K_s multiplier | Not started |
