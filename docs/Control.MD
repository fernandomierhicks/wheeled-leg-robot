# Control Architecture — Wheeled-Leg Robot (master_sim_jump)

**Reference robot:** Ascento (ETH Zurich, ICRA 2019) — same cascade architecture.

---

## Architecture Overview

```
Outer loops (500 Hz)                  Inner loop (500 Hz)          Actuators
────────────────────                  ───────────────────          ─────────
[Velocity PI] ──θ_ref──▶             [Balance LQR] ──τ_sym─▶     τ_L = τ_sym + τ_yaw
[Yaw PI]      ──τ_yaw──▶─────────────────────────────────────▶   τ_R = τ_sym − τ_yaw
                          State: [φ−θ_ref, φ̇, v_wheel−v_ref]      (+ FF1/FF2 additive)

Independent (500 Hz)
────────────────────
[Hip Servo PD]      ──────────────────────────────────────────▶  τ_hip (position tracking)
[Hip Impedance]     ──────────────────────────────────────────▶  τ_hip (S8 / BALANCE: compliant)
[Jump FSM]          ── overrides hip mode & wheel hold ────────▶  τ_hip, τ_wheel (during jump)

Feedforward (500 Hz, appended after LQR + hip outputs)
───────────────────────────────────────────────────────
[FF1 Hip reaction]  ── τ_hip_total, l_eff ──▶  τ_ff1 = −α₁·τ_hip·(r/l_eff)   ──▶  τ_L, τ_R
[FF2 Gravity comp]  ── pitch, m_b, l_eff ───▶  τ_ff2 = α₂·m·g·l·sin(φ)        ──▶  τ_L, τ_R
[FF4 Centripetal]   ── v_fwd, ω_yaw ────────▶  θ_ff4 = −α₄·v·ω/g              ──▶  θ_ref offset
```

| # | Controller | Type | State / Input | Output | Rate |
|---|------------|------|---------------|--------|------|
| 1 | Balance LQR | LQR 3-state | [φ−θ_ref, φ̇, v_wheel−v_ref] | τ_sym | 500 Hz |
| 2 | Velocity PI | PI + FF | v_desired − v_measured | θ_ref | 500 Hz |
| 3 | Yaw PI | PI | ω_desired − ω_measured | τ_yaw | 500 Hz |
| 4 | Hip Servo PD | Position PD | q_tgt − q_hip, dq_tgt − dq_hip | τ_hip | 500 Hz |
| 4 | Hip Impedance + Roll | Spring-damper + roll PD | q_hip − q_nom ± δq_roll, dq_hip | τ_hip | 500 Hz |
| 5 | Jump FSM | 6-state machine | hip position, IMU az | hip mode override | 500 Hz |
| 6 | FF1 Hip reaction cancel | Feedforward | τ_hip_total, l_eff | τ_wheel additive | 500 Hz |
| 6 | FF2 Gravity compensation | Feedforward | pitch, m_b, l_eff | τ_wheel additive | 500 Hz |
| 6 | FF4 Centripetal | Feedforward | v_fwd, ω_yaw | θ_ref additive | 500 Hz |

---

## Control Loop Tick Sequence (sim_loop.py, 500 Hz)

Each tick executes in this exact order:

1. **Sensor read + noise** — pitch, pitch_rate, wheel velocities, IMU accel from body state; Gaussian noise added per `NoiseParams`
2. **Latency buffers** — push to sensor delay ring buffer; pop delayed readings; apply matrix predictor (Smith-style) to partially cancel phase lag
3. **Jump FSM** — update state machine; returns hip mode, adaptive suspension scale, wheel-hold flag
4. **Velocity PI** — frozen during CROUCH/EXTEND; produces θ_ref
5. **FF4 centripetal offset** — adds −α₄·v·ω/g to θ_ref (currently disabled, α₄=0)
6. **Balance LQR** — uses delayed+predicted state; produces τ_sym
7. **Yaw PI** — produces τ_yaw
8. **Wheel hold damping** — velocity damping applied to wheels during CROUCH/EXTEND only
9. **Actuator delay + motor taper** — ring buffer delays τ_sym/τ_yaw; speed-dependent torque limit applied; writes to wheel actuators
10. **Hip control** — dispatch: jump torque override | impedance | position servo; writes to hip actuators
11. **Feedforward (FF1, FF2)** — shared IK call for l_eff; additive torques added to wheel commands; re-clamped to motor_taper limits
12. **Battery model** — integrates currents, tracks SoC and temperature

---

## Controller Details

### 1. Balance LQR (inner, 500 Hz)

**State vector:**
```
x = [pitch − θ_ref, pitch_rate, wheel_vel_avg − v_ref]   ← 3-state
```

`wheel_pos` is NOT in the state. Position hold in BALANCE mode is the Velocity PI's job (via θ_ref). This keeps the LQR compatible with intentional motion in DRIVE mode.

**Control law:**
```python
u = −K @ x
K = [K_pitch, K_pitch_rate, K_vel]
Q = diag([Q_pitch, Q_pitch_rate, Q_vel])
R = scalar
```

**`v_ref` coupling:** The third state is `(wheel_vel − v_ref)`, where `v_ref = v_desired / wheel_r`. The LQR does not penalize wheel velocity when the robot is intentionally moving at the commanded speed. Passing `v_ref=0` during DRIVE incorrectly fights the PI's lean command.

**Gain scheduling (Ascento-style):**

The 4-bar stroke (~62°) changes effective pendulum length l_eff by ~50 mm.
Gains are precomputed at 3 operating points and interpolated online:

```python
K_table = {
    'retracted': compute_lqr_gain(q_hip=Q_RET),
    'nominal':   compute_lqr_gain(q_hip=Q_NOM),
    'extended':  compute_lqr_gain(q_hip=Q_EXT),
}

# Online at 500 Hz:
alpha = (q_hip_avg - Q_RET) / (Q_EXT - Q_RET)
K = (1 - alpha) * K_table['retracted'] + alpha * K_table['extended']
```

**Current gains (params.py baseline):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Q_PITCH | 0.01 | pitch cost weight |
| Q_PITCH_RATE | 0.1884 | dominant damping |
| Q_VEL | 0.00508442 | velocity tracking penalty |
| R | 100.0 | control effort cost |
| torque_limit | 7.0 N·m | wheel motor clamp |

---

### 2. Velocity PI (outer, 500 Hz)

Converts velocity command into a lean angle setpoint for the LQR.

**Control law:**
```python
v_err = v_desired - v_measured          # v_measured = wheel_vel_delayed * wheel_r
integral = clamp(integral + v_err*dt, -int_max, +int_max)
dv_cmd_dt = (v_desired - prev_v_desired) / dt
theta_ref = Kp*v_err + Ki*integral + Kff*dv_cmd_dt
theta_ref = clamp(theta_ref, -theta_max, +theta_max)
theta_ref = rate_limited(theta_ref, theta_ref_rate_limit)
```

**Current gains (params.py baseline):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kp | 0.2 rad/(m/s) | proportional gain |
| Ki | 0.1 rad/m | integral gain |
| Kff | 0.1049 s²·rad/m | acceleration feedforward (≈ 1/g) |
| theta_max | 0.698 rad | ±40° hard clamp |
| theta_ref_rate_limit | 1.745 rad/s | 100°/s max lean slew rate |
| int_max | 1.0 rad·s | anti-windup limit |

**Mode behavior:**
- **BALANCE:** `v_desired=0` — PI integrates drift into a restoring lean; achieves position hold indirectly
- **DRIVE:** `v_desired` set by joystick/command — PI drives to target speed

**Integrator reset:** When sign of `v_desired` flips (direction reversal), integrator is zeroed to prevent windup carryover.

**Freeze during jump:** VelocityPI is disabled during CROUCH and EXTEND phases. θ_ref is held constant; integrator does not update. Prevents velocity error from fighting the jump impulse.

---

### 3. Yaw PI (outer, 500 Hz)

Produces differential torque for turning. Fully orthogonal to balance — τ_yaw cancels in the wheel torque average and is invisible to the LQR/VelocityPI.

**Control law:**
```python
err = omega_desired - omega_measured    # omega_measured = data.qvel[dof_yaw] (world-frame)
integral = clamp(integral + err*dt, -int_max, +int_max)
tau_yaw = Kp*err + Ki*integral
tau_yaw = clamp(tau_yaw, -torque_max, +torque_max)

# Differential mixing:
tau_L = tau_sym + tau_yaw
tau_R = tau_sym - tau_yaw
```

Yaw rate is read from the free joint world-frame qvel — no pitch contamination.

**Current gains (params.py baseline):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kp | 0.1 N·m/(rad/s) | proportional gain |
| Ki | 0.2 N·m/rad | integral gain |
| torque_max | 0.5 N·m | differential torque clamp |
| omega_cmd_max | 4.0 rad/s | commanded yaw rate limit |
| int_max | 0.5 N·m·s | anti-windup |

---

### 4. Hip Control (independent, 500 Hz)

Two modes; the jump FSM and scenario flags control which is active.

#### Mode A — Position Servo PD (fixed-position scenarios)

Stiff PD that reliably tracks commanded hip angle. Used for scenarios with explicit position targets (sinusoidal cycling, staircase profiles, jump CROUCH/RETRACT phases).

```python
tau_hip = KP * (q_tgt - q_hip) + KD * (dq_tgt - dq_hip)
tau_hip = clamp(tau_hip, -torque_limit, +torque_limit)
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| HIP_POSITION_KP | 50.0 N·m/rad | ω_n ≈ 31 rad/s >> leg cycle frequency |
| HIP_POSITION_KD | 3.0 N·m·s/rad | near-critical damping |
| torque_limit | 7.0 N·m | |

#### Mode B — Impedance + Roll Leveling (terrain compliance, BALANCE/FLYING/LANDING)

Compliant spring-damper with differential roll correction. Used in S8 terrain compliance scenario and during FLYING/LANDING phases of the jump.

```python
# Roll leveling — differential hip offset
roll = atan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]² + q[2]²))  # from body quat
roll_rate = data.qvel[dof_roll]

delta_q = K_roll * roll + D_roll * roll_rate
q_nom_L = clamp(q_hip_sym + delta_q, Q_EXT, Q_RET)
q_nom_R = clamp(q_hip_sym - delta_q, Q_EXT, Q_RET)

# Spring-damper per leg:
tau_hip = clamp(-(K_s*(q_hip - q_nom) + B_s*dq_hip), -torque_limit, +torque_limit)
```

Roll leveling is orthogonal to pitch/yaw — δq has zero effect on average hip height.

**Current gains (params.py baseline — zero-latency):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| K_s | 20.0 N·m/rad | spring stiffness |
| B_s | 0.5 N·m·s/rad | damping |
| K_roll | 10.0 rad/rad | roll leveling gain |
| D_roll | 0.15 rad·s/rad | roll rate damping |
| torque_limit | 5.0 N·m | |

**Adaptive suspension during jump:**

| Phase | susp_scale | Effect |
|-------|-----------|--------|
| FLYING | 0.15 | Soften ~7× for freefall |
| LANDING | 0.15 → 1.0 ramp over 0.75 s | Gradually stiffen on impact |
| All others | 1.0 | Full stiffness |

---

### 5. Jump State Machine (500 Hz)

Six-state FSM controlling the complete jump sequence. Overrides hip mode during CROUCH/EXTEND/RETRACT; controls adaptive suspension during FLYING/LANDING.

**States and transitions:**

```
BALANCE ──[triggered]──▶ CROUCH ──[t ≥ crouch_time]──▶ EXTEND
                                                            │
            ┌───────────────────────────────────────────────┘
            ▼ hip near Q_EXT (or timeout)
         RETRACT ──[hip near Q_NOM]──▶ FLYING ──[IMU az spike + min_airborne]──▶ LANDING
                                                                                      │
            ┌─────────────────────────────────────────────────────────────────────────┘
            ▼ |pitch| < 10° for 0.5s (or timeout)
         SETTLED ──[immediate]──▶ BALANCE
```

**Per-phase hip control:**

| Phase | Hip Mode | Target | Behavior |
|-------|----------|--------|----------|
| BALANCE / SETTLED | Impedance | Q_NOM | Compliant spring-damper |
| CROUCH | Position servo | Q_NOM → Q_RET (linear ramp) | Stiff PD, ramps down over `crouch_time` |
| EXTEND | Torque override | — | Direct torque from motor model (see below) |
| RETRACT | Position servo | Q_EXT → Q_NOM (linear ramp) | Stiff PD, ramps up over `retract_time` |
| FLYING | Impedance | Q_NOM | Softened (susp_scale=0.15) |
| LANDING | Impedance | Q_NOM | Ramping stiffness (0.15 → 1.0 over 0.75 s) |

**EXTEND torque calculation:**
```python
ramp_in    = min(1.0, (t - start) / ramp_up_s)
ramp_out   = max(0.0, (q_hip - Q_EXT) / ramp_down_rad)   # tapers near Q_EXT
speed_lim  = max(0.0, 1.0 - abs(dq_hip) / omega_max)     # motor speed limit
tau = -max_torque * ramp_in * ramp_out * speed_lim
```

**Jump parameters (params.gains.jump):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| crouch_time | 0.1 s | CROUCH phase duration |
| max_torque | 7.0 N·m | Peak EXTEND torque |
| omega_max | 18.85 rad/s | AK45-10 no-load speed (KV75, 24 V, 10:1) |
| ramp_up_s | 0.01 s | Torque ramp-in at EXTEND start |
| ramp_down_rad | 0.05 rad | Hip zone before Q_EXT where torque tapers |
| retract_time | 0.01 s | RETRACT phase duration |
| retract_done_rad | 0.05 rad | Hip must be within this of Q_NOM to enter FLYING |
| min_airborne_s | 0.10 s | Minimum airborne time before landing can trigger |
| extend_timeout_s | 1.0 s | Max time in EXTEND before aborting to SETTLED |
| settle_pitch_deg | 10.0 deg | Pitch threshold for SETTLED |
| settle_time_s | 0.5 s | Duration pitch must stay below threshold |
| landing_timeout_s | 0.5 s | Max time in LANDING before forcing SETTLED |
| wheel_hold_gain | 0.15 N·m·s/rad | Velocity damping on wheels during CROUCH/EXTEND |
| freefall_scale | 0.15 | Impedance scale during FLYING |
| landing_ramp_s | 0.75 s | Duration of impedance ramp-up during LANDING |

**Liftoff detection:** Position-based — hip reaches Q_EXT ± ramp_down_rad. No IMU dependency.

**Landing detection:** IMU az spike above `landing_az_threshold` AND `t - flying_start ≥ min_airborne_s` (prevents false landing on vibration at liftoff).

---

### 6. Feedforward Terms (500 Hz)

All three FF terms run after the LQR and hip outputs are committed to `data.ctrl`.
FF1 and FF2 share a single IK call for l_eff.

#### FF1 — Hip Reaction Torque Cancellation

Every hip torque command creates an equal-and-opposite reaction on the body. FF1 cancels this proactively in the same tick rather than waiting for LQR feedback.

```python
tau_hip_total = ctrl[hip_L] + ctrl[hip_R]
l_eff         = abs(IK(hip_q_avg)['W_z'])
tau_ff1       = -ff1_alpha * tau_hip_total * (wheel_r / l_eff)
ctrl[wheel_L] += tau_ff1 / 2
ctrl[wheel_R] += tau_ff1 / 2
```

Current `ff1_alpha = 0.5` (partial cancellation).

#### FF2 — Gravity Compensation

When the robot leans, gravity exerts `m_b·g·l_eff·sin(pitch)` on the wheels. FF2 pre-applies this rather than chasing it with feedback.

```python
tau_ff2 = ff2_alpha * m_b * g * l_eff * sin(pitch)
ctrl[wheel_L] += tau_ff2 / 2
ctrl[wheel_R] += tau_ff2 / 2
```

Current `ff2_alpha = 1.0`.

**Caution:** The LQR linearized model already contains an implicit gravity term (`m_b·g·l_eff / denom`). At small angles where linearization is accurate, `ff2_alpha=1.0` may partially double-count gravity. The term is most beneficial at large lean angles where `sin(pitch) ≈ pitch` breaks down.

#### FF4 — Centripetal Turn Coupling

During high-speed turns, centripetal acceleration creates a pitching moment. FF4 compensates with a lean angle offset.

```python
theta_ff4 = -ff4_alpha * v_fwd * yaw_rate / 9.81
theta_ref_total = theta_ref + theta_ff4
```

Current `ff4_alpha = 0.0` (disabled).

---

## Sensor Latency and Prediction

A two-stage ring-buffer delay model simulates real hardware timing:

| Delay | Value | Source |
|-------|-------|--------|
| Sensor delay | 0.002 s (~1 control tick) | CAN + IMU read latency |
| Actuator delay | 0.001 s | CAN write latency |

**Smith-style matrix predictor:**

After popping the delayed sensor readings, the controller propagates them forward using stored torque history:

```
x̂(t) = A_d^n × x(t−n) + Σ_{k=0}^{n-1} A_d^{n−1−k} × B_d × u(t−n+k)
```

A_d and B_d are the discretized (ZOH, 500 Hz) state matrices evaluated at nominal hip position. This partially cancels the phase lag that would otherwise cost stability margin.

---

## Optimizer Scenarios

All scenarios run at 500 Hz controller rate (2 kHz physics, CTRL_STEPS=4). Noise model active by default. The shared `run()` / `sim_loop.py` backend ensures identical physics across all scenarios.

| S# | Name | Controller under test | Key stimulus | Metric |
|:--:|------|----------------------|--------------|--------|
| 1 | `1_LQR_pitch_step` | Balance LQR | Fixed legs; +5° pitch step + ±4 N kicks; VelocityPI OFF | ISE pitch (rad²·s) |
| 2 | `2_leg_height_gain_sched` | Balance LQR + gain scheduler | Legs cycling full stroke; ±4 N kicks | ISE pitch (rad²·s) |
| 3 | `3_VEL_PI_disturbance` | Velocity PI | Fixed legs; v_desired=0; ±1 N kicks | Wheel travel (m) |
| 4 | `4_VEL_PI_staircase` | Velocity PI | Fixed legs; 0→0.3→0.6→1.0→−0.5→−1.0 m/s | vel_track_rms (m/s) |
| 5 | `5_VEL_PI_leg_cycling` | Velocity PI | Legs cycling + staircase + kicks + bumps | vel_rms + pitch_rms |
| 6 | `6_YAW_PI_turn` | Yaw PI | Fixed legs; ω_desired=1 rad/s pure turn | yaw_track_rms (rad/s) |
| 7 | `7_DRIVE_TURN` | Yaw PI + Velocity PI | v=0.3 m/s + ω=0.5 rad/s simultaneous | Combined vel + yaw rms |
| 8 | `8_terrain_compliance` | Hip Impedance + Roll | v=1.0 m/s; single 5 cm left bump at x=1.2 m | Peak roll angle (deg) |
| 9 | `9_jump` | Jump FSM | Full jump sequence from BALANCE | Jump height (mm) |

**Optimizer targets:** LQR → S2 · VelocityPI → S5 · YawPI → S6 · Suspension → S8 · Jump → S9

---

## Parameter Hierarchy

All parameters are immutable at runtime via frozen `SimParams` dataclasses. Optimizers create modified copies via `dataclasses.replace()` — no global state mutation.

```
params
├── robot          → RobotGeometry: Q_RET, Q_EXT, Q_NOM, masses, wheel_r, dimensions
├── gains
│   ├── lqr        → LQRGains: Q_pitch, Q_pitch_rate, Q_vel, R
│   ├── velocity_pi→ VelocityPIGains: Kp, Ki, Kff, theta_max, rate_limit, int_max
│   ├── yaw_pi     → YawPIGains: Kp, Ki, torque_max, omega_cmd_max, int_max
│   ├── suspension → SuspensionGains: K_s, B_s, K_roll, D_roll, freefall_scale, landing_ramp_s
│   ├── jump       → JumpGains: crouch_time, max_torque, omega_max, timings, thresholds
│   └── feedforward→ FeedforwardGains: ff1_alpha, ff2_alpha, ff4_alpha
├── timing         → SimTiming: sim_timestep (0.0005 s), ctrl_hz (500 Hz)
├── latency        → LatencyParams: sensor_delay_s, actuator_delay_s
├── noise          → NoiseParams: pitch_std, pitch_rate_std, accel_std, roll_std_rad
├── motors         → MotorParams: KV, Kt, speed limits, thermal model (hip + wheel)
└── battery        → BatteryParams: capacity, V/I curves, thermal model
```

---

## Why We Do NOT Use LQI (LQR + Integrator on Pitch)

### The Velocity PI already IS the integrator

In BALANCE mode (`v_desired = 0`), the Velocity PI integrates wheel velocity error:
```
∫ v_err dt  →  θ_ref  →  LQR lean command
```
Any pitch offset that causes drift is continuously integrated and corrected via the θ_ref lean command — indirect integral action on position without touching the LQR.

### Why LQI would conflict

Adding a pitch integrator inside the LQR creates two competing integrators:
```
LQI pitch integrator  →  wheel torque  →  velocity drift
Velocity PI integrator →  θ_ref        →  wheel torque
```

Both try to null pitch error via different paths. This causes integrator wind-up interactions, potential instability, and much harder tuning.

### Ascento precedent

Ascento (ETH Zurich, ICRA 2019) uses pure LQR for balance without integral action on pitch. The outer Velocity PI handles steady-state correction. This architecture is confirmed correct — do not add LQI.

---

## Knee Spring Analysis — Why It Does Not Help Jump Height

A torsional spring at the knee joint (K=12 N·m/rad) was simulated to store energy during CROUCH and release it during EXTEND. **Conclusion: disabled (`use_knee_spring=False`), logic retained in code.**

### Why it fails — motor speed limit interaction

The AK45-10 hip motor's torque tapers with speed:
```
speed_limit = 1 − |ω_hip| / omega_max        (omega_max = 18.85 rad/s)
torque = max_torque × speed_limit
```

Without the spring, the motor runs at near-full torque throughout the 40° extension stroke, arriving at Q_EXT at ~17.8 rad/s — nearly optimal on its own.

The spring adds up to **10.8 N·m** (both legs) at the start of EXTEND on top of the motor's 14 N·m. This doubles the initial torque, hitting `omega_max` after only ~25° of the 40° stroke. Motor torque drops to zero. For the remaining 15° of stroke, neither motor nor spring provides meaningful force.

Energy comparison (per-leg estimates, I_leg ≈ 0.031 kg·m²):

| | KE at Q_EXT | Source |
|---|---|---|
| Without spring | 4.9 J/leg | Motor full torque, full stroke |
| Spring energy stored | +1.2 J/leg | Released in first 25° |
| Motor work lost (last 15°) | −1.8 J/leg | Motor at zero after omega_max hit |
| **Net with spring** | **~4.3 J/leg** | Slightly worse |

### When a spring could help

1. **Motor already hits `omega_max` before Q_EXT without the spring** — so the spring fills a gap. Requires higher gear ratio or longer stroke.
2. **Latch-and-release (catapult)**: spring pre-loaded slowly at stall, released by a mechanical latch. Decouples storage speed from release speed.
3. **Series Elastic Actuator**: spring in series with the motor shaft; motor pre-winds spring at low speed.
4. **Lower K_spring (~6 N·m/rad)**: halving stiffness halves initial over-torque, allowing the motor to stay in range longer. Small net gain (~1.2 J) possible but not compelling.
