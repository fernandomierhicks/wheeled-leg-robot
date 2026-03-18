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

**Optimized gains (balance-only run, 8496 evals):**
| Parameter | Value |
|-----------|-------|
| Q_PITCH | 0.654 |
| Q_PITCH_RATE | 0.134 |
| Q_VEL | 0.0269 |
| R | 9.275 |
| K_nominal (at Q_NOM) | [-12.672, -1.754, -0.003] |

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

**Optimized gains (balance-only, 3032 evals, sign-corrected 2026-03-18):**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Kp_v | 0.299 | rad/(m/s) |
| Ki_v | 0.500 | rad/m — may benefit from wider range (was at upper bound) |
| fitness | 0.142 m | wheel travel over 5s balance (was 2.94 m without PI) |

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
- Best: Q=[0.654, 0.134, 0.0269], R=9.275, K_nominal=[-12.672, -1.754, -0.003]
- RMS pitch 0.113°, max pitch 0.363°, zero liftoff

---

### Phase 2 — Velocity PI + Balance Position Hold ✅ COMPLETE

**Files:** `scenarios.py`, `optimize_lqr.py`, `replay.py`, `sim_config.py`

- `VelocityPI` class wired into all balance scenarios (v_desired=0)
- Correct sign: `vel_est_ms = wheel_vel * WHEEL_R` (positive = forward)
- (1+8)-ES optimizer over KP_V/KI_V with fixed LQR Q/R
- Best: KP_V=0.299, KI_V=0.500, fitness=0.142 m (20× improvement)
- `replay.py`: 4-panel telemetry, v_desired slider (±1 m/s), Drive ON/OFF toggle, XYZ world frame

**Next step for Phase 2:** Re-run optimizer with all 5 scenarios active (balance, disturbance, drive_slow, drive_med, obstacle) and wider KI_V range (up to 2.0). See HANDOFF.md.

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

## Tuning Summary

| Controller | Method | Parameters | Status |
|------------|--------|-----------|--------|
| Balance LQR | (1+8)-ES | Q_pitch, Q_pitch_rate, Q_vel, R | ✅ Done |
| Gain scheduling | Offline at 3 points | l_eff at Q_RET/NOM/EXT | ✅ Done |
| Velocity PI | (1+8)-ES | Kp_v, Ki_v | ✅ Balance-only done; combined scenarios next |
| Yaw PI | Manual visual | Kp_yaw, Ki_yaw | Not started |
| Leg impedance | Critical damping formula | K_s, B_s | Placeholder values only |
| Jump recovery | 1D search | Landing K_s multiplier | Not started |
