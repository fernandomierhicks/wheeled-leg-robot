# Control Architecture — Wheeled-Leg Robot

**Status:** Design complete. Implementation starts Phase 1 (3-state LQR).
**Reference robot:** Ascento (ETH Zurich, ICRA 2019) — same cascade architecture.

---

## Architecture Overview

### 4 Controllers — 2 LQRs + 2 PIDs

```
Outer loops (50 Hz)                 Inner loop (500 Hz)          Actuators
─────────────────────               ───────────────────          ─────────
[Velocity PI] ──θ_ref──▶            [Balance LQR] ──τ_sym─▶     τ_L = τ_sym + τ_yaw
[Yaw PI]      ──τ_yaw──▶────────────────────────────────────▶   τ_R = τ_sym − τ_yaw
                                    State: [φ−θ_ref, φ̇, v−v_ref]

Independent (100 Hz)
──────────────────────
[Leg Impedance PD] ──────────────────────────────────────────▶  τ_hip (both legs)
```

| # | Controller | Type | State / Input | Output | Rate |
|---|------------|------|--------------|--------|------|
| 1 | Balance LQR | LQR 3-state | [pitch−θ_ref, pitch_rate, wheel_vel_avg−v_ref] | τ_sym | 500 Hz |
| 2 | Leg Impedance | PD impedance | [q_hip−q_nom, dq_hip] | τ_hip | 100 Hz |
| 3 | Velocity | PI | v_desired − v_measured | θ_ref (lean cmd) | 50 Hz |
| 4 | Yaw | PI | ω_yaw_desired − ω_yaw_measured | τ_yaw | 50 Hz |

---

## Controller Details

### 1. Balance LQR (inner, 500 Hz)

**State vector change from Session 4:**

```
OLD (broken for drive):  x = [pitch_err, pitch_rate, wheel_pos, wheel_vel]   ← 4-state
NEW:                     x = [pitch − θ_ref, pitch_rate, wheel_vel_avg − v_ref]  ← 3-state
```

Removing `wheel_pos` eliminates the restoring force that fights intentional motion.
Position hold in BALANCE mode is handled by the outer loop (θ_ref derived from drift).

**LQR formulation:**
```
u = −K @ x
K = [K_pitch, K_pitch_rate, K_vel]   ← expected ~[-35, -7, -3]
Q = diag([Q_pitch, Q_pitch_rate, Q_vel])
R = scalar
```

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

### 2. Velocity PI (outer, 50 Hz)

Converts velocity command into a lean angle setpoint for the LQR:

```python
v_err = v_desired − v_measured
θ_ref = Kp_v * v_err + Ki_v * integral(v_err)
θ_ref = clamp(θ_ref, −0.26 rad, +0.26 rad)   # ±15° hard limit
```

In BALANCE mode: `θ_ref = K_pos * wheel_pos_drift` (gentle position hold), `v_ref = 0`.
In DRIVE mode: `θ_ref` from PI above, `v_ref = v_desired` (feedforward to LQR).

Starting gains: Kp_v = 0.04 rad/(m/s), Ki_v = 0.008 rad/m.

---

### 3. Yaw PI (outer, 50 Hz)

Produces differential torque for turning. Blends with drive.

```python
yaw_err = ω_yaw_desired − ω_yaw_measured   # from BNO086 gyro Z
τ_yaw = Kp_yaw * yaw_err + Ki_yaw * integral(yaw_err)
τ_yaw = clamp(τ_yaw, −0.5 N·m, +0.5 N·m)

τ_L = τ_sym + τ_yaw
τ_R = τ_sym − τ_yaw
```

Starting gains: Kp_yaw = 0.3, Ki_yaw = 0.05.

---

### 4. Leg Impedance PD (independent, 100 Hz)

Decoupled from balance. Controls hip motor as a spring-damper:

```python
τ_hip = −K_s * (q_hip − q_nom) − B_s * dq_hip
```

| Setting | K_s (N·m/rad) | B_s (N·m·s/rad) |
|---------|--------------|-----------------|
| Soft    | 1.0          | 0.3             |
| Medium  | 5.0          | 0.8             |
| Stiff   | 15.0         | 1.5             |

`SUSPENSION_SOFTNESS` scalar 0→1 interpolates K_s and B_s.
`q_nom` = commanded leg height (adjustable by user).

**Terrain adaptation:** When unexpected pitch disturbance detected (|pitch − θ_ref| > 0.05 rad
while θ_ref ≈ 0), reduce K_s by 30% for 200ms to absorb impact passively.

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

| Mode | Balance LQR | Leg K_s | Entry Trigger |
|------|-------------|---------|---------------|
| BALANCE | Normal | User setting | default / stop |
| DRIVE | Normal + v_ref | User setting | drive command |
| TURN | Normal + v_ref | User setting | turn command |
| JUMP | Active, stiff | K_S_STIFF | jump command |
| FLYING | K_pitch × 2 | Neutral q_nom | contact force < 0.1 N for 50ms |
| LANDING | K_pitch × 2 | K_s × 0.3 | IMU Z accel > 1.5g |
| SETTLED | Normal | Ramp to user over 1s | |pitch| < 2° for 500ms |

---

## Implementation Phases

### Phase 1 — 3-State LQR + Gain Scheduling
**Files:** `lqr_design.py`, `sim_config.py`, `run_scenario_visual.py`

| Step | Action | Pass Criterion |
|------|--------|---------------|
| 1.1 | Add `q_hip` param to `compute_lqr_gain()`, recompute l_eff via IK | l_eff ~0.17m at Q_NOM |
| 1.2 | Change A/B to 3×3/3×1 (remove wheel_pos row/col) | A is 3×3, open-loop unstable |
| 1.3 | Add `compute_gain_table()` — computes K at Q_RET, Q_NOM, Q_EXT | 3 distinct K arrays |
| 1.4 | Add `interpolate_gains(K_table, q_hip)` — linear interp | Midpoint = average of extremes |
| 1.5 | Update `sim_config.py` — replace LQR_K with LQR_K_TABLE | Import without error |
| 1.6 | Update controller in `run_scenario_visual.py` to 3-state + interpolation | Self-balance RMS pitch < 0.15° |

**Tuning:** Re-run `optimize_self_balance.py` with 4D search (Q_pitch, Q_pitch_rate, Q_vel, R).
Verify balance at all 3 leg heights.

---

### Phase 2 — Velocity PI + Drive Mode
**Files:** `run_scenario_visual.py`, `viewer.py`

| Step | Action | Pass Criterion |
|------|--------|---------------|
| 2.1 | Add `VelocityPI` class with integrator, ±15° clamp, 0.5 rad/s rate limit | Unit test: θ_ref ramps then decays |
| 2.2 | Add v_ref feedforward to LQR state | — |
| 2.3 | Create DRIVE scenario: v_desired=0.5 m/s at t=2s | Reach 0.5 m/s in <3s, pitch < 3° |
| 2.4 | Add W/S keys + HUD (v_desired, v_measured, θ_ref) to viewer | Manual drive works |
| 2.5 | Add v_desired ramp 0.5 m/s² on key hold/release | Smooth accel/decel |

**Tuning:** 5×5 grid sweep Kp_v ∈ [0.01, 0.1], Ki_v ∈ [0.001, 0.05].
Fitness = settle_time + overshoot + oscillation_count.

---

### Phase 3 — Yaw PI + Turn Mode
**Files:** `run_scenario_visual.py`, `viewer.py`

| Step | Action | Pass Criterion |
|------|--------|---------------|
| 3.1 | Add `YawPI` class, ±0.5 N·m clamp | Unit test: converges on ω step |
| 3.2 | Apply differential mixing τ_L/τ_R | — |
| 3.3 | TURN scenario: ω=1 rad/s for 6.28s | 360° with < 10cm drift |
| 3.4 | DRIVE+TURN scenario: v=0.3 + ω=0.5 | Curved path, pitch < 3° |
| 3.5 | Add A/D keys + yaw HUD to viewer | Manual turn works |

**Tuning:** Manual Kp_yaw/Ki_yaw — start [0.3, 0.05], adjust to eliminate oscillation.
Check cross-coupling: turning must not disturb pitch by more than 2°.

---

### Phase 4 — Leg Impedance Suspension
**Files:** `run_scenario_visual.py`, `viewer.py`, `sim_config.py`

| Step | Action | Pass Criterion |
|------|--------|---------------|
| 4.1 | Add K_S/B_S params + SUSPENSION_SOFTNESS to sim_config | — |
| 4.2 | Add `ImpedanceController` class | Returns to q_nom after perturbation |
| 4.3 | Run at 100 Hz (every 5 timesteps), apply to both hips | Balance holds while legs move |
| 4.4 | Terrain disturbance: reduce K_s 30% for 200ms on unexpected pitch | 5N impulse → Z deflect < 2cm |
| 4.5 | Add Q/E (leg height) + [ / ] (softness) keys to viewer | Manual control works |

**Tuning:** Verify critical damping: B_s > 2√(K_s × I_leg). No optimization needed.

---

### Phase 5 — Jump Recovery State Machine
**Files:** `run_scenario_visual.py`, `viewer.py`

| Step | Action | Pass Criterion |
|------|--------|---------------|
| 5.1 | Add `RobotMode` enum (BALANCE/DRIVE/TURN/JUMP/FLYING/LANDING/SETTLED) | — |
| 5.2 | FLYING detection: contact force < 0.1N for 50ms | Mode switches on 5cm drop |
| 5.3 | LANDING detection: IMU Z accel > 1.5g after FLYING | Spike triggers LANDING |
| 5.4 | LANDING: K_s × 0.3 for 500ms, K_pitch × 2.0 | Land from 15cm, stable in 2s |
| 5.5 | SETTLED: |pitch| < 2° for 500ms → BALANCE, ramp K_s over 1s | Stable recovery |
| 5.6 | JUMP: stiffen legs, extend hips, enter FLYING on liftoff | Jump height ≈ baseline1 |
| 5.7 | Add SPACE key for jump, show mode in HUD | Manual jump-land works |

**Tuning:** 1D search on landing K_s multiplier [0.1–0.5]. Minimize time-to-stable from 10cm drop.

---

### Phase 6 — End-to-End Optimization
**Files:** `optimize_self_balance.py`, `test_behavioral.py`

| Step | Action |
|------|--------|
| 6.1 | Create compound scenario: balance 5s → drive 5s → turn 3s → stop 5s |
| 6.2 | Multi-objective fitness: RMS pitch + velocity error + yaw error + control effort |
| 6.3 | Differential evolution on 6 params: Q_pitch, Q_pitch_rate, Q_vel, R, Kp_v, Ki_v |
| 6.4 | Validate at 3 leg heights |
| 6.5 | Write final gains to sim_config.py with comments |

---

## Haiku Agent Prompt Template

Use this when spawning a Haiku agent for a single implementation step:

```
Robot: 2-wheel inverted pendulum, 4-bar legs, MuJoCo simulation (Windows)
Working dir: simulation/mujoco/LQR_controller/

Context: [paste relevant architecture section above]
Files already modified: [list]

Phase [N], Step [M.K]:
File: [filename]
Action: [exact action from step table]
Success criterion: [test from step table]

Rules:
- Only modify the specified file
- Run the file after editing and report output
- Do not add features beyond what the step describes
```

---

## Tuning Summary Table

| Controller | Method | Parameters | Tool |
|------------|--------|-----------|------|
| Balance LQR | Differential evolution | Q_pitch, Q_pitch_rate, Q_vel, R | `optimize_self_balance.py` |
| Gain scheduling | Offline at 3 points | l_eff at Q_RET/NOM/EXT | `lqr_design.py` |
| Velocity PI | 5×5 grid sweep | Kp_v, Ki_v | `lqr_parameter_sweep.py` |
| Yaw PI | Manual visual | Kp_yaw, Ki_yaw | viewer |
| Leg impedance | Critical damping formula | K_s, B_s | manual check |
| Jump recovery | 1D search | Landing K_s multiplier | `test_behavioral.py` |
| All together | Differential evolution 6D | Q×4 + Kp_v + Ki_v | compound scenario |
