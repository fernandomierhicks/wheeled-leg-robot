# Latencies.MD — System Latency Budget

This document maps every latency source in the real robot, the current zero-latency
simulation, and the Arduino-HIL configuration. It is the reference for deciding what
to emulate in sim before firmware work begins.

---

## 1. Control Loop Reference Frame

The balance loop runs at **500 Hz → 2 ms period**. All latencies below are expressed
both in milliseconds and in **loop counts** (1 loop = 2 ms) because that is the unit
that matters for stability: a pure delay of N loops reduces the phase margin by
N × 360° × f_ctrl / f_crossover. At a crossover of ~10 Hz (typical for this class of
balancing robot) each 2 ms loop costs ~3.6° of phase margin.

---

## 2. Real Robot — Full Latency Chain

```
  BNO086 internal fusion
        │ ~5 ms
        ▼
  I2C read (400 kHz, ~20 bytes)
        │ ~0.4 ms
        ▼
  Arduino RA4M1 — compute LQR + PI loops
        │ < 0.1 ms  (3×3 matrix multiply at 48 MHz)
        ▼
  CAN TX to ODESC (1 Mbit/s, standard frame)     CAN TX to AK45-10 (MIT CAN)
        │ ~0.15 ms                                       │ ~0.15 ms
        ▼                                                ▼
  ODESC FOC current loop (~8 kHz)              AK45-10 integrated FOC (~10 kHz)
        │ ~0.5 ms                                       │ ~0.3 ms
        ▼                                                ▼
  Wheel motor electrical lag (τ_elec)          Hip motor — leg motion
        │ ~2 ms  (L/R for 5065 130KV)                  │ ~1 ms
        ▼                                                ▼
  Torque at tyre contact patch                 Torque at knee joint
```

### Summary table

| Stage | Component | Latency | Loops |
|-------|-----------|---------|-------|
| Sensor fusion | BNO086 internal DSP | ~5.0 ms | 2.5 |
| Sensor read | I2C @ 400 kHz | ~0.4 ms | 0.2 |
| Computation | RA4M1 LQR + VelocityPI + YawPI | < 0.1 ms | 0.05 |
| Wheel command | CAN TX to ODESC | ~0.15 ms | 0.08 |
| Wheel actuator | ODESC FOC lag | ~0.5 ms | 0.25 |
| Wheel actuator | Motor electrical (τ_elec) | ~2.0 ms | 1.0 |
| Hip command | CAN TX to AK45-10 | ~0.15 ms | 0.08 |
| Hip actuator | AK45-10 integrated FOC lag | ~0.3 ms | 0.15 |
| **Total (wheel path)** | | **~8.2 ms** | **~4.1 loops** |
| **Total (hip path)** | | **~6.0 ms** | **~3.0 loops** |

### Key implications for the real robot

- **~4 loop delay on wheel torque** is the dominant risk. At 10 Hz crossover this
  costs ~14° of phase margin. Gains tuned in a zero-latency sim will be
  overconfident — likely too aggressive, causing oscillation or fall.
- **BNO086 fusion delay (5 ms) is fixed** — cannot be reduced. It is the dominant
  single contributor. Some mitigation: use raw gyro + complementary filter instead
  of the Game Rotation Vector report, reducing sensor lag to ~0.4 ms at the cost of
  drift. Worth evaluating once firmware is running.
- **CAN bus is fast** — 0.15 ms per frame is negligible. Two motors per bus (wheel L+R
  on ODESC, hip L+R on two AK45-10 addresses) adds no meaningful additional delay.
- **Hip path is 2 ms faster** than wheel path (no motor electrical lag because AK45-10
  has integrated driver and lower-inductance winding). Suspension impedance loop
  is therefore slightly less phase-limited than the balance loop.

---

## 3. Current Simulation — Zero Latency

The MuJoCo sim in `scenarios.py` / `sandbox_fastchart.py` has **no modelled latency anywhere**:

| Stage | Sim behaviour | Real behaviour | Error |
|-------|--------------|----------------|-------|
| IMU reading | `data.xquat` / `data.qvel` read instantly | 5.4 ms total | **−5.4 ms** |
| Computation | Same timestep | < 0.1 ms | negligible |
| Wheel torque | Written to `data.ctrl` instantly | 2.65 ms (FOC + elec) | **−2.65 ms** |
| Hip torque | Written to `data.ctrl` instantly | 0.45 ms (FOC) | **−0.45 ms** |
| **Total wheel path** | **0 ms** | **~8.2 ms** | **−8.2 ms** |

**Consequence:** all currently-tuned gains (LQR Q/R, VelocityPI, YawPI, suspension)
assume zero delay. The real robot will have ~4 loop delays. The gains will need to
be re-tuned or de-rated after introducing latency into the sim.

---

## 4. Arduino HIL Configuration

In HIL mode the Arduino runs the real firmware. MuJoCo runs on the PC and acts as
the plant. The Arduino receives simulated sensor state over USB serial and returns
computed torque commands.

```
  MuJoCo sim (PC)
        │  USB serial TX — state packet (pitch, pitch_rate, vel, yaw_rate, roll, hip_q)
        │  ~1–2 ms  (USB Full Speed, 1 ms microframe + serialisation)
        ▼
  Arduino RA4M1 — runs real firmware LQR + PI loops
        │  < 0.1 ms  (same as real robot)
        ▼
  USB serial TX — command packet (tau_L, tau_R, tau_hip_L, tau_hip_R)
        │  ~1–2 ms
        ▼
  MuJoCo sim — applies torques to data.ctrl
```

### HIL latency summary

| Stage | Latency | Notes |
|-------|---------|-------|
| PC → Arduino USB TX | ~1–2 ms | USB CDC, 1 ms frame granularity |
| Arduino firmware compute | < 0.1 ms | Same as real robot |
| Arduino → PC USB TX | ~1–2 ms | Second USB frame |
| Sim applies torque | ~0 ms | Instant within MuJoCo step |
| **Total round-trip** | **~2–4 ms** | **1–2 loop delays** |

### How HIL compares to real robot

| Latency source | Real robot | HIL sim | Delta |
|----------------|-----------|---------|-------|
| Sensor fusion (BNO086) | 5.0 ms | 0 ms — state sent directly | **−5 ms** |
| I2C read | 0.4 ms | 0 ms — replaced by USB | **−0.4 ms** |
| USB round-trip | — | 2–4 ms | **+2–4 ms** |
| ODESC FOC lag | 0.5 ms | 0 ms — no real motor | **−0.5 ms** |
| Motor electrical τ | 2.0 ms | 0 ms — no real motor | **−2.0 ms** |
| **Total wheel path** | **~8.2 ms** | **~2–4 ms** | **−4–6 ms** |

**HIL is still ~2–3 loop delays faster than the real robot** because the BNO086
fusion delay and motor electrical lag are absent. Gains validated in HIL will still
be slightly overconfident compared to the full robot.

### What HIL does validate

- Firmware logic (state machine, mode switching, CAN framing) is correct
- USB packet format and timing between PC and Arduino
- LQR + PI numerics match the Python sim
- Discrete-time integration behaviour at 500 Hz matches MuJoCo timestep

### What HIL does NOT validate

- BNO086 fusion delay (5 ms) — the largest single gap
- Motor electrical lag (2 ms) — second largest
- ODESC FOC dynamics
- Sensor noise from real IMU (sim uses Gaussian model)
- CAN bus timing (not present in HIL path)

---

## 5. How Delays Are Modelled in Sim

Work is done in a **new folder `simulation/mujoco/latency_sensitivity/`** — a full
copy of `LQR_Control_optimization/`. The original folder is left untouched as a
known-good zero-latency baseline.

### Implementation approach — physics-step ring buffers

Delays are expressed in **seconds** in `sim_config.py` and converted to buffer
depths at runtime:

```python
_n_sens = round(SENSOR_DELAY_S   / model.opt.timestep)   # e.g. 0.005 / 0.0005 = 10 steps
_n_act  = round(ACTUATOR_DELAY_S / model.opt.timestep)   # e.g. 0.0025 / 0.0005 = 5 steps
```

A `collections.deque` of fixed length holds the history. Each control tick:
1. Push the new reading/command onto the right of the deque
2. Pop the left (oldest) element as the delayed value seen by the controller/actuator

```python
# Sensor side (scenarios.py / sandbox_fastchart.py)
_sens_buf.append((pitch, pitch_rate, wheel_vel))
_pitch_d, _pitch_rate_d, _wheel_vel_d = _sens_buf[0]   # oldest = delayed

# Actuator side
_ctrl_buf.append((tau_sym - tau_yaw, tau_sym + tau_yaw))
_tau_L_d, _tau_R_d = _ctrl_buf[0]
data.ctrl[act_wheel_L] = motor_taper(_tau_L_d, data.qvel[d_whl_L], v_batt)
```

`motor_taper` uses **current** `omega_wheel` (not delayed) — correct, taper is
applied at the moment the command arrives at the motor.

Hip torque is **not delayed** — AK45-10 FOC lag (~0.3 ms) rounds to 0 steps at the
current 0.5 ms physics timestep.

### Delay constants (initial estimates)

```python
# sim_config.py  — latency_sensitivity/ only
USE_LATENCY_MODEL  = True
SENSOR_DELAY_S     = 0.005    # BNO086 fusion ~5 ms + I2C ~0.4 ms  → 10 steps = 5.0 ms
ACTUATOR_DELAY_S   = 0.0025   # ODESC FOC ~0.5 ms + τ_elec ~2 ms   →  5 steps = 2.5 ms
```

Set `USE_LATENCY_MODEL = False` to revert to zero-latency behaviour without changing
buffer code — useful for A/B comparisons.

### Delay granularity vs physics rate

Granularity = `SIM_TIMESTEP`. `CTRL_STEPS` recomputes automatically if the rate changes.

| Physics rate | Timestep | 2.3 ms rounded to | Optimizer speed |
|---|---|---|---|
| 2 kHz (current) | 0.5 ms | 2.0 or 2.5 ms | 1× baseline |
| 4 kHz | 0.25 ms | 2.25 ms | ~2× slower |
| 10 kHz | 0.1 ms | 2.3 ms exact | ~5× slower |

For overnight optimizer runs 4 kHz is the practical upper limit.

### BNO086 at 400 Hz

The Game Rotation Vector max rate is **400 Hz (2.5 ms period)**, not 500 Hz.
`CTRL_HZ` stays at 500 in sim for optimizer continuity. The ~0.5 ms average
staleness from the mismatch is negligible vs the 5 ms fusion lag and is handled in
firmware by re-using the last IMU sample on ticks with no new reading.

### Plugging in hardware measurements

Once hardware is running:
1. Log `bno086_report_us` and `ctrl_applied_us` on every firmware tick
2. Compute mean latency from the difference
3. Update `SENSOR_DELAY_S` in `sim_config.py` — buffer depth auto-adjusts
4. Re-run all optimizers in `latency_sensitivity/`

---

## 6. Phase Margin Estimate

Approximate Bode analysis assuming crossover at 10 Hz:

| Configuration | Total delay (wheel) | Phase lost | Estimated margin |
|---------------|---------------------|------------|------------------|
| Current sim | 0 ms | 0° | ~60° (typical LQR) |
| With sensor+motor delay in sim | ~8 ms | ~29° | ~31° — acceptable |
| Real robot (no mitigation) | ~8.2 ms | ~30° | ~30° — marginal |
| Real robot + raw gyro (no fusion) | ~3.2 ms | ~12° | ~48° — comfortable |

Margin below ~20° is generally unstable for balance robots under disturbance.
The current tuned gains have ~60° margin in sim — comfortable headroom to absorb
the real-robot delay once it is introduced into the sim and gains are re-tuned.
