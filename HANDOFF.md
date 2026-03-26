# HANDOFF — Feedforward & State Estimation Improvements

Status: **Planning** — options documented, not yet implemented.

---

## Current Baseline

### Existing feedforward elements

| # | Name | Where | What it does |
|---|------|-------|-------------|
| 1 | Pitch equilibrium FF | `controllers/lqr.py` — `get_equilibrium_pitch(hip_q_avg)` | Shifts LQR linearization point as leg angle changes so the controller always balances around the correct tilt |
| 2 | Lean acceleration FF | `controllers/velocity_pi.py` — `Kff * dv_cmd/dt` | Applies `theta_ref ≈ a/g` immediately on velocity command steps instead of waiting for PI to integrate |
| 3 | Roll leveling FF | `controllers/hip.py` — `K_roll * roll + D_roll * roll_rate` | Differential hip offsets to keep the body level on uneven terrain |

### State estimation

- **No EKF.** State estimation is a deterministic matrix predictor that forward-propagates delayed sensor readings through the discretized linear model using stored torque history. It compensates for sensor + actuator latency but has no noise model, no measurement fusion, and no disturbance estimation.
- **No live CoM calculation.** Mass distribution is baked into the LQR gain table at build time. The controller has no runtime knowledge of where CoM actually is.

---

## Proposed Improvements

### FF1 — Hip Reaction Torque Cancellation

**Priority: HIGH** — biggest bang-for-buck

**Problem:** Every hip torque command creates an equal-and-opposite reaction torque on the body. The LQR sees this as a pitch disturbance and corrects it, but only after the disturbance has already occurred (one control loop late at minimum). During jump crouch, suspension response, and roll leveling, this reaction torque is the dominant disturbance source.

**Approach:**
- Read the hip torque command `tau_hip` (already computed each tick)
- Feed a cancellation torque into the wheel motors: `tau_wheel_ff = -tau_hip * (r_wheel / l_eff)` scaled by the geometric ratio
- Apply it in the same control tick, before the LQR output is summed

**Where to implement:** `sim_loop.py`, between hip torque computation and wheel torque summation.

**Risks:**
- Geometric ratio `l_eff` changes with leg angle — must use current `l_eff` from IK, not a constant
- Sign convention must be verified carefully (hip extension vs. retraction)
- May need a tunable gain `0 < alpha <= 1` rather than full cancellation if model mismatch causes overcorrection

---

### FF2 — Gravity Compensation Torque on Wheels

**Priority: HIGH**

**Problem:** When the robot leans (pitch != 0), gravity exerts a torque on the pendulum. The LQR handles this through feedback, but during large lean angles (velocity tracking, jump recovery) the feedback is always chasing the gravitational load.

**Approach:**
- Compute `tau_gravity_ff = m_body * g * l_eff(q_hip) * sin(pitch)`
- Add to wheel torque command as feedforward, reducing the error signal the LQR must correct
- `l_eff` comes from the existing IK/gain-scheduling infrastructure

**Where to implement:** `sim_loop.py`, added to wheel torque after LQR output.

**Risks:**
- The LQR already has an implicit gravity term in its linearized model (the `alpha` coefficient). Adding explicit gravity FF on top may cause double-counting. May need to reduce LQR Q-weight on pitch proportionally, or scale this FF term with a tunable gain.
- Only beneficial at larger lean angles where `sin(pitch) ≈ pitch` breaks down

---

### FF3 — CoM Shift Compensation

**Priority: MEDIUM**

**Problem:** As hip angle changes, the leg mass moves and the whole-body CoM shifts along X. The LQR linearization assumes a fixed pivot geometry for each gain-table entry, but between table entries the CoM shift is uncompensated. This causes a transient pitch disturbance during leg transitions (visible as a "lurch" when entering/exiting crouch).

**Approach:**
- Compute CoM_x as a function of hip angle using the known link masses and geometry (forward kinematics already available in `physics.py`)
- Convert CoM_x shift into a pitch reference offset: `delta_theta = arctan(delta_CoM_x / l_eff)`
- Add this offset to `pitch_ff` in the LQR state vector

**Where to implement:** New function in `physics.py`, called from `lqr.py` or `sim_loop.py`.

**Risks:**
- Requires accurate mass properties for each link (currently only total `m_body` is used)
- Small effect when leg motion is slow; most valuable during fast leg transitions (jump)
- Could be approximated as a lookup table rather than computed every tick

---

### FF4 — Centripetal Coupling During Turns

**Priority: MEDIUM**

**Problem:** At forward speed, yaw rotation creates a centripetal acceleration that tilts the robot. The yaw PI and roll leveling operate independently of the balance controller, so this cross-coupling is uncompensated.

**Approach:**
- Compute centripetal lean: `theta_centripetal = v * omega_yaw / g`
- Subtract from LQR pitch reference so the robot leans into turns
- Optionally add a roll-rate feedforward term from yaw command

**Where to implement:** `sim_loop.py`, modifying `theta_ref` before passing to LQR.

**Risks:**
- Only matters at meaningful speed + yaw rate combinations. At low speeds this is negligible.
- Sign must match coordinate convention (lean into the turn = negative roll on right turn)

---

### SE1 — Extended Kalman Filter

**Priority: LOW** (current predictor works; EKF is a larger architectural change)

**Problem:** The matrix predictor assumes the linear model is perfect. Any model mismatch (unmodeled friction, mass errors, leg flexibility) accumulates as prediction error. There is no mechanism to detect or correct this drift.

**Approach:**
- Replace the matrix predictor with an EKF that fuses:
  - IMU: pitch, pitch_rate (delayed, noisy)
  - Wheel encoders: wheel_vel (delayed)
  - Hip encoders: hip_q (undelayed, low noise)
- State vector: `[pitch, pitch_rate, wheel_vel, disturbance_torque]` — the disturbance state gives free disturbance estimation
- Process model: same linearized pendulum dynamics already in `lqr.py`
- Measurement model: identity on measured states with appropriate delays

**Where to implement:** New file `controllers/ekf.py`, replacing the predictor block in `sim_loop.py`.

**Benefits beyond current predictor:**
- Graceful degradation under model mismatch
- Disturbance torque estimate (could feed into FF1 as a learned correction)
- Noise-optimal state estimates (currently just raw delayed+predicted values)

**Risks:**
- Tuning Q and R covariance matrices adds complexity
- Must handle the variable-delay nature of the sensor pipeline
- Computational cost higher than current predictor (matrix multiplies per tick)
- Overkill if the current predictor + feedforward terms are sufficient

---

## Implementation Order

Recommended sequence based on impact and independence:

```
1. FF1  Hip reaction torque cancellation     (high impact, simple, independent)
2. FF2  Gravity compensation on wheels       (high impact, needs careful gain tuning)
3. FF3  CoM shift compensation               (medium, most valuable once FF1+FF2 are in)
4. FF4  Centripetal turn coupling             (medium, independent, only matters at speed)
5. SE1  EKF                                  (low priority, large scope, do last if needed)
```

Each can be implemented and tested independently. FF1 and FF2 can be done in parallel.

---

## Validation Plan

For each feedforward term:
1. Add the FF term with a tunable gain (default 0, so it's a no-op)
2. Run the relevant scenario (S08 for suspension, S09 for disturbance, S10 for jump)
3. Compare pitch tracking error RMS with FF gain = 0 vs. FF gain = 1
4. Sweep gain 0 to 1 to find optimal value (may not be 1.0 due to model mismatch)
5. Check that no scenario regresses
