# LQR Integration Guide

## Overview

The LQR controller has been integrated into the MuJoCo viewer with **PID/LQR blending** to smoothly transition from legacy PID control to optimal LQR control.

---

## Key Components

### 1. **lqr_design.py** — LQR Gain Computation
Computes optimal feedback gains using:
- Linearized inverted pendulum dynamics (Grasser et al. 2002)
- Continuous-time Algebraic Riccati Equation (CARE)
- State vector: `[pitch, pitch_rate, wheel_pos, wheel_vel]`
- LQR gains stored in `sim_config.py`

**Run it to verify the model:**
```bash
python lqr_design.py
```

### 2. **sim_config.py** — LQR Gains
```python
LQR_K      = np.array([-108.9959, -19.2330, -3.1623, -2.3265])
LQR_Q_DIAG = [100.0, 1.0, 1.0, 0.1]      # State cost matrix
LQR_R_VAL  = 0.1                          # Control cost
```

The PID gains are preserved for reference and rollback.

### 3. **viewer.py** — Blending Control
Modified viewer with:
- **Wheel position tracking**: Integrates wheel velocities to track cumulative position
- **LQR control law**: Computes optimal torque from state
- **PID control law**: Legacy proportional-derivative control
- **Blending slider**: Real-time fade between PID and LQR (0.0 → 1.0)

---

## How to Use

### Launch the Viewer
```bash
python viewer.py
```

### Control the Blend (LQR Fade-In)
A **blue slider** appears at the top of the matplotlib telemetry window:

```
[slider] ─────●───────  "LQR Blend"  0.00
```

**How to use it:**
1. **Slider at 0.0**: Pure PID control (robot balances with original PD gains)
2. **Slide to 0.5**: 50/50 blend — PID + LQR contribution
3. **Slide to 1.0**: Pure LQR control (optimal feedback)

### Monitor the Effect
Watch the **Motion tab** (Pitch, Wheel Torque graphs):
- At 0.0: Smooth PID response
- At 0.5: Transition behavior
- At 1.0: Sharper, more aggressive LQR response

---

## Expected Behavior

### Pitch Stabilization (all blend values)
The robot should remain balanced in neutral stance. If it falls:
- **At blend=0.0 (PID)**: Tune PITCH_KP / PITCH_KD
- **At blend=1.0 (LQR)**: Check if Q[0,0] (pitch cost) is too high or low

### Wheel Motion
- **At blend=0.0**: Minimal wheel motion during balancing
- **At blend=1.0**: More active wheel damping (expected)

### Control Torque Magnitude
- **At blend=0.0**: u_pid = KP × pitch_error + KD × pitch_rate
- **At blend=1.0**: u_lqr = -K @ state (more sophisticated)

---

## Testing Checklist

- [ ] **Blend=0.0**: Robot balances normally (PID baseline)
- [ ] **Blend=0.5**: Smooth transition, stable
- [ ] **Blend=1.0**: Robot still balances (LQR stable)
- [ ] **Fwd/Bwd buttons**: Work at all blend values
- [ ] **Jump**: Works at all blend values
- [ ] **Telemetry**: Plots update smoothly

---

## Tuning the LQR Gains

If the system is unstable or oscillates at full blend:

1. **Edit lqr_design.py**, modify Q and R:
   ```python
   Q = np.diag([100.0, 1.0, 1.0, 0.1])  # Higher Q[0,0] = aggressive pitch control
   R = np.array([[0.1]])                 # Lower R = less penalty on torque
   ```

2. **Recompute gains:**
   ```bash
   python lqr_design.py
   ```
   Copy the new `LQR_K` values into `sim_config.py`

3. **Test in viewer:**
   ```bash
   python viewer.py
   # Slide to blend=1.0 to test new gains
   ```

---

## State Vector Reference

**LQR state in control law:**
```
x = [pitch_err, pitch_rate, wheel_pos, wheel_vel]
   [    rad      rad/s       m       m/s      ]
```

**Control output:**
```
u = -K @ x  [N·m]
```

---

## Integration Test

To verify the control law is executing correctly:
```bash
python test_lqr_integration.py
```

This runs a 2-second open-loop simulation and shows:
- PID control output (u_pid)
- LQR control output (u_lqr) computed but unused
- Wheel position tracking
- State evolution

---

## Next Steps

After verifying stable operation at blend=1.0:
1. Implement **firmware on UNO R4 WiFi** using the same LQR gains
2. Add real IMU (BNO086) feedback
3. Test on hardware with initial blend=0.0, then gradually increase blend

---

## References

- **Linearization**: Grasser et al. 2002, "JOE: A Mobile Robot for Education"
- **LQR Theory**: Optimal control, continuous-time Riccati equation
- **Inverted Pendulum**: Standard wheeled robot balance problem

