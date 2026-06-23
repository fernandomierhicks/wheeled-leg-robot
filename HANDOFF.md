# Safe LQR + Nested Controller Implementation Plan

## Context

The robot is in flatsat (benchtop) configuration — no wheels mounted, motors free-spinning. `control_loop.cpp` is an empty stub. `STATE_RUNNING` currently only sends hip PD setpoints from the radio. All controllers defined in `docs/Control.MD` need to be implemented from scratch, safely.

The goal is a phased implementation where each controller is tested in isolation before the next is layered on, with explicit safety guards at every stage.

---

## Safety Principles

### Hardware risks
- **Wheel motors**: free-spinning in air is safe; risk is runaway spin (no load, any torque = unlimited acceleration) or unexpectedly large torque. Must governor both torque AND velocity.
- **Hip motors**: currently working and calibrated. Risk is full-torque extension (jump FSM). Keep jump FSM disabled until robot is secured on the ground.
- **Personal**: no risk on flatsat from wheel motors. Hip motors can pinch fingers — never reach into the 4-bar while in STATE_RUNNING.

### Software safety mechanisms to build in (all in `control_loop.cpp`)
1. **`PARAM_LQR_TORQUE_LIMIT`** — separate, lower torque clamp for testing (default 1.0 Nm; hard max is 7.0 Nm). Adjustable from GUI without reflash.
2. **`PARAM_SIM_PITCH_RAD`** — when non-zero, overrides `g_state.pitch_rad` with this value. Lets you inject fake pitch to verify LQR direction with zero risk of real instability.
3. **`PARAM_LQR_ENABLE`** — master enable flag for wheel torque output (default 0). Wheels get zero torque when disabled; LQR still runs so you can log computed torque.
4. **Wheel velocity soft-governor** — even a small constant torque will spin free wheels to infinity with no load. In the torque output path, before calling `wheel_motors_send()`, check `wm_L.vel_turns_s` and `wm_R.vel_turns_s`. If a wheel exceeds `PARAM_WHEEL_VEL_LIMIT_TURNS_S` (default 3 turns/s ≈ ~1.4 m/s for Ø150 mm wheel) *in the commanded torque direction*, clamp that wheel's torque to zero. This is a per-tick soft-governor, not an ESTOP — the LQR can still recover once speed drops.
5. **Pitch magnitude watchdog** — if `|pitch| > 50°` for more than 200 ms, ESTOP with `FAULT_PITCH_WATCHDOG`. Prevents a tipping robot from spinning wheels into the bench.
6. **Wheel runaway watchdog** — if `|wheel_vel_avg| > PARAM_WHEEL_VEL_LIMIT_TURNS_S * 2` (governor failed), ESTOP with `FAULT_WHEEL_RUNAWAY`. Hard backup.
7. **Radio disarm always works** — CH10 drop triggers `stateMachine_disarm_running()`; already wired.

New fault codes to add to `comm_protocol.h` / `robot_state.h`:
- `FAULT_PITCH_WATCHDOG` — robot tipped past safe angle
- `FAULT_WHEEL_RUNAWAY` — wheel velocity exceeded hard limit

---

## Implementation Phases

### ~~Phase 1 — Wire up control_loop.cpp skeleton~~ ✅ COMPLETE

The main `loop()` already busy-waits at the bottom (`while (micros() - t_start < 2000) {}`), giving exactly 500 Hz. `stateMachine_update()` → `on_running()` is therefore already called at 500 Hz. **No `IntervalTimer` needed.**

~~Work to do:~~
- ~~Rename / fill in `isr_500hz()` → call it `controlLoop_run()`, called from `on_running()` in `state_machine.cpp`~~
- ~~Add the safety parameters to `param_registry`~~
- ~~Add `FAULT_PITCH_WATCHDOG` and `FAULT_WHEEL_RUNAWAY` to fault codes~~
- ~~Add the pitch watchdog and wheel runaway watchdog (both just call `stateMachine_request_estop()` with the right fault code)~~
- ~~Log `tau_sym = 0` to `g_state.cmd_l/cmd_r` so GUI shows it~~

**Test**: flash, arm, watch telemetry. `cmd_l` / `cmd_r` should be 0. Trigger pitch watchdog by tilting past 50°; confirm ESTOP and correct fault code visible in GUI.

---

### Phase 2 — LQR with fake pitch injection ✅ IMPLEMENTED — test sequence in progress

Implement the 3-state LQR in `controlLoop_run()`:
```
x = [pitch - theta_ref,  pitch_rate,  wheel_vel_avg - v_ref]
tau_sym = -(K_pitch * x[0] + K_pitch_rate * x[1] + K_vel * x[2])
tau_sym = clamp(tau_sym, -PARAM_LQR_TORQUE_LIMIT, +PARAM_LQR_TORQUE_LIMIT)
```

K gains (nominal hip, params.py baseline): K_pitch=−9.771, K_pitch_rate=−1.881, K_vel=−0.00713. Hardcoded `theta_ref = 0`, `v_ref = 0` for isolation.

**Safe test sequence:**
- [x] 1. `PARAM_LQR_ENABLE = 0`. Confirm computed `tau_sym` appears in telemetry.
- [x] 2. `PARAM_SIM_PITCH_RAD = +0.1` (6° forward lean). Verify `tau_sym` is positive (forward wheel torque to correct). Check sign.
- [x] 3. `PARAM_SIM_PITCH_RAD = -0.1`. Verify `tau_sym` flips negative.
- [x] 4. `PARAM_SIM_PITCH_RAD = 0`, tilt robot physically. Verify real IMU pitch produces correct-sign torque in telemetry.
- [x] 5. `PARAM_LQR_ENABLE = 1`, 1 Nm limit. Tilt robot — confirm wheels move in the correcting direction (free-spinning).

---

### Phase 3 — Velocity PI

Implement on top of working LQR:
```
v_err = v_desired - wheel_vel_avg
integral += v_err * dt  (anti-windup clamp)
theta_ref = Kp*v_err + Ki*integral + Kff*dv_cmd_dt
theta_ref = clamp(theta_ref, -theta_max, +theta_max)
```

`wheel_vel_avg` from `wm_L.vel_turns_s` / `wm_R.vel_turns_s` × `wheel_r`. Gains from Control.MD table.

**Safe test** (flatsat, wheels free-spinning):
1. `v_desired = 0` (BALANCE mode). Push robot by hand; confirm wheels resist.
2. `v_desired = 0.1 m/s`. Confirm wheels spin up slowly. Watch `theta_ref` in telemetry — should be small and stable.

---

### Phase 4 — Yaw PI

```
tau_yaw = Kp*(omega_desired - imu_yaw_rate()) + Ki*integral
tau_L = tau_sym + tau_yaw
tau_R = tau_sym - tau_yaw
```

Need `imu_yaw_rate()` getter added to `IMU.h` (BNO086 already outputs it).

**Safe test** (flatsat): Command `omega_desired = 0.5 rad/s`. Confirm left wheel faster than right (verify sign vs. robot frame: +Y = left, +Z = up, right-hand rule → positive yaw = counterclockwise from above → left wheel slower, right faster). Adjust sign if needed.

---

### Phase 5 — Hip gain scheduling

Implement 3-point LQR gain interpolation from Control.MD:
```
alpha = (q_hip_avg - Q_RET) / (Q_EXT - Q_RET)
K = (1 - alpha) * K_retracted + alpha * K_extended
```

**Safe test** (flatsat): sweep hip position via radio CH3, confirm K values in telemetry change with leg height.

---

### Phase 6 — Feedforward terms (FF1, FF2)

Add after LQR + yaw verified:
- **FF2 first** (gravity comp, pitch angle only). Start `ff2_alpha = 0`, ramp up.
- **FF1 second** (hip reaction cancel, needs hip torque readback from CAN).
- FF4 stays `alpha = 0` until driving on floor.

**Safe test**: compare `tau_sym` in telemetry with and without FF terms at a fixed fake pitch angle.

---

### Phase 7 — Jump FSM (NOT on flatsat)

Only implement and test once:
- Robot is on the floor, secured or in a cradle
- You are behind a barrier or at arm's length
- Start with CROUCH phase only (slow, controlled). Add EXTEND only after CROUCH is verified.
- Use low `max_torque` parameter initially.

---

## Critical Files to Modify

| File | Changes |
|------|---------|
| `firmware/robot_teensy/teensy/src/control_loop.cpp` | All controller logic |
| `firmware/robot_teensy/shared/comm_protocol.h` | Add `FAULT_PITCH_WATCHDOG`, `FAULT_WHEEL_RUNAWAY`; telemetry fields for `tau_sym`, `theta_ref`, `tau_yaw` |
| `firmware/robot_teensy/teensy/src/param_registry` | Add `PARAM_LQR_ENABLE`, `PARAM_SIM_PITCH_RAD`, `PARAM_LQR_TORQUE_LIMIT`, `PARAM_WHEEL_VEL_LIMIT_TURNS_S` |
| `firmware/robot_teensy/teensy/src/state_machine.cpp` | `on_running()` calls `controlLoop_run()` instead of sending hip directly |
| `firmware/robot_teensy/teensy/src/IMU.h/.cpp` | Add `imu_yaw_rate()` getter |

## End-to-End Verification Checklist

- [ ] Flash firmware. Boot. Calibrate. Arm via CH10.
- [ ] GUI shows `tau_sym` in telemetry at 50 Hz = 0 when upright.
- [ ] With `SIM_PITCH = 0.1`, tau is positive and proportional to gain × error.
- [ ] With `LQR_ENABLE = 1`, `TORQUE_LIMIT = 1 Nm`: tilt robot — wheels spin in correcting direction.
- [ ] ESTOP from radio disarm (CH10 drop): fault code = none / normal disarm.
- [ ] ESTOP from pitch watchdog (tilt past 50°): fault code = `FAULT_PITCH_WATCHDOG` visible in GUI.
- [ ] ESTOP from wheel runaway: fault code = `FAULT_WHEEL_RUNAWAY`.
- [ ] No runaway: release upright robot → wheels settle to zero torque within ~1 s.
