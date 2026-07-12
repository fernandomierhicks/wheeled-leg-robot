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

### Phase 3 — Velocity PI ✅ IMPLEMENTED — NOT VERIFIED (robot mode required)

New params: `vel_pi_en`, `vel_pi_kp`=0.2, `vel_pi_ki`=0.1, `vel_pi_kff`=0.1049, `vel_pi_theta_max`=0.698, `vel_pi_rate_lim`=1.745, `vel_pi_int_max`=1.0, `v_cmd_ms`=0.0

**Note:** Flatsat verification is not possible — K_vel positive feedback spins wheels without pendulum dynamics, causing vel PI integrator to wind up. All steps require upright balancing robot.

**Safe test** (robot upright and balancing):
- [ ] 1. `vel_pi_en=0` — confirm `theta_ref=0` in telemetry.
- [ ] 2. `vel_pi_en=1`, `v_cmd_ms=0` — push robot by hand; confirm wheels resist drift.
- [ ] 3. `v_cmd_ms=0.1` — wheels spin up slowly; `theta_ref` goes slightly positive, `vel_err_integral` accumulates then stabilizes.

---

### Phase 4 — Yaw PI ✅ IMPLEMENTED — verification pending

New params: `yaw_pi_en`, `yaw_pi_kp`=0.1, `yaw_pi_ki`=0.2, `yaw_pi_torque_max`=0.5, `yaw_pi_int_max`=0.5, `omega_cmd_rds`=0.0

`imu_yaw_rate()` was already in IMU.h — no changes needed there.

**Safe test** (flatsat):
- [ ] 1. `yaw_pi_en=1`, `omega_cmd_rds=0.5` — confirm one wheel faster than the other.
- [ ] 2. Verify sign: +Y = left, +Z = up, right-hand rule → positive yaw = CCW from above → right wheel faster. Flip sign of `omega_cmd_rds` or `yaw_pi_kp` if backwards.

---

### Phase 5 — Hip gain scheduling ✅ IMPLEMENTED — verification pending

No new params. Gains hardcoded from `lqr.py` self-test (Q_pitch=0.01, Q_pitch_rate=0.1884, Q_vel=0.00508442, R=100.0):

| Position | K_pitch | K_rate | K_vel |
|---|---|---|---|
| Retracted (α=0) | −13.050 | −2.181 | −0.00713 |
| Extended  (α=1) | −7.929  | −1.691 | −0.00713 |

`alpha` derived from calibrated hip position span — coordinate-system agnostic, defaults to 0.5 if calibration not done.

**Safe test** (flatsat):
- [ ] 1. Calibrate hips. Sweep CH3 from retracted to extended — confirm `gain_sched_alpha` moves 0→1 in telemetry.

---

### Phase 6 — Feedforward terms (FF1, FF2) ✅ IMPLEMENTED — verification pending

New params: `ff2_alpha`=0.0, `ff1_alpha`=0.0, `ff1_kt_hip`=1.2732 N·m/A

`l_eff` linearly interpolated from IK-computed values: 0.183 m (retracted) → 0.296 m (extended). FF4 hardcoded 0 until floor driving.

**Safe test** (flatsat):
- [ ] 1. Set `sim_pitch_rad=0.1`. Note baseline `tau_sym`.
- [ ] 2. `ff2_alpha=0.1` — confirm `ff2_out` non-zero in telemetry, `tau_sym` shifts. Ramp up slowly toward 1.0.
- [ ] 3. `ff1_alpha=0.1` — watch `ff1_out`; verify sign plausible (should oppose hip torque effect). Increase cautiously.

---

### Phase 7 — Jump FSM (NOT on flatsat)

Only implement and test once:
- Robot is on the floor, secured or in a cradle
- You are behind a barrier or at arm's length
- Start with CROUCH phase only (slow, controlled). Add EXTEND only after CROUCH is verified.
- Use low `max_torque` parameter initially.

---

## Critical Files (all modified — implementation complete)

| File | Status |
|------|--------|
| `firmware/robot_teensy/teensy/src/control_loop.cpp` | ✅ All phases 1–6 implemented |
| `firmware/robot_teensy/shared/comm_protocol.h` | ✅ Fault codes, telemetry fields all present (TELEM_VERSION 6) |
| `firmware/robot_teensy/teensy/lib/ParamRegistry/param_ids.h` | ✅ All params added (0x0400–0x0414) |
| `firmware/robot_teensy/teensy/lib/ParamRegistry/param_registry.cpp` | ✅ All params registered with defaults |
| `firmware/robot_teensy/teensy/src/main.cpp` | ✅ `build_health_flags()` updated for VEL_PI_SAT and YAW_PI_SAT |
| `firmware/robot_teensy/teensy/src/state_machine.cpp` | ✅ `on_running()` calls `controlLoop_run()` |
| `firmware/robot_teensy/teensy/lib/IMU/IMU.h` | ✅ `imu_yaw_rate()` already present, no changes needed |

## End-to-End Verification Checklist

**Phase 2 (LQR)** — verified:
- [x] Flash firmware. Boot. Calibrate. Arm via CH10.
- [x] GUI shows `tau_sym` in telemetry at 50 Hz = 0 when upright.
- [x] With `sim_pitch_rad=0.1`, tau is positive and proportional to gain × error.
- [x] With `lqr_enable=1`, `lqr_torque_limit=1 Nm`: tilt robot — wheels spin in correcting direction.
- [x] ESTOP from radio disarm (CH10 drop): normal disarm.
- [x] ESTOP from pitch watchdog (tilt past 50°): fault code = `FAULT_PITCH_WATCHDOG` visible in GUI.
- [x] ESTOP from wheel runaway: fault code = `FAULT_WHEEL_RUNAWAY`.
- [x] No runaway: release upright robot → wheels settle to zero torque within ~1 s.

**Phase 3 (Velocity PI)** — NOT VERIFIED (robot mode required):
- [ ] `vel_pi_en=0` → `theta_ref=0` in telemetry
- [ ] `vel_pi_en=1`, `v_cmd_ms=0` → wheels resist hand-push
- [ ] `v_cmd_ms=0.1` → wheels spin up, `theta_ref` positive, integral stabilizes

**Phase 4 (Yaw PI)** — pending:
- [ ] `yaw_pi_en=1`, `omega_cmd_rds=0.5` → differential wheel speed visible
- [ ] Sign correct (right wheel faster for positive omega); adjust if not

**Phase 5 (Gain scheduling)** — pending:
- [ ] Post-calibration: sweep CH3, `gain_sched_alpha` tracks 0→1 in telemetry

**Phase 6 (Feedforward)** — pending:
- [ ] `ff2_alpha` ramped up from 0 at fixed `sim_pitch_rad`; `ff2_out` visible in telemetry
- [ ] `ff1_alpha` enabled last; sign verified before increasing

**Phase 7 (Jump FSM)** — NOT flatsat. Implement only when robot is on floor, secured.
