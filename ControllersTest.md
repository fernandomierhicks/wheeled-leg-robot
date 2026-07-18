# Controller Hardware Verification Checklist

All controllers (LQR, Velocity PI, Yaw PI, gain scheduling, feedforward, jump FSM) are implemented in firmware. This is the checklist for verifying them on real hardware — no more implementation work, just bring-up and sign/behavior checks.

---

## Safety Reminders

- **Wheel motors**: free-spinning in air is safe standalone, but risk is runaway spin (no load, any torque = unlimited acceleration). Keep `PARAM_LQR_TORQUE_LIMIT` low (0.1 Nm) until behavior is confirmed correct — 1 Nm was tried and spun the wheels too fast to observe.
- **Hip motors**: never reach into the 4-bar while in `STATE_RUNNING` — pinch risk.
- **Jump FSM**: only test on the floor, secured or in a cradle, from behind a barrier or at arm's length. Never on flatsat.
- Radio disarm (CH10 drop) should always work as an emergency stop — confirm this first, before anything else.

---

## Phase 1 — Motor Direction Sanity Check (flatsat OK)

Before closing any control loop, confirm each motor physically moves the direction the firmware/telemetry thinks it does. Use the per-motor bench-test tabs (Hip Motors, Wheel Motors) in the GUI.

- [x] Hip left: jog to a small positive position — leg should visibly extend (hip angle moves toward α=1, per gain-scheduling convention). Confirm telemetry position sign matches the observed motion direction.
- [x] Hip right: same check — should mirror hip left's extend/retract sense.
- [x] Wheel left: command a small positive open-loop torque/velocity — wheel should spin the direction that would drive the robot forward (+X) if it were on the ground. Confirm telemetry velocity sign matches the observed spin direction.
- [x] Wheel right: same check — should spin the same rotational sense as wheel left (both wheels drive the robot forward together).
- [ ] Any mismatch: fix the motor direction/sign convention (CAN axis config or firmware) before proceeding to Phase 2 — LQR sign checks assume this is already correct.

---

## Phase 2 — LQR

- [x] Flash firmware. Boot. Calibrate. Arm via CH10.
- [x] GUI shows `tau_sym` in telemetry at 50 Hz = 0 when upright.
- [x] `PARAM_LQR_ENABLE=0`. Confirm `tau_sym` still computes and appears in telemetry (no wheel motion).
- [x] `PARAM_SIM_PITCH_RAD=+0.1` (6° forward lean) — `tau_sym` positive.
- [x] `PARAM_SIM_PITCH_RAD=-0.1` — `tau_sym` flips negative.
- [x] `PARAM_SIM_PITCH_RAD=0`, tilt robot physically — real IMU pitch produces correct-sign torque in telemetry.
- [x] `PARAM_LQR_ENABLE=1`, `PARAM_LQR_TORQUE_LIMIT=0.1 Nm`. Tilt robot — wheels spin in correcting direction (free-spinning).
- [x] ESTOP from radio disarm (CH10 drop): normal disarm.
- [x] ESTOP from pitch watchdog (tilt past 50°): fault code = `FAULT_PITCH_WATCHDOG` visible in GUI.
- [x] ESTOP from wheel runaway: fault code = `FAULT_WHEEL_RUNAWAY`.
- [ ] No runaway: release upright robot → wheels settle to zero torque within ~1 s.

---

## Phase 3 — Velocity PI (requires robot upright and balancing — not flatsat)

- [ ] `vel_pi_en=0` — confirm `theta_ref=0` in telemetry.
- [ ] `vel_pi_en=1`, `v_cmd_ms=0` — push robot by hand; confirm wheels resist drift.
- [ ] `v_cmd_ms=0.1` — wheels spin up slowly; `theta_ref` goes slightly positive, `vel_err_integral` accumulates then stabilizes.

---

## Phase 4 — Yaw PI (flatsat OK)

- [ ] `yaw_pi_en=1`, `omega_cmd_rds=0.5` — confirm one wheel spins faster than the other.
- [ ] Verify sign: positive yaw command should drive the right wheel faster (CCW from above per +Y=left, +Z=up right-hand rule). If backwards, flip sign of `omega_cmd_rds` or `yaw_pi_kp`.

---

## Phase 5 — Hip Gain Scheduling (flatsat OK)

- [ ] Calibrate hips.
- [ ] Sweep CH3 from retracted to extended — confirm `gain_sched_alpha` moves 0→1 in telemetry.

---

## Phase 6 — Feedforward (FF1, FF2) (flatsat OK)

- [ ] Set `sim_pitch_rad=0.1`. Note baseline `tau_sym`.
- [ ] `ff2_alpha=0.1` — confirm `ff2_out` non-zero in telemetry, `tau_sym` shifts. Ramp up slowly toward 1.0.
- [ ] `ff1_alpha=0.1` — watch `ff1_out`; verify sign plausible (should oppose hip torque effect). Increase cautiously.

---

## Phase 7 — Jump FSM (NOT flatsat — floor only, secured, behind barrier)

- [ ] Start with CROUCH phase only (slow, controlled).
- [ ] Add EXTEND only after CROUCH is verified.
- [ ] Use low `max_torque` parameter initially.
