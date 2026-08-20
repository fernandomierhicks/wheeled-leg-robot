#pragma once

constexpr float MOTOR_TRQ_MAX = 7.0f;  // [N·m] hard per-wheel clamp

void controlLoop_init();   // called once in setup()
void controlLoop_run();    // called every tick from on_running() — 500 Hz
void controlLoop_reset();  // called on RUNNING entry — clears integrators/rate-limit state so each arm starts clean
void controlLoop_disarm_plant_id();  // clear the non-persistent identification chirp arm
void controlLoop_reset_hip_ramp();  // called on RUNNING entry, except when returning from JUMPING
// STANDING_UP calls this only after its measured hip-settle gate. It guarantees
// the LQR catch and RUNNING handoff retain full hip stiffness instead of
// re-entering the ordinary arm-in gain ramp.
void controlLoop_complete_hip_ramp();

// Ephemeral jump controls. The velocity offset is added to the pilot's live
// v_cmd_ms (never replaces or persists it). Jump handoff keeps the ordinary
// RUNNING stack but enables its scoped LQR-gain/authority overrides. Both are
// cleared by controlLoop_reset().
void controlLoop_set_velocity_command_offset(float offset_ms);
void controlLoop_set_jump_handoff_active(bool active);

// The wheel speed governor's limit currently in force [turns/s]: wm_vel_limit
// normally, the STANDING_UP-scoped standup_vel_limit, or the active jump
// handoff's jmp_handoff_vel_lim. Zero overrides inherit the normal value.
// Shared by the soft governor, runaway watchdog's 2x trip, and telemetry's
// velocity-limited health bits so all three report the same number.
float controlLoop_wheel_vel_limit();

// Hip height override: while set, controlLoop_run() holds the legs at this
// normalized height (0 = retracted) instead of tracking radio_hip_cmd (CH3).
// STANDING_UP uses it to pin the legs at the calibrated retracted endpoint
// while the RUNNING controllers balance, so the pilot cannot move them mid-catch.
// The rate-limiter shadow is kept in sync with the override, so clearing it on
// the RUNNING handoff resumes the normal CH3 slew from where the legs are —
// no position step. Cleared by controlLoop_reset() and on RUNNING entry;
// nothing outside controlLoop_run() reads it.
void controlLoop_set_hip_override(float t);
void controlLoop_clear_hip_override();

// Hip torque ramp-DOWN on disarm (RUNNING -> STANDBY), same rate/param as the
// arm-in ramp. controlLoop_reset_hip_disarm_ramp() is called once, on the
// transition; controlLoop_run_hip_disarm_ramp() is called every tick while
// ramping and returns true while still in progress, false once kp/tff have
// reached zero (caller should then release the setpoint).
void controlLoop_reset_hip_disarm_ramp();
bool controlLoop_run_hip_disarm_ramp();

// Gentle 1 s ESTOP ramp: holds the last commanded hip position and tapers hip
// kp/kd/tff to zero, instead of an instant cutoff, whenever ESTOP is entered
// while a hip was actively being commanded (RUNNING, JUMPING, CALIBRATION,
// MANUAL). Wheels are not ramped — they cut power immediately on ESTOP.
// Reset snapshots whatever was active at that instant; is a no-op for any
// axis that had nothing active. Call controlLoop_estop_ramp_has_hip() right
// after reset() to know whether state_machine.cpp should defer the hip MIT
// exit until the ramp finishes, or run it immediately.
void controlLoop_reset_estop_ramp();
bool controlLoop_run_estop_ramp();  // true while still ramping
bool controlLoop_estop_ramp_has_hip();
