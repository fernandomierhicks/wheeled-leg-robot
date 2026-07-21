#pragma once

constexpr float MOTOR_TRQ_MAX = 7.0f;  // [N·m] hard per-wheel clamp

void controlLoop_init();   // called once in setup()
void controlLoop_run();    // called every tick from on_running() — 500 Hz
void controlLoop_reset();  // called on RUNNING entry — clears integrators/rate-limit state so each arm starts clean
void controlLoop_reset_hip_ramp();  // called on RUNNING entry, except when returning from JUMPING

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
