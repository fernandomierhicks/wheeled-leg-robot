#pragma once

void stateMachine_init();
void stateMachine_update();

// Called by the command handler to request a mode transition.
bool stateMachine_request_manual();
bool stateMachine_exit_manual();

// Request entry into STATE_RUNNING (armed). Only takes effect from STANDBY and
// requires valid in-RAM hip limits unless the calibration bypass is enabled.
bool stateMachine_request_running();

// Request exit from STATE_RUNNING back to STANDBY. Call when CH10 drops.
bool stateMachine_disarm_running();

// Request entry into STATE_CALIBRATION. Only takes effect from STANDBY.
bool stateMachine_request_calibration();

// Gracefully cancel an active radio-triggered calibration through DISARMING.
bool stateMachine_disarm_calibration();

// Request entry into STATE_JUMPING (3 s jump sequence). Only from STATE_RUNNING.
// Plays a fanfare, then auto-returns to STATE_RUNNING. Called when CH6 goes 1000→2000.
bool stateMachine_request_jump();

// Called by the command handler to request a reset out of ESTOP back to
// STARTUP, clearing the fault code and re-running the startup checks.
bool stateMachine_request_reset();

// Called by the command handler to immediately force STATE_ESTOP with
// fault_code = FAULT_HUMAN_ESTOP, from any non-ESTOP state.
bool stateMachine_request_estop();

// Skip STARTUP re-init for SOFT severity faults; transitions ESTOP → STANDBY
// directly. No-op (with a warning log) if the current fault is not SOFT.
bool stateMachine_request_soft_clear();

// Trigger a ~1 s CMD_REJECT transient (buzzer + red blink) then auto-return to
// the originating state. Called internally by req_running() on denied arm attempt.
void stateMachine_request_cmd_reject();

// Compatibility hook for the hip poller. ESTOP is now immediate and this is
// always false; normal torque ramp-down is represented by STATE_DISARMING.
bool stateMachine_is_estop_hip_ramping();

// Called by the command handler each time a GUI command packet arrives
// (the GUI sends CMD_ID_PING at 10 Hz as a heartbeat). Feeds the MANUAL-mode
// watchdog: if no call for >MANUAL_GUI_TIMEOUT_MS (state_machine.cpp,
// currently 500 ms) while in MANUAL, the state machine auto-exits to STANDBY.
void stateMachine_ping_gui_watchdog();

// Milliseconds since the last GUI command packet (whatever the ping above
// last stamped). Reused by main.cpp radio_update()'s GUI-motion-control
// watchdog (tuning.md §1d) at a tighter timeout than MANUAL_GUI_TIMEOUT_MS.
uint32_t stateMachine_ms_since_gui_packet();
