#pragma once

void stateMachine_init();
void stateMachine_update();

// Called by the command handler to request a mode transition.
void stateMachine_request_manual();
void stateMachine_exit_manual();

// Request entry into STATE_RUNNING (armed). Only takes effect from STANDBY and
// only if PARAM_CALIB_DONE == 1. Call when CH10 goes above the arm threshold.
void stateMachine_request_running();

// Request exit from STATE_RUNNING back to STANDBY. Call when CH10 drops.
void stateMachine_disarm_running();

// Request entry into STATE_CALIBRATION. Only takes effect from STANDBY.
void stateMachine_request_calibration();

// Request entry into STATE_JUMPING (3 s jump sequence). Only from STATE_RUNNING.
// Plays a fanfare, then auto-returns to STATE_RUNNING. Called when CH6 goes 1000→2000.
void stateMachine_request_jump();

// Called by the command handler to request a reset out of ESTOP back to
// STARTUP, clearing the fault code and re-running the startup checks.
void stateMachine_request_reset();

// Called by the command handler to immediately force STATE_ESTOP with
// fault_code = FAULT_HUMAN_ESTOP, from any non-ESTOP state.
void stateMachine_request_estop();

// Skip STARTUP re-init for SOFT severity faults; transitions ESTOP → STANDBY
// directly. No-op (with a warning log) if the current fault is not SOFT.
void stateMachine_request_soft_clear();

// Trigger a ~1 s CMD_REJECT transient (buzzer + red blink) then auto-return to
// the originating state. Called internally by req_running() on denied arm attempt.
void stateMachine_request_cmd_reject();

// Called by the command handler each time a GUI command packet arrives
// (the GUI sends CMD_ID_PING at 10 Hz as a heartbeat). Feeds the MANUAL-mode
// watchdog: if no call for >MANUAL_GUI_TIMEOUT_MS (state_machine.cpp,
// currently 500 ms) while in MANUAL, the state machine auto-exits to STANDBY.
void stateMachine_ping_gui_watchdog();
