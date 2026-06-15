#pragma once

void stateMachine_init();
void stateMachine_update();

// Called by the command handler to request a mode transition.
void stateMachine_request_manual();
void stateMachine_exit_manual();

// Request entry into STATE_CALIBRATION. Only takes effect from STANDBY.
void stateMachine_request_calibration();

// Called by the command handler to request a reset out of ESTOP back to
// STARTUP, clearing the fault code and re-running the startup checks.
void stateMachine_request_reset();

// Called by the command handler to immediately force STATE_ESTOP with
// fault_code = FAULT_HUMAN_ESTOP, from any non-ESTOP state.
void stateMachine_request_estop();
