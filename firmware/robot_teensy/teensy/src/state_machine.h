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

// Called by the command handler to request a reset out of ESTOP back to
// STARTUP, clearing the fault code and re-running the startup checks.
void stateMachine_request_reset();

// Called by the command handler to immediately force STATE_ESTOP with
// fault_code = FAULT_HUMAN_ESTOP, from any non-ESTOP state.
void stateMachine_request_estop();
