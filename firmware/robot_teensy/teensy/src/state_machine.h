#pragma once

void stateMachine_init();
void stateMachine_update();

// Called by the command handler to request a mode transition.
void stateMachine_request_manual();
void stateMachine_exit_manual();

// Called by the command handler to request a reset out of ESTOP back to
// STARTUP, clearing the fault code and re-running the startup checks.
void stateMachine_request_reset();
