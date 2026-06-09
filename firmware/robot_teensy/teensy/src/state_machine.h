#pragma once

void stateMachine_init();
void stateMachine_update();

// Called by the command handler to request a mode transition.
void stateMachine_request_manual();
void stateMachine_exit_manual();
