#pragma once
// No-op stand-ins for app-layer symbols (defined in src/, which
// env:test_teensy excludes from the build) that hardware-driver libs call
// into. Bench tests exercise the driver in isolation and don't want the
// logging destinations or state machine that a real symbol would pull in.
//
// Include in any test .cpp whose libs call these.

#include <Arduino.h>
#include "comm_protocol.h"
#include "state_machine.h"

void comm_log(uint8_t, const char*, ...) {}
void stateMachine_request_estop() {}
