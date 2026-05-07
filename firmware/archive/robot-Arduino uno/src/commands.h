#pragma once
// commands.h — UDP command receiver (10 Hz).
//
// Command types:
//   DRIVE  (1): v_cmd, omega_cmd, hip_q_target  — 13 bytes
//   MODE   (2): mode                             — 2 bytes
//   GAIN   (3): gain_id, value                   — 6 bytes
//   PING   (4): (empty)                          — 1 byte

#include "robot_state.h"

// Initialise the command UDP socket.  Call once in setup() after wifi_init().
void commands_init();

// Poll for incoming command packets and apply to state.
// Call every COMMAND_DIV ticks from the main loop.
void commands_receive(RobotState& state);
