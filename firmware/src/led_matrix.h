#pragma once
// led_matrix.h — 12×8 LED matrix status display for UNO R4 WiFi.
//
// Layout (each status = 2-row horizontal bar, full 12-col width):
//   Rows 0-1: WiFi      solid=connected, blink=connecting, off=disconnected
//   Rows 2-3: Host      solid=host connected
//   Rows 4-5: Heartbeat toggles at ~3 Hz (proves main loop alive)
//   Rows 6-7: Fault     solid=fault active

#include "robot_state.h"

void led_matrix_init();
void led_matrix_update(const RobotState *state);
