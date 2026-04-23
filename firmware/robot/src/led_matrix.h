#pragma once
// led_matrix.h — 12×8 LED matrix icon display for UNO R4 WiFi.
//
// Shows a single full-screen icon based on robot state (priority order):
//   FAULT     → blinking X
//   Overrun   → flash "!" for ~1 s
//   IMU lost  → "?"
//   IDLE      → smiley face :)
//   BALANCE   → animated heartbeat sweep
//   DRIVE     → right-pointing arrow →
//
// WiFi overlay (bottom-right 3×3): wave arcs when connected, blink when connecting.
// Host overlay (bottom-left dot): solid when host PC linked.

#include "robot_state.h"

void led_matrix_init();
void led_matrix_update(const RobotState *state);
