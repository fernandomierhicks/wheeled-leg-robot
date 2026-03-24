#pragma once
// imu.h — BNO086 IMU interface (SPI via patched Adafruit lib, ~400 Hz GRV + Gyro).

#include "robot_state.h"

// Initialise BNO086 over SPI.  Blocks up to ~10 s on failure (3 retries).
// Returns true if sensor is responding and reports are enabled.
bool imu_init();

// Non-blocking poll.  Call every tick.
// Updates state->pitch, pitch_rate, roll, roll_rate, imu_ok, imu_last_ms.
void imu_poll(RobotState *state);
