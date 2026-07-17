#pragma once
// IMU.h — BNO086 driver (SPI, Game Rotation Vector + Gyro, ~400 Hz)
//
// Call imu_init() once at startup, then imu_update() every control tick.
// Motion must not be enabled until imu_state() == ImuState::NOMINAL.

#include <stdint.h>

enum class ImuState : uint8_t {
    NOT_READY    = 0,  // imu_init() not yet called
    INITIALIZING = 1,  // begin_SPI() in progress / waiting for retry
    NOMINAL      = 2,  // data flowing, packet loss < 10%
    DEGRADED     = 3,  // data flowing but >= 10% packet loss
    ERROR        = 4,  // init failed or sensor silent > 100 ms; retrying every 1 s
};

// Call once; sets state to INITIALIZING. Actual init runs inside imu_update().
void imu_init();

// Non-blocking poll — call every control tick. Drives state machine + data.
void imu_update();

ImuState imu_state();
// Game Rotation Vector — no magnetometer, immune to motor field disturbance
float    imu_pitch();           // rad  — positive = lean forward (+X down)
float    imu_roll();            // rad  — positive = lean right
float    imu_yaw();             // rad
// Rotation Vector — magnetometer fused, absolute heading reference
float    imu_pitch_mag();       // rad
float    imu_roll_mag();        // rad
float    imu_yaw_mag();         // rad
// Gyro (shared between both fusion paths)
float    imu_pitch_rate();      // rad/s (gyro Y)
float    imu_roll_rate();       // rad/s (gyro X)
float    imu_yaw_rate();        // rad/s (gyro Z)
float    imu_packet_loss();     // 0.0–1.0, rolling 1-second window
uint32_t imu_last_update_ms();  // millis() of last good GRV packet
