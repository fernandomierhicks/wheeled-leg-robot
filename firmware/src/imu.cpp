// imu.cpp — BNO086 SPI driver: Game Rotation Vector + Gyro reports.
//
// Ported from components/characterization/IMU/include/bno086_common.h
// Pin wiring: CS=D10, INT=D2, RST=D3, SPI bus on D11-D13, PS0+PS1 bridged.

#include <Arduino.h>
#include <SPI.h>
#include <SparkFun_BNO08x_Arduino_Library.h>
#include "config.h"
#include "imu.h"

// ── Module state ────────────────────────────────────────────────────────────
static BNO08x sensor;

// ── Quaternion → pitch/roll ─────────────────────────────────────────────────
// BNO086 Game Rotation Vector outputs (qi, qj, qk, qr) in NED-ish frame.
// We convert to our body-frame pitch (rotation about Y, positive = forward lean)
// and roll (rotation about X, positive = lean right).
//
// Using small-angle-safe atan2 formulas (valid full range):
//   pitch = atan2(2(qr*qj - qi*qk), 1 - 2(qi*qi + qj*qj))
//   roll  = asin(clamp(2(qr*qi + qj*qk), -1, 1))

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static void quat_to_pitch_roll(float qi, float qj, float qk, float qr,
                               float *pitch, float *roll) {
    // Pitch (about Y axis) — positive = leaning forward (+X down)
    float sinp = 2.0f * (qr * qj - qi * qk);
    float cosp = 1.0f - 2.0f * (qi * qi + qj * qj);
    *pitch = atan2f(sinp, cosp);

    // Roll (about X axis) — positive = leaning right
    float sinr = 2.0f * (qr * qi + qj * qk);
    *roll = asinf(clampf(sinr, -1.0f, 1.0f));
}

// ── Enable sensor reports ───────────────────────────────────────────────────
static bool enable_reports() {
    uint32_t interval_us = 1000000UL / IMU_RATE_HZ;  // 2000 µs for 500 Hz

    if (!sensor.enableGameRotationVector(interval_us)) {
        Serial.println("[IMU] Failed to enable Game Rotation Vector");
        return false;
    }
    if (!sensor.enableGyro(interval_us)) {
        Serial.println("[IMU] Failed to enable Gyro");
        return false;
    }

    Serial.print("[IMU] Reports enabled at ");
    Serial.print(IMU_RATE_HZ);
    Serial.println(" Hz (GRV + Gyro)");
    return true;
}

// ── Public API ──────────────────────────────────────────────────────────────

bool imu_init() {
    // Retry up to 3 times (matches characterization code)
    for (int attempt = 1; attempt <= 3; attempt++) {
        if (sensor.beginSPI(PIN_IMU_CS, PIN_IMU_INT, PIN_IMU_RST,
                            3000000, SPI)) {
            Serial.print("[IMU] BNO086 init OK (attempt ");
            Serial.print(attempt);
            Serial.println(")");

            // Consume post-init reset and let SH2 handshake settle
            delay(100);
            if (sensor.wasReset()) {
                Serial.println("[IMU] Post-init reset consumed");
            }
            for (int i = 0; i < 10; i++) {
                sensor.getSensorEvent();
                delay(10);
            }

            return enable_reports();
        }

        Serial.print("[IMU] BNO086 not detected (attempt ");
        Serial.print(attempt);
        Serial.println("/3)");
        Serial.println("[IMU]   CS=D10, INT=D2, RST=D3, PS0+PS1 bridged?");
        delay(3000);
    }

    Serial.println("[IMU] FATAL: init failed after 3 attempts");
    return false;
}

void imu_poll(RobotState *state) {
    // Handle unexpected resets (re-enable reports)
    if (sensor.wasReset()) {
        Serial.println("[IMU] Sensor reset detected — re-enabling reports");
        enable_reports();
    }

    // Non-blocking: getSensorEvent returns true when new data is available
    if (!sensor.getSensorEvent()) {
        return;  // no new data this tick — keep previous values
    }

    uint8_t report = sensor.getSensorEventID();

    if (report == SENSOR_REPORTID_GAME_ROTATION_VECTOR) {
        float qi = sensor.getGameQuatI();
        float qj = sensor.getGameQuatJ();
        float qk = sensor.getGameQuatK();
        float qr = sensor.getGameQuatReal();

        quat_to_pitch_roll(qi, qj, qk, qr,
                           &state->pitch, &state->roll);

        state->imu_ok = true;
        state->imu_last_ms = millis();
    }
    else if (report == SENSOR_REPORTID_GYROSCOPE_CALIBRATED) {
        // BNO086 gyro is in rad/s, axes: x=roll_rate, y=pitch_rate, z=yaw
        state->pitch_rate = sensor.getGyroY();
        state->roll_rate  = sensor.getGyroX();
    }
}
