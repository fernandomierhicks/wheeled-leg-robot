// imu.cpp — BNO086 SPI driver using patched Adafruit library (3 MHz SPI,
// tight-spin INT poll) for ~400 Hz Game Rotation Vector + Gyro reports.
//
// Pin wiring: CS=D10, INT=D2, RST=D3, SPI bus on D11-D13, PS0+PS1 bridged.

#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_BNO08x.h>
#include "config.h"
#include "imu.h"

// ── Module state ────────────────────────────────────────────────────────────
static Adafruit_BNO08x sensor(PIN_IMU_RST);
static sh2_SensorValue_t sensorValue;

// INT pin check via digitalRead — safe on all register layouts.
// The ~3 µs overhead is negligible vs. SPI transaction time.
static inline bool imu_int_asserted() {
    return digitalRead(PIN_IMU_INT) == LOW;  // INT is active-low
}

// ── Quaternion → pitch/roll ─────────────────────────────────────────────────
// BNO086 Game Rotation Vector outputs (i, j, k, real) in NED-ish frame.
// We convert to our body-frame pitch (rotation about Y, positive = forward lean)
// and roll (rotation about X, positive = lean right).

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static void quat_to_euler(float qi, float qj, float qk, float qr,
                          float *pitch, float *roll, float *yaw) {
    // Pitch (about Y axis) — positive = leaning forward (+X down)
    float sinp = 2.0f * (qr * qj - qi * qk);
    float cosp = 1.0f - 2.0f * (qi * qi + qj * qj);
    *pitch = atan2f(sinp, cosp);

    // Roll (about X axis) — positive = leaning right
    float sinr = 2.0f * (qr * qi + qj * qk);
    *roll = asinf(clampf(sinr, -1.0f, 1.0f));

    // Yaw (about Z axis) — mag-fused heading
    float siny = 2.0f * (qr * qk + qi * qj);
    float cosy = 1.0f - 2.0f * (qj * qj + qk * qk);
    *yaw = atan2f(siny, cosy);
}

// ── Enable sensor reports ───────────────────────────────────────────────────
static bool enable_reports() {
    uint32_t interval_us = 1000000UL / IMU_RATE_HZ;  // 2500 µs for 400 Hz

    if (!sensor.enableReport(SH2_ROTATION_VECTOR, interval_us)) {
        Serial.println("[IMU] Failed to enable Rotation Vector");
        return false;
    }
    if (!sensor.enableReport(SH2_GYROSCOPE_CALIBRATED, interval_us)) {
        Serial.println("[IMU] Failed to enable Gyro");
        return false;
    }

    Serial.print("[IMU] Reports enabled at ");
    Serial.print(IMU_RATE_HZ);
    Serial.println(" Hz (RV + Gyro)");
    return true;
}

// ── Public API ──────────────────────────────────────────────────────────────

bool imu_init() {
    // Retry up to 3 times (matches characterization code)
    for (int attempt = 1; attempt <= 3; attempt++) {
        if (sensor.begin_SPI(PIN_IMU_CS, PIN_IMU_INT)) {
            Serial.print("[IMU] BNO086 init OK (Adafruit lib, attempt ");
            Serial.print(attempt);
            Serial.println(")");

            // Consume post-init reset and let SH2 handshake settle
            delay(100);
            if (sensor.wasReset()) {
                Serial.println("[IMU] Post-init reset consumed");
                sensor.enableReport(SH2_ROTATION_VECTOR, 100000);  // dummy to ack reset
            }
            for (int i = 0; i < 10; i++) {
                sensor.getSensorEvent(&sensorValue);
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

    // Drain queued events (GRV + Gyro = 2 events per cycle, cap at 4).
    for (int i = 0; i < 4; i++) {
        if (!imu_int_asserted()) break;       // no more data waiting
        if (!sensor.getSensorEvent(&sensorValue)) break;   // SPI read failed

        uint8_t report = sensorValue.sensorId;

        if (report == SH2_ROTATION_VECTOR) {
            float qi = sensorValue.un.rotationVector.i;
            float qj = sensorValue.un.rotationVector.j;
            float qk = sensorValue.un.rotationVector.k;
            float qr = sensorValue.un.rotationVector.real;

            quat_to_euler(qi, qj, qk, qr,
                          &state->pitch, &state->roll, &state->yaw);

            state->imu_ok = true;
            state->imu_last_ms = millis();
        }
        else if (report == SH2_GYROSCOPE_CALIBRATED) {
            state->pitch_rate = sensorValue.un.gyroscope.y;
            state->roll_rate  = sensorValue.un.gyroscope.x;
        }
    }
}
