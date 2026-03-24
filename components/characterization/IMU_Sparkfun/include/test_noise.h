#pragma once
#include "bno086_common.h"

// Test 5 — Noise Density
// Phase A (30s): GRV quaternion → fused pitch/roll noise at rest
// Phase B (60s): Raw gyro ONLY → gyro noise std-dev, Allan variance, PSD
// Phase C (30s): Raw accel ONLY → accel noise std-dev, Allan variance, PSD
//
// IMPORTANT: The SparkFun library's enableReport() waits for INT assertion
// before sending SHTP config. If no reports are active, INT never fires and
// enableReport() silently fails. Therefore GRV stays enabled at low rate
// throughout all phases as a "keepalive". We use getSensorEventID() to filter
// events by report type so only the desired data is recorded.
//
// Raw gyro and accel are collected in separate phases to avoid the stale-cache
// problem (getSensorEvent fires for ANY report; getters return cached values
// for the report that didn't just arrive).

#define NOISE_GRV_DURATION_MS    30000   // 30s for fused noise
#define NOISE_GYRO_DURATION_MS   60000   // 60s for gyro noise + Allan variance
#define NOISE_ACCEL_DURATION_MS  30000   // 30s for accel noise
#define NOISE_REPORT_INTERVAL    3       // 3ms → ~333Hz
#define NOISE_GRV_KEEPALIVE_MS   100     // 10Hz keepalive for INT pin

// Report IDs for filtering
#define RID_GRV   0x08   // SENSOR_REPORTID_GAME_ROTATION_VECTOR
#define RID_GYRO  0x15   // SENSOR_REPORTID_RAW_GYROSCOPE
#define RID_ACCEL 0x14   // SENSOR_REPORTID_RAW_ACCELEROMETER

void run_test5_noise() {
    char line[96];

    // ================================================================
    // Phase A: GRV noise (30s)
    // ================================================================
    Serial.println("[TEST5] Phase A: GRV fused noise (30s)");
    Serial.println("[TEST5] Place IMU flat and stationary.");
    delay(2000);  // give user time to stop touching it

    imu.enableGameRotationVector(NOISE_REPORT_INTERVAL);
    delay(500);

    // Discard first 3s of GRV data (let fusion converge)
    Serial.println("[TEST5] Discarding 3s warmup...");
    uint32_t warmup_end = millis() + 3000;
    while (millis() < warmup_end) {
        if (int_asserted()) imu.getSensorEvent();
    }

    Serial.println("[TEST5A] ---CSV_START---");
    Serial.println("idx,timestamp_us,qi,qj,qk,qr,accuracy");

    uint32_t start = millis();
    uint32_t t0_us = micros();
    uint32_t idx = 0;

    while (millis() - start < NOISE_GRV_DURATION_MS) {
        if (!int_asserted()) continue;

        if (imu.getSensorEvent()) {
            if (imu.getSensorEventID() != RID_GRV) continue;
            uint32_t t = micros() - t0_us;
            int n = snprintf(line, sizeof(line), "%lu,%lu,",
                             (unsigned long)idx, (unsigned long)t);
            dtostrf(imu.getGameQuatI(), 1, 6, line + n); n = strlen(line);
            line[n++] = ',';
            dtostrf(imu.getGameQuatJ(), 1, 6, line + n); n = strlen(line);
            line[n++] = ',';
            dtostrf(imu.getGameQuatK(), 1, 6, line + n); n = strlen(line);
            line[n++] = ',';
            dtostrf(imu.getGameQuatReal(), 1, 6, line + n); n = strlen(line);
            line[n++] = ',';
            n += snprintf(line + n, sizeof(line) - n, "%u", imu.getQuatAccuracy());
            Serial.println(line);
            idx++;
        }
    }

    Serial.println("[TEST5A] ---CSV_END---");
    Serial.print("[TEST5A] Samples: "); Serial.println(idx);

    // ================================================================
    // Phase B: Raw gyro ONLY (60s)
    // ================================================================
    // Slow down GRV to 10Hz keepalive (just enough to keep INT firing
    // so enableReport works), then enable raw gyro at full speed.
    Serial.println("[TEST5] Switching to gyro-only...");
    imu.enableGameRotationVector(NOISE_GRV_KEEPALIVE_MS);
    delay(100);
    imu.enableRawGyro(NOISE_REPORT_INTERVAL);
    delay(500);

    Serial.println("[TEST5] Phase B: Raw gyro noise (60s, gyro-only)");

    // Discard 1s warmup
    warmup_end = millis() + 1000;
    while (millis() < warmup_end) {
        if (int_asserted()) imu.getSensorEvent();
    }

    Serial.println("[TEST5B] ---CSV_START---");
    Serial.println("idx,timestamp_us,gx,gy,gz");

    start = millis();
    t0_us = micros();
    idx = 0;

    while (millis() - start < NOISE_GYRO_DURATION_MS) {
        if (!int_asserted()) continue;

        if (imu.getSensorEvent()) {
            if (imu.getSensorEventID() != RID_GYRO) continue;
            uint32_t t = micros() - t0_us;
            snprintf(line, sizeof(line), "%lu,%lu,%d,%d,%d",
                     (unsigned long)idx, (unsigned long)t,
                     imu.getRawGyroX(), imu.getRawGyroY(), imu.getRawGyroZ());
            Serial.println(line);
            idx++;
        }
    }

    Serial.println("[TEST5B] ---CSV_END---");
    Serial.print("[TEST5B] Samples: "); Serial.println(idx);

    // ================================================================
    // Phase C: Raw accel ONLY (30s)
    // ================================================================
    // Disable gyro (interval=0 should work since GRV keepalive keeps INT alive)
    Serial.println("[TEST5] Switching to accel-only...");
    imu.enableRawGyro(0);
    delay(100);
    imu.enableRawAccelerometer(NOISE_REPORT_INTERVAL);
    delay(500);

    Serial.println("[TEST5] Phase C: Raw accel noise (30s, accel-only)");

    warmup_end = millis() + 1000;
    while (millis() < warmup_end) {
        if (int_asserted()) imu.getSensorEvent();
    }

    Serial.println("[TEST5C] ---CSV_START---");
    Serial.println("idx,timestamp_us,ax,ay,az");

    start = millis();
    t0_us = micros();
    idx = 0;

    while (millis() - start < NOISE_ACCEL_DURATION_MS) {
        if (!int_asserted()) continue;

        if (imu.getSensorEvent()) {
            if (imu.getSensorEventID() != RID_ACCEL) continue;
            uint32_t t = micros() - t0_us;
            snprintf(line, sizeof(line), "%lu,%lu,%d,%d,%d",
                     (unsigned long)idx, (unsigned long)t,
                     imu.getRawAccelX(), imu.getRawAccelY(), imu.getRawAccelZ());
            Serial.println(line);
            idx++;
        }
    }

    Serial.println("[TEST5C] ---CSV_END---");
    Serial.print("[TEST5C] Samples: "); Serial.println(idx);

    // Clean up: disable raw reports, restore GRV at normal rate
    imu.enableRawAccelerometer(0);
    imu.enableGameRotationVector(NOISE_REPORT_INTERVAL);

    Serial.println("[TEST5] Done. Total duration ~125s.");
}
