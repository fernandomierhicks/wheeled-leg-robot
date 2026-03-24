#pragma once
#include "bno086_common.h"

// Test 3 — Static GRV Drift (5 minutes)
// Streams quaternion + accuracy at ~333Hz for 5 minutes.
// IMU must be placed flat and stationary for the entire duration.
// Analysis trims the first 60s (BNO086 calibration convergence),
// then fits linear regression on pitch/roll vs time to get drift rate.
//
// Uses the same INT-gating + getSensorEventID() pattern as Test 5.

#define DRIFT_DURATION_MS    300000  // 5 minutes
#define DRIFT_REPORT_INTERVAL    3  // 3ms -> ~333Hz

// Report ID for filtering
#define DRIFT_RID_GRV  0x08  // SENSOR_REPORTID_GAME_ROTATION_VECTOR

void run_test3_static_drift() {
    char line[96];

    Serial.println("[TEST3] Static GRV Drift (5 min)");
    Serial.println("[TEST3] Place IMU flat and stationary. Starting in 3s...");
    delay(3000);

    imu.enableGameRotationVector(DRIFT_REPORT_INTERVAL);
    delay(500);

    // Discard 3s warmup (let SH2 reports stabilize)
    Serial.println("[TEST3] Discarding 3s warmup...");
    uint32_t warmup_end = millis() + 3000;
    while (millis() < warmup_end) {
        if (int_asserted()) imu.getSensorEvent();
    }

    Serial.println("[TEST3] ---CSV_START---");
    Serial.println("idx,timestamp_us,qi,qj,qk,qr,accuracy");

    uint32_t start = millis();
    uint32_t t0_us = micros();
    uint32_t idx = 0;

    while (millis() - start < DRIFT_DURATION_MS) {
        if (!int_asserted()) continue;

        if (imu.getSensorEvent()) {
            if (imu.getSensorEventID() != DRIFT_RID_GRV) continue;
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

    Serial.println("[TEST3] ---CSV_END---");
    Serial.print("[TEST3] Samples: "); Serial.println(idx);
    Serial.println("[TEST3] Done.");
}
