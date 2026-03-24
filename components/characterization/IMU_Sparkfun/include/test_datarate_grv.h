#pragma once
#include "bno086_common.h"

// ---------- Fast mode: timestamps only, buffered in RAM ----------
// Removes serial as a bottleneck so we measure the BNO086's true output rate.
// RA4M1 has 32KB SRAM; 5000 × 4 bytes = 20KB — fits with margin.
// Share one buffer between tests (only one runs at a time).
// 2500 × 4 = 10KB — fits alongside WiFi/SPI stacks in 32KB SRAM.
static const uint16_t MAX_FAST_SAMPLES = 2500;
static uint32_t ts_buf[MAX_FAST_SAMPLES];

// ---------- INT-pin fast-check via direct port register ----------
// digitalRead() costs ~3µs on RA4M1 due to pin table lookup.
// Direct register read is ~0.1µs.  D2 = P104 = Port 1, Bit 4.
// RA4M1 PORT1 base = 0x40040020, PIDR offset = 0x06 (pin input data register)
#define PORT1_PIDR  (*(volatile const uint16_t *)0x40040026)
static inline bool int_asserted() {
    return (PORT1_PIDR & (1U << 4)) == 0;  // H_INTN active-low
}

void run_test1_grv_datarate() {
    Serial.println("[TEST1] GRV Max Datarate (fast mode — timestamps buffered in RAM)");

    imu.enableGameRotationVector(1);  // 1ms = request max rate from BNO086
    delay(200);  // let BNO086 start generating reports

    // ---- Phase 1: collect timestamps as fast as possible ----
    Serial.println("[TEST1] Collecting for 10s...");
    uint16_t idx = 0;
    uint32_t start = micros();
    uint32_t end_time = millis() + 10000;

    while (millis() < end_time) {
        // Skip SPI poll if INT pin is not asserted (no data ready)
        if (!int_asserted()) continue;

        if (imu.getSensorEvent()) {
            if (idx < MAX_FAST_SAMPLES) {
                ts_buf[idx] = micros() - start;
            }
            idx++;
        }
    }

    uint16_t captured = (idx < MAX_FAST_SAMPLES) ? idx : MAX_FAST_SAMPLES;

    // ---- Phase 2: dump buffered timestamps over serial ----
    Serial.println("[TEST1] ---CSV_START---");
    Serial.println("idx,timestamp_us");
    char line[32];
    for (uint16_t i = 0; i < captured; i++) {
        snprintf(line, sizeof(line), "%u,%lu", i, (unsigned long)ts_buf[i]);
        Serial.println(line);
    }
    Serial.println("[TEST1] ---CSV_END---");

    Serial.print("[TEST1] Total events: "); Serial.println(idx);
    Serial.print("[TEST1] Buffered:     "); Serial.println(captured);
    Serial.print("[TEST1] Effective Hz:  ");
    if (captured > 1) {
        float elapsed_s = (ts_buf[captured - 1] - ts_buf[0]) / 1e6f;
        Serial.println((captured - 1) / elapsed_s, 1);
    } else {
        Serial.println("N/A");
    }
    Serial.println("[TEST1] Done.");
}

// ---------- Full mode: quaternion data printed live ----------
// Rate will be limited by serial throughput, but captures actual quat values.
void run_test1_grv_datarate_full() {
    Serial.println("[TEST1F] GRV Datarate — full quaternion output (serial-limited)");

    imu.enableGameRotationVector(1);
    delay(200);

    Serial.println("[TEST1F] ---CSV_START---");
    Serial.println("idx,timestamp_us,qi,qj,qk,qr");

    uint32_t start = millis();
    uint32_t idx = 0;
    char line[80];

    while (millis() - start < 10000) {
        if (!int_asserted()) continue;

        if (imu.getSensorEvent()) {
            // Single snprintf + println instead of 10+ Serial.print calls
            int n = snprintf(line, sizeof(line), "%lu,%lu,", (unsigned long)idx, (unsigned long)micros());
            // dtostrf for floats (snprintf %f not available on all Arduino cores)
            dtostrf(imu.getGameQuatI(), 1, 4, line + n); n = strlen(line);
            line[n++] = ',';
            dtostrf(imu.getGameQuatJ(), 1, 4, line + n); n = strlen(line);
            line[n++] = ',';
            dtostrf(imu.getGameQuatK(), 1, 4, line + n); n = strlen(line);
            line[n++] = ',';
            dtostrf(imu.getGameQuatReal(), 1, 4, line + n);
            Serial.println(line);
            idx++;
        }
    }

    Serial.println("[TEST1F] ---CSV_END---");
    Serial.print("[TEST1F] Samples: "); Serial.println(idx);
    Serial.println("[TEST1F] Done.");
}
