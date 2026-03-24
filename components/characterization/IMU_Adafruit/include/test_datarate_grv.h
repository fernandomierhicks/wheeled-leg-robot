#pragma once
#include "bno086_common.h"

// Shared RAM buffers for all speed tests (only one runs at a time).
// 1500 x 4 bytes = 6KB timestamps + 1500 x 4 bytes = 6KB data = 12KB total.
// Fits in 32KB RA4M1 SRAM alongside WiFi/SPI/OTA stacks.
// At ~400 Hz max, 1500 samples covers ~3.75s of data — enough for rate measurement.
static const uint16_t MAX_FAST_SAMPLES = 1500;
static uint32_t ts_buf[MAX_FAST_SAMPLES];

// Union for data values — reused across tests
static union {
    int16_t  i16[MAX_FAST_SAMPLES];   // raw gyro Y (2 bytes each)
    float    f32[MAX_FAST_SAMPLES];    // calibrated gyro Y (4 bytes each)
} data_buf;

// ---------- Test 1: Game Rotation Vector max datarate ----------
// Requests GRV at 1ms (max rate), buffers timestamps in RAM,
// then dumps over serial to eliminate serial as bottleneck.

static void enable_grv_reports() {
    bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 1000);  // 1000us = 1ms = request max
}

void run_test1_grv_datarate() {
    Serial.println("[TEST1] GRV Max Datarate (Adafruit lib — timestamps buffered in RAM)");

    activeReportEnabler = enable_grv_reports;
    enable_grv_reports();
    delay(200);  // let BNO086 start generating reports

    // ---- Phase 1: collect timestamps as fast as possible ----
    Serial.println("[TEST1] Collecting for 10s...");
    uint16_t idx = 0;
    uint32_t start = micros();
    uint32_t end_time = millis() + 10000;

    while (millis() < end_time) {
        if (!int_asserted()) continue;

        if (bno08x.getSensorEvent(&sensorValue)) {
            if (sensorValue.sensorId == SH2_GAME_ROTATION_VECTOR) {
                if (idx < MAX_FAST_SAMPLES) {
                    ts_buf[idx] = micros() - start;
                }
                idx++;
            }
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
    activeReportEnabler = nullptr;
}
