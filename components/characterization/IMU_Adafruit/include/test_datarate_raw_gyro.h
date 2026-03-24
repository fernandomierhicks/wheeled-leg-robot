#pragma once
#include "bno086_common.h"

// Uses shared ts_buf[] and data_buf from test_datarate_grv.h

// ---------- Test 2: Raw Gyroscope max datarate ----------
// Requests SH2_RAW_GYROSCOPE at 1ms (max rate).
// Raw gyro is uncalibrated ADC output — fastest possible gyro path.

static void enable_raw_gyro_reports() {
    bno08x.enableReport(SH2_RAW_GYROSCOPE, 1000);  // 1000us = 1ms
}

void run_test2_raw_gyro_datarate() {
    Serial.println("[TEST2] Raw Gyro-Y Max Datarate (Adafruit lib — timestamps buffered in RAM)");

    activeReportEnabler = enable_raw_gyro_reports;
    enable_raw_gyro_reports();
    delay(200);

    Serial.println("[TEST2] Collecting for 10s...");
    uint16_t idx = 0;
    uint32_t start = micros();
    uint32_t end_time = millis() + 10000;

    while (millis() < end_time) {
        if (!int_asserted()) continue;

        if (bno08x.getSensorEvent(&sensorValue)) {
            if (sensorValue.sensorId == SH2_RAW_GYROSCOPE) {
                if (idx < MAX_FAST_SAMPLES) {
                    ts_buf[idx] = micros() - start;
                    data_buf.i16[idx] = sensorValue.un.rawGyroscope.y;
                }
                idx++;
            }
        }
    }

    uint16_t captured = (idx < MAX_FAST_SAMPLES) ? idx : MAX_FAST_SAMPLES;

    Serial.println("[TEST2] ---CSV_START---");
    Serial.println("idx,timestamp_us,raw_gy");
    char line[48];
    for (uint16_t i = 0; i < captured; i++) {
        snprintf(line, sizeof(line), "%u,%lu,%d", i, (unsigned long)ts_buf[i], data_buf.i16[i]);
        Serial.println(line);
    }
    Serial.println("[TEST2] ---CSV_END---");

    Serial.print("[TEST2] Total events: "); Serial.println(idx);
    Serial.print("[TEST2] Buffered:     "); Serial.println(captured);
    Serial.print("[TEST2] Effective Hz:  ");
    if (captured > 1) {
        float elapsed_s = (ts_buf[captured - 1] - ts_buf[0]) / 1e6f;
        Serial.println((captured - 1) / elapsed_s, 1);
    } else {
        Serial.println("N/A");
    }
    Serial.println("[TEST2] Done.");
    activeReportEnabler = nullptr;
}
