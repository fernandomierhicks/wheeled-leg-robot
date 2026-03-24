#pragma once
#include "bno086_common.h"

// Uses shared ts_buf[] and data_buf from test_datarate_grv.h

// ---------- Test 3: Calibrated Gyroscope max datarate ----------
// SH2_GYROSCOPE_CALIBRATED gives calibrated rad/s with bias removed
// by the BNO's fusion engine. Expected to be slower than raw gyro.

static void enable_cal_gyro_reports() {
    bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 1000);  // 1000us = 1ms
}

void run_test3_cal_gyro_datarate() {
    Serial.println("[TEST3] Calibrated Gyro-Y Max Datarate (Adafruit lib — timestamps buffered in RAM)");

    activeReportEnabler = enable_cal_gyro_reports;
    enable_cal_gyro_reports();
    delay(200);

    Serial.println("[TEST3] Collecting for 10s...");
    uint16_t idx = 0;
    uint32_t start = micros();
    uint32_t end_time = millis() + 10000;

    while (millis() < end_time) {
        if (!int_asserted()) continue;

        if (bno08x.getSensorEvent(&sensorValue)) {
            if (sensorValue.sensorId == SH2_GYROSCOPE_CALIBRATED) {
                if (idx < MAX_FAST_SAMPLES) {
                    ts_buf[idx] = micros() - start;
                    data_buf.f32[idx] = sensorValue.un.gyroscope.y;
                }
                idx++;
            }
        }
    }

    uint16_t captured = (idx < MAX_FAST_SAMPLES) ? idx : MAX_FAST_SAMPLES;

    Serial.println("[TEST3] ---CSV_START---");
    Serial.println("idx,timestamp_us,cal_gy_rads");
    char line[64];
    char fbuf[16];
    for (uint16_t i = 0; i < captured; i++) {
        dtostrf(data_buf.f32[i], 1, 6, fbuf);
        snprintf(line, sizeof(line), "%u,%lu,%s", i, (unsigned long)ts_buf[i], fbuf);
        Serial.println(line);
    }
    Serial.println("[TEST3] ---CSV_END---");

    Serial.print("[TEST3] Total events: "); Serial.println(idx);
    Serial.print("[TEST3] Buffered:     "); Serial.println(captured);
    Serial.print("[TEST3] Effective Hz:  ");
    if (captured > 1) {
        float elapsed_s = (ts_buf[captured - 1] - ts_buf[0]) / 1e6f;
        Serial.println((captured - 1) / elapsed_s, 1);
    } else {
        Serial.println("N/A");
    }
    Serial.println("[TEST3] Done.");
    activeReportEnabler = nullptr;
}
