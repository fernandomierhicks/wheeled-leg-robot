#pragma once
#include "bno086_common.h"

// Reuse ts_buf from test_datarate_grv.h as raw storage (only one test runs at a time).
// Pack: ts_buf[2i] = t_start, ts_buf[2i+1] = (duration_us << 1) | had_data
// Max samples = MAX_FAST_SAMPLES / 2
static inline uint16_t max_spi_samples() { return MAX_FAST_SAMPLES / 2; }

void run_test4_spi_latency() {
    imu.enableGameRotationVector(10);  // 100Hz — ensure data available frequently
    delay(100);  // let first reports arrive

    Serial.println("[TEST4] SPI Transaction Latency (buffered)");
    Serial.println("[TEST4] Duration: 5s");
    Serial.println("[TEST4] Collecting...");

    // ---- Phase 1: collect timing data without serial overhead ----
    uint16_t count = 0;
    uint16_t max_n = max_spi_samples();
    uint32_t end_time = millis() + 5000;

    while (millis() < end_time && count < max_n) {
        uint32_t t0 = micros();
        bool had = imu.getSensorEvent();
        if (had) {
            volatile float qi = imu.getGameQuatI();
            volatile float qj = imu.getGameQuatJ();
            volatile float qk = imu.getGameQuatK();
            volatile float qr = imu.getGameQuatReal();
            (void)qi; (void)qj; (void)qk; (void)qr;
        }
        uint32_t t1 = micros();

        ts_buf[2 * count]     = t0;
        ts_buf[2 * count + 1] = ((t1 - t0) << 1) | (had ? 1 : 0);
        count++;
    }

    // ---- Phase 2: dump over serial ----
    Serial.println("[TEST4] ---CSV_START---");
    Serial.println("idx,t_start_us,duration_us,had_data");
    char line[48];
    for (uint16_t i = 0; i < count; i++) {
        uint32_t dur_packed = ts_buf[2 * i + 1];
        snprintf(line, sizeof(line), "%u,%lu,%lu,%u",
                 i, (unsigned long)ts_buf[2 * i],
                 (unsigned long)(dur_packed >> 1), (unsigned)(dur_packed & 1));
        Serial.println(line);
    }
    Serial.println("[TEST4] ---CSV_END---");

    Serial.print("[TEST4] Samples: "); Serial.println(count);
    Serial.println("[TEST4] Done.");
}
