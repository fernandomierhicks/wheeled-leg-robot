#pragma once
#include "bno086_common.h"

// ---------- Test 6: End-to-End Latency ----------
// Measures ISR-to-read delay and SPI read time using interrupt on INT pin.
// Buffers results in shared ts_buf (4 words per sample):
//   [4i+0] = t_int_us      (ISR timestamp)
//   [4i+1] = t_read_start   (main loop responds)
//   [4i+2] = t_read_done    (after SPI read completes)
//   [4i+3] = bno_ts low32   (BNO086 internal timestamp, 100µs ticks)
// Max samples = MAX_FAST_SAMPLES / 4 = 625 (need ~500 for 100Hz × 5s)

static volatile bool     e2e_int_fired = false;
static volatile uint32_t e2e_t_int     = 0;

static void e2e_isr() {
    e2e_t_int   = micros();
    e2e_int_fired = true;
}

void run_test6_e2e_latency() {
    Serial.println("[TEST6] End-to-End Latency (buffered)");
    Serial.println("[TEST6] Duration: 5s @ 100Hz GRV");

    imu.enableGameRotationVector(10);  // 10ms = 100Hz
    delay(200);  // let reports start flowing

    // Attach ISR on INT pin (D2, FALLING = H_INTN active-low)
    e2e_int_fired = false;
    attachInterrupt(digitalPinToInterrupt(BNO_INT_PIN), e2e_isr, FALLING);

    Serial.println("[TEST6] Collecting...");

    // ---- Phase 1: collect in RAM ----
    uint16_t count = 0;
    uint16_t max_n = MAX_FAST_SAMPLES / 4;
    uint32_t end_time = millis() + 5000;

    while (millis() < end_time && count < max_n) {
        if (!e2e_int_fired) continue;

        e2e_int_fired = false;
        uint32_t t_isr = e2e_t_int;  // copy volatile

        uint32_t t_rs = micros();
        imu.getSensorEvent();
        volatile float qi = imu.getGameQuatI();  // force full read
        (void)qi;
        uint64_t bno_ts = imu.getTimeStamp();
        uint32_t t_rd = micros();

        ts_buf[4 * count]     = t_isr;
        ts_buf[4 * count + 1] = t_rs;
        ts_buf[4 * count + 2] = t_rd;
        ts_buf[4 * count + 3] = (uint32_t)(bno_ts & 0xFFFFFFFF);
        count++;
    }

    detachInterrupt(digitalPinToInterrupt(BNO_INT_PIN));

    // ---- Phase 2: dump over serial ----
    Serial.println("[TEST6] ---CSV_START---");
    Serial.println("idx,t_int_us,t_read_start_us,t_read_done_us,isr_to_read_us,read_us,bno_ts");
    char line[96];
    for (uint16_t i = 0; i < count; i++) {
        uint32_t t_isr = ts_buf[4 * i];
        uint32_t t_rs  = ts_buf[4 * i + 1];
        uint32_t t_rd  = ts_buf[4 * i + 2];
        uint32_t bts   = ts_buf[4 * i + 3];
        snprintf(line, sizeof(line), "%u,%lu,%lu,%lu,%lu,%lu,%lu",
                 i,
                 (unsigned long)t_isr,
                 (unsigned long)t_rs,
                 (unsigned long)t_rd,
                 (unsigned long)(t_rs - t_isr),
                 (unsigned long)(t_rd - t_rs),
                 (unsigned long)bts);
        Serial.println(line);
    }
    Serial.println("[TEST6] ---CSV_END---");

    Serial.print("[TEST6] Samples: "); Serial.println(count);
    Serial.println("[TEST6] Done.");
}
