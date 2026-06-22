#include <Arduino.h>
#include <unity.h>
#include "esp32_link.h"
#include "comm_protocol.h"
#include "config.h"
#include "../test_led.h"

void setUp(void) {}
void tearDown(void) {}

// ── Tests ─────────────────────────────────────────────────────────────────────

void test_baud_config(void) {
    TEST_ASSERT_EQUAL_UINT32(1200000UL, ESP32_BAUD);
}

void test_frame_size(void) {
    // TelemetryPayload V4: 128 B (see static_assert in comm_protocol.h)
    // Frame overhead: start + type + ver + src + seq + len×2 + crc + end = 9 B
    // Total on wire: 137 B → at 1.2 Mbaud ≈ 1.14 ms < 2000 µs loop budget
    TEST_ASSERT_EQUAL(128, sizeof(TelemetryPayload));
    TEST_ASSERT_EQUAL(137, 9 + (int)sizeof(TelemetryPayload));
}

void test_serial5_init(void) {
    g_esp32.begin();
    TEST_ASSERT_TRUE(true);  // reaching here means Serial5.begin() did not hang
}

void test_send_no_stall(void) {
    TelemetryPayload t = {};
    t.timestamp_ms = millis();
    t.pitch_rad    = 0.1f;
    t.robot_state  = (uint8_t)2;  // STATE_RUNNING

    uint32_t before   = g_esp32.tx_count();
    uint32_t deadline = millis() + 100;

    for (int i = 0; i < 10; i++) {
        t.timestamp_ms = millis();
        g_esp32.send_telemetry(t);
        delay(2);
    }

    TEST_ASSERT_EQUAL_UINT32(before + 10, g_esp32.tx_count());
    TEST_ASSERT_MESSAGE(millis() < deadline, "send_telemetry stalled — TX buffer may be full");
}

// ── Entry ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    delay(500);
    test_led_begin();

    UNITY_BEGIN();
    RUN_TEST(test_baud_config);
    RUN_TEST(test_frame_size);
    RUN_TEST(test_serial5_init);
    RUN_TEST(test_send_no_stall);
    test_led_done(UNITY_END());

    Serial.println();
    Serial.println("--- Streaming telemetry on Serial5 (TX=pin20, RX=pin21) at 500 Hz ---");
    Serial.println("--- Connect ESP32 RX→pin20, GND→GND (3.3 V logic, no level shift) ---");
    Serial.println("--- tx_count | Hz   | pitch (rad) | state                          ---");
}

// ── Live stream ───────────────────────────────────────────────────────────────

void loop() {
    static uint32_t last_send_us = 0;
    static uint32_t last_print   = 0;
    static uint32_t count_snap   = 0;

    uint32_t now_us = micros();

    if (now_us - last_send_us >= 2000) {
        last_send_us = now_us;

        TelemetryPayload t = {};
        float phase        = millis() * 0.001f * TWO_PI * 0.5f;  // 0.5 Hz sine
        t.timestamp_ms     = millis();
        t.pitch_rad        = 0.05f * sinf(phase);
        t.pitch_rate_rads  = 0.05f * TWO_PI * 0.5f * cosf(phase);
        t.robot_state      = (uint8_t)2;  // STATE_RUNNING

        g_esp32.send_telemetry(t);
        g_esp32.update();  // pump any incoming ACKs from ESP32
    }

    uint32_t now_ms = millis();
    if (now_ms - last_print >= 500) {
        uint32_t cnt = g_esp32.tx_count();
        float hz     = (cnt - count_snap) * 1000.0f / (now_ms - last_print);
        count_snap   = cnt;
        last_print   = now_ms;

        static uint8_t row = 0;
        if (row % 10 == 0) {
            Serial.println();
            Serial.println("  tx_count |   Hz  | pitch (rad) | state");
            Serial.println("-----------+-------+-------------+-------");
        }
        row++;

        float pitch = 0.05f * sinf(millis() * 0.001f * TWO_PI * 0.5f);
        Serial.printf("%10lu | %5.1f | %+11.4f | %d\n", cnt, hz, pitch, 2);
    }
}
