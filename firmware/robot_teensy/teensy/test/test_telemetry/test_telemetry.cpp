#include <Arduino.h>
#include <unity.h>
#include "comm_protocol.h"
#include "CommLink.h"
#include "esp32_link.h"

void setUp(void) {}
void tearDown(void) {}

// ── Loopback Stream ───────────────────────────────────────────────────────────

class LoopbackStream : public Stream {
    uint8_t _buf[512];
    int _head = 0, _tail = 0;
public:
    size_t write(const uint8_t* buf, size_t len) override {
        for (size_t i = 0; i < len; i++) _buf[_tail++ % 512] = buf[i];
        return len;
    }
    size_t write(uint8_t b) override { return write(&b, 1); }
    int available() override { return _tail - _head; }
    int read()      override { return (_head < _tail) ? _buf[_head++ % 512] : -1; }
    int peek()      override { return (_head < _tail) ? _buf[_head  % 512] : -1; }
};

// ── ACK tracking ─────────────────────────────────────────────────────────────

static volatile uint32_t s_ack_count = 0;

static void on_ack(uint8_t type, uint8_t /*ver*/, uint8_t src,
                   const uint8_t* /*payload*/, uint16_t /*len*/) {
    if (type == COMM_TYPE_ACK && src == COMM_SRC_ESP32)
        ++s_ack_count;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

void test_payload_size(void) {
    // 1×uint32 + 9×float + 1×uint8 = 4 + 36 + 1 = 41 bytes
    TEST_ASSERT_EQUAL(41, sizeof(TelemetryPayload));
}

void test_frame_constants(void) {
    TEST_ASSERT_EQUAL_HEX8(0xFF, COMM_START);
    TEST_ASSERT_EQUAL_HEX8(0xFE, COMM_END);
}

static bool     s_rx_got;
static uint8_t  s_rx_type;
static uint8_t  s_rx_src;
static uint16_t s_rx_len;
static uint8_t  s_rx_buf[COMM_MAX_PAYLOAD];

static void on_packet(uint8_t type, uint8_t ver, uint8_t src,
                      const uint8_t* payload, uint16_t len) {
    s_rx_got  = true;
    s_rx_type = type;
    s_rx_src  = src;
    s_rx_len  = len;
    for (uint16_t i = 0; i < len && i < COMM_MAX_PAYLOAD; i++)
        s_rx_buf[i] = payload[i];
    (void)ver;
}

void test_telemetry_roundtrip(void) {
    s_rx_got = false;

    LoopbackStream ls;
    CommLink cl(ls, COMM_SRC_TEENSY);
    cl.onPacket(on_packet);

    TelemetryPayload tx = {};
    tx.timestamp_ms     = 99999;
    tx.pitch_rad        = 0.123f;
    tx.wheel_vel_avg_ms = -0.5f;
    tx.robot_state      = 2;

    cl.send(COMM_TYPE_TELEMETRY, TELEM_PAYLOAD_V1, &tx, sizeof(tx));
    cl.update();

    TEST_ASSERT_TRUE(s_rx_got);
    TEST_ASSERT_EQUAL(COMM_TYPE_TELEMETRY, s_rx_type);
    TEST_ASSERT_EQUAL(COMM_SRC_TEENSY, s_rx_src);
    TEST_ASSERT_EQUAL(sizeof(TelemetryPayload), s_rx_len);

    TelemetryPayload rx;
    memcpy(&rx, s_rx_buf, sizeof(rx));
    TEST_ASSERT_EQUAL_UINT32(99999, rx.timestamp_ms);
    TEST_ASSERT_EQUAL_FLOAT(0.123f, rx.pitch_rad);
    TEST_ASSERT_EQUAL_FLOAT(-0.5f, rx.wheel_vel_avg_ms);
    TEST_ASSERT_EQUAL_UINT8(2, rx.robot_state);
}

void test_bad_checksum_dropped(void) {
    s_rx_got = false;

    LoopbackStream ls;
    CommLink cl(ls, COMM_SRC_TEENSY);
    cl.onPacket(on_packet);

    TelemetryPayload tx = {};
    cl.send(COMM_TYPE_TELEMETRY, TELEM_PAYLOAD_V1, &tx, sizeof(tx));

    uint8_t corrupt = ls.read();
    (void)corrupt;
    ls.write((uint8_t)0xAB);

    cl.update();
    TEST_ASSERT_FALSE(s_rx_got);
}

void test_uart_tx_no_stall(void) {
    g_esp32.begin();

    TelemetryPayload t = {};
    t.pitch_rad   = 0.1f;
    t.robot_state = 2;

    uint32_t before   = g_esp32.tx_count();
    uint32_t deadline = millis() + 200;

    for (int i = 0; i < 10; i++) {
        t.timestamp_ms = millis();
        g_esp32.send_telemetry(t);
        delay(2);
    }

    TEST_ASSERT_EQUAL_UINT32(before + 10, g_esp32.tx_count());
    TEST_ASSERT_MESSAGE(millis() < deadline, "Serial5 TX stalled");
}

// ── Entry ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    delay(500);

    UNITY_BEGIN();
    RUN_TEST(test_payload_size);
    RUN_TEST(test_frame_constants);
    RUN_TEST(test_telemetry_roundtrip);
    RUN_TEST(test_bad_checksum_dropped);
    RUN_TEST(test_uart_tx_no_stall);
    UNITY_END();

    g_esp32.onCommand(on_ack);

    Serial.println();
    Serial.println("--- Streaming telemetry on Serial5 (TX=pin20) at 500 Hz ---");
    Serial.println("--- Listening for ACKs from ESP32 (RX=pin21)           ---");
    Serial.println("  tx_count |  ack_rx  | ack_rate (%)");
    Serial.println("-----------+----------+--------------");
}

// ── Live stream ───────────────────────────────────────────────────────────────

void loop() {
    static uint32_t last_send_us = 0;
    static uint32_t last_print   = 0;
    static uint32_t tx_snap      = 0;
    static uint32_t ack_snap     = 0;

    g_esp32.update();

    uint32_t now_us = micros();
    if (now_us - last_send_us >= 2000) {
        last_send_us = now_us;

        TelemetryPayload t = {};
        t.timestamp_ms = millis();
        t.pitch_rad    = 0.05f * sinf(millis() * 0.001f * TWO_PI * 0.5f);
        t.robot_state  = 2;
        g_esp32.send_telemetry(t);
    }

    uint32_t now_ms = millis();
    if (now_ms - last_print >= 1000) {
        uint32_t tx  = g_esp32.tx_count();
        uint32_t ack = s_ack_count;
        uint32_t dtx = tx  - tx_snap;
        uint32_t dak = ack - ack_snap;
        tx_snap  = tx;
        ack_snap = ack;
        float rate = (dtx > 0) ? (100.0f * dak / dtx) : 0.0f;
        Serial.printf("%10lu | %8lu | %10.1f %%\n", tx, ack, rate);
        last_print = now_ms;
    }
}
