#include <unity.h>
#include "comm_protocol.h"
#include "CommLink.h"

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

// ── Tests ─────────────────────────────────────────────────────────────────────

void test_payload_size(void) {
    TEST_ASSERT_EQUAL(33, sizeof(TelemetryPayload));
}

void test_frame_constants(void) {
    TEST_ASSERT_EQUAL_HEX8(0xFF, COMM_START);
    TEST_ASSERT_EQUAL_HEX8(0xFE, COMM_END);
}

static bool     s_rx_got;
static uint8_t  s_rx_type;
static uint16_t s_rx_len;
static uint8_t  s_rx_buf[COMM_MAX_PAYLOAD];

static void on_packet(uint8_t type, uint8_t ver, uint8_t src,
                      const uint8_t* payload, uint16_t len) {
    s_rx_got  = true;
    s_rx_type = type;
    s_rx_len  = len;
    for (uint16_t i = 0; i < len && i < COMM_MAX_PAYLOAD; i++)
        s_rx_buf[i] = payload[i];
    (void)ver; (void)src;
}

void test_telemetry_roundtrip(void) {
    s_rx_got = false;

    LoopbackStream ls;
    CommLink cl(ls, COMM_SRC_ESP32);
    cl.onPacket(on_packet);

    TelemetryPayload tx = {};
    tx.timestamp_ms = 42000;
    tx.pitch_rad    = -0.25f;

    cl.send(COMM_TYPE_TELEMETRY, TELEM_PAYLOAD_V1, &tx, sizeof(tx));
    cl.update();

    TEST_ASSERT_TRUE(s_rx_got);
    TEST_ASSERT_EQUAL(COMM_TYPE_TELEMETRY, s_rx_type);
    TEST_ASSERT_EQUAL(sizeof(TelemetryPayload), s_rx_len);

    TelemetryPayload rx;
    memcpy(&rx, s_rx_buf, sizeof(rx));
    TEST_ASSERT_EQUAL_UINT32(42000, rx.timestamp_ms);
    TEST_ASSERT_EQUAL_FLOAT(-0.25f, rx.pitch_rad);
}

void test_zero_payload(void) {
    s_rx_got = false;

    LoopbackStream ls;
    CommLink cl(ls, COMM_SRC_ESP32);
    cl.onPacket(on_packet);

    cl.send(COMM_TYPE_ACK, 1, nullptr, 0);
    cl.update();

    TEST_ASSERT_TRUE(s_rx_got);
    TEST_ASSERT_EQUAL(COMM_TYPE_ACK, s_rx_type);
    TEST_ASSERT_EQUAL(0, s_rx_len);
}

void setup() {
    UNITY_BEGIN();
    RUN_TEST(test_payload_size);
    RUN_TEST(test_frame_constants);
    RUN_TEST(test_telemetry_roundtrip);
    RUN_TEST(test_zero_payload);
    UNITY_END();
}
void loop() {}
