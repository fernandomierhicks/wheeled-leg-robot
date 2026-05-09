#include <Arduino.h>
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

// ── Cross-device link (Serial2: GPIO16=RX2←Teensy pin20, GPIO17=TX2→Teensy pin21) ────

#define TEENSY_LINK_BAUD  1200000UL
#define TEENSY_LINK_RX    16   // GPIO16 (RX2) ← Teensy pin 20
#define TEENSY_LINK_TX    17   // GPIO17 (TX2) → Teensy pin 21

static CommLink*  s_link     = nullptr;
static uint32_t   s_rx_count = 0;
static uint32_t   s_ack_tx   = 0;

static const uint8_t kEmpty = 0;

static void on_teensy_packet(uint8_t type, uint8_t /*ver*/, uint8_t src,
                              const uint8_t* /*payload*/, uint16_t /*len*/) {
    if (type == COMM_TYPE_TELEMETRY && src == COMM_SRC_TEENSY) {
        ++s_rx_count;
        s_link->send(COMM_TYPE_ACK, 1, &kEmpty, 0);
        ++s_ack_tx;
    }
}

static void init_uart_link() {
    if (!s_link) {
        Serial2.begin(TEENSY_LINK_BAUD, SERIAL_8N1, TEENSY_LINK_RX, TEENSY_LINK_TX);
        s_link = new CommLink(Serial2, COMM_SRC_ESP32);
        s_link->onPacket(on_teensy_packet);
    }
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

    cl.send(COMM_TYPE_ACK, 1, &kEmpty, 0);
    cl.update();

    TEST_ASSERT_TRUE(s_rx_got);
    TEST_ASSERT_EQUAL(COMM_TYPE_ACK, s_rx_type);
    TEST_ASSERT_EQUAL(0, s_rx_len);
}

void test_uart_rx_from_teensy(void) {
    init_uart_link();
    s_rx_count = 0;

    uint32_t deadline = millis() + 3000;
    while (millis() < deadline && s_rx_count == 0)
        s_link->update();

    TEST_ASSERT_MESSAGE(s_rx_count > 0,
                        "no TELEM from Teensy within 3 s — check UART wiring");
}

// ── Entry ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    delay(500);

    UNITY_BEGIN();
    RUN_TEST(test_payload_size);
    RUN_TEST(test_frame_constants);
    RUN_TEST(test_telemetry_roundtrip);
    RUN_TEST(test_zero_payload);
    RUN_TEST(test_uart_rx_from_teensy);
    UNITY_END();

    Serial.println();
    Serial.println("--- ACK-ing every TELEM from Teensy (RX2=GPIO16, TX2=GPIO17) ---");
    Serial.println("  rx_count |  ack_tx");
    Serial.println("-----------+---------");
}

// ── ACK loop ──────────────────────────────────────────────────────────────────

void loop() {
    static uint32_t last_print = 0;

    if (s_link) s_link->update();

    uint32_t now_ms = millis();
    if (now_ms - last_print >= 1000) {
        last_print = now_ms;
        Serial.printf("%10lu | %8lu\n", s_rx_count, s_ack_tx);
    }
}
