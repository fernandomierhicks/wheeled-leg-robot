// CommLink USB demo + test — proves the binary packet protocol over USB Serial.
//
// Unity phase (setup):
//   1. test_log_tx          — send a LOG packet; verify tx counter increments and Serial doesn't stall
//   2. test_telemetry_tx    — send a TELEMETRY packet; same check
//   3. test_rx_parse        — loopback: encode a packet, feed raw bytes back through CommLink, verify callback fires
//
// Live phase (loop):
//   Streams LOG + TELEMETRY packets to USB at ~2 Hz each.
//   Listens for any valid framed packet arriving from the PC.
//   Open the GUI Flash & Monitor tab — the "Last Packet" inspector should update
//   with each received packet, proving Teensy → PC binary framing works end-to-end.
//   If the PC side sends a framed command back, the LED turns green.
//
// LED:
//   white strobe  — Unity tests running
//   cyan pulse    — streaming to PC, no response from PC yet
//   green pulse   — at least one valid packet received from PC (bidirectional)
//   red blink     — TX stall (Serial not draining)

#include <Arduino.h>
#include <unity.h>
#include <math.h>
#include "CommLink.h"
#include "comm_protocol.h"
#include "config.h"
#include "../test_led.h"

// ── Globals ───────────────────────────────────────────────────────────────────

static CommLink g_link(Serial, COMM_SRC_TEENSY);

static volatile uint32_t g_tx_count = 0;
static volatile uint32_t g_rx_count = 0;
static volatile uint32_t g_last_rx_ms = 0;

static void _send_log(uint8_t level, const char* msg) {
    uint8_t buf[63];
    buf[0] = level;
    size_t n = strlen(msg);
    if (n > 62) n = 62;
    memcpy(buf + 1, msg, n);
    g_link.send(COMM_TYPE_LOG, LOG_PAYLOAD_V1, buf, 1 + n);
    g_tx_count++;
}

static void _send_telemetry(float phase) {
    TelemetryPayload t = {};
    t.timestamp_ms    = millis();
    t.pitch_rad       = 0.30f * sinf(phase);
    t.pitch_rate_rads = 0.30f * TWO_PI * 0.25f * cosf(phase);
    t.roll_rad        = 0.05f * sinf(phase * 1.3f);
    t.yaw_rad         = 0.02f * sinf(phase * 0.7f);
    t.robot_state     = 2;   // RUNNING
    g_link.send(COMM_TYPE_TELEMETRY, TELEM_PAYLOAD_V1, &t, sizeof(t));
    g_tx_count++;
}

// ── Loopback stream ───────────────────────────────────────────────────────────

class LoopbackStream : public Stream {
    uint8_t _buf[256];
    uint16_t _head = 0, _tail = 0;
public:
    size_t write(const uint8_t* b, size_t n) override {
        for (size_t i = 0; i < n; i++) _buf[_tail++ % 256] = b[i];
        return n;
    }
    size_t write(uint8_t b) override { return write(&b, 1); }
    int available() override { return (int)(_tail - _head); }
    int read()      override { return (_head < _tail) ? _buf[_head++ % 256] : -1; }
    int peek()      override { return (_head < _tail) ? _buf[_head  % 256] : -1; }
};

// ── Unity tests ───────────────────────────────────────────────────────────────

void setUp(void) {}
void tearDown(void) {}

// Sending a LOG packet must not stall and must increment the tx counter.
void test_log_tx(void) {
    uint32_t before   = g_tx_count;
    uint32_t deadline = millis() + 50;

    _send_log(LOG_LEVEL_INFO, "test_log_tx: CommLink LOG over USB Serial");

    TEST_ASSERT_EQUAL_UINT32(before + 1, g_tx_count);
    TEST_ASSERT_MESSAGE(millis() < deadline, "Serial.write() stalled — USB not draining");
}

// Sending a TELEMETRY packet must not stall.
void test_telemetry_tx(void) {
    uint32_t before   = g_tx_count;
    uint32_t deadline = millis() + 50;

    _send_telemetry(0.0f);

    TEST_ASSERT_EQUAL_UINT32(before + 1, g_tx_count);
    TEST_ASSERT_MESSAGE(millis() < deadline, "Serial.write() stalled on TELEMETRY");
}

// Encode a TELEMETRY packet into a LoopbackStream, parse it back, verify roundtrip.
void test_rx_parse(void) {
    static bool got = false;
    static uint8_t got_type = 0;
    static uint16_t got_len = 0;
    static uint32_t got_ts  = 0;

    LoopbackStream ls;
    CommLink cl(ls, COMM_SRC_TEENSY);
    cl.onPacket([](uint8_t type, uint8_t /*ver*/, uint8_t /*src*/,
                   const uint8_t* payload, uint16_t len) {
        got      = true;
        got_type = type;
        got_len  = len;
        if (len >= 4) memcpy(&got_ts, payload, 4);
    });

    TelemetryPayload tx = {};
    tx.timestamp_ms = 0xDEADBEEF;
    tx.pitch_rad    = 1.23f;
    tx.robot_state  = 2;
    cl.send(COMM_TYPE_TELEMETRY, TELEM_PAYLOAD_V1, &tx, sizeof(tx));
    cl.update();

    TEST_ASSERT_TRUE_MESSAGE(got,  "parser callback never fired — framing broken");
    TEST_ASSERT_EQUAL_HEX8(COMM_TYPE_TELEMETRY, got_type);
    TEST_ASSERT_EQUAL_UINT16(sizeof(TelemetryPayload), got_len);
    TEST_ASSERT_EQUAL_HEX32(0xDEADBEEF, got_ts);
}

// ── Incoming packet handler (live phase) ─────────────────────────────────────

static void _on_rx(uint8_t type, uint8_t /*ver*/, uint8_t src,
                   const uint8_t* /*payload*/, uint16_t len) {
    g_rx_count++;
    g_last_rx_ms = millis();

    char msg[56];
    snprintf(msg, sizeof(msg), "RX type=0x%02X src=0x%02X len=%u count=%lu",
             type, src, (unsigned)len, (unsigned long)g_rx_count);
    _send_log(LOG_LEVEL_INFO, msg);
}

// ── Entry ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    delay(1000);
    test_led_begin();   // white strobe while tests run

    UNITY_BEGIN();
    RUN_TEST(test_log_tx);
    RUN_TEST(test_telemetry_tx);
    RUN_TEST(test_rx_parse);
    int failures = UNITY_END();
    test_led_done(failures);

    if (failures > 0) return;  // stay on red blink — don't enter live phase

    // Wire the live RX handler and announce we're streaming
    g_link.onPacket(_on_rx);
    _send_log(LOG_LEVEL_INFO,
              "LIVE: streaming TELEM+LOG at ~2 Hz — watch GUI packet inspector");

    Serial.println();
    Serial.println("=== CommLink USB live phase ===");
    Serial.println("Binary packets streaming on this port.");
    Serial.println("Open GUI Flash & Monitor — 'Last Packet' inspector should update.");
    Serial.println("Send a framed COMMAND packet from the PC to prove bidirectional.");
    Serial.println("  tx_count | rx_count | uptime(s)");
    Serial.println("-----------+----------+----------");

    // Cyan pulse: streaming but no PC response yet
    _tled::led.pulse(0, 180, 255, 1500);
}

// ── Live streaming loop ───────────────────────────────────────────────────────

void loop() {
    static uint32_t last_telem_ms = 0;
    static uint32_t last_log_ms   = 0;
    static uint32_t last_print_ms = 0;
    static bool     linked        = false;

    uint32_t now = millis();

    g_link.update();  // pump incoming bytes

    // Telemetry at ~2 Hz
    if (now - last_telem_ms >= 500) {
        last_telem_ms = now;
        float phase = now * 0.001f * TWO_PI * 0.25f;
        _send_telemetry(phase);
    }

    // Heartbeat LOG every 5 s
    if (now - last_log_ms >= 5000) {
        last_log_ms = now;
        char msg[56];
        snprintf(msg, sizeof(msg), "heartbeat tx=%lu rx=%lu up=%lus",
                 (unsigned long)g_tx_count, (unsigned long)g_rx_count,
                 (unsigned long)(now / 1000));
        _send_log(LOG_LEVEL_INFO, msg);
    }

    // Flip to green once we've received at least one packet from the PC
    if (!linked && g_rx_count > 0 && now - g_last_rx_ms > 400) {
        linked = true;
        _tled::led.pulse(0, 255, 0, 1500);   // green = bidirectional confirmed
    }

    // Human-readable status every 2 s
    if (now - last_print_ms >= 2000) {
        last_print_ms = now;
        Serial.printf("%10lu | %8lu | %9lu\n",
                      (unsigned long)g_tx_count,
                      (unsigned long)g_rx_count,
                      (unsigned long)(now / 1000));
    }
}
