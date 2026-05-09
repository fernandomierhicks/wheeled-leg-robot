#include <Arduino.h>
#include <unity.h>

// Serial5: TX=pin20 → ESP32 GPIO16(RX2), RX=pin21 ← ESP32 GPIO17(TX2)
#define LINK Serial5
#define LINK_BAUD 115200

void setUp(void) {}
void tearDown(void) {}

static uint8_t s_tx = 1;

void test_uart_echo(void) {
    // Send 0x01 every 200 ms until we receive 0x02 back (up to 5 s)
    uint32_t deadline = millis() + 5000;
    while (millis() < deadline) {
        while (LINK.available()) LINK.read();  // flush stale bytes
        LINK.write((uint8_t)0x01);
        Serial.println("Sent    0x01");
        uint32_t t = millis() + 200;
        while (millis() < t) {
            if (LINK.available()) {
                uint8_t b = LINK.read();
                Serial.printf("Received 0x%02X\n", b);
                TEST_ASSERT_EQUAL_HEX8(0x02, b);  // sent 0x01, expect 0x02
                return;
            }
        }
    }
    TEST_FAIL_MESSAGE("No reply from ESP32 within 5 s — check UART wiring");
}

void setup() {
    Serial.begin(115200);
    LINK.begin(LINK_BAUD);
    delay(500);

    UNITY_BEGIN();
    RUN_TEST(test_uart_echo);
    UNITY_END();

    Serial.println();
    Serial.println("--- UART HW: Teensy sends incrementing bytes, ESP32 replies +1 ---");
    Serial.println("  sent  | received");
    Serial.println("--------+----------");
}

void loop() {
    while (LINK.available()) LINK.read();  // flush before send

    LINK.write(s_tx);
    Serial.printf("Sent    0x%02X\n", s_tx);

    uint32_t deadline = millis() + 500;
    bool got = false;
    while (millis() < deadline) {
        if (LINK.available()) {
            uint8_t b = LINK.read();
            Serial.printf("Received 0x%02X\n", b);
            got = true;
            break;
        }
    }
    if (!got) Serial.println("Received (timeout)");

    s_tx = (s_tx >= 0xFE) ? 1 : s_tx + 1;
    delay(500);
}
