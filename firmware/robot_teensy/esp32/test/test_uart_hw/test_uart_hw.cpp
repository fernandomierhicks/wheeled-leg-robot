#include <Arduino.h>
#include <unity.h>
#include "config.h"

// Serial2: RX=TEENSY_UART_RX ← Teensy pin20, TX=TEENSY_UART_TX → Teensy pin21
// Baud, RX, TX pin constants come from config.h so this test tracks production config.

void setUp(void) {}
void tearDown(void) {}

void test_uart_echo(void) {
    // Wait up to 5 s for any byte from Teensy — value doesn't matter, just proves the wire works
    uint32_t deadline = millis() + 5000;
    while (millis() < deadline) {
        if (Serial2.available()) {
            uint8_t b = Serial2.read();
            uint8_t reply = (b >= 0xFE) ? 1 : b + 1;
            Serial.printf("Received 0x%02X — sending 0x%02X\n", b, reply);
            Serial2.write(reply);
            return;  // PASS
        }
    }
    TEST_FAIL_MESSAGE("No byte from Teensy within 5 s — check UART wiring");
}

void setup() {
    Serial.begin(115200);
    Serial2.setRxBufferSize(1024);
    Serial2.begin(TEENSY_UART_BAUD, SERIAL_8N1, TEENSY_UART_RX, TEENSY_UART_TX);
    delay(500);

    UNITY_BEGIN();
    RUN_TEST(test_uart_echo);
    UNITY_END();

    Serial.println();
    Serial.println("--- UART HW: ESP32 echoes every byte from Teensy as +1 ---");
}

void loop() {
    static uint32_t last_wait_print = 0;

    if (Serial2.available()) {
        uint8_t b = Serial2.read();
        uint8_t reply = (b >= 0xFE) ? 1 : b + 1;
        Serial.printf("Received 0x%02X — sending 0x%02X\n", b, reply);
        Serial2.write(reply);
        last_wait_print = millis();
    } else if (millis() - last_wait_print >= 1000) {
        last_wait_print = millis();
        Serial.println("Waiting for Teensy...");
    }
}
