#include <Arduino.h>
#include "config.h"
#include "ak45_uart.h"

// AK45-1: Serial2  TX=pin8  RX=pin7
// AK45-2: Serial3  TX=pin14 RX=pin15
static AK45Uart ak1(Serial2);
static AK45Uart ak2(Serial3);

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}
    Serial.println("AK45 UART demo — hip motors on Serial2 (pin 8/7) + Serial3 (pin 14/15)");

    ak1.begin(AK45_UART_BAUD);
    ak2.begin(AK45_UART_BAUD);
}

static void print_motor(const char* label, const AK45Uart& ak) {
    const AK45UartState& s = ak.state();
    Serial.print(label);
    if (!s.ok) { Serial.println(": no response"); return; }
    Serial.printf(":  E=%.4f rad  M=%.4f rad  RAW=%d\n",
                  s.e_angle_rad, s.m_angle_rad, s.raw);
}

void loop() {
    static uint32_t last_ms = 0;
    uint32_t now = millis();
    if (now - last_ms < 500) return;
    last_ms = now;

    ak1.poll();
    ak2.poll();
    print_motor("AK45-1", ak1);
    print_motor("AK45-2", ak2);
}
