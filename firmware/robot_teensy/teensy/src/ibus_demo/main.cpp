#include <Arduino.h>
#include "config.h"
#include "IBus.h"

// Serial4 RX = pin 16 (PIN_IBUS_RX). TX pin unused — iBUS is receive-only.
static IBus ibus(Serial4);

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}   // wait for USB serial on Teensy
    Serial.println("iBUS hello-world — FlySky receiver on Serial4 pin 16");

    ibus.begin(115200);
}

void loop() {
    bool new_packet = ibus.update();

    static uint32_t last_print_ms = 0;
    uint32_t now = millis();
    if (now - last_print_ms < 200) return;
    last_print_ms = now;

    if (!ibus.alive()) {
        Serial.println("iBUS: no signal");
        return;
    }

    // Print all 14 channels on one line, 5 Hz
    Serial.print(new_packet ? "NEW  ch: " : "     ch: ");
    for (uint8_t i = 1; i <= IBUS_NUM_CH; i++) {
        Serial.print(ibus.channel(i));
        if (i < IBUS_NUM_CH) Serial.print("  ");
    }
    Serial.println();
}
