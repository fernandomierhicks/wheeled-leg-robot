#include "esp32_link.h"
#include <Arduino.h>
#include "config.h"

Esp32Link g_esp32;

Esp32Link::Esp32Link() : _link(Serial5, COMM_SRC_TEENSY), _tx_count(0) {}

void Esp32Link::begin() {
    Serial5.begin(ESP32_BAUD);
}

void Esp32Link::send_telemetry(const TelemetryPayload& t) {
    _link.send(COMM_TYPE_TELEMETRY, TELEM_PAYLOAD_V2, &t, sizeof(t));
    ++_tx_count;
}

void Esp32Link::update() {
    _link.update();
}

void Esp32Link::onCommand(CommPacketCb cb) {
    _link.onPacket(cb);
}
