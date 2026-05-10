#pragma once
#include <Arduino.h>

// FlySky iBUS driver — non-blocking, 32-byte framing over hardware UART.
// Call begin() once, then update() each loop. Channel values (1000–2000 µs)
// are safe to read any time; they hold the last valid packet until the next.

static constexpr uint8_t  IBUS_NUM_CH      = 14;
static constexpr uint16_t IBUS_CH_MIN      = 1000;
static constexpr uint16_t IBUS_CH_MID      = 1500;
static constexpr uint16_t IBUS_CH_MAX      = 2000;

class IBus {
public:
    explicit IBus(HardwareSerial& serial) : _serial(serial) {}

    void begin(uint32_t baud = 115200) {
        _serial.begin(baud);
        for (uint8_t i = 0; i < IBUS_NUM_CH; i++) _ch[i] = IBUS_CH_MID;
    }

    // Call every loop. Returns true if a new valid packet was decoded this call.
    bool update() {
        bool got_packet = false;
        while (_serial.available()) {
            uint8_t b = _serial.read();

            if (_idx == 0 && b != 0x20) continue;   // wait for length byte
            if (_idx == 1 && b != 0x40) { _idx = 0; continue; } // bad cmd byte

            _buf[_idx++] = b;
            if (_idx < 32) continue;
            _idx = 0;

            if (!_verify_checksum()) continue;

            for (uint8_t i = 0; i < IBUS_NUM_CH; i++)
                _ch[i] = (uint16_t)(_buf[2 + i * 2] | (_buf[3 + i * 2] << 8));

            _last_packet_ms = millis();
            got_packet = true;
        }
        return got_packet;
    }

    uint16_t channel(uint8_t n) const {
        return (n < IBUS_NUM_CH) ? _ch[n] : IBUS_CH_MID;
    }

    // True if a packet arrived within the last `timeout_ms` milliseconds.
    bool alive(uint32_t timeout_ms = 500) const {
        return (millis() - _last_packet_ms) < timeout_ms;
    }

private:
    bool _verify_checksum() const {
        uint16_t chk = 0xFFFF;
        for (uint8_t i = 0; i < 30; i++) chk -= _buf[i];
        uint16_t rx = (uint16_t)(_buf[30] | (_buf[31] << 8));
        return chk == rx;
    }

    HardwareSerial& _serial;
    uint8_t  _buf[32]        = {};
    uint8_t  _idx            = 0;
    uint16_t _ch[IBUS_NUM_CH]= {};
    uint32_t _last_packet_ms = 0;
};
