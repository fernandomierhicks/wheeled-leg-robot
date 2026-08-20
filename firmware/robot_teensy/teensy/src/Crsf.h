#pragma once
// CRSF receiver + telemetry emitter for the Teensy, over one hardware UART.
//
// Replaces IBus.h. The public surface is deliberately identical — begin(),
// update(), channel(n) 1-indexed returning 0 on signal loss, alive(timeout) —
// so the ~20 call sites in main.cpp change name and nothing else. Every one of
// those sites is safety-relevant (arm interlock, rescue combo, calibration
// combo), and a port that reshaped them all at once would be a port nobody
// could review.
//
// All parsing and every liveness rule lives in crsf_protocol.h, which is
// Arduino-free and unit-tested natively. This file only supplies a serial port
// and a clock.
//
// Wiring: unlike iBUS this link is bidirectional. Both TX and RX must be
// connected — telemetry, and eventually the command tunnel, ride the return
// path. On Serial4 that is pin 17 (TX) alongside the existing pin 16 (RX).

#include <Arduino.h>
#include "crsf_protocol.h"

class Crsf {
public:
    explicit Crsf(HardwareSerial& serial) : _serial(serial) {}

    void begin(uint32_t baud = CRSF_BAUD) {
        _serial.begin(baud);
        _core.reset();
    }

    // Call every loop. Returns true if at least one valid frame was decoded.
    bool update() {
        const uint32_t now = millis();
        bool got = false;
        // Bounded so a stuck or flooded UART cannot stall the 500 Hz control
        // loop. At 420 kbaud a 2 ms tick carries ~105 bytes; 256 is generous
        // headroom that still terminates.
        uint16_t budget = 256;
        while (_serial.available() && budget--) {
            if (_core.feed((uint8_t)_serial.read(), now)) got = true;
        }
        return got;
    }

    // 1-indexed (CH1..CH16). Returns 0 on signal loss — see the FAILSAFE
    // CONTRACT in crsf_protocol.h before using anything else here in an arm,
    // rescue or calibration test.
    uint16_t channel(uint8_t n) const { return _core.channel_us(n, millis()); }

    bool alive(uint32_t timeout_ms = CRSF_LINK_TIMEOUT_MS) const {
        return _core.alive(millis(), timeout_ms);
    }

    // Diagnostics only. Does NOT zero on signal loss.
    uint16_t channel_raw(uint8_t n) const { return _core.channel_us_raw(n); }

    const CrsfLinkStats& link() const { return _core.link(); }
    bool     have_link_stats() const  { return _core.have_link_stats(); }
    uint32_t crc_errors() const       { return _core.crc_errors(); }

    // ── Telemetry ────────────────────────────────────────────────────────────
    // Writes a prebuilt frame back to the receiver. Non-blocking: if the TX
    // FIFO cannot take the whole frame the frame is DROPPED rather than
    // partially written. A half-frame would desync the receiver's parser, and
    // dropping telemetry is free while stalling the control loop is not.
    bool send(const uint8_t* frame, uint8_t len) {
        if ((int)len > _serial.availableForWrite()) { _tx_drops++; return false; }
        _serial.write(frame, len);
        _tx_frames++;
        return true;
    }

    uint32_t tx_frames() const { return _tx_frames; }
    uint32_t tx_drops() const  { return _tx_drops; }

private:
    HardwareSerial& _serial;
    CrsfCore _core;
    uint32_t _tx_frames = 0;
    uint32_t _tx_drops  = 0;
};
