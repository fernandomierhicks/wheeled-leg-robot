#pragma once
#include <Arduino.h>

// FlySky iBUS driver — non-blocking, 32-byte framing over hardware UART.
// Call begin() once, then update() each loop. Channel values (1000–2000 µs)
// are safe to read any time; returns 0 on signal loss.
//
// Signal-loss detection — two independent indicators, both must be false for alive():
//  1. CH7/CH8 sentinel: FS-iA6B outputs ~700 µs on these channels when TX is lost.
//     Normal range is 1000–2000, so <900 is unambiguous.
//  2. Noise on CH11–CH14: extended channels with no physical TX input are stable
//     when TX is bound and noisy when TX is lost. CH9/CH10 are valid channels,
//     CH11–CH14 are unused and reliable noise indicators.

static constexpr uint8_t  IBUS_NUM_CH        = 14;
static constexpr uint8_t  IBUS_EXT_CH_START  = 10;   // CH11 onwards (CH9/CH10 are valid)
static constexpr uint16_t IBUS_CH_MIN        = 1000;
static constexpr uint16_t IBUS_CH_MID        = 1500;
static constexpr uint16_t IBUS_CH_MAX        = 2000;
static constexpr uint16_t IBUS_SENTINEL_THR  = 900;  // CH7/CH8 below this = TX lost
static constexpr uint16_t IBUS_EXT_NOISE_THR = 50;   // µs change = noise threshold
static constexpr uint8_t  IBUS_WARMUP_PKTS   = 5;    // ignore noise check on first N packets

class IBus {
public:
    explicit IBus(HardwareSerial& serial) : _serial(serial) {}

    void begin(uint32_t baud = 115200) {
        _serial.begin(baud);
        for (uint8_t i = 0; i < IBUS_NUM_CH; i++) _ch[i] = IBUS_CH_MID;
        for (uint8_t i = IBUS_EXT_CH_START; i < IBUS_NUM_CH; i++) _ext_prev[i - IBUS_EXT_CH_START] = IBUS_CH_MID;
    }

    // Call every loop. Returns true if a new valid packet was decoded this call.
    bool update() {
        bool got_packet = false;
        while (_serial.available()) {
            uint8_t b = _serial.read();

            if (_idx == 0 && b != 0x20) continue;
            if (_idx == 1 && b != 0x40) { _idx = 0; continue; }

            _buf[_idx++] = b;
            if (_idx < 32) continue;
            _idx = 0;

            if (!_verify_checksum()) continue;

            for (uint8_t i = 0; i < IBUS_NUM_CH; i++)
                _ch[i] = (uint16_t)(_buf[2 + i * 2] | (_buf[3 + i * 2] << 8));

            _last_packet_ms = millis();
            got_packet = true;

            // CH7/CH8 sentinel: FS-iA6B drops these to ~700 µs when TX is lost
            _sentinel_lost = (_ch[6] < IBUS_SENTINEL_THR || _ch[7] < IBUS_SENTINEL_THR);

            if (_pkt_count >= IBUS_WARMUP_PKTS) {
                bool noisy = false;
                for (uint8_t i = IBUS_EXT_CH_START; i < IBUS_NUM_CH; i++) {
                    uint16_t prev = _ext_prev[i - IBUS_EXT_CH_START];
                    uint16_t curr = _ch[i];
                    if ((curr > prev ? curr - prev : prev - curr) > IBUS_EXT_NOISE_THR) {
                        noisy = true;
                        break;
                    }
                }
                _signal_noisy = noisy;
            }

            for (uint8_t i = IBUS_EXT_CH_START; i < IBUS_NUM_CH; i++)
                _ext_prev[i - IBUS_EXT_CH_START] = _ch[i];

            if (_pkt_count < IBUS_WARMUP_PKTS) _pkt_count++;
        }
        return got_packet;
    }

    // Channel number is 1-indexed (CH1–CH14), matching TX and GUI labeling.
    // Returns 0 on signal loss; IBUS_CH_MID for out-of-range n.
    uint16_t channel(uint8_t n) const {
        if (!alive()) return 0;
        if (n < 1 || n > IBUS_NUM_CH) return IBUS_CH_MID;
        return _ch[n - 1];
    }

    bool alive(uint32_t timeout_ms = 100) const {
        if ((millis() - _last_packet_ms) >= timeout_ms) return false;
        if (_pkt_count < IBUS_WARMUP_PKTS) return false;
        return !_sentinel_lost && !_signal_noisy;
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
    uint16_t _ext_prev[IBUS_NUM_CH - IBUS_EXT_CH_START] = {};
    bool     _sentinel_lost  = false;
    bool     _signal_noisy   = false;
    uint8_t  _pkt_count      = 0;
};
