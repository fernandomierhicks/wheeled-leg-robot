#include "CommLink.h"

#ifdef ARDUINO
#include <Arduino.h>
#else
#include <cstring>
#endif

CommLink::CommLink(Stream& stream, uint8_t source_id)
    : _s(stream), _src(source_id), _cb(nullptr), _seq_tx(0),
      _parse_start_ms(0), _rx_drops(0)
{
    _reset_parser();
}

void CommLink::onPacket(CommPacketCb cb) {
    _cb = cb;
}

void CommLink::send(uint8_t type, uint8_t version, const void* payload, uint16_t len) {
    if (len > COMM_MAX_PAYLOAD) return;  // overflow guard — caller passed oversized payload
    uint8_t frame[10 + COMM_MAX_PAYLOAD];

    uint8_t seq    = _seq_tx++;
    uint8_t len_lo = (uint8_t)(len & 0xFF);
    uint8_t len_hi = (uint8_t)(len >> 8);

    uint8_t crc = type ^ version ^ _src ^ seq ^ len_lo ^ len_hi;
    const uint8_t* p = (const uint8_t*)payload;
    for (uint16_t i = 0; i < len; i++) crc ^= p[i];

    frame[0] = COMM_START_A;
    frame[1] = COMM_START_B;
    frame[2] = type;
    frame[3] = version;
    frame[4] = _src;
    frame[5] = seq;
    frame[6] = len_lo;
    frame[7] = len_hi;
    memcpy(frame + 8, p, len);
    frame[8 + len] = crc;
    frame[9 + len] = COMM_END;

    _s.write(frame, 10 + len);
}

void CommLink::update() {
    // Fix 1: timeout — if stuck mid-frame, abandon and count a drop.
    // Protects against Teensy reboot mid-frame, FIFO overflow tail loss, etc.
#ifdef ARDUINO
    if (_ps != PS_IDLE &&
        (uint32_t)(millis() - _parse_start_ms) > COMM_PARSE_TIMEOUT_MS) {
        ++_rx_drops;
        _reset_parser();
    }
#endif

    while (_s.available()) {
        uint8_t b = (uint8_t)_s.read();
        switch (_ps) {
            case PS_IDLE:
                if (b == COMM_START_A) {
#ifdef ARDUINO
                    _parse_start_ms = millis();  // start timeout clock
#endif
                    _ps = PS_MAGIC2;
                }
                break;
            case PS_MAGIC2:
                if (b == COMM_START_B) {
                    _ps = PS_TYPE;
                } else {
                    ++_rx_drops;
                    _reset_parser();
                    // if this byte is itself a start byte, don't discard it
                    if (b == COMM_START_A) _ps = PS_MAGIC2;
                }
                break;
            case PS_TYPE:
                _rx_type = b; _rx_crc  = b; _ps = PS_VER; break;
            case PS_VER:
                _rx_ver  = b; _rx_crc ^= b; _ps = PS_SRC; break;
            case PS_SRC:
                _rx_src  = b; _rx_crc ^= b; _ps = PS_SEQ; break;
            case PS_SEQ:
                _rx_seq  = b; _rx_crc ^= b; _ps = PS_LEN0; break;
            case PS_LEN0:
                _rx_len  = b; _rx_crc ^= b; _ps = PS_LEN1; break;
            case PS_LEN1:
                _rx_len |= ((uint16_t)b << 8);
                _rx_crc ^= b;
                _rx_idx  = 0;
                // Fix 2: length guard — a corrupted length field can lock the parser
                // in PS_PAYLOAD for up to 65535 bytes (~500 ms at 1.2 Mbaud).
                if (_rx_len > COMM_MAX_PAYLOAD) {
                    ++_rx_drops;
                    _reset_parser();
                    break;
                }
                _ps = (_rx_len == 0) ? PS_CHECKSUM : PS_PAYLOAD;
                break;
            case PS_PAYLOAD:
                if (_rx_idx < COMM_MAX_PAYLOAD) {
                    _rx_buf[_rx_idx] = b;
                    _rx_crc ^= b;
                }
                if (++_rx_idx >= _rx_len) _ps = PS_CHECKSUM;
                break;
            case PS_CHECKSUM:
                // Fix 4: count bad-checksum drops explicitly
                if (b != _rx_crc) {
                    ++_rx_drops;
                    _reset_parser();
                } else {
                    _ps = PS_END;
                }
                break;
            case PS_END:
                if (b == COMM_END) {
                    if (_cb) _cb(_rx_type, _rx_ver, _rx_src, _rx_buf, _rx_len);
                } else {
                    // Fix 4: bad END byte — frame was corrupted after checksum
                    ++_rx_drops;
                }
                _reset_parser();
                break;
        }
    }
}

void CommLink::_reset_parser() {
    _ps      = PS_IDLE;
    _rx_type = _rx_ver = _rx_src = _rx_seq = 0;
    _rx_len  = _rx_idx = _rx_crc = 0;
}
