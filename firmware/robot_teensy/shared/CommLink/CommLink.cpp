#include "CommLink.h"

#ifdef ARDUINO
#include <Arduino.h>
#else
#include <cstring>
#endif

CommLink::CommLink(Stream& stream, uint8_t source_id)
    : _s(stream), _src(source_id), _cb(nullptr), _seq_tx(0)
{
    _reset_parser();
}

void CommLink::onPacket(CommPacketCb cb) {
    _cb = cb;
}

void CommLink::send(uint8_t type, uint8_t version, const void* payload, uint16_t len) {
    uint8_t frame[9 + COMM_MAX_PAYLOAD];

    uint8_t seq    = _seq_tx++;
    uint8_t len_lo = (uint8_t)(len & 0xFF);
    uint8_t len_hi = (uint8_t)(len >> 8);

    uint8_t crc = type ^ version ^ _src ^ seq ^ len_lo ^ len_hi;
    const uint8_t* p = (const uint8_t*)payload;
    for (uint16_t i = 0; i < len; i++) crc ^= p[i];

    frame[0] = COMM_START;
    frame[1] = type;
    frame[2] = version;
    frame[3] = _src;
    frame[4] = seq;
    frame[5] = len_lo;
    frame[6] = len_hi;
    memcpy(frame + 7, p, len);
    frame[7 + len] = crc;
    frame[8 + len] = COMM_END;

    _s.write(frame, 9 + len);
}

void CommLink::update() {
    while (_s.available()) {
        uint8_t b = (uint8_t)_s.read();
        switch (_ps) {
            case PS_IDLE:
                if (b == COMM_START) _ps = PS_TYPE;
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
                _ps = (b == _rx_crc) ? PS_END : PS_IDLE;
                break;
            case PS_END:
                if (b == COMM_END && _cb)
                    _cb(_rx_type, _rx_ver, _rx_src, _rx_buf, _rx_len);
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
