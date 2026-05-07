#pragma once
#include <stdint.h>
#include "comm_protocol.h"

#ifndef ARDUINO
// Minimal Stream stub so this header compiles on native (unit-test) builds
#include <cstring>
class Stream {
public:
    virtual int    available()                          = 0;
    virtual int    read()                               = 0;
    virtual size_t write(const uint8_t* buf, size_t n) = 0;
    virtual size_t write(uint8_t b) { return write(&b, 1); }
    virtual void   flush() {}
    virtual        ~Stream() {}
};
#else
#include <Stream.h>
#endif

#ifndef COMM_MAX_PAYLOAD
#define COMM_MAX_PAYLOAD 128
#endif

typedef void (*CommPacketCb)(uint8_t type, uint8_t version, uint8_t source,
                             const uint8_t* payload, uint16_t len);

class CommLink {
public:
    CommLink(Stream& stream, uint8_t source_id);

    // Assemble and transmit one framed packet. Full frame is written in a
    // single write() call so UDPStream can wrap it in one datagram.
    void send(uint8_t type, uint8_t version, const void* payload, uint16_t len);

    // Drive the receive parser — call every main-loop iteration.
    void update();

    // Register callback invoked on every fully-validated received packet.
    void onPacket(CommPacketCb cb);

private:
    Stream&      _s;
    uint8_t      _src;
    CommPacketCb _cb;
    uint8_t      _seq_tx;

    enum ParserState : uint8_t {
        PS_IDLE, PS_TYPE, PS_VER, PS_SRC, PS_SEQ,
        PS_LEN0, PS_LEN1, PS_PAYLOAD, PS_CHECKSUM, PS_END
    };

    ParserState _ps;
    uint8_t     _rx_type, _rx_ver, _rx_src, _rx_seq;
    uint16_t    _rx_len, _rx_idx;
    uint8_t     _rx_crc;
    uint8_t     _rx_buf[COMM_MAX_PAYLOAD];

    void _reset_parser();
};
