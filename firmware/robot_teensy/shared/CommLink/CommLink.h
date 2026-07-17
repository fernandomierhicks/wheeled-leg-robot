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
// Must be >= sizeof(TelemetryPayload).  See propagation checklist in comm_protocol.h.
// Current V5 payload = 244 bytes.  Bump this BEFORE growing any payload struct.
#define COMM_MAX_PAYLOAD 512
#endif
#ifdef __cplusplus
static_assert(COMM_MAX_PAYLOAD >= sizeof(TelemetryPayload),
    "COMM_MAX_PAYLOAD is smaller than TelemetryPayload — CommLink::send() would overflow its stack buffer");
static_assert(LOG_CHUNK_DATA + sizeof(LogDataHeader) <= COMM_MAX_PAYLOAD,
    "LOG_DATA frame exceeds COMM_MAX_PAYLOAD");
#endif

// If a frame start is seen but the frame does not complete within this many ms, the
// parser resets and increments rx_drops().  Must be > (frame_bytes * 10 / baud_bps * 1000)
// for the slowest expected baud.  At 1.2 Mbaud a max frame (137 B) takes ~1.1 ms; 10 ms
// gives ~9× margin while recovering within one 2 ms telemetry interval.
#ifndef COMM_PARSE_TIMEOUT_MS
#define COMM_PARSE_TIMEOUT_MS 10
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

    // Count of frames discarded due to bad checksum, bad framing, length overflow, or timeout.
    uint32_t rx_drops() const { return _rx_drops; }

    // Total frames lost on this link, from seq discontinuities between
    // consecutive valid frames (audit W3). Never reset.
    uint32_t rx_seq_gaps() const { return _rx_seq_gaps; }

    // Seq of the frame currently being delivered — only meaningful when read
    // from inside the onPacket callback (which runs synchronously in update()).
    uint8_t last_rx_seq() const { return _rx_seq; }

private:
    Stream&      _s;
    uint8_t      _src;
    CommPacketCb _cb;
    uint8_t      _seq_tx;

    enum ParserState : uint8_t {
        PS_IDLE, PS_MAGIC2, PS_TYPE, PS_VER, PS_SRC, PS_SEQ,
        PS_LEN0, PS_LEN1, PS_PAYLOAD, PS_CHECKSUM, PS_END
    };

    ParserState _ps;
    uint8_t     _rx_type, _rx_ver, _rx_src, _rx_seq;
    uint16_t    _rx_len, _rx_idx;
    uint8_t     _rx_crc;
    uint8_t     _rx_buf[COMM_MAX_PAYLOAD];
    uint32_t    _parse_start_ms;  // millis() when current frame start was seen
    uint32_t    _rx_drops;        // total frames discarded (never reset)
    uint32_t    _rx_seq_gaps;     // total frames lost per seq discontinuities (never reset)
    uint8_t     _rx_last_seq;     // seq of the previous valid frame
    bool        _rx_seq_valid;    // _rx_last_seq holds a real value

    void _reset_parser();         // resets state machine; does NOT touch _rx_drops/_rx_seq_gaps
};
