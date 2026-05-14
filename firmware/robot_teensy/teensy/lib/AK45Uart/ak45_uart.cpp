#include "ak45_uart.h"

// Precomputed frame bytes from datasheet §5.3.1.1 (pp69-70).
// Frame format: [0x02][LEN][0x14 + ASCII cmd][CRC_H][CRC_L][0x03]
// CRC algorithm: see manual page 33.
static const uint8_t CMD_ENCODER[] = {
    0x02, 0x08, 0x14,
    'e','n','c','o','d','e','r',
    0xB0, 0x4C, 0x03
};

static const uint8_t FRAME_HEADER = 0x02;
static const uint8_t FRAME_TAIL   = 0x03;
static const uint8_t COMM_PRINT   = 0x15;

// ── helpers ──────────────────────────────────────────────────────────────────

// Find needle in a null-terminated haystack; return pointer past the needle,
// or nullptr if not found.
static const char* find_after(const char* hay, const char* needle) {
    const char* p = strstr(hay, needle);
    return p ? p + strlen(needle) : nullptr;
}

// ── AK45Uart ─────────────────────────────────────────────────────────────────

AK45Uart::AK45Uart(HardwareSerial& port) : _port(port), _state{} {}

void AK45Uart::begin(uint32_t baud) {
    _port.begin(baud);
}

bool AK45Uart::poll(uint32_t timeout_ms) {
    while (_port.available()) _port.read();   // flush stale RX bytes
    _port.write(CMD_ENCODER, sizeof(CMD_ENCODER));
    _port.flush();

    // Motor sends a diagnostic "show encoder now" frame before the data frame —
    // keep reading frames until one parses or the window expires.
    uint32_t deadline = millis() + timeout_ms;
    uint8_t buf[128];
    while (millis() < deadline) {
        uint32_t remaining = deadline - millis();
        uint8_t n = read_frame(buf, sizeof(buf), remaining);
        if (n == 0) break;
        if (parse_response(buf, n)) {
            _state.ok = true;
            _state.last_fb_ms = millis();
            return true;
        }
    }
    _state.ok = false;
    return false;
}

// Read bytes until FRAME_TAIL or timeout; returns total byte count (0 = fail).
uint8_t AK45Uart::read_frame(uint8_t* buf, uint8_t maxlen, uint32_t timeout_ms) {
    uint32_t deadline = millis() + timeout_ms;
    uint8_t  n       = 0;
    bool     started = false;

    while (millis() < deadline) {
        if (!_port.available()) continue;
        uint8_t b = (uint8_t)_port.read();
        if (!started) {
            if (b == FRAME_HEADER) { started = true; buf[n++] = b; }
            continue;
        }
        if (n < maxlen) buf[n++] = b;
        if (b == FRAME_TAIL) return n;
    }
    return 0;
}

// Parse a COMM_PRINT frame whose payload is the ASCII string:
//   "E ANGLE :5.243810    M ANGLE: -19.792570    RAW: 2541"
bool AK45Uart::parse_response(const uint8_t* buf, uint8_t len) {
    // Minimum: header + len + type + some payload + crc(2) + tail = 7 bytes
    if (len < 7 || buf[2] != COMM_PRINT) return false;

    uint8_t payload_len = buf[1] - 1;              // data frame len minus type byte
    uint8_t payload_end = 3 + payload_len;
    if (payload_end + 3 > len) return false;       // incomplete frame

    // Copy payload into a null-terminated scratch buffer for string ops.
    char scratch[64];
    uint8_t copy_len = (payload_len < sizeof(scratch) - 1) ? payload_len
                                                            : sizeof(scratch) - 1;
    memcpy(scratch, &buf[3], copy_len);
    scratch[copy_len] = '\0';

    const char* p;

    p = find_after(scratch, "E ANGLE :");
    if (!p) p = find_after(scratch, "E ANGLE:");
    if (!p) return false;
    _state.e_angle_rad = strtof(p, nullptr);

    p = find_after(scratch, "M ANGLE:");
    if (!p) return false;
    _state.m_angle_rad = strtof(p, nullptr);

    p = find_after(scratch, "RAW:");
    if (!p) return false;
    _state.raw = (int16_t)strtol(p, nullptr, 10);

    return true;
}
