#pragma once
#include <Arduino.h>

// AK45-10 UART driver — CubeMars MIT Serial terminal protocol (manual §5.3.1).
// Each instance owns one HardwareSerial port (standard 2-wire TX/RX).
//
// Typical wiring (from pinout.MD):
//   AK45-1 → Serial2  TX=pin8  RX=pin7
//   AK45-2 → Serial3  TX=pin14 RX=pin15

struct AK45UartState {
    float    e_angle_rad;   // encoder angle        [0 .. 2π  rad]
    float    m_angle_rad;   // mechanical angle     [unlimited rad]
    int16_t  raw;           // raw encoder count    [0 .. 16383]
    uint32_t last_fb_ms;    // millis() of last good frame
    bool     ok;            // true when last poll() succeeded
};

class AK45Uart {
public:
    explicit AK45Uart(HardwareSerial& port);

    // Open the serial port. baud defaults to 921600 (CubeMars standard).
    void begin(uint32_t baud = 921600UL);

    // Send one "encoder" request and block up to timeout_ms for a response.
    // Returns true on fresh data; populates state on success.
    bool poll(uint32_t timeout_ms = 50);

    const AK45UartState& state() const { return _state; }

private:
    HardwareSerial& _port;
    AK45UartState   _state;

    uint8_t read_frame(uint8_t* buf, uint8_t maxlen, uint32_t timeout_ms);
    bool    parse_response(const uint8_t* buf, uint8_t len);
};
