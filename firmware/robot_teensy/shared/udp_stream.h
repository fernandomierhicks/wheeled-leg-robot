#pragma once

#ifdef ARDUINO_ARCH_ESP32
#include <Arduino.h>
#include <WiFiUdp.h>

// Thin Stream adapter for WiFiUDP.
//
// CommLink::send() writes the entire framed packet in one write() call, so
// each call maps 1:1 to a UDP datagram — no partial-datagram buffering needed.
//
// Usage:
//   UDPStream udp;
//   udp.beginListen(5005);           // receive on port 5005
//   udp.beginSend("192.168.1.100", 5005);
//   CommLink link(udp, COMM_SRC_ESP32);
class UDPStream : public Stream {
public:
    UDPStream() {}

    void beginSend(const char* host, uint16_t port) {
        _host_valid = _host.fromString(host);
        _port = port;
    }

    void beginSend(IPAddress host, uint16_t port) {
        _host = host;
        _host_valid = true;
        _port = port;
    }

    void beginListen(uint16_t port) {
        _udp.begin(port);
    }

    size_t write(const uint8_t* buf, size_t len) override {
        if (!_host_valid) {
            _fail_count++;
            return 0;
        }
        _udp.beginPacket(_host, _port);
        size_t n = _udp.write(buf, len);
        if (!_udp.endPacket()) _fail_count++;
        return n;
    }

    size_t write(uint8_t b) override { return write(&b, 1); }

    int available() override {
        int a = _udp.available();
        return (a > 0) ? a : _udp.parsePacket();
    }

    int read()  override { return _udp.read(); }
    int peek()  override { return -1; }
    void flush() override {}

    // Count of endPacket() failures (0 = success in Arduino's WiFiUDP API).
    uint32_t failCount() const { return _fail_count; }

private:
    WiFiUDP     _udp;
    IPAddress   _host;
    bool        _host_valid = false;
    uint16_t    _port = 0;
    uint32_t    _fail_count = 0;
};

#endif // ARDUINO_ARCH_ESP32
