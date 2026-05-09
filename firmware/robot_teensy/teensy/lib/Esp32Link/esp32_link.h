#pragma once
#include "CommLink.h"
#include "comm_protocol.h"

class Esp32Link {
public:
    Esp32Link();
    void     begin();
    void     send_telemetry(const TelemetryPayload& t);
    void     update();
    void     onCommand(CommPacketCb cb);
    uint32_t tx_count() const { return _tx_count; }

private:
    CommLink _link;
    uint32_t _tx_count;
};

extern Esp32Link g_esp32;
