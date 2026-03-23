// commands.cpp — UDP command receiver (10 Hz).
//
// Listens on COMMAND_PORT for binary command packets from the dashboard.
// On first packet received, latches the sender IP as the telemetry target.

#include "commands.h"
#include "config.h"
#include <Arduino.h>
#include <WiFiS3.h>
#include <WiFiUdp.h>

// Dashboard address — owned by telemetry.cpp
extern IPAddress g_dashboard_ip;
extern bool      g_dashboard_known;

// Command type IDs (must match Python dashboard)
enum CmdType : uint8_t {
    CMD_DRIVE = 1,
    CMD_MODE  = 2,
    CMD_GAIN  = 3,
    CMD_PING  = 4,
};

static WiFiUDP s_udp;
static bool    s_init_ok = false;
static uint8_t s_buf[16];  // max command size

void commands_init() {
    if (s_udp.begin(COMMAND_PORT)) {
        s_init_ok = true;
        Serial.print("[Cmd]   Listening on UDP :");
        Serial.println(COMMAND_PORT);
    } else {
        Serial.println("[Cmd]   FAILED to bind UDP");
    }
}

void commands_receive(RobotState& state) {
    if (!s_init_ok) return;

    int pkt_size = s_udp.parsePacket();
    if (pkt_size < 1) return;

    // Latch dashboard IP on first contact
    if (!g_dashboard_known) {
        g_dashboard_ip    = s_udp.remoteIP();
        g_dashboard_known = true;
        Serial.print("[Cmd]   Dashboard at ");
        Serial.println(g_dashboard_ip);
    }

    int n = s_udp.read(s_buf, sizeof(s_buf));
    if (n < 1) return;

    state.host_connected = true;

    switch (s_buf[0]) {
    case CMD_DRIVE: {
        if (n < 13) break;
        float v, omega, hip;
        memcpy(&v,     &s_buf[1],  4);
        memcpy(&omega, &s_buf[5],  4);
        memcpy(&hip,   &s_buf[9],  4);
        state.v_cmd       = v;
        state.omega_cmd   = omega;
        state.hip_q_target = hip;
        break;
    }
    case CMD_MODE: {
        if (n < 2) break;
        uint8_t m = s_buf[1];
        if (m <= static_cast<uint8_t>(Mode::FAULT)) {
            state.mode = static_cast<Mode>(m);
            Serial.print("[Cmd]   Mode → ");
            Serial.println(m);
        }
        break;
    }
    case CMD_GAIN: {
        if (n < 6) break;
        uint8_t gid = s_buf[1];
        float val;
        memcpy(&val, &s_buf[2], 4);
        Serial.print("[Cmd]   Gain ");
        Serial.print(gid);
        Serial.print(" → ");
        Serial.println(val, 6);
        // TODO: apply gain by ID once controllers are implemented
        break;
    }
    case CMD_PING:
        // Dashboard discovery — IP already latched above
        Serial.println("[Cmd]   Ping from dashboard");
        break;
    }
}
