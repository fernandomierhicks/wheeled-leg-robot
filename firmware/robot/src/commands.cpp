// commands.cpp — UDP command receiver (10 Hz).
//
// Listens on COMMAND_PORT for binary command packets from the dashboard.
// On first packet received, latches the sender IP as the telemetry target.

#include "commands.h"
#include "config.h"
#include "odesc_can.h"
#include <Arduino.h>

#if USE_WIFI
// WiFi mode: command receive is handled by wifi_fast.cpp.
// This file is only compiled for USB-UART fallback.
#include <WiFiS3.h>
#include <WiFiUdp.h>

// Dashboard address — owned by wifi_fast.cpp in WiFi mode
extern IPAddress g_dashboard_ip;
extern bool      g_dashboard_known;
#endif

#if !USE_WIFI
// ── USB-UART mode: binary serial command parser ───────────────────────────
//
// Frame format (dashboard → firmware):
//   [0xBB][0xCC][cmd_type][payload...][XOR checksum]
//   XOR = cmd_type ^ payload[0] ^ payload[1] ^ ...
//
// Commands:
//   CMD_PING  (4): no payload
//   CMD_MODE  (2): 1 byte  (mode enum)
//   CMD_DRIVE (1): 12 bytes (float v, float omega, float hip_target)

static constexpr uint8_t SERIAL_SYNC1 = 0xBB;
static constexpr uint8_t SERIAL_SYNC2 = 0xCC;

// Command type IDs (must match Python dashboard)
enum CmdType : uint8_t {
    CMD_DRIVE         = 1,
    CMD_MODE          = 2,
    CMD_GAIN          = 3,
    CMD_PING          = 4,
    CMD_HIP           = 5,
    CMD_ODRIVE_ENABLE = 7,   // 1 byte: ctrl_mode (2=velocity, 3=position)
    CMD_ODRIVE_DISABLE= 8,   // no payload
    CMD_ODRIVE_VEL    = 9,   // 1 float: velocity [turns/s] for both axes
    CMD_ODRIVE_POS    = 10,  // 1 float: position [turns] for both axes
};

static uint8_t payload_bytes(uint8_t cmd) {
    switch (cmd) {
        case CMD_PING:           return 0;
        case CMD_ODRIVE_DISABLE: return 0;
        case CMD_MODE:           return 1;
        case CMD_ODRIVE_ENABLE:  return 1;  // ctrl_mode byte
        case CMD_DRIVE:          return 12; // 3 × float32
        case CMD_ODRIVE_VEL:     return 4;  // 1 × float32
        case CMD_ODRIVE_POS:     return 4;  // 1 × float32
        default:                 return 0xFF; // unknown — discard
    }
}

static void process_cmd(uint8_t cmd, const uint8_t* buf, uint8_t len, RobotState& state) {
    state.host_connected = true;
    switch (cmd) {
    case CMD_PING:
        Serial.println("[Cmd]  Ping");
        break;
    case CMD_MODE: {
        uint8_t m = buf[0];
        if (m <= static_cast<uint8_t>(Mode::FAULT)) {
            state.mode = static_cast<Mode>(m);
            Serial.print("[Cmd]  Mode -> ");
            Serial.println(m);
        }
        break;
    }
    case CMD_DRIVE: {
        if (len < 12) break;
        float v, omega, hip;
        memcpy(&v,     buf,     4);
        memcpy(&omega, buf + 4, 4);
        memcpy(&hip,   buf + 8, 4);
        state.v_cmd        = v;
        state.omega_cmd    = omega;
        state.hip_q_target = hip;
        break;
    }
    case CMD_ODRIVE_ENABLE: {
        uint8_t ctrl_mode = buf[0]; // 2=velocity, 3=position
        state.odrive_ctrl_mode = ctrl_mode;
        state.odrive_vel_cmd   = 0.0f;
        state.odrive_pos_cmd   = 0.0f;
        if (ctrl_mode == 2) {
            odesc_can_enable_velocity();
        } else if (ctrl_mode == 3) {
            odesc_can_enable_position();
        }
        Serial.print("[Cmd]  ODrive enable ctrl_mode=");
        Serial.println(ctrl_mode);
        break;
    }
    case CMD_ODRIVE_DISABLE:
        state.odrive_ctrl_mode = 0;
        state.odrive_vel_cmd   = 0.0f;
        state.odrive_pos_cmd   = 0.0f;
        odesc_can_disable();
        Serial.println("[Cmd]  ODrive disable");
        break;
    case CMD_ODRIVE_VEL: {
        if (len < 4) break;
        float vel;
        memcpy(&vel, buf, 4);
        state.odrive_vel_cmd = vel;
        Serial.print("[Cmd]  ODrive vel=");
        Serial.println(vel, 3);
        break;
    }
    case CMD_ODRIVE_POS: {
        if (len < 4) break;
        float pos;
        memcpy(&pos, buf, 4);
        state.odrive_pos_cmd = pos;
        Serial.print("[Cmd]  ODrive pos=");
        Serial.println(pos, 3);
        break;
    }
    default:
        break;
    }
}

static uint8_t s_parse_state = 0;  // 0=sync1, 1=sync2, 2=cmd, 3=payload, 4=checksum
static uint8_t s_cmd_type   = 0;
static uint8_t s_payload[16];
static uint8_t s_payload_pos = 0;
static uint8_t s_payload_len = 0;

void commands_init() {
    Serial.println("[Cmd]  USB-UART serial command receiver ready (0xBB 0xCC framing)");
}

void commands_receive(RobotState& state) {
    while (Serial.available()) {
        uint8_t b = (uint8_t)Serial.read();
        switch (s_parse_state) {
        case 0:  // wait for sync1
            if (b == SERIAL_SYNC1) s_parse_state = 1;
            break;
        case 1:  // wait for sync2
            if      (b == SERIAL_SYNC2) s_parse_state = 2;
            else if (b == SERIAL_SYNC1) s_parse_state = 1;  // could be 0xBB 0xBB...
            else                        s_parse_state = 0;
            break;
        case 2: {  // read command type
            s_cmd_type   = b;
            s_payload_len = payload_bytes(b);
            s_payload_pos = 0;
            if (s_payload_len == 0xFF) { s_parse_state = 0; break; }
            s_parse_state = (s_payload_len == 0) ? 4 : 3;
            break;
        }
        case 3:  // read payload bytes
            if (s_payload_pos < sizeof(s_payload))
                s_payload[s_payload_pos] = b;
            s_payload_pos++;
            if (s_payload_pos >= s_payload_len) s_parse_state = 4;
            break;
        case 4: {  // read + verify XOR checksum
            uint8_t xor_ck = s_cmd_type;
            for (uint8_t i = 0; i < s_payload_len; i++) xor_ck ^= s_payload[i];
            if (xor_ck == b)
                process_cmd(s_cmd_type, s_payload, s_payload_len, state);
            s_parse_state = 0;
            break;
        }
        }
    }
}

#else
// ── WiFi mode: UDP command receiver (legacy path, unused when wifi_fast active)

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
            Serial.print("[Cmd]   Mode -> ");
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
        Serial.print(" -> ");
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
#endif // USE_WIFI
