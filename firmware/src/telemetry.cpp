// telemetry.cpp — UDP + Serial telemetry sender (50 Hz).
//
// Sends a 65-byte binary packet to the dashboard PC via UDP and/or Serial.
// UDP: dashboard IP is learned automatically when dashboard sends first
//      command/ping to COMMAND_PORT.  Until then, no UDP telemetry is sent.
// Serial: always sent with framing [0xAA][0x55][65 bytes][XOR checksum].

#include "telemetry.h"
#include "config.h"
#include <Arduino.h>

#if !USE_WIFI
// USB-UART mode: telemetry over Serial only.  No WiFi/UDP.

void telemetry_init() {
    Serial.println("[Telem] Serial-only mode (USB-UART)");
}
#else
// WiFi mode: telemetry send is handled by wifi_fast.cpp.
// telemetry_init/telemetry_send are unused but kept for reference.
#include <WiFiS3.h>
#include <WiFiUdp.h>

static WiFiUDP s_udp;
static bool    s_init_ok = false;

// Dashboard address — owned by wifi_fast.cpp in WiFi mode.
extern IPAddress g_dashboard_ip;
extern bool      g_dashboard_known;

void telemetry_init() {
    s_udp.begin(0);
    s_init_ok = true;
    Serial.println("[Telem] Ready (waiting for dashboard)");
}
#endif

void telemetry_send(const RobotState& state) {
    TelemetryPacket pkt;
    pkt.timestamp_ms  = millis();
    pkt.mode          = static_cast<uint8_t>(state.mode);
    pkt.pitch         = state.pitch;
    pkt.pitch_rate    = state.pitch_rate;
    pkt.roll          = state.roll;
    pkt.yaw           = state.yaw;
    pkt.wheel_vel_avg = state.wheel_vel_avg;
    pkt.v_cmd         = state.v_cmd;
    pkt.theta_ref     = state.theta_ref;
    pkt.tau_sym       = state.tau_sym;
    pkt.tau_yaw       = state.tau_yaw;
    pkt.tau_wheel_L   = state.tau_wheel_L;
    pkt.tau_wheel_R   = state.tau_wheel_R;
    pkt.hip_q_avg     = state.hip_q_avg;
    pkt.tau_hip_L     = state.tau_hip_L;
    pkt.tau_hip_R     = state.tau_hip_R;
    pkt.dt_us         = static_cast<float>(state.dt_us);
    pkt.debug_sine    = state.debug_sine;

#if USE_WIFI
    // UDP send handled by wifi_fast.cpp in slack time — not here.
    if (s_init_ok && g_dashboard_known) {
        s_udp.beginPacket(g_dashboard_ip, TELEMETRY_PORT);
        s_udp.write(reinterpret_cast<const uint8_t*>(&pkt), sizeof(pkt));
        s_udp.endPacket();
    }
#endif

    // ── Serial (framed binary: [0xAA][0x55][packet][XOR checksum]) ──
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(&pkt);
    uint8_t xor_ck = 0;
    for (size_t i = 0; i < sizeof(pkt); i++) xor_ck ^= raw[i];
    static const uint8_t sync[2] = {0xAA, 0x55};
    Serial.write(sync, 2);
    Serial.write(raw, sizeof(pkt));
    Serial.write(xor_ck);
}
