// wifi_fast.cpp — Slack-time WiFi with double-buffered telemetry.
//
// All UDP I/O happens in the idle spin between ticks, never inside the
// control path.  A double-buffered TelemetryPacket ensures the control
// loop never stalls waiting for WiFi.

#include "wifi_fast.h"

#if USE_WIFI

#include "config.h"
#include "telemetry.h"    // TelemetryPacket struct
#include <Arduino.h>
#include <WiFiS3.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>

// ── Double-buffered telemetry ────────────────────────────────────────────────
static TelemetryPacket s_pkt[2];
static uint8_t  s_write_idx    = 0;      // buffer being written by control path
static bool     s_send_pending = false;  // a filled buffer is waiting for UDP send

// ── UDP sockets ──────────────────────────────────────────────────────────────
static WiFiUDP s_telem_udp;   // telemetry send socket
static WiFiUDP s_cmd_udp;     // command receive socket

// ── Dashboard auto-discovery ─────────────────────────────────────────────────
static IPAddress s_dashboard_ip(0, 0, 0, 0);
static bool      s_dashboard_known = false;

// ── State ────────────────────────────────────────────────────────────────────
static bool s_wifi_ok = false;

// ── Profiling ────────────────────────────────────────────────────────────────
static uint32_t s_last_send_us  = 0;
static uint32_t s_last_recv_us  = 0;
static uint32_t s_send_skips    = 0;
static uint32_t s_last_recv_tick = 0xFFFFFFFF;  // dedup repeated polls in one tick

// ── Helpers ──────────────────────────────────────────────────────────────────
static inline uint32_t slack_remaining_us(uint32_t tick_start_us) {
    uint32_t elapsed = micros() - tick_start_us;
    return (elapsed < LOOP_PERIOD_US) ? (LOOP_PERIOD_US - elapsed) : 0;
}

// ── Init ─────────────────────────────────────────────────────────────────────
bool wifi_fast_init() {
    Serial.print("[WiFi] Connecting to ");
    Serial.println(WIFI_SSID);

    for (int attempt = 0; attempt < 3; attempt++) {
        Serial.print("[WiFi] Attempt ");
        Serial.println(attempt + 1);
        if (WiFi.begin(WIFI_SSID, WIFI_PASS) == WL_CONNECTED) {
            Serial.print("[WiFi] Connected — IP: ");
            Serial.println(WiFi.localIP());

            // OTA
            ArduinoOTA.begin(WiFi.localIP(), OTA_HOSTNAME, OTA_PASSWORD,
                             InternalStorage);
            Serial.println("[OTA]  Ready");

            // Telemetry send socket (ephemeral source port)
            s_telem_udp.begin(0);

            // Command receive socket
            if (s_cmd_udp.begin(COMMAND_PORT)) {
                Serial.print("[Cmd]  Listening on UDP :");
                Serial.println(COMMAND_PORT);
            } else {
                Serial.println("[Cmd]  FAILED to bind UDP");
            }

            s_wifi_ok = true;
            Serial.println("[WiFi] Slack-time WiFi ready");
            return true;
        }
        delay(1000);
    }

    Serial.println("[WiFi] FAILED — continuing without WiFi");
    return false;
}

// ── Fill telemetry back-buffer (called from tick, no UDP) ────────────────────
void wifi_fast_fill_telemetry(const RobotState& state) {
    // If previous buffer wasn't sent, count as skip
    if (s_send_pending) {
        s_send_skips++;
    }

    // Fill current write buffer
    TelemetryPacket& pkt = s_pkt[s_write_idx];
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

    // Swap: next fill goes to other buffer; this one is ready to send
    s_write_idx ^= 1;
    s_send_pending = true;
}

// ── Slack-time UDP telemetry send ────────────────────────────────────────────
bool wifi_try_send(uint32_t tick_start_us) {
    if (!s_send_pending) return false;
    if (!s_wifi_ok || !s_dashboard_known) return false;
    if (slack_remaining_us(tick_start_us) < WIFI_SEND_MIN_SLACK_US) return false;

    uint32_t t0 = micros();

    // Send the buffer that was just filled (one behind current write_idx)
    uint8_t send_idx = s_write_idx ^ 1;
    s_telem_udp.beginPacket(s_dashboard_ip, TELEMETRY_PORT);
    s_telem_udp.write(reinterpret_cast<const uint8_t*>(&s_pkt[send_idx]),
                      sizeof(TelemetryPacket));
    s_telem_udp.endPacket();

    s_send_pending = false;
    s_last_send_us = micros() - t0;
    return true;
}

// ── Slack-time UDP command receive ───────────────────────────────────────────
bool wifi_try_receive(RobotState& state, uint32_t tick, uint32_t tick_start_us) {
    if (!s_wifi_ok) return false;
    // 10 Hz: only poll every COMMAND_RECV_DIV ticks
    if (tick % COMMAND_RECV_DIV != 0) return false;
    // Only poll once per eligible tick (idle spin calls us many times)
    if (tick == s_last_recv_tick) return false;
    if (slack_remaining_us(tick_start_us) < WIFI_RECV_MIN_SLACK_US) return false;

    s_last_recv_tick = tick;
    uint32_t t0 = micros();

    int pkt_size = s_cmd_udp.parsePacket();
    if (pkt_size < 1) {
        s_last_recv_us = micros() - t0;
        return false;
    }

    // Latch dashboard IP on first contact
    if (!s_dashboard_known) {
        s_dashboard_ip    = s_cmd_udp.remoteIP();
        s_dashboard_known = true;
        Serial.print("[Cmd]  Dashboard at ");
        Serial.println(s_dashboard_ip);
    }

    static uint8_t buf[16];
    int n = s_cmd_udp.read(buf, sizeof(buf));
    if (n < 1) {
        s_last_recv_us = micros() - t0;
        return false;
    }

    state.host_connected = true;

    // Command type IDs (must match Python dashboard)
    enum : uint8_t { CMD_DRIVE = 1, CMD_MODE = 2, CMD_GAIN = 3, CMD_PING = 4 };

    switch (buf[0]) {
    case CMD_DRIVE: {
        if (n < 13) break;
        float v, omega, hip;
        memcpy(&v,     &buf[1],  4);
        memcpy(&omega, &buf[5],  4);
        memcpy(&hip,   &buf[9],  4);
        state.v_cmd        = v;
        state.omega_cmd    = omega;
        state.hip_q_target = hip;
        break;
    }
    case CMD_MODE: {
        if (n < 2) break;
        uint8_t m = buf[1];
        if (m <= static_cast<uint8_t>(Mode::FAULT)) {
            state.mode = static_cast<Mode>(m);
            Serial.print("[Cmd]  Mode -> ");
            Serial.println(m);
        }
        break;
    }
    case CMD_GAIN: {
        if (n < 6) break;
        uint8_t gid = buf[1];
        float val;
        memcpy(&val, &buf[2], 4);
        Serial.print("[Cmd]  Gain ");
        Serial.print(gid);
        Serial.print(" -> ");
        Serial.println(val, 6);
        // TODO: apply gain by ID
        break;
    }
    case CMD_PING:
        Serial.println("[Cmd]  Ping from dashboard");
        break;
    }

    s_last_recv_us = micros() - t0;
    return true;
}

// ── OTA poll ─────────────────────────────────────────────────────────────────
void wifi_fast_ota_poll() {
    if (s_wifi_ok) {
        ArduinoOTA.poll();
    }
}

// ── Accessors ────────────────────────────────────────────────────────────────
bool     wifi_fast_connected() { return s_wifi_ok; }
uint32_t wifi_last_send_us()   { return s_last_send_us; }
uint32_t wifi_last_recv_us()   { return s_last_recv_us; }
uint32_t wifi_send_skips()     { return s_send_skips; }

#endif // USE_WIFI
