// wifi_fast.cpp — Slack-time WiFi with double-buffered telemetry.
//
// All UDP I/O happens in the idle spin between ticks, never inside the
// control path.  A double-buffered TelemetryPacket ensures the control
// loop never stalls waiting for WiFi.

#include "wifi_fast.h"

#if USE_WIFI

#include "config.h"
#include "telemetry.h"    // TelemetryPacket struct
#include "ak45_can.h"
#include "wheel_motors.h"
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
    pkt.hip_q_L       = state.hip_q_L;
    pkt.tau_hip_L     = state.tau_hip_L;
    pkt.tau_hip_R     = state.tau_hip_R;
    pkt.hip_q_R       = state.hip_q_R;
    pkt.dt_us         = static_cast<float>(state.dt_us);
    pkt.debug_sine    = state.debug_sine;
    pkt.wheel_pos_L   = state.wheel_pos_L;
    pkt.wheel_vel_L   = state.wheel_vel_L;
    pkt.wheel_pos_R   = state.wheel_pos_R;
    pkt.wheel_vel_R   = state.wheel_vel_R;
    pkt.status_flags  = (state.wheel_ok ? 0x01 : 0x00)
                      | (state.imu_ok   ? 0x02 : 0x00)
                      | ((state.odrive_axis_state & 0x0F) << 2)
                      | (state.odrive_axis_error ? 0x40 : 0x00);
    pkt.odrive_flags_R = (state.odrive_axis_state_R & 0x0F)
                       | (state.odrive_axis_error_R ? 0x10 : 0x00);

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

    static uint8_t buf[24];
    int n = s_cmd_udp.read(buf, sizeof(buf));
    if (n < 1) {
        s_last_recv_us = micros() - t0;
        return false;
    }

    state.host_connected = true;

    // Command type IDs (must match Python dashboard)
    enum : uint8_t {
        CMD_DRIVE = 1, CMD_MODE = 2, CMD_GAIN = 3, CMD_PING = 4, CMD_HIP = 5,
        CMD_ODRIVE_ENABLE = 7, CMD_ODRIVE_DISABLE = 8,
        CMD_ODRIVE_VEL = 9, CMD_ODRIVE_POS = 10, CMD_ODRIVE_CLEAR = 11,
        CMD_ODRIVE_TORQUE = 12,
        CMD_ODRIVE_VEL_M = 13, CMD_ODRIVE_POS_M = 14, CMD_ODRIVE_TORQUE_M = 15,
        // AK45 flat commands (motor_id in buf[1], payload from buf[2])
        CMD_HIP_VEL = 16,  // motor_id + vel_f32 [rad/s] + kd_f32
        CMD_HIP_POS = 17,  // motor_id + pos_f32 [rad] + kp_f32 + kd_f32
        CMD_HIP_MIT = 18,  // motor_id + p_f32 + v_f32 + kp_f32 + kd_f32 + t_ff_f32
    };

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
    case CMD_ODRIVE_ENABLE: {
        if (n < 2) break;
        uint8_t ctrl_mode = buf[1]; // 1=torque, 2=velocity, 3=position
        state.odrive_ctrl_mode = ctrl_mode;
        state.odrive_vel_L = state.odrive_vel_R = 0.0f;
        state.odrive_pos_L = state.odrive_pos_R = 0.0f;
        state.odrive_tau_L = state.odrive_tau_R = 0.0f;
        if      (ctrl_mode == 1) wheel_motors_set_mode(WheelMode::TORQUE);
        else if (ctrl_mode == 2) wheel_motors_set_mode(WheelMode::VELOCITY);
        else if (ctrl_mode == 3) wheel_motors_set_mode(WheelMode::POSITION);
        Serial.print("[Cmd]  WheelMode enable ctrl="); Serial.println(ctrl_mode);
        break;
    }
    case CMD_ODRIVE_DISABLE:
        state.odrive_ctrl_mode = 0;
        state.odrive_vel_L = state.odrive_vel_R = 0.0f;
        state.odrive_pos_L = state.odrive_pos_R = 0.0f;
        state.odrive_tau_L = state.odrive_tau_R = 0.0f;
        wheel_motors_set_mode(WheelMode::IDLE);
        Serial.println("[Cmd]  WheelMode disable");
        break;
    case CMD_ODRIVE_VEL:
        if (n < 5) break;
        { float v; memcpy(&v, &buf[1], 4); state.odrive_vel_L = state.odrive_vel_R = v; }
        break;
    case CMD_ODRIVE_POS:
        if (n < 5) break;
        { float p; memcpy(&p, &buf[1], 4); state.odrive_pos_L = state.odrive_pos_R = p; }
        break;
    case CMD_ODRIVE_TORQUE:
        if (n < 5) break;
        { float t; memcpy(&t, &buf[1], 4); state.odrive_tau_L = state.odrive_tau_R = t; }
        break;
    case CMD_ODRIVE_VEL_M:
        if (n < 6) break;
        { float v; memcpy(&v, &buf[2], 4);
          if (buf[1] == 0) state.odrive_vel_L = v; else state.odrive_vel_R = v; }
        break;
    case CMD_ODRIVE_POS_M:
        if (n < 6) break;
        { float p; memcpy(&p, &buf[2], 4);
          if (buf[1] == 0) state.odrive_pos_L = p; else state.odrive_pos_R = p; }
        break;
    case CMD_ODRIVE_TORQUE_M:
        if (n < 6) break;
        { float t; memcpy(&t, &buf[2], 4);
          if (buf[1] == 0) state.odrive_tau_L = t; else state.odrive_tau_R = t; }
        break;
    case CMD_ODRIVE_CLEAR:
        wheel_motors_clear_errors();
        break;
    case CMD_HIP: {
        // [CMD_HIP][motor_id][sub_cmd][...payload...]
        // motor_id: 1=L, 2=R, 3=Both
        // sub_cmd:  0=disable, 1=enable, 2=set_zero, 3=mit_raw, 4=velocity, 5=position
        if (n < 3) break;
        uint8_t mid = buf[1];
        uint8_t sub = buf[2];
        if (sub == 0) {  // disable
            if (mid == 1 || mid == 3) ak45_disable(CAN_ID_HIP_L);
            if (mid == 2 || mid == 3) ak45_disable(CAN_ID_HIP_R);
            state.hip_enabled     = false;
            state.hip_direct_mode = 0;
        } else if (sub == 1) {  // enable
            if (mid == 1 || mid == 3) ak45_enable(CAN_ID_HIP_L);
            if (mid == 2 || mid == 3) ak45_enable(CAN_ID_HIP_R);
            state.hip_enabled = true;
        } else if (sub == 2) {  // set zero
            if (mid == 1 || mid == 3) ak45_set_zero(CAN_ID_HIP_L);
            if (mid == 2 || mid == 3) ak45_set_zero(CAN_ID_HIP_R);
        } else if (sub == 3) {  // MIT raw → write state, main.cpp repeats every tick
            if (n < 23) break;
            float p, v, kp, kd, t_ff;
            memcpy(&p,    &buf[3],  4);
            memcpy(&v,    &buf[7],  4);
            memcpy(&kp,   &buf[11], 4);
            memcpy(&kd,   &buf[15], 4);
            memcpy(&t_ff, &buf[19], 4);
            state.hip_kp = kp; state.hip_kd = kd;
            state.hip_direct_mode = 3;
            if (mid == 1 || mid == 3) { state.hip_pos_L = p; state.hip_vel_L = v; state.hip_t_ff_L = t_ff; }
            if (mid == 2 || mid == 3) { state.hip_pos_R = p; state.hip_vel_R = v; state.hip_t_ff_R = t_ff; }
        } else if (sub == 4) {  // velocity mode
            if (n < 11) break;
            float vel, kd;
            memcpy(&vel, &buf[3], 4);
            memcpy(&kd,  &buf[7], 4);
            state.hip_kd = kd;
            state.hip_direct_mode = 1;
            if (mid == 1 || mid == 3) state.hip_vel_L = vel;
            if (mid == 2 || mid == 3) state.hip_vel_R = vel;
        } else if (sub == 5) {  // position mode
            if (n < 15) break;
            float pos, kp, kd;
            memcpy(&pos, &buf[3],  4);
            memcpy(&kp,  &buf[7],  4);
            memcpy(&kd,  &buf[11], 4);
            state.hip_kp = kp; state.hip_kd = kd;
            state.hip_direct_mode = 2;
            if (mid == 1 || mid == 3) state.hip_pos_L = pos;
            if (mid == 2 || mid == 3) state.hip_pos_R = pos;
        }
        break;
    }
    case CMD_HIP_VEL: {
        // [CMD_HIP_VEL][motor_id][vel_f32][kd_f32]
        if (n < 10) break;
        float vel, kd;
        memcpy(&vel, &buf[2], 4);
        memcpy(&kd,  &buf[6], 4);
        state.hip_kd = kd;
        state.hip_direct_mode = 1;
        if (buf[1] == 1 || buf[1] == 3) state.hip_vel_L = vel;
        if (buf[1] == 2 || buf[1] == 3) state.hip_vel_R = vel;
        break;
    }
    case CMD_HIP_POS: {
        // [CMD_HIP_POS][motor_id][pos_f32][kp_f32][kd_f32]
        if (n < 14) break;
        float pos, kp, kd;
        memcpy(&pos, &buf[2], 4);
        memcpy(&kp,  &buf[6], 4);
        memcpy(&kd,  &buf[10], 4);
        state.hip_kp = kp; state.hip_kd = kd;
        state.hip_direct_mode = 2;
        if (buf[1] == 1 || buf[1] == 3) state.hip_pos_L = pos;
        if (buf[1] == 2 || buf[1] == 3) state.hip_pos_R = pos;
        break;
    }
    case CMD_HIP_MIT: {
        // [CMD_HIP_MIT][motor_id][p_f32][v_f32][kp_f32][kd_f32][t_ff_f32]
        if (n < 22) break;
        float p, v, kp, kd, t_ff;
        memcpy(&p,    &buf[2],  4);
        memcpy(&v,    &buf[6],  4);
        memcpy(&kp,   &buf[10], 4);
        memcpy(&kd,   &buf[14], 4);
        memcpy(&t_ff, &buf[18], 4);
        state.hip_kp = kp; state.hip_kd = kd;
        state.hip_direct_mode = 3;
        if (buf[1] == 1 || buf[1] == 3) { state.hip_pos_L = p; state.hip_vel_L = v; state.hip_t_ff_L = t_ff; }
        if (buf[1] == 2 || buf[1] == 3) { state.hip_pos_R = p; state.hip_vel_R = v; state.hip_t_ff_R = t_ff; }
        break;
    }
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
