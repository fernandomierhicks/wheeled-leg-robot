#include "wheel_motors.h"
#include "config.h"
#include "param_registry.h"
#include "comm_protocol.h"  // comm_log — driver messages must be visible over WiFi too (audit W7)
#include <Arduino.h>
#include <FlexCAN_T4.h>

// ODrive CAN command IDs (5-bit, ORed into bits [4:0] of the 11-bit frame ID)
#define CMD_HEARTBEAT       0x001
#define CMD_ENCODER_EST     0x009
#define CMD_SET_AXIS_STATE  0x007
#define CMD_SET_CTRL_MODE   0x00B
#define CMD_SET_INPUT_POS   0x00C
#define CMD_SET_INPUT_VEL   0x00D
#define CMD_SET_INPUT_TRQ   0x00E
#define CMD_CLEAR_ERRORS    0x018
#define CMD_GET_VBUS        0x017

// ODrive axis state values
#define AXIS_IDLE           1u
#define AXIS_CLOSED_LOOP    8u

// ODrive control mode values
#define CTRL_TORQUE         1u
#define CTRL_VELOCITY       2u
#define CTRL_POSITION       3u

// ODrive input mode
#define INPUT_PASSTHROUGH   1u

WheelAxisState wm_L    = {};
WheelAxisState wm_R    = {};
WheelMode      wm_mode = WheelMode::IDLE;

// CAN3 on Teensy 4.1 uses pins 30 (RX) and 31 (TX) — matches config.h PIN_CAN3_*.
static FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16> can3;

// ── helpers ──────────────────────────────────────────────────────────────────

static void send_frame(uint8_t node_id, uint8_t cmd_id, const void* data, uint8_t len) {
    CAN_message_t msg = {};
    msg.id  = ((uint32_t)node_id << 5) | cmd_id;
    msg.len = len;
    memcpy(msg.buf, data, len);
    can3.write(msg);
}

// ── CAN RX callback (called from FlexCAN_T4 ISR) ─────────────────────────────

static void rx_callback(const CAN_message_t& msg) {
    uint8_t node_id = (msg.id >> 5) & 0x3F;
    uint8_t cmd_id  = msg.id & 0x1F;

    WheelAxisState* ax;
    if      (node_id == ODESC_NODE_L) ax = &wm_L;
    else if (node_id == ODESC_NODE_R) ax = &wm_R;
    else return;

    if (cmd_id == CMD_ENCODER_EST && msg.len >= 8) {
        float pos, vel;
        memcpy(&pos, msg.buf + 0, 4);
        memcpy(&vel, msg.buf + 4, 4);
        // Right motor is physically mirrored — negate so positive = robot forward.
        if (node_id == ODESC_NODE_R) { pos = -pos; vel = -vel; }
        ax->pos_turns   = pos;
        ax->vel_turns_s = vel;
        ax->last_fb_ms  = millis();
        ax->ever_heard  = true;
    } else if (cmd_id == CMD_HEARTBEAT && msg.len >= 5) {
        uint32_t err;
        memcpy(&err, msg.buf + 0, 4);
        ax->error      = err;
        ax->axis_state = msg.buf[4];
        ax->last_hb_ms = millis();
    } else if (cmd_id == CMD_GET_VBUS && msg.len >= 4) {
        memcpy(&ax->vbus, msg.buf + 0, 4);
    }
}

// ── public API ────────────────────────────────────────────────────────────────

bool wheel_motors_init() {
    can3.begin();
    can3.setBaudRate(CAN_BAUD);
    can3.setMaxMB(16);
    can3.enableFIFO();
    can3.enableFIFOInterrupt();
    can3.onReceive(rx_callback);
    comm_log(LOG_LEVEL_INFO, "WheelMotors: CAN3 init OK %lu kbps node_L=%u node_R=%u",
             (unsigned long)(CAN_BAUD / 1000), (unsigned)ODESC_NODE_L, (unsigned)ODESC_NODE_R);
    return true;
}

void wheel_motors_poll() {
    // FlexCAN_T4 interrupt-driven — no polling call needed.
    uint32_t now = millis();
    uint32_t enc_timeout = (uint32_t)param_get(PARAM_WM_ENC_TIMEOUT_MS);
    // ever_heard guard: without it, ok is spuriously true for the first
    // enc_timeout ms after boot (last_fb_ms == 0) even with no ODrive present.
    wm_L.ok = wm_L.ever_heard && (now - wm_L.last_fb_ms) < enc_timeout;
    wm_R.ok = wm_R.ever_heard && (now - wm_R.last_fb_ms) < enc_timeout;

    bool l_bad = (param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f) && (!wm_L.ok || wm_L.error);
    bool r_bad = (param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f) && (!wm_R.ok || wm_R.error);
    bool fault = (l_bad || r_bad);
    if (fault && wm_mode != WheelMode::IDLE) {
        comm_log(LOG_LEVEL_ERROR, "WheelMotors FAULT -> IDLE  L(ok=%d err=0x%lX)  R(ok=%d err=0x%lX)",
                 (int)wm_L.ok, (unsigned long)wm_L.error, (int)wm_R.ok, (unsigned long)wm_R.error);
        wheel_motors_set_mode(WheelMode::IDLE);
    }
}

void wheel_motors_set_mode(WheelMode mode) {
    bool l_en = param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f;
    bool r_en = param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f;
    if (mode == WheelMode::IDLE) {
        uint32_t s = AXIS_IDLE;
        if (l_en) send_frame(ODESC_NODE_L, CMD_SET_AXIS_STATE, &s, 4);
        delayMicroseconds(CAN_INTER_FRAME_US);
        if (r_en) send_frame(ODESC_NODE_R, CMD_SET_AXIS_STATE, &s, 4);
    } else {
        uint32_t ctrl;
        switch (mode) {
            case WheelMode::VELOCITY: ctrl = CTRL_VELOCITY; break;
            case WheelMode::POSITION: ctrl = CTRL_POSITION; break;
            case WheelMode::TORQUE:
            default:                  ctrl = CTRL_TORQUE;   break;
        }
        uint32_t ctrl_data[2] = { ctrl, INPUT_PASSTHROUGH };
        if (l_en) send_frame(ODESC_NODE_L, CMD_SET_CTRL_MODE, ctrl_data, 8);
        delayMicroseconds(CAN_INTER_FRAME_US);
        if (r_en) send_frame(ODESC_NODE_R, CMD_SET_CTRL_MODE, ctrl_data, 8);
        delayMicroseconds(CAN_INTER_FRAME_US);
        uint32_t s = AXIS_CLOSED_LOOP;
        if (l_en) send_frame(ODESC_NODE_L, CMD_SET_AXIS_STATE, &s, 4);
        delayMicroseconds(CAN_INTER_FRAME_US);
        if (r_en) send_frame(ODESC_NODE_R, CMD_SET_AXIS_STATE, &s, 4);
    }
    wm_mode = mode;
}

void wheel_motors_send(float L, float R) {
    // Right motor is physically mirrored — apply sign convention once here.
    float L_hw =  L;
    float R_hw = -R;
    bool l_en = param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f;
    bool r_en = param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f;

    switch (wm_mode) {
        case WheelMode::IDLE:
            break;

        case WheelMode::VELOCITY: {
            // rad/s → turns/s
            float vel_L[2] = { L_hw / TWO_PI, 0.0f };
            float vel_R[2] = { R_hw / TWO_PI, 0.0f };
            if (l_en) send_frame(ODESC_NODE_L, CMD_SET_INPUT_VEL, vel_L, 8);
            delayMicroseconds(CAN_INTER_FRAME_US);
            if (r_en) send_frame(ODESC_NODE_R, CMD_SET_INPUT_VEL, vel_R, 8);
            break;
        }

        case WheelMode::POSITION: {
            // rad → turns; vel_ff and torque_ff both zero
            float pos_L = L_hw / TWO_PI;
            float pos_R = R_hw / TWO_PI;
            uint8_t buf[8] = {};
            if (l_en) {
                memcpy(buf + 0, &pos_L, 4);
                send_frame(ODESC_NODE_L, CMD_SET_INPUT_POS, buf, 8);
            }
            delayMicroseconds(CAN_INTER_FRAME_US);
            if (r_en) {
                memcpy(buf + 0, &pos_R, 4);
                send_frame(ODESC_NODE_R, CMD_SET_INPUT_POS, buf, 8);
            }
            break;
        }

        case WheelMode::TORQUE: {
            if (l_en) send_frame(ODESC_NODE_L, CMD_SET_INPUT_TRQ, &L_hw, 4);
            delayMicroseconds(CAN_INTER_FRAME_US);
            if (r_en) send_frame(ODESC_NODE_R, CMD_SET_INPUT_TRQ, &R_hw, 4);
            break;
        }
    }
}

void wheel_motors_pet_watchdog() {
    // 50 Hz divider (audit D4): petting at the full 500 Hz call rate put a
    // standing ~1000 frames/s (~13% of CAN3) on the bus for no benefit — the
    // ODrive watchdog only needs a frame well inside its timeout window.
    // Coverage in the other modes: TORQUE is petted by the control loop's
    // unconditional wheel_motors_send(); VELOCITY/POSITION are petted by the
    // 50 Hz wheel_motors_request_vbus() poll (any CAN frame addressed to the
    // axis feeds the ODrive watchdog) — a zero-vel keepalive there would
    // stomp the GUI's live setpoint, so IDLE keeps the only explicit pet.
    static uint8_t div = 0;
    if (++div < 10) return;
    div = 0;
    if (wm_mode == WheelMode::IDLE) {
        float zero[2] = { 0.0f, 0.0f };
        if (param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f) send_frame(ODESC_NODE_L, CMD_SET_INPUT_VEL, zero, 8);
        delayMicroseconds(CAN_INTER_FRAME_US);
        if (param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f) send_frame(ODESC_NODE_R, CMD_SET_INPUT_VEL, zero, 8);
    }
}

void wheel_motors_request_vbus() {
    CAN_message_t msg = {};
    msg.flags.remote = 1;
    msg.len = 8;
    if (param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f) {
        msg.id = ((uint32_t)ODESC_NODE_L << 5) | CMD_GET_VBUS;
        can3.write(msg);
    }
    delayMicroseconds(CAN_INTER_FRAME_US);
    if (param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f) {
        msg.id = ((uint32_t)ODESC_NODE_R << 5) | CMD_GET_VBUS;
        can3.write(msg);
    }
}

void wheel_motors_clear_errors() {
    uint32_t ident = 0;
    if (param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f) send_frame(ODESC_NODE_L, CMD_CLEAR_ERRORS, &ident, 4);
    delayMicroseconds(CAN_INTER_FRAME_US);
    if (param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f) send_frame(ODESC_NODE_R, CMD_CLEAR_ERRORS, &ident, 4);
    wm_L.error = 0;
    wm_R.error = 0;
    comm_log(LOG_LEVEL_INFO, "WheelMotors: clear_errors sent");
}

bool wheel_motors_ok() {
    bool l_ok = (param_get(PARAM_WHEEL_L_ENABLE) < 0.5f) || (wm_L.ok && !wm_L.error);
    bool r_ok = (param_get(PARAM_WHEEL_R_ENABLE) < 0.5f) || (wm_R.ok && !wm_R.error);
    return l_ok && r_ok;
}
