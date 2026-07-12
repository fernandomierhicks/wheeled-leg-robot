#include "wheel_motors.h"
#include "config.h"
#include "param_registry.h"
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
    Serial.print("[WheelMotors] CAN3 init OK  ");
    Serial.print(CAN_BAUD / 1000);
    Serial.print(" kbps  node_L=");
    Serial.print(ODESC_NODE_L);
    Serial.print("  node_R=");
    Serial.println(ODESC_NODE_R);
    return true;
}

void wheel_motors_poll() {
    // FlexCAN_T4 interrupt-driven — no polling call needed.
    uint32_t now = millis();
    uint32_t enc_timeout = (uint32_t)param_get(PARAM_WM_ENC_TIMEOUT_MS);
    wm_L.ok = (now - wm_L.last_fb_ms) < enc_timeout;
    wm_R.ok = (now - wm_R.last_fb_ms) < enc_timeout;

    bool fault = (!wm_L.ok || !wm_R.ok || wm_L.error || wm_R.error);
    if (fault && wm_mode != WheelMode::IDLE) {
        Serial.print("[WheelMotors] FAULT → IDLE");
        if (!wm_L.ok)   Serial.print("  L_timeout");
        if (!wm_R.ok)   Serial.print("  R_timeout");
        if (wm_L.error) { Serial.print("  L_err=0x"); Serial.print(wm_L.error, HEX); }
        if (wm_R.error) { Serial.print("  R_err=0x"); Serial.print(wm_R.error, HEX); }
        Serial.println();
        wheel_motors_set_mode(WheelMode::IDLE);
    }
}

void wheel_motors_set_mode(WheelMode mode) {
    if (mode == WheelMode::IDLE) {
        uint32_t s = AXIS_IDLE;
        send_frame(ODESC_NODE_L, CMD_SET_AXIS_STATE, &s, 4);
        delayMicroseconds(CAN_INTER_FRAME_US);
        send_frame(ODESC_NODE_R, CMD_SET_AXIS_STATE, &s, 4);
    } else {
        uint32_t ctrl;
        switch (mode) {
            case WheelMode::VELOCITY: ctrl = CTRL_VELOCITY; break;
            case WheelMode::POSITION: ctrl = CTRL_POSITION; break;
            case WheelMode::TORQUE:
            default:                  ctrl = CTRL_TORQUE;   break;
        }
        uint32_t ctrl_data[2] = { ctrl, INPUT_PASSTHROUGH };
        send_frame(ODESC_NODE_L, CMD_SET_CTRL_MODE, ctrl_data, 8);
        delayMicroseconds(CAN_INTER_FRAME_US);
        send_frame(ODESC_NODE_R, CMD_SET_CTRL_MODE, ctrl_data, 8);
        delayMicroseconds(CAN_INTER_FRAME_US);
        uint32_t s = AXIS_CLOSED_LOOP;
        send_frame(ODESC_NODE_L, CMD_SET_AXIS_STATE, &s, 4);
        delayMicroseconds(CAN_INTER_FRAME_US);
        send_frame(ODESC_NODE_R, CMD_SET_AXIS_STATE, &s, 4);
    }
    wm_mode = mode;
}

void wheel_motors_send(float L, float R) {
    // Right motor is physically mirrored — apply sign convention once here.
    float L_hw =  L;
    float R_hw = -R;

    switch (wm_mode) {
        case WheelMode::IDLE:
            break;

        case WheelMode::VELOCITY: {
            // rad/s → turns/s
            float vel_L[2] = { L_hw / TWO_PI, 0.0f };
            float vel_R[2] = { R_hw / TWO_PI, 0.0f };
            send_frame(ODESC_NODE_L, CMD_SET_INPUT_VEL, vel_L, 8);
            delayMicroseconds(CAN_INTER_FRAME_US);
            send_frame(ODESC_NODE_R, CMD_SET_INPUT_VEL, vel_R, 8);
            break;
        }

        case WheelMode::POSITION: {
            // rad → turns; vel_ff and torque_ff both zero
            float pos_L = L_hw / TWO_PI;
            float pos_R = R_hw / TWO_PI;
            uint8_t buf[8] = {};
            memcpy(buf + 0, &pos_L, 4);
            send_frame(ODESC_NODE_L, CMD_SET_INPUT_POS, buf, 8);
            delayMicroseconds(CAN_INTER_FRAME_US);
            memcpy(buf + 0, &pos_R, 4);
            send_frame(ODESC_NODE_R, CMD_SET_INPUT_POS, buf, 8);
            break;
        }

        case WheelMode::TORQUE: {
            send_frame(ODESC_NODE_L, CMD_SET_INPUT_TRQ, &L_hw, 4);
            delayMicroseconds(CAN_INTER_FRAME_US);
            send_frame(ODESC_NODE_R, CMD_SET_INPUT_TRQ, &R_hw, 4);
            break;
        }
    }
}

void wheel_motors_pet_watchdog() {
    if (wm_mode == WheelMode::IDLE) {
        float zero[2] = { 0.0f, 0.0f };
        send_frame(ODESC_NODE_L, CMD_SET_INPUT_VEL, zero, 8);
        delayMicroseconds(CAN_INTER_FRAME_US);
        send_frame(ODESC_NODE_R, CMD_SET_INPUT_VEL, zero, 8);
    }
}

void wheel_motors_request_vbus() {
    CAN_message_t msg = {};
    msg.flags.remote = 1;
    msg.len = 8;
    msg.id = ((uint32_t)ODESC_NODE_L << 5) | CMD_GET_VBUS;
    can3.write(msg);
    delayMicroseconds(CAN_INTER_FRAME_US);
    msg.id = ((uint32_t)ODESC_NODE_R << 5) | CMD_GET_VBUS;
    can3.write(msg);
}

void wheel_motors_clear_errors() {
    uint32_t ident = 0;
    send_frame(ODESC_NODE_L, CMD_CLEAR_ERRORS, &ident, 4);
    delayMicroseconds(CAN_INTER_FRAME_US);
    send_frame(ODESC_NODE_R, CMD_CLEAR_ERRORS, &ident, 4);
    wm_L.error = 0;
    wm_R.error = 0;
    Serial.println("[WheelMotors] clear_errors sent");
}
