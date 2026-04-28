#include "wheel_motors.h"
#include "config.h"
#include <Arduino.h>

WheelAxisState wm_L   = {};
WheelAxisState wm_R   = {};
WheelMode      wm_mode = WheelMode::IDLE;

// ── Real CAN implementation ────────────────────────────────────────────────
#ifndef NO_CAN

#include <Arduino_CAN.h>
#include <ODriveCAN.h>
#include <ODriveHardwareCAN.hpp>

static ODriveCAN odrive_L(wrap_can_intf(CAN), ODESC_NODE_L);
static ODriveCAN odrive_R(wrap_can_intf(CAN), ODESC_NODE_R);

// ── Required by ODriveHardwareCAN: dispatch incoming frames by node ID ──────

void onCanMessage(const CanMsg& msg) {
    uint8_t node_id = (msg.id >> 5) & 0x3F;
    if      (node_id == ODESC_NODE_R) onReceive(msg, odrive_R);
    else if (node_id == ODESC_NODE_L) onReceive(msg, odrive_L);
}

// ── Encoder feedback callbacks ──────────────────────────────────────────────

static void on_feedback_L(Get_Encoder_Estimates_msg_t& msg, void*) {
    wm_L.pos_rad    = msg.Pos_Estimate * TWO_PI;
    wm_L.vel_rad_s  = msg.Vel_Estimate * TWO_PI;
    wm_L.last_fb_ms = millis();
}

static void on_feedback_R(Get_Encoder_Estimates_msg_t& msg, void*) {
    wm_R.pos_rad    = msg.Pos_Estimate * TWO_PI;
    wm_R.vel_rad_s  = msg.Vel_Estimate * TWO_PI;
    wm_R.last_fb_ms = millis();
}

// ── Heartbeat callbacks ─────────────────────────────────────────────────────

static void on_heartbeat_L(Heartbeat_msg_t& msg, void*) {
    wm_L.error      = msg.Axis_Error;
    wm_L.axis_state = (uint8_t)msg.Axis_State;
    wm_L.last_hb_ms = millis();
}

static void on_heartbeat_R(Heartbeat_msg_t& msg, void*) {
    wm_R.error      = msg.Axis_Error;
    wm_R.axis_state = (uint8_t)msg.Axis_State;
    wm_R.last_hb_ms = millis();
}

// ── Public API ──────────────────────────────────────────────────────────────

bool wheel_motors_init() {
    if (!CAN.begin(CAN_BAUD)) {
        Serial.println("[WheelMotors] ERROR: CAN.begin() failed");
        return false;
    }
    odrive_L.onFeedback(on_feedback_L, nullptr);
    odrive_R.onFeedback(on_feedback_R, nullptr);
    odrive_L.onStatus(on_heartbeat_L, nullptr);
    odrive_R.onStatus(on_heartbeat_R, nullptr);
    Serial.print("[WheelMotors] CAN init OK  ");
    Serial.print(CAN_BAUD / 1000);
    Serial.print(" kbps  node_L=");
    Serial.print(ODESC_NODE_L);
    Serial.print("  node_R=");
    Serial.println(ODESC_NODE_R);
    return true;
}

void wheel_motors_poll() {
    pumpEvents(CAN);

    uint32_t now = millis();
    wm_L.ok = (now - wm_L.last_fb_ms) < CAN_TIMEOUT_MS;
    wm_R.ok = (now - wm_R.last_fb_ms) < CAN_TIMEOUT_MS;

    // Auto-IDLE on any CAN fault (timeout or ODrive error)
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
        odrive_L.setState(AXIS_STATE_IDLE);
        delayMicroseconds(CAN_INTER_FRAME_US);
        odrive_R.setState(AXIS_STATE_IDLE);
    } else {
        uint8_t ctrl;
        switch (mode) {
            case WheelMode::VELOCITY: ctrl = CONTROL_MODE_VELOCITY_CONTROL; break;
            case WheelMode::POSITION: ctrl = CONTROL_MODE_POSITION_CONTROL; break;
            case WheelMode::TORQUE:
            default:                  ctrl = CONTROL_MODE_TORQUE_CONTROL;   break;
        }
        odrive_L.setControllerMode(ctrl, INPUT_MODE_PASSTHROUGH);
        delayMicroseconds(CAN_INTER_FRAME_US);
        odrive_R.setControllerMode(ctrl, INPUT_MODE_PASSTHROUGH);
        delayMicroseconds(CAN_INTER_FRAME_US);
        odrive_L.setState(AXIS_STATE_CLOSED_LOOP_CONTROL);
        delayMicroseconds(CAN_INTER_FRAME_US);
        odrive_R.setState(AXIS_STATE_CLOSED_LOOP_CONTROL);
    }
    wm_mode = mode;
}

void wheel_motors_send(float L, float R) {
    switch (wm_mode) {
        case WheelMode::IDLE:
            break;
        case WheelMode::VELOCITY:
            odrive_L.setVelocity(L / TWO_PI);   // rad/s → turns/s
            delayMicroseconds(CAN_INTER_FRAME_US);
            odrive_R.setVelocity(R / TWO_PI);
            break;
        case WheelMode::POSITION:
            odrive_L.setPosition(L / TWO_PI);   // rad → turns
            delayMicroseconds(CAN_INTER_FRAME_US);
            odrive_R.setPosition(R / TWO_PI);
            break;
        case WheelMode::TORQUE:
            odrive_L.setTorque(L);
            delayMicroseconds(CAN_INTER_FRAME_US);
            odrive_R.setTorque(R);
            break;
    }
}

void wheel_motors_pet_watchdog() {
    // In active modes, the regular send() already keeps the ODrive alive.
    // In IDLE, send zero-velocity so the ODrive watchdog (if enabled) doesn't trip.
    if (wm_mode == WheelMode::IDLE) {
        odrive_L.setVelocity(0.0f);
        delayMicroseconds(CAN_INTER_FRAME_US);
        odrive_R.setVelocity(0.0f);
    }
}

void wheel_motors_clear_errors() {
    odrive_L.clearErrors();
    odrive_R.clearErrors();
    wm_L.error = 0;
    wm_R.error = 0;
    Serial.println("[WheelMotors] clear_errors sent");
}

// ── NO_CAN stubs (uno_r4_wifi env, software-only builds) ───────────────────
#else

bool wheel_motors_init()              { return true; }
void wheel_motors_poll()              {}
void wheel_motors_set_mode(WheelMode) {}
void wheel_motors_send(float, float)  {}
void wheel_motors_pet_watchdog()      {}
void wheel_motors_clear_errors()      {}

#endif // NO_CAN
