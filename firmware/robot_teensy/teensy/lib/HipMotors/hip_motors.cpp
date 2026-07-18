#include "hip_motors.h"
#include "config.h"
#include "robot_state.h"
#include "comm_protocol.h"
#include "param_registry.h"
#include "state_machine.h"
#include <Arduino.h>
#include <FlexCAN_T4.h>

// Maximum allowed position step per command (rad). Larger jumps trigger ESTOP.
#define MAX_HIP_DELTA_RAD  1.5708f  // 90 deg

// AK45-10 MIT Cheetah protocol parameter limits
#define P_MIN   -12.5f
#define P_MAX    12.5f
#define V_MIN   -65.0f
#define V_MAX    65.0f
#define KP_MIN    0.0f
#define KP_MAX  500.0f
#define KD_MIN    0.0f
#define KD_MAX    5.0f
#define T_MIN   -18.0f
#define T_MAX    18.0f
#define I_MIN   -20.0f
#define I_MAX    20.0f

// Motor silently drops out of MIT mode after ~4 s without re-entry; use 3 s margin.
#define MIT_REENTER_MS  3000u

// A cached setpoint not refreshed within this window is considered stale and
// poll() reverts to the safe zero-torque ping.
#define HIP_SETPOINT_TIMEOUT_MS  5000u

HipAxisState hm_L = {};
HipAxisState hm_R = {};
HipSetpoint  hm_sp_L = {};
HipSetpoint  hm_sp_R = {};
HipLimits    hm_limits_L = {};
HipLimits    hm_limits_R = {};

// CAN2 on Teensy 4.1 uses pins 1 (TX) and 0 (RX) — matches config.h PIN_CAN2_*.
static FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> can2;
static uint32_t last_enter_ms = 0;

// ── helpers ──────────────────────────────────────────────────────────────────

static uint16_t float_to_uint(float x, float x_min, float x_max, int bits) {
    uint32_t max_val = (1u << bits) - 1;
    if (x < x_min) x = x_min;
    if (x > x_max) x = x_max;
    return (uint16_t)((x - x_min) / (x_max - x_min) * (float)max_val);
}

static float uint_to_float(uint16_t x, float x_min, float x_max, int bits) {
    uint32_t max_val = (1u << bits) - 1;
    return (float)x / (float)max_val * (x_max - x_min) + x_min;
}

// Clamp a commanded position to the calibrated software limits, if valid.
static float clamp_to_limits(float pos, const HipLimits& lim) {
    if (!lim.valid) return pos;
    if (pos < lim.min_rad) return lim.min_rad;
    if (pos > lim.max_rad) return lim.max_rad;
    return pos;
}

static void send_raw(uint32_t id, const uint8_t data[8]) {
    CAN_message_t msg = {};
    msg.id  = id;
    msg.len = 8;
    memcpy(msg.buf, data, 8);
    can2.write(msg);
}

static void pack_and_send(uint32_t id, float pos, float vel, float kp, float kd, float torque) {
    // Guard against large position jumps — fault and suppress the frame.
    const HipAxisState* ax = (id == AK45_ID_L) ? &hm_L : &hm_R;
    if (ax->ever_heard) {
        float delta = pos - ax->pos_rad;
        if (delta < 0) delta = -delta;
        if (delta > MAX_HIP_DELTA_RAD) {
            const char* side = (id == AK45_ID_L) ? "L" : "R";
            comm_log(LOG_LEVEL_ERROR, "FAULT: hip %s pos jump %.3f rad > %.3f", side, delta, MAX_HIP_DELTA_RAD);
            // Request the transition through the FSM rather than writing
            // g_state.state directly — a direct write desyncs it from the
            // StateMachine library's own private currentState index, which
            // then keeps re-running the state we were actually in (masking
            // this fault entirely; see calibration double-run bug).
            g_state.fault_code = FAULT_HIP_LARGE_POS_CMD;
            stateMachine_request_estop();
            return;
        }
    }

    // Left hip is physically mirrored relative to right — negate pos/vel/torque
    // here, once, so every caller (calibration, jump FSM, GUI MIT frames, radio
    // hip command) can work in one consistent frame where positive means the
    // same physical direction on both sides, same as WheelMotors does for R.
    float pos_hw = pos, vel_hw = vel, torque_hw = torque;
    if (id == AK45_ID_L) { pos_hw = -pos_hw; vel_hw = -vel_hw; torque_hw = -torque_hw; }

    uint16_t p   = float_to_uint(pos_hw,    P_MIN,  P_MAX,  16);
    uint16_t v   = float_to_uint(vel_hw,    V_MIN,  V_MAX,  12);
    uint16_t kp_ = float_to_uint(kp,        KP_MIN, KP_MAX, 12);
    uint16_t kd_ = float_to_uint(kd,        KD_MIN, KD_MAX, 12);
    uint16_t t   = float_to_uint(torque_hw, T_MIN,  T_MAX,  12);

    uint8_t buf[8];
    buf[0] = p >> 8;
    buf[1] = p & 0xFF;
    buf[2] = v >> 4;
    buf[3] = ((v & 0xF) << 4) | (kp_ >> 8);
    buf[4] = kp_ & 0xFF;
    buf[5] = kd_ >> 4;
    buf[6] = ((kd_ & 0xF) << 4) | (t >> 8);
    buf[7] = t & 0xFF;
    send_raw(id, buf);
}

// ── CAN RX callback (called from FlexCAN_T4 ISR) ─────────────────────────────

static void rx_callback(const CAN_message_t& msg) {
    if (msg.len < 6) return;

    HipAxisState* ax;
    if      (msg.id == AK45_ID_L) ax = &hm_L;
    else if (msg.id == AK45_ID_R) ax = &hm_R;
    else return;

    uint16_t raw_pos = ((uint16_t)msg.buf[1] << 8) | msg.buf[2];
    uint16_t raw_vel = ((uint16_t)msg.buf[3] << 4) | (msg.buf[4] >> 4);
    uint16_t raw_cur = ((uint16_t)(msg.buf[4] & 0xF) << 8) | msg.buf[5];

    float pos = uint_to_float(raw_pos, P_MIN, P_MAX, 16);
    float vel = uint_to_float(raw_vel, V_MIN, V_MAX, 12);
    float cur = uint_to_float(raw_cur, I_MIN, I_MAX, 12);
    // Left hip is physically mirrored relative to right — negate feedback here
    // so downstream code (gain scheduling, FF1 current sum, telemetry, GUI)
    // sees a consistent frame; must match the TX-side flip in pack_and_send().
    if (msg.id == AK45_ID_L) { pos = -pos; vel = -vel; cur = -cur; }
    ax->pos_rad    = pos;
    ax->vel_rad_s  = vel;
    ax->current_A  = cur;
    ax->last_fb_ms = millis();
    ax->ever_heard = true;
}

// ── public API ────────────────────────────────────────────────────────────────

bool hip_motors_init() {
    can2.begin();
    can2.setBaudRate(CAN_BAUD);
    can2.setMaxMB(16);
    can2.enableFIFO();
    can2.enableFIFOInterrupt();
    can2.onReceive(rx_callback);
    Serial.print("[HipMotors] CAN2 init OK  ");
    Serial.print(CAN_BAUD / 1000);
    Serial.print(" kbps  id_L=0x");
    Serial.print(AK45_ID_L, HEX);
    Serial.print("  id_R=0x");
    Serial.println(AK45_ID_R, HEX);
    return true;
}

void hip_motors_poll() {
    uint32_t now = millis();
    hm_L.ok = hm_L.ever_heard && (now - hm_L.last_fb_ms) < HIP_CAN_TIMEOUT_MS;
    hm_R.ok = hm_R.ever_heard && (now - hm_R.last_fb_ms) < HIP_CAN_TIMEOUT_MS;

    if ((hm_L.mit_active || hm_R.mit_active) && (now - last_enter_ms) >= MIT_REENTER_MS) {
        hip_motors_enter_mit();
    }

    // ESTOP or a stale (unrefreshed) setpoint both fall back to the safe ping —
    // except during the brief, time-bounded gentle-cutoff ramp (see
    // state_machine.cpp on_estop()), which needs its tapering setpoint to
    // survive for its ~1 s duration instead of being zeroed the instant
    // STATE_ESTOP is entered.
    if (g_state.state == STATE_ESTOP && !stateMachine_is_estop_hip_ramping()) {
        hm_sp_L.active = false;
        hm_sp_R.active = false;
    }
    if (hm_sp_L.active && (now - hm_sp_L.last_cmd_ms) > HIP_SETPOINT_TIMEOUT_MS) hm_sp_L.active = false;
    if (hm_sp_R.active && (now - hm_sp_R.last_cmd_ms) > HIP_SETPOINT_TIMEOUT_MS) hm_sp_R.active = false;

    // While a setpoint is active, re-send it every tick so it isn't overridden
    // by the zero-torque ping below. Otherwise ping with current-position +
    // zero-torque so the AK45 returns feedback every frame. Each side is gated
    // independently so a disabled/absent motor never gets CAN traffic.
    if (hm_L.mit_active) {
        if (hm_sp_L.active)
            pack_and_send(AK45_ID_L, hm_sp_L.p, hm_sp_L.v, hm_sp_L.kp, hm_sp_L.kd, hm_sp_L.tff);
        else
            pack_and_send(AK45_ID_L, hm_L.pos_rad, 0.0f, 0.0f, 0.0f, 0.0f);
        delayMicroseconds(CAN_INTER_FRAME_US);
    }
    if (hm_R.mit_active) {
        if (hm_sp_R.active)
            pack_and_send(AK45_ID_R, hm_sp_R.p, hm_sp_R.v, hm_sp_R.kp, hm_sp_R.kd, hm_sp_R.tff);
        else
            pack_and_send(AK45_ID_R, hm_R.pos_rad, 0.0f, 0.0f, 0.0f, 0.0f);
    }
}

void hip_motors_enter_mit() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC};
    bool l_en = param_get(PARAM_HIP_L_ENABLE) >= 0.5f;
    bool r_en = param_get(PARAM_HIP_R_ENABLE) >= 0.5f;
    if (l_en) {
        send_raw(AK45_ID_L, cmd);
        delayMicroseconds(CAN_INTER_FRAME_US);
    }
    if (r_en) send_raw(AK45_ID_R, cmd);
    hm_L.mit_active = l_en;
    hm_R.mit_active = r_en;
    last_enter_ms   = millis();
}

void hip_motors_exit_mit() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD};
    if (hm_L.mit_active) {
        send_raw(AK45_ID_L, cmd);
        delayMicroseconds(CAN_INTER_FRAME_US);
    }
    if (hm_R.mit_active) send_raw(AK45_ID_R, cmd);
    hm_L.mit_active = false;
    hm_R.mit_active = false;
    hip_motors_clear_setpoints();
}

void hip_motors_zero() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE};
    bool l_en = param_get(PARAM_HIP_L_ENABLE) >= 0.5f;
    bool r_en = param_get(PARAM_HIP_R_ENABLE) >= 0.5f;
    if (l_en) {
        send_raw(AK45_ID_L, cmd);
        delayMicroseconds(CAN_INTER_FRAME_US);
    }
    if (r_en) send_raw(AK45_ID_R, cmd);
    comm_log(LOG_LEVEL_INFO, "Hip encoders zeroed (L=%d R=%d)", (int)l_en, (int)r_en);
}

void hip_motor_zero_L() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE};
    send_raw(AK45_ID_L, cmd);
    comm_log(LOG_LEVEL_INFO, "Hip encoder zeroed (L)");
}

void hip_motor_zero_R() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE};
    send_raw(AK45_ID_R, cmd);
    comm_log(LOG_LEVEL_INFO, "Hip encoder zeroed (R)");
}

void hip_motors_send(float pos_L, float vel_L, float kp_L, float kd_L, float trq_L,
                     float pos_R, float vel_R, float kp_R, float kd_R, float trq_R) {
    // Gate each motor independently — with only one leg enabled (bench config)
    // the active side must still receive its frame.
    if (hm_L.mit_active) {
        pack_and_send(AK45_ID_L, clamp_to_limits(pos_L, hm_limits_L), vel_L, kp_L, kd_L, trq_L);
        delayMicroseconds(CAN_INTER_FRAME_US);
    }
    if (hm_R.mit_active)
        pack_and_send(AK45_ID_R, clamp_to_limits(pos_R, hm_limits_R), vel_R, kp_R, kd_R, trq_R);
}

void hip_motor_send_L(float pos, float vel, float kp, float kd, float torque) {
    if (!hm_L.mit_active) return;
    pack_and_send(AK45_ID_L, clamp_to_limits(pos, hm_limits_L), vel, kp, kd, torque);
}

void hip_motor_send_R(float pos, float vel, float kp, float kd, float torque) {
    if (!hm_R.mit_active) return;
    pack_and_send(AK45_ID_R, clamp_to_limits(pos, hm_limits_R), vel, kp, kd, torque);
}

void hip_motors_set_setpoint_L(float pos, float vel, float kp, float kd, float torque) {
    hm_sp_L = {clamp_to_limits(pos, hm_limits_L), vel, kp, kd, torque, true, millis()};
}

void hip_motors_set_setpoint_R(float pos, float vel, float kp, float kd, float torque) {
    hm_sp_R = {clamp_to_limits(pos, hm_limits_R), vel, kp, kd, torque, true, millis()};
}

void hip_motors_clear_setpoints() {
    hm_sp_L.active = false;
    hm_sp_R.active = false;
}

void hip_cmd_to_setpoints(float t, float* pos_L, float* pos_R) {
    float span_L = hm_limits_L.max_rad - hm_limits_L.min_rad;
    float span_R = hm_limits_R.max_rad - hm_limits_R.min_rad;
    float dir_L  = param_get(PARAM_CALIB_L_SEEK_DIR);
    float dir_R  = param_get(PARAM_CALIB_R_SEEK_DIR);
    *pos_L = (dir_L > 0.0f) ? (hm_limits_L.max_rad - t * span_L)
                             : (hm_limits_L.min_rad + t * span_L);
    *pos_R = (dir_R > 0.0f) ? (hm_limits_R.max_rad - t * span_R)
                             : (hm_limits_R.min_rad + t * span_R);
}

bool hip_motors_ok() {
    bool l_ok = (param_get(PARAM_HIP_L_ENABLE) < 0.5f) || hm_L.ok;
    bool r_ok = (param_get(PARAM_HIP_R_ENABLE) < 0.5f) || hm_R.ok;
    return l_ok && r_ok;
}
