#include "hip_motors.h"
#include "config.h"
#include <Arduino.h>
#include <FlexCAN_T4.h>

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

HipAxisState hm_L = {};
HipAxisState hm_R = {};

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

static void send_raw(uint32_t id, const uint8_t data[8]) {
    CAN_message_t msg = {};
    msg.id  = id;
    msg.len = 8;
    memcpy(msg.buf, data, 8);
    can2.write(msg);
}

static void pack_and_send(uint32_t id, float pos, float vel, float kp, float kd, float torque) {
    uint16_t p   = float_to_uint(pos,    P_MIN,  P_MAX,  16);
    uint16_t v   = float_to_uint(vel,    V_MIN,  V_MAX,  12);
    uint16_t kp_ = float_to_uint(kp,     KP_MIN, KP_MAX, 12);
    uint16_t kd_ = float_to_uint(kd,     KD_MIN, KD_MAX, 12);
    uint16_t t   = float_to_uint(torque, T_MIN,  T_MAX,  12);

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

    ax->pos_rad    = uint_to_float(raw_pos, P_MIN, P_MAX, 16);
    ax->vel_rad_s  = uint_to_float(raw_vel, V_MIN, V_MAX, 12);
    ax->current_A  = uint_to_float(raw_cur, I_MIN, I_MAX, 12);
    ax->last_fb_ms = millis();
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
    hm_L.ok = (now - hm_L.last_fb_ms) < CAN_TIMEOUT_MS;
    hm_R.ok = (now - hm_R.last_fb_ms) < CAN_TIMEOUT_MS;

    if (hm_L.mit_active && (now - last_enter_ms) >= MIT_REENTER_MS) {
        hip_motors_enter_mit();
    }
}

void hip_motors_enter_mit() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC};
    send_raw(AK45_ID_L, cmd);
    delayMicroseconds(CAN_INTER_FRAME_US);
    send_raw(AK45_ID_R, cmd);
    hm_L.mit_active = true;
    hm_R.mit_active = true;
    last_enter_ms   = millis();
}

void hip_motors_exit_mit() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD};
    send_raw(AK45_ID_L, cmd);
    delayMicroseconds(CAN_INTER_FRAME_US);
    send_raw(AK45_ID_R, cmd);
    hm_L.mit_active = false;
    hm_R.mit_active = false;
}

void hip_motors_zero() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE};
    send_raw(AK45_ID_L, cmd);
    delayMicroseconds(CAN_INTER_FRAME_US);
    send_raw(AK45_ID_R, cmd);
    Serial.println("[HipMotors] encoder zeroed");
}

void hip_motors_send(float pos_L, float vel_L, float kp_L, float kd_L, float trq_L,
                     float pos_R, float vel_R, float kp_R, float kd_R, float trq_R) {
    if (!hm_L.mit_active) return;
    pack_and_send(AK45_ID_L, pos_L, vel_L, kp_L, kd_L, trq_L);
    delayMicroseconds(CAN_INTER_FRAME_US);
    pack_and_send(AK45_ID_R, pos_R, vel_R, kp_R, kd_R, trq_R);
}
