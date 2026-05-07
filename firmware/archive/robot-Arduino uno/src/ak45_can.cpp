// ak45_can.cpp — CubeMars AK45-10 MIT CAN driver for hip motors.
//
// Protocol reference: CubeMars MIT CAN manual v1.0.18 §5.3
// Build with -DNO_CAN to compile no-op stubs (bench testing without hardware).

#include "ak45_can.h"
#include "config.h"
#include <Arduino.h>

#ifdef NO_CAN
// ── No-op stubs (no CAN hardware) ─────────────────────────────────────────
void ak45_init()                                                      {}
void ak45_enable(uint8_t id)                                          { (void)id; }
void ak45_disable(uint8_t id)                                         { (void)id; }
void ak45_set_zero(uint8_t id)                                        { (void)id; }
void ak45_send_cmd(uint8_t id, float p, float v, float kp, float kd, float t)
                                                                      { (void)id; (void)p; (void)v; (void)kp; (void)kd; (void)t; }
void ak45_parse_rx(uint32_t id, const uint8_t* d, RobotState& s)     { (void)id; (void)d; (void)s; }
bool ak45_hip_ok()                                                    { return false; }

#else
#include <Arduino_CAN.h>

// ── AK45-10 physical ranges ─────────────────────────────────────────────────
static constexpr float P_MIN  = -12.5f;  static constexpr float P_MAX  =  12.5f;
static constexpr float V_MIN  = -20.0f;  static constexpr float V_MAX  =  20.0f;
static constexpr float T_MIN  =  -8.0f;  static constexpr float T_MAX  =   8.0f;
static constexpr float KP_MIN =   0.0f;  static constexpr float KP_MAX = 500.0f;
static constexpr float KD_MIN =   0.0f;  static constexpr float KD_MAX =   5.0f;

// ── Internal state ─────────────────────────────────────────────────────────
static uint32_t s_last_rx_L_ms = 0;
static uint32_t s_last_rx_R_ms = 0;

// ── Encode / decode helpers ────────────────────────────────────────────────

static inline uint16_t float_to_uint(float x, float x_min, float x_max, int bits) {
    float span = x_max - x_min;
    float offset = x - x_min;
    if (offset < 0.0f)       offset = 0.0f;
    if (offset > span)        offset = span;
    return (uint16_t)(offset / span * (float)((1 << bits) - 1));
}

static inline float uint_to_float(uint16_t x_int, float x_min, float x_max, int bits) {
    float span = x_max - x_min;
    return (float)x_int / (float)((1 << bits) - 1) * span + x_min;
}

// ── Send helper ────────────────────────────────────────────────────────────

static inline void can_send_ak(uint8_t id, const uint8_t* data, uint8_t len) {
    CanMsg msg(CanStandardId(id), len, data);
    CAN.write(msg);
}

// ── Public API implementation ──────────────────────────────────────────────

void ak45_init() {
    // CAN bus already initialised by odesc_can_init(); nothing to do.
    Serial.println("[AK45] driver ready");
}

void ak45_enable(uint8_t id) {
    uint8_t data[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC};
    can_send_ak(id, data, 8);
}

void ak45_disable(uint8_t id) {
    uint8_t data[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD};
    can_send_ak(id, data, 8);
}

void ak45_set_zero(uint8_t id) {
    uint8_t data[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE};
    can_send_ak(id, data, 8);
}

void ak45_send_cmd(uint8_t id, float p_des, float v_des,
                   float kp, float kd, float t_ff) {
    uint16_t p_int  = float_to_uint(p_des, P_MIN,  P_MAX,  16);
    uint16_t v_int  = float_to_uint(v_des, V_MIN,  V_MAX,  12);
    uint16_t kp_int = float_to_uint(kp,    KP_MIN, KP_MAX, 12);
    uint16_t kd_int = float_to_uint(kd,    KD_MIN, KD_MAX, 12);
    uint16_t t_int  = float_to_uint(t_ff,  T_MIN,  T_MAX,  12);

    uint8_t data[8];
    data[0] = (uint8_t)(p_int  >> 8);
    data[1] = (uint8_t)(p_int  & 0xFF);
    data[2] = (uint8_t)(v_int  >> 4);
    data[3] = (uint8_t)((v_int  & 0xF) << 4 | (kp_int >> 8));
    data[4] = (uint8_t)(kp_int & 0xFF);
    data[5] = (uint8_t)(kd_int >> 4);
    data[6] = (uint8_t)((kd_int & 0xF) << 4 | (t_int  >> 8));
    data[7] = (uint8_t)(t_int  & 0xFF);

    can_send_ak(id, data, 8);
}

void ak45_parse_rx(uint32_t can_id, const uint8_t* data, RobotState& state) {
    // Unpack per §5.3 RX frame layout
    uint16_t p_int  = ((uint16_t)data[1] << 8) | data[2];
    uint16_t v_int  = ((uint16_t)data[3] << 4) | (data[4] >> 4);

    float pos = uint_to_float(p_int, P_MIN, P_MAX, 16);
    float vel = uint_to_float(v_int, V_MIN, V_MAX, 12);
    float temp_c = (float)data[6] - 40.0f;
    uint8_t err  = data[7];

    uint32_t now = millis();
    if (can_id == CAN_ID_HIP_L) {
        state.hip_q_L  = pos;
        state.hip_dq_L = vel;
        s_last_rx_L_ms = now;
    } else if (can_id == CAN_ID_HIP_R) {
        state.hip_q_R  = pos;
        state.hip_dq_R = vel;
        s_last_rx_R_ms = now;
    }
    state.hip_temp  = temp_c;
    state.hip_error = err;

    state.hip_q_avg  = 0.5f * (state.hip_q_L + state.hip_q_R);
    state.hip_ok     = ak45_hip_ok();
    state.hip_last_ms = max(s_last_rx_L_ms, s_last_rx_R_ms);
}

bool ak45_hip_ok() {
    uint32_t now = millis();
    return (now - s_last_rx_L_ms) < CAN_TIMEOUT_MS &&
           (now - s_last_rx_R_ms) < CAN_TIMEOUT_MS;
}

#endif // !NO_CAN
