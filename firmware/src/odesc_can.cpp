// odesc_can.cpp — ODESC (ODrive v3.x) CAN driver for wheel motors.
//
// Protocol reference: ODrive CAN protocol v0.5
//   arb_id = (node_id << 5) | cmd_id
//   All payloads little-endian.

#include "odesc_can.h"
#include "config.h"
#include <Arduino.h>
#include <Arduino_CAN.h>

// ── ODrive CAN command IDs (5-bit) ─────────────────────────────────────────
static constexpr uint8_t CMD_HEARTBEAT        = 0x01;
static constexpr uint8_t CMD_ESTOP            = 0x02;
static constexpr uint8_t CMD_SET_AXIS_STATE   = 0x07;
static constexpr uint8_t CMD_ENCODER_ESTIMATE = 0x09;
static constexpr uint8_t CMD_SET_INPUT_TORQUE = 0x0E;
static constexpr uint8_t CMD_CLEAR_ERRORS     = 0x12;

// ── ODrive axis states ─────────────────────────────────────────────────────
static constexpr uint32_t AXIS_STATE_IDLE              = 1;
static constexpr uint32_t AXIS_STATE_CLOSED_LOOP       = 8;

// ── Helpers ────────────────────────────────────────────────────────────────

static inline uint32_t arb_id(uint8_t node_id, uint8_t cmd_id) {
    return ((uint32_t)node_id << 5) | cmd_id;
}

static inline uint8_t node_from_id(uint32_t id) {
    return (uint8_t)((id >> 5) & 0x3F);
}

static inline uint8_t cmd_from_id(uint32_t id) {
    return (uint8_t)(id & 0x1F);
}

/// Pack a float into a CAN data buffer at byte offset (little-endian).
static inline void pack_float(uint8_t* buf, uint8_t offset, float val) {
    memcpy(buf + offset, &val, sizeof(float));
}

/// Unpack a float from CAN data buffer at byte offset (little-endian).
static inline float unpack_float(const uint8_t* buf, uint8_t offset) {
    float val;
    memcpy(&val, buf + offset, sizeof(float));
    return val;
}

/// Unpack a uint32 from CAN data buffer at byte offset (little-endian).
static inline uint32_t unpack_u32(const uint8_t* buf, uint8_t offset) {
    uint32_t val;
    memcpy(&val, buf + offset, sizeof(uint32_t));
    return val;
}

// ── Internal state ─────────────────────────────────────────────────────────
static uint32_t s_last_heartbeat_L_ms = 0;
static uint32_t s_last_heartbeat_R_ms = 0;
static uint32_t s_last_encoder_L_ms   = 0;
static uint32_t s_last_encoder_R_ms   = 0;

// ── Send helpers ───────────────────────────────────────────────────────────

/// Send a raw CAN frame.  Returns true on success.
static bool can_send(uint32_t id, const uint8_t* data, uint8_t len) {
    CanMsg msg(CanStandardId(id), len, data);
    return CAN.write(msg) > 0;
}

/// Set axis state on one ODESC node.
static void set_axis_state(uint8_t node_id, uint32_t requested_state) {
    uint8_t data[4];
    memcpy(data, &requested_state, 4);
    can_send(arb_id(node_id, CMD_SET_AXIS_STATE), data, 4);
}

/// Send torque command to one ODESC node.
static void send_torque(uint8_t node_id, float torque) {
    uint8_t data[4];
    pack_float(data, 0, torque);
    can_send(arb_id(node_id, CMD_SET_INPUT_TORQUE), data, 4);
}

// ── Public API implementation ──────────────────────────────────────────────

bool odesc_can_init() {
    if (!CAN.begin(CanBitRate::BR_1000k)) {
        Serial.println("[ODESC] CAN init FAILED");
        return false;
    }
    Serial.println("[ODESC] CAN bus 1 Mbps OK");
    return true;
}

void odesc_can_poll(RobotState& state) {
    // Drain all queued CAN frames (typically 0–4 per tick at 500 Hz).
    while (CAN.available()) {
        CanMsg msg = CAN.read();
        uint32_t std_id = msg.getStandardId();
        uint8_t node = node_from_id(std_id);
        uint8_t cmd  = cmd_from_id(std_id);

        // ── Encoder estimates (pos in turns, vel in turns/s) ──
        if (cmd == CMD_ENCODER_ESTIMATE && msg.data_length >= 8) {
            float pos_turns = unpack_float(msg.data, 0);
            float vel_turns = unpack_float(msg.data, 4);
            // Convert turns → radians
            float pos_rad = pos_turns * 6.2831853f;  // 2*PI
            float vel_rad = vel_turns * 6.2831853f;

            if (node == ODESC_NODE_L) {
                state.wheel_pos_L = pos_rad;
                state.wheel_vel_L = vel_rad;
                s_last_encoder_L_ms = millis();
            } else if (node == ODESC_NODE_R) {
                state.wheel_pos_R = pos_rad;
                state.wheel_vel_R = vel_rad;
                s_last_encoder_R_ms = millis();
            }
        }

        // ── Heartbeat (error + state) ──
        else if (cmd == CMD_HEARTBEAT && msg.data_length >= 5) {
            uint32_t axis_error = unpack_u32(msg.data, 0);
            uint8_t  axis_state = msg.data[4];

            if (node == ODESC_NODE_L) {
                s_last_heartbeat_L_ms = millis();
                if (axis_error != 0) {
                    Serial.print("[ODESC] L error=0x");
                    Serial.println(axis_error, HEX);
                }
            } else if (node == ODESC_NODE_R) {
                s_last_heartbeat_R_ms = millis();
                if (axis_error != 0) {
                    Serial.print("[ODESC] R error=0x");
                    Serial.println(axis_error, HEX);
                }
            }
            (void)axis_state;  // available if needed for state machine
        }
    }

    // Update averages and watchdog
    state.wheel_vel_avg = 0.5f * (state.wheel_vel_L + state.wheel_vel_R);

    uint32_t now = millis();
    bool enc_L_ok = (now - s_last_encoder_L_ms) < CAN_TIMEOUT_MS;
    bool enc_R_ok = (now - s_last_encoder_R_ms) < CAN_TIMEOUT_MS;
    state.wheel_ok = enc_L_ok && enc_R_ok;
    state.wheel_last_ms = max(s_last_encoder_L_ms, s_last_encoder_R_ms);
}

void odesc_can_send_torque(float tau_L, float tau_R) {
    send_torque(ODESC_NODE_L, tau_L);
    send_torque(ODESC_NODE_R, tau_R);
}

void odesc_can_enable() {
    Serial.println("[ODESC] Enabling closed-loop control");
    set_axis_state(ODESC_NODE_L, AXIS_STATE_CLOSED_LOOP);
    set_axis_state(ODESC_NODE_R, AXIS_STATE_CLOSED_LOOP);
}

void odesc_can_disable() {
    Serial.println("[ODESC] Disabling — axes to IDLE");
    set_axis_state(ODESC_NODE_L, AXIS_STATE_IDLE);
    set_axis_state(ODESC_NODE_R, AXIS_STATE_IDLE);
}

void odesc_can_estop() {
    uint8_t empty[1] = {0};
    can_send(arb_id(ODESC_NODE_L, CMD_ESTOP), empty, 0);
    can_send(arb_id(ODESC_NODE_R, CMD_ESTOP), empty, 0);
    Serial.println("[ODESC] ESTOP sent");
}
