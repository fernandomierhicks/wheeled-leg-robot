// odesc_can.cpp — ODESC (ODrive v3.x) CAN driver for wheel motors.
//
// Protocol reference: ODrive CAN protocol v0.5
//   arb_id = (node_id << 5) | cmd_id
//   All payloads little-endian.
//
// Build with -DNO_CAN to compile no-op stubs (bench testing without hardware).

#include "odesc_can.h"
#include "ak45_can.h"
#include "config.h"
#include <Arduino.h>

#ifdef NO_CAN
// ── No-op stubs (no CAN hardware) ─────────────────────────────────────────
bool odesc_can_init()                              { Serial.println("[ODESC] NO_CAN — skipped"); return true; }
void odesc_can_poll(RobotState& state)             { state.wheel_vel_avg = 0.0f; state.wheel_ok = false; }
void odesc_can_send_torque(float tau_L, float tau_R) { (void)tau_L; (void)tau_R; }
void odesc_can_enable()                            {}
void odesc_can_disable()                           {}
void odesc_can_estop()                             {}

#else
#include <Arduino_CAN.h>

// ── ODrive CAN command IDs (5-bit) ─────────────────────────────────────────
static constexpr uint8_t CMD_HEARTBEAT            = 0x01;
static constexpr uint8_t CMD_ESTOP                = 0x02;
static constexpr uint8_t CMD_SET_AXIS_STATE       = 0x07;
static constexpr uint8_t CMD_ENCODER_ESTIMATE     = 0x09;
static constexpr uint8_t CMD_SET_CONTROLLER_MODES = 0x0B;
static constexpr uint8_t CMD_SET_INPUT_POS        = 0x0C;
static constexpr uint8_t CMD_SET_INPUT_VEL        = 0x0D;
static constexpr uint8_t CMD_SET_INPUT_TORQUE     = 0x0E;
static constexpr uint8_t CMD_CLEAR_ERRORS         = 0x18;

// ── ODrive axis states ─────────────────────────────────────────────────────
static constexpr uint32_t AXIS_STATE_IDLE              = 1;
static constexpr uint32_t AXIS_STATE_CLOSED_LOOP       = 8;

// ── ODrive control / input modes ───────────────────────────────────────────
static constexpr uint32_t CTRL_MODE_TORQUE   = 1;
static constexpr uint32_t CTRL_MODE_VELOCITY = 2;
static constexpr uint32_t CTRL_MODE_POSITION = 3;
static constexpr uint32_t INPUT_MODE_PASSTHROUGH = 1;

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
static bool     s_encoder_L_seen      = false;  // first-frame diagnostic flag
static bool     s_encoder_R_seen      = false;

// ── Error decoder ──────────────────────────────────────────────────────────

static void print_axis_error(const char* axis_label, uint32_t err) {
    if (err == 0) return;
    Serial.print("[ODESC] ");
    Serial.print(axis_label);
    Serial.print(" axis_error=0x");
    Serial.print(err, HEX);
    Serial.print(" :");
    if (err & 0x00000001) Serial.print(" INVALID_STATE");
    if (err & 0x00000002) Serial.print(" DC_BUS_UNDER_VOLTAGE");
    if (err & 0x00000004) Serial.print(" DC_BUS_OVER_VOLTAGE");
    if (err & 0x00000008) Serial.print(" CURRENT_MEASUREMENT_TIMEOUT");
    if (err & 0x00000010) Serial.print(" BRAKE_RESISTOR_DISARMED");
    if (err & 0x00000020) Serial.print(" MOTOR_DISARMED");
    if (err & 0x00000040) Serial.print(" MOTOR_FAILED");
    if (err & 0x00000080) Serial.print(" SENSORLESS_ESTIMATOR_FAILED");
    if (err & 0x00000100) Serial.print(" ENCODER_FAILED");
    if (err & 0x00000200) Serial.print(" CONTROLLER_FAILED");
    if (err & 0x00000400) Serial.print(" POS_CTRL_DURING_SENSORLESS");
    if (err & 0x00000800) Serial.print(" WATCHDOG_TIMER_EXPIRED");
    if (err & 0x00001000) Serial.print(" MIN_ENDSTOP_PRESSED");
    if (err & 0x00002000) Serial.print(" MAX_ENDSTOP_PRESSED");
    if (err & 0x00004000) Serial.print(" ESTOP_REQUESTED");
    if (err & 0x00020000) Serial.print(" HOMING_WITHOUT_ENDSTOP");
    if (err & 0x00040000) Serial.print(" OVER_TEMP");
    if (err & 0x00080000) Serial.print(" UNKNOWN_POSITION");
    Serial.println();
}

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
    // Pet watchdogs immediately so axes don't fault before the first tick fires.
    odesc_can_pet_watchdog();
    return true;
}

void odesc_can_pet_watchdog() {
    uint8_t data[8] = {0};  // vel=0.0, torque_ff=0.0
    can_send(arb_id(ODESC_NODE_L, CMD_SET_INPUT_VEL), data, 8);
    can_send(arb_id(ODESC_NODE_R, CMD_SET_INPUT_VEL), data, 8);
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
                if (!s_encoder_L_seen) {
                    s_encoder_L_seen = true;
                    Serial.print("[ODESC] Encoder L first frame: pos=");
                    Serial.print(pos_turns, 4);
                    Serial.print(" turns  vel=");
                    Serial.print(vel_turns, 4);
                    Serial.println(" turns/s");
                }
            } else if (node == ODESC_NODE_R) {
                state.wheel_pos_R = pos_rad;
                state.wheel_vel_R = vel_rad;
                s_last_encoder_R_ms = millis();
                if (!s_encoder_R_seen) {
                    s_encoder_R_seen = true;
                    Serial.print("[ODESC] Encoder R first frame: pos=");
                    Serial.print(pos_turns, 4);
                    Serial.print(" turns  vel=");
                    Serial.print(vel_turns, 4);
                    Serial.println(" turns/s");
                }
            }
        }

        // ── AK45-10 hip motor reply ──
        else if (std_id == CAN_ID_HIP_L || std_id == CAN_ID_HIP_R) {
            if (msg.data_length >= 6)
                ak45_parse_rx(std_id, msg.data, state);
        }

        // ── Heartbeat (error + state) ──
        else if (cmd == CMD_HEARTBEAT && msg.data_length >= 5) {
            uint32_t axis_error = unpack_u32(msg.data, 0);
            uint8_t  axis_state = msg.data[4];

            if (node == ODESC_NODE_L) {
                s_last_heartbeat_L_ms = millis();
                state.odrive_axis_state = axis_state;
                if (axis_error != state.odrive_axis_error) {
                    state.odrive_axis_error = axis_error;
                    print_axis_error("L", axis_error);
                }
            } else if (node == ODESC_NODE_R) {
                s_last_heartbeat_R_ms = millis();
                state.odrive_axis_state_R = axis_state;
                if (axis_error != state.odrive_axis_error_R) {
                    state.odrive_axis_error_R = axis_error;
                    print_axis_error("R", axis_error);
                }
            }
        }
    }

    // Update averages and watchdog
    state.wheel_vel_avg = 0.5f * (state.wheel_vel_L + state.wheel_vel_R);

    uint32_t now = millis();
    bool enc_L_ok = (now - s_last_encoder_L_ms) < CAN_TIMEOUT_MS;
    bool enc_R_ok = (now - s_last_encoder_R_ms) < CAN_TIMEOUT_MS;
    // wheel_ok requires L always; R only if it has ever responded (single-axis safe)
    state.wheel_ok = enc_L_ok && (s_last_encoder_R_ms == 0 || enc_R_ok);
    state.wheel_last_ms = max(s_last_encoder_L_ms, s_last_encoder_R_ms);
}

void odesc_can_send_torque(float tau_L, float tau_R) {
    send_torque(ODESC_NODE_L, tau_L);
    send_torque(ODESC_NODE_R, tau_R);
}

void odesc_can_send_velocity(float vel_L_turns_s, float vel_R_turns_s) {
    uint8_t data[8];
    float ff = 0.0f;
    pack_float(data, 0, vel_L_turns_s);
    pack_float(data, 4, ff);
    can_send(arb_id(ODESC_NODE_L, CMD_SET_INPUT_VEL), data, 8);
    pack_float(data, 0, vel_R_turns_s);
    can_send(arb_id(ODESC_NODE_R, CMD_SET_INPUT_VEL), data, 8);
}

void odesc_can_send_position(float pos_L_turns, float pos_R_turns) {
    uint8_t data[8];
    int16_t vel_ff = 0, tau_ff = 0;
    pack_float(data, 0, pos_L_turns);
    memcpy(data + 4, &vel_ff, 2);
    memcpy(data + 6, &tau_ff, 2);
    can_send(arb_id(ODESC_NODE_L, CMD_SET_INPUT_POS), data, 8);
    pack_float(data, 0, pos_R_turns);
    can_send(arb_id(ODESC_NODE_R, CMD_SET_INPUT_POS), data, 8);
}

void odesc_can_set_control_mode(uint32_t ctrl_mode, uint32_t input_mode) {
    uint8_t data[8];
    memcpy(data,     &ctrl_mode,  4);
    memcpy(data + 4, &input_mode, 4);
    can_send(arb_id(ODESC_NODE_L, CMD_SET_CONTROLLER_MODES), data, 8);
    can_send(arb_id(ODESC_NODE_R, CMD_SET_CONTROLLER_MODES), data, 8);
}

void odesc_can_clear_errors() {
    uint8_t empty[1] = {0};
    bool ok_L = can_send(arb_id(ODESC_NODE_L, CMD_CLEAR_ERRORS), empty, 0);
    bool ok_R = can_send(arb_id(ODESC_NODE_R, CMD_CLEAR_ERRORS), empty, 0);
    Serial.print("[ODESC] CLEAR_ERRORS L=");
    Serial.print(ok_L ? "OK" : "FAIL");
    Serial.print("  R=");
    Serial.println(ok_R ? "OK" : "FAIL");
}

// Pet both watchdogs + drain RX for up to ms milliseconds.
// Returns true if both axes confirmed CLOSED_LOOP within the window.
static bool _pet_and_wait(bool& ok_L, bool& ok_R, uint32_t ms) {
    uint8_t vel_data[8] = {0};  // 0.0f vel, 0.0f ff
    uint32_t t0 = millis();
    while ((millis() - t0) < ms) {
        can_send(arb_id(ODESC_NODE_L, CMD_SET_INPUT_VEL), vel_data, 8);
        can_send(arb_id(ODESC_NODE_R, CMD_SET_INPUT_VEL), vel_data, 8);
        while (CAN.available()) {
            CanMsg msg = CAN.read();
            uint8_t node = node_from_id(msg.getStandardId());
            uint8_t cmd  = cmd_from_id(msg.getStandardId());
            if (cmd == CMD_HEARTBEAT && msg.data_length >= 5) {
                if (node == ODESC_NODE_L && msg.data[4] == AXIS_STATE_CLOSED_LOOP) ok_L = true;
                if (node == ODESC_NODE_R && msg.data[4] == AXIS_STATE_CLOSED_LOOP) ok_R = true;
            }
        }
        delay(10);
    }
    return ok_L && ok_R;
}

static void _enable_closed_loop(uint32_t ctrl_mode, const char* label) {
    uint8_t mode_data[8];
    memcpy(mode_data,     &ctrl_mode,            4);
    memcpy(mode_data + 4, &INPUT_MODE_PASSTHROUGH, 4);
    uint8_t empty[1] = {0};
    bool ok_L = false, ok_R = false;

    // Every delay is replaced with _pet_and_wait so watchdogs are fed
    // continuously throughout the enable sequence.
    can_send(arb_id(ODESC_NODE_L, CMD_CLEAR_ERRORS), empty, 0);
    can_send(arb_id(ODESC_NODE_R, CMD_CLEAR_ERRORS), empty, 0);
    _pet_and_wait(ok_L, ok_R, 50);

    ok_L = ok_R = false;  // reset — CLOSED_LOOP not requested yet
    can_send(arb_id(ODESC_NODE_L, CMD_SET_CONTROLLER_MODES), mode_data, 8);
    can_send(arb_id(ODESC_NODE_R, CMD_SET_CONTROLLER_MODES), mode_data, 8);
    _pet_and_wait(ok_L, ok_R, 30);

    ok_L = ok_R = false;
    can_send(arb_id(ODESC_NODE_L, CMD_SET_AXIS_STATE),
             (const uint8_t*)&AXIS_STATE_CLOSED_LOOP, 4);
    can_send(arb_id(ODESC_NODE_R, CMD_SET_AXIS_STATE),
             (const uint8_t*)&AXIS_STATE_CLOSED_LOOP, 4);
    _pet_and_wait(ok_L, ok_R, 2000);  // wait up to 2 s for both to confirm

    Serial.print("[ODESC] enable (");
    Serial.print(label);
    Serial.print(") L=");
    Serial.print(ok_L ? "CLOSED_LOOP" : "TIMEOUT");
    Serial.print("  R=");
    Serial.println(ok_R ? "CLOSED_LOOP" : "TIMEOUT");
}

void odesc_can_enable_velocity() {
    Serial.println("[ODESC] Enable: velocity mode");
    _enable_closed_loop(CTRL_MODE_VELOCITY, "vel");
}

void odesc_can_enable_position() {
    Serial.println("[ODESC] Enable: position mode");
    _enable_closed_loop(CTRL_MODE_POSITION, "pos");
}

void odesc_can_enable_torque() {
    Serial.println("[ODESC] Enable: torque mode");
    odesc_can_set_control_mode(CTRL_MODE_TORQUE, INPUT_MODE_PASSTHROUGH);
    set_axis_state(ODESC_NODE_L, AXIS_STATE_CLOSED_LOOP);
    set_axis_state(ODESC_NODE_R, AXIS_STATE_CLOSED_LOOP);
}

void odesc_can_disable() {
    Serial.println("[ODESC] Disable — axes to IDLE");
    set_axis_state(ODESC_NODE_L, AXIS_STATE_IDLE);
    set_axis_state(ODESC_NODE_R, AXIS_STATE_IDLE);
}

void odesc_can_estop() {
    uint8_t empty[1] = {0};
    can_send(arb_id(ODESC_NODE_L, CMD_ESTOP), empty, 0);
    can_send(arb_id(ODESC_NODE_R, CMD_ESTOP), empty, 0);
    Serial.println("[ODESC] ESTOP sent");
}

#endif // !NO_CAN
