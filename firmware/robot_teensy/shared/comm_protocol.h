#pragma once
#include <stdint.h>

// ── Frame constants ───────────────────────────────────────────────────────────
#define COMM_START  0xFF
#define COMM_END    0xFE

// ── Source IDs ────────────────────────────────────────────────────────────────
#define COMM_SRC_TEENSY  0x01
#define COMM_SRC_ESP32   0x02
#define COMM_SRC_PC      0x03

// ── Packet types ──────────────────────────────────────────────────────────────
#define COMM_TYPE_TELEMETRY  0x01
#define COMM_TYPE_COMMAND    0x02
#define COMM_TYPE_ACK        0x03
#define COMM_TYPE_LOG        0x04

// ── Log levels ────────────────────────────────────────────────────────────────
#define LOG_LEVEL_INFO   0x01
#define LOG_LEVEL_WARN   0x02
#define LOG_LEVEL_ERROR  0x03
#define LOG_PAYLOAD_V1   1
// Log payload: uint8_t level, char msg[] (variable, no null terminator)

// ── Frame layout (9 bytes overhead) ──────────────────────────────────────────
//
//   [COMM_START]          1 byte  — 0xFF
//   [type]                1 byte  — COMM_TYPE_*
//   [version]             1 byte  — payload struct version for this type
//   [source]              1 byte  — COMM_SRC_*
//   [seq]                 1 byte  — rolling 0-255 tx counter
//   [len_lo][len_hi]      2 bytes — payload length, little-endian
//   [...payload...]       len bytes
//   [checksum]            1 byte  — XOR(type,version,source,seq,len_lo,len_hi,payload[0..n-1])
//   [COMM_END]            1 byte  — 0xFE

// ── Fault codes (robot_state == STATE_ESTOP → fault_code says why) ────────────
// IMPORTANT: when adding/changing a fault code below, also update:
//   - software/gui/flash_monitor.py   _FAULT_NAMES and _FAULT_DESCRIPTIONS dicts
//   - esp32/src/main.cpp               fault_description()
//   - firmware/robot_teensy/README.md  fault code table
#define FAULT_NONE               0x00
#define FAULT_IMU_ERROR          0x01  // IMU reported ERROR during startup
#define FAULT_HIP_INIT_TIMEOUT   0x02  // no CAN reply from hip motors within 2 s of boot
#define FAULT_HIP_FEEDBACK_LOST  0x03  // hip CAN feedback timed out during operation
#define FAULT_HIP_LARGE_POS_CMD  0x04  // commanded position jump exceeded MAX_HIP_DELTA_RAD
#define FAULT_CALIBRATION_TIMEOUT 0x05 // hardstop not found within CALIB_SAFETY_BOUND_RAD
#define FAULT_HUMAN_ESTOP        0x06  // ESTOP requested by user via GUI button

// ── Payload: telemetry ────────────────────────────────────────────────────────
#define TELEM_PAYLOAD_V1  1

typedef struct __attribute__((packed)) {
    uint32_t timestamp_ms;
    float    pitch_rad;
    float    pitch_rate_rads;
    float    wheel_vel_avg_ms;
    float    hip_l_pos_rad;
    float    hip_r_pos_rad;
    float    cmd_l;
    float    cmd_r;
    float    roll_rad;
    float    yaw_rad;
    uint8_t  robot_state;   // matches RobotStateEnum
    uint8_t  fault_code;    // FAULT_* — non-zero only when robot_state == STATE_ESTOP
    float    test_val;      // dummy 2 Hz sine wave for pipeline testing
    float    hip_l_current_a;  // hip L phase current [A]
    float    hip_r_current_a;  // hip R phase current [A]
} TelemetryPayload;         // 54 bytes

// ── Payload: command ──────────────────────────────────────────────────────────
#define CMD_PAYLOAD_V1  1

typedef struct __attribute__((packed)) {
    uint8_t cmd_id;
    uint8_t data[8];
} CommandPayload;           // 9 bytes

// ── Command IDs ───────────────────────────────────────────────────────────────
// MIRROR: software/gui/hip_motors.py  _CMD_ID_* constants must stay in sync
#define CMD_ID_SET_MODE   0x01  // payload: uint8_t target_state (RobotStateEnum)
#define CMD_ID_HIP        0x05  // payload: uint8_t motor_id, uint8_t sub_cmd [, 5×float]
#define CMD_ID_REBOOT     0x06  // payload: none — triggers a full MCU reset (reruns setup())

// Hip motor IDs (CMD_ID_HIP payload byte 1)
#define HIP_MOTOR_BOTH    0x00
#define HIP_MOTOR_L       0x01
#define HIP_MOTOR_R       0x02

// Hip sub-commands (CMD_ID_HIP payload byte 2)
#define HIP_SUB_DISABLE   0x00
#define HIP_SUB_ENABLE    0x01
#define HIP_SUB_ZERO      0x02
#define HIP_SUB_MIT       0x03  // + float p_rad, vel_rad_s, kp, kd, tff  (20 bytes)
