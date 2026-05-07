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
} TelemetryPayload;         // 41 bytes

// ── Payload: command (fields TBD) ─────────────────────────────────────────────
#define CMD_PAYLOAD_V1  1

typedef struct __attribute__((packed)) {
    uint8_t cmd_id;
    uint8_t data[8];
} CommandPayload;           // 9 bytes
