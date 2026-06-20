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
#define COMM_TYPE_TELEMETRY    0x01
#define COMM_TYPE_COMMAND      0x02
#define COMM_TYPE_ACK          0x03
#define COMM_TYPE_LOG          0x04
#define COMM_TYPE_CALIB_EVENT  0x05
#define COMM_TYPE_PARAM_REPORT 0x06  // Teensy→GUI: current value of one param
#define COMM_TYPE_TOF          0x07  // ESP32→Teensy: raw ToF distances (TofPayload)

// ── Calibration event sub-types ───────────────────────────────────────────────
#define CALIB_EVENT_PAYLOAD_V1   1
#define CALIB_EVENT_START        0x01  // both axes: seek begins
#define CALIB_EVENT_BOTTOM_FOUND 0x02  // axis: bottom hardstop found & zeroed
#define CALIB_EVENT_LIMITS       0x03  // axis: top hardstop found, limits computed
#define CALIB_EVENT_DONE         0x04  // axis: returned home, holding
#define CALIB_EVENT_FAULT        0x05  // axis: hardstop not found within safety bound

typedef struct __attribute__((packed)) {
    uint8_t axis;     // HIP_MOTOR_BOTH/L/R
    uint8_t event;    // CALIB_EVENT_*
    float   pos_rad;  // measured position at the event
    float   min_rad;  // computed lower limit (LIMITS/DONE only, else 0)
    float   max_rad;  // computed upper limit (LIMITS/DONE only, else 0)
} CalibEventPayload;  // 14 bytes

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

// ── Payload: ToF distances (ESP32→Teensy, COMM_TYPE_TOF) ─────────────────────
#define TOF_PAYLOAD_V1  1

typedef struct __attribute__((packed)) {
    uint16_t dist_mm[4];      // raw distance per sensor (0xFFFF = no data/invalid)
    uint16_t front_min_mm;    // min(dist[0], dist[1]) — forward sensors
    uint16_t rear_min_mm;     // min(dist[2], dist[3]) — backward sensors
} TofPayload;                 // 12 bytes

// ── Payload: telemetry ────────────────────────────────────────────────────────
// IMPORTANT: when bumping the version, also update the version check in
//   esp32/src/main.cpp  on_teensy_packet() — search for TELEM_PAYLOAD_V*
#define TELEM_PAYLOAD_V2  2
#define TELEM_PAYLOAD_V3  3
#define TELEM_PAYLOAD_V4  4

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
    uint8_t  robot_state;        // matches RobotStateEnum
    uint8_t  fault_code;         // FAULT_* — non-zero only when robot_state == STATE_ESTOP
    float    test_val;           // dummy 2 Hz sine wave for pipeline testing
    float    hip_l_current_a;    // hip L phase current [A]
    float    hip_r_current_a;    // hip R phase current [A]
    uint16_t ibus_ch[14];        // RC channels 0–13, 1000–2000 µs (1500 = center)
    uint8_t  ibus_alive;         // 1 = packet received within 500 ms, 0 = link lost
    // V3 additions — wheel motor telemetry (35 bytes, total 118)
    float    wm_l_vel_turns_s;   // left wheel velocity  [turns/s]
    float    wm_r_vel_turns_s;   // right wheel velocity [turns/s]
    float    wm_l_pos_turns;     // left wheel position  [turns]
    float    wm_r_pos_turns;     // right wheel position [turns]
    float    wm_l_vbus;          // left ODrive bus voltage  [V]
    float    wm_r_vbus;          // right ODrive bus voltage [V]
    uint32_t wm_l_error;         // left ODrive Axis_Error word
    uint32_t wm_r_error;         // right ODrive Axis_Error word
    uint8_t  wm_l_state;         // left ODrive Axis_State  (1=IDLE, 8=CLOSED_LOOP)
    uint8_t  wm_r_state;         // right ODrive Axis_State
    uint8_t  wm_mode;            // current WheelMode (0=IDLE,1=VEL,2=POS,3=TRQ)
    // V4 additions — ToF obstacle sensor data relayed from ESP32 (10 bytes, total 128)
    uint16_t tof_dist_mm[4];     // raw distances from sensors 0-3 [mm], 0xFFFF = no data
    uint16_t tof_age_ms;         // ms since last valid ToF packet from ESP32, 0xFFFF = never
} TelemetryPayload;              // 128 bytes (V4)

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
#define CMD_ID_WHEEL      0x07  // payload: uint8_t sub_cmd [, data]

// Wheel sub-commands (CMD_ID_WHEEL payload byte 1)
#define WHEEL_SUB_SET_MODE     0x01  // payload: uint8_t mode (WheelMode)
#define WHEEL_SUB_SEND         0x02  // payload: float L, float R
#define WHEEL_SUB_CLEAR_ERRORS 0x03  // no payload
#define CMD_ID_PARAM_SET  0x10  // payload: uint16_t param_id, float value  (6 bytes after cmd_id)
#define CMD_ID_PARAM_GET  0x11  // payload: uint16_t param_id  (0xFFFF = dump all)

// ── Payload: param report (COMM_TYPE_PARAM_REPORT) ───────────────────────────
#define PARAM_REPORT_PAYLOAD_V1  1

typedef struct __attribute__((packed)) {
    uint16_t param_id;
    float    value;
    float    min_val;
    float    max_val;
    uint8_t  flags;
    char     name[20];
} ParamReportPayload;  // 35 bytes

// Hip motor IDs (CMD_ID_HIP payload byte 1)
#define HIP_MOTOR_BOTH    0x00
#define HIP_MOTOR_L       0x01
#define HIP_MOTOR_R       0x02

// Hip sub-commands (CMD_ID_HIP payload byte 2)
#define HIP_SUB_DISABLE   0x00
#define HIP_SUB_ENABLE    0x01
#define HIP_SUB_ZERO      0x02
#define HIP_SUB_MIT       0x03  // + float p_rad, vel_rad_s, kp, kd, tff  (20 bytes)

// ── Logging / calib-event helpers (implemented in main.cpp) ──────────────────
#ifdef __cplusplus
void comm_log(uint8_t level, const char* fmt, ...);
void comm_send_calib_event(uint8_t axis, uint8_t event,
                            float pos_rad, float min_rad, float max_rad);
#endif
