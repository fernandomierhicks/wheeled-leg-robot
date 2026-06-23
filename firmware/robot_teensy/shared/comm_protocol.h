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
#define FAULT_HUMAN_ESTOP        0x06  // ESTOP requested by user via GUI button or radio
#define FAULT_PARAM_OUT_OF_BOUNDS 0x07 // param write rejected — value outside [min, max]
#define FAULT_PITCH_WATCHDOG     0x08  // |pitch| > 50° for > 200 ms
#define FAULT_WHEEL_RUNAWAY      0x09  // wheel velocity exceeded 2× soft governor limit

// ── Fault severity tiers (used by state machine + GUI recovery panel) ────────
// IMPORTANT: when adding fault codes, update fault_severity() below too.
typedef enum {
    FAULT_SEVERITY_SOFT,        // ESTOP → STANDBY directly, no re-init required
    FAULT_SEVERITY_REPOSITION,  // robot fell or calib failed; reposition then reset
    FAULT_SEVERITY_GUI_FIX,     // bad param; fix in GUI before reset
    FAULT_SEVERITY_REBOOT,      // hardware dropout; power-cycle required
} fault_severity_t;

#ifdef __cplusplus
inline fault_severity_t fault_severity(uint8_t code) {
    switch (code) {
        case FAULT_HUMAN_ESTOP:
        case FAULT_WHEEL_RUNAWAY:         return FAULT_SEVERITY_SOFT;
        case FAULT_PITCH_WATCHDOG:
        case FAULT_CALIBRATION_TIMEOUT:   return FAULT_SEVERITY_REPOSITION;
        case FAULT_PARAM_OUT_OF_BOUNDS:
        case FAULT_HIP_LARGE_POS_CMD:     return FAULT_SEVERITY_GUI_FIX;
        default:                          return FAULT_SEVERITY_REBOOT;
    }
}
#endif

// ── Payload: ToF distances (ESP32→Teensy, COMM_TYPE_TOF) ─────────────────────
#define TOF_PAYLOAD_V1  1

typedef struct __attribute__((packed)) {
    uint16_t dist_mm[4];      // raw distance per sensor (0xFFFF = no data/invalid)
    uint16_t front_min_mm;    // min(dist[0], dist[1]) — forward sensors
    uint16_t rear_min_mm;     // min(dist[2], dist[3]) — backward sensors
} TofPayload;                 // 12 bytes

// ── Payload: telemetry ────────────────────────────────────────────────────────
//
// PROPAGATION CHECKLIST — touch ALL of these when adding/removing fields or bumping version:
//
//  1. shared/CommLink/CommLink.h
//       COMM_MAX_PAYLOAD must be > sizeof(TelemetryPayload) + 9 (frame overhead).
//       Failing this silently drops every telemetry packet on the ESP32 USB forward path
//       because Serial.write() may block and the ESP32 UART HW FIFO is only 128 bytes.
//       Current V4 payload = 128 bytes; COMM_MAX_PAYLOAD is currently 256.
//
//  2. esp32/src/main.cpp  on_teensy_packet()
//       a) Bump TELEM_VERSION here — the version check in on_teensy_packet() is automatic.
//       b) `len >= sizeof(TelemetryPayload)` guard — automatically correct if struct grows.
//       c) Add a new volatile g_telem_* variable for each new field.
//       d) Copy the field out of `pkt` into the new g_telem_* variable.
//       e) Pass the variable to the relevant draw function in update_display().
//
//  3. teensy/src/main.cpp  send_telemetry()
//       Fill the new struct field from the appropriate g_state / sensor variable.
//
//  4. software/gui/flash_monitor.py  PacketDecoder._parse()
//       a) Add a new `if length >= N:` block (N = new total struct size) to unpack
//          the new fields with struct.unpack_from() at the correct byte offset.
//       b) Add the new key(s) to the info dict so tabs can consume them via TelemetryBus.
//       c) For field additions: add a new `if length >= N:` block at the correct offset.
//          For breaking changes (remove/reorder fields): bump TELEM_VERSION — the GUI and
//          ESP32 will reject mismatched packets with a clear error until both are reflashed.
//          Also update _TELEM_VERSION in flash_monitor.py (search "must match TELEM_VERSION").
//
//  5. (optional) software/gui/raw_data_tab.py
//       Add new field rows to the "Telemetry Payload" grid if you want live inspection.
//
// Byte-offset map (packed, no padding — verify with static_assert or python struct.calcsize):
//   [0]    uint32  timestamp_ms
//   [4]    float×9 pitch_rad … yaw_rad
//   [40]   uint8   robot_state
//   [41]   uint8   fault_code
//   [42]   float×3 test_val, hip_l_current_a, hip_r_current_a
//   [54]   uint16×14 ibus_ch[14]
//   [82]   uint8   ibus_alive
//   [83]   float×6 wm_l_vel … wm_r_vbus    ← V3 start
//   [107]  uint32×2 wm_l_error, wm_r_error
//   [115]  uint8×3  wm_l_state, wm_r_state, wm_mode
//   [118]  uint16×4 tof_dist_mm[4]          ← V4 start
//   [126]  uint16   tof_age_ms
//   [128]  ← end, sizeof = 128 bytes
//
#define TELEM_VERSION  4  // bump when adding/removing struct fields; triggers mismatch errors

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
} TelemetryPayload;  // 128 bytes — TELEM_VERSION 4
#ifdef __cplusplus
static_assert(sizeof(TelemetryPayload) == 128,
    "TelemetryPayload size changed — bump TELEM_VERSION, update COMM_MAX_PAYLOAD, and see PROPAGATION CHECKLIST");
#endif

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
