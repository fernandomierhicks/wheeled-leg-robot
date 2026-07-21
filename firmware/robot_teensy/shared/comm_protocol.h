#pragma once
#include <stdint.h>

// ── Bandwidth budget ─────────────────────────────────────────────────────────
// Teensy→ESP32 (UART, 4 Mbaud):  50 Hz × 2 frames × 130 bytes ≈  13 kB/s  — UART easily handles this.
// ESP32→PC     (USB serial CP2102): same 13 kB/s forwarded over Serial.
//   CP2102 max is 921600 baud = 92 kB/s.  Serial.begin() must be ≥ 921600 or the TX buffer
//   backs up, blocks CommLink::update(), starves the UART RX ring buffer, and causes CRC drops + lag.
//   If TelemetryPayload grows significantly, recalculate: bytes/frame × 50 Hz × 2 < 92 kB/s.
//   CP2102N supports up to 3 Mbaud if more headroom is ever needed.
//
// ── Frame constants ───────────────────────────────────────────────────────────
#define COMM_START_A 0xAA  // two-byte magic reduces false-sync probability to ~1/65536
#define COMM_START_B 0x55
#define COMM_END     0xEF
// Legacy aliases — remove after all unit tests are updated
#define COMM_START   COMM_START_A

// ── Source IDs ────────────────────────────────────────────────────────────────
#define COMM_SRC_TEENSY  0x01
#define COMM_SRC_ESP32   0x02
#define COMM_SRC_PC      0x03

// ── Packet types ──────────────────────────────────────────────────────────────
#define COMM_TYPE_TELEMETRY    0x01  // deprecated — replaced by TELEM_A + TELEM_B
#define COMM_TYPE_COMMAND      0x02
#define COMM_TYPE_ACK          0x03
#define COMM_TYPE_LOG          0x04
#define COMM_TYPE_CALIB_EVENT  0x05
#define COMM_TYPE_PARAM_REPORT 0x06  // Teensy→GUI: current value of one param
#define COMM_TYPE_TOF          0x07  // ESP32→Teensy: raw ToF distances (TofPayload)
#define COMM_TYPE_TELEM_A      0x10  // first 118 bytes of TelemetryPayload (frame = 128 bytes)
#define COMM_TYPE_TELEM_B      0x11  // next  120 bytes of TelemetryPayload (frame = 130 bytes)
#define COMM_TYPE_LOG_INFO     0x12  // Teensy→PC: SD-log directory / transfer metadata (LogInfoPayload)
#define COMM_TYPE_LOG_DATA     0x13  // Teensy→PC: SD-log file chunk (LogDataHeader + raw bytes)
#define COMM_TYPE_WIFI_DIAG    0x14  // ESP32→PC only: link/loop diagnostics (WifiDiagPayload)
#define COMM_TYPE_TELEM_FULL_WIFI 0x15  // ESP32→PC, WiFi-only: full TelemetryPayload as one datagram (WIFI_TELEM_COMBINED)
#define COMM_TYPE_ESP32_STATUS 0x16  // ESP32→Teensy: 5 Hz link heartbeat + ESP32 health (Esp32StatusPayload)

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

// ── Frame layout (10 bytes overhead) ─────────────────────────────────────────
//
//   [COMM_START_A]        1 byte  — 0xAA  ┐ two-byte magic; false-sync prob ~1/65536
//   [COMM_START_B]        1 byte  — 0x55  ┘
//   [type]                1 byte  — COMM_TYPE_*
//   [version]             1 byte  — payload struct version for this type
//   [source]              1 byte  — COMM_SRC_*
//   [seq]                 1 byte  — rolling 0-255 tx counter
//   [len_lo][len_hi]      2 bytes — payload length, little-endian
//   [...payload...]       len bytes
//   [checksum]            1 byte  — CRC-8 (poly 0x07, init 0x00, MSB-first) over
//                                   type,version,source,seq,len_lo,len_hi,payload[0..n-1]
//                                   (was 1-byte XOR before 2026-07-13 — flag-day change,
//                                    all three ends must be updated together)
//   [COMM_END]            1 byte  — 0xEF

// ── Fault codes (robot_state == STATE_ESTOP → fault_code says why) ────────────
// IMPORTANT: when adding/changing a fault code below, also update:
//   - software/gui/tabs/telem_format.py   _FAULT_NAMES and _FAULT_DESCRIPTIONS dicts
//   - esp32/src/main.cpp               fault_description()
//   - firmware/robot_teensy/README.md  fault code table
#define FAULT_NONE               0x00
#define FAULT_IMU_ERROR          0x01  // IMU reported ERROR during startup
#define FAULT_HIP_INIT_TIMEOUT   0x02  // no CAN reply from hip motors within 2 s of boot
#define FAULT_HIP_FEEDBACK_LOST  0x03  // hip CAN feedback timed out during operation
#define FAULT_HIP_LARGE_POS_CMD  0x04  // commanded position jump exceeded MAX_HIP_DELTA_RAD
#define FAULT_CALIBRATION_TIMEOUT 0x05 // hardstop not found within CALIB_SAFETY_BOUND_RAD
#define FAULT_HUMAN_ESTOP        0x06  // ESTOP requested by user via GUI button or radio
// 0x07 reserved — was FAULT_PARAM_OUT_OF_BOUNDS (removed; out-of-range writes always clamp)
#define FAULT_PITCH_WATCHDOG     0x08  // |pitch| > 50° for > 200 ms
#define FAULT_WHEEL_RUNAWAY      0x09  // wheel velocity exceeded 2× soft governor limit
#define FAULT_IMU_LOST           0x0A  // IMU left NOMINAL while RUNNING/JUMPING (silence or heavy loss)
#define FAULT_WHEEL_FEEDBACK_LOST 0x0B // wheel encoder timeout or ODrive error during operation
#define FAULT_WHEEL_INIT_TIMEOUT 0x0C  // no CAN reply from wheel motors within 2 s of boot
#define FAULT_STANDUP_FAILED     0x0D  // standup denied (pitch out of recoverable range) or exhausted retries/diverged

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
        case FAULT_CALIBRATION_TIMEOUT:
        case FAULT_STANDUP_FAILED:        return FAULT_SEVERITY_REPOSITION;
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

// ── Payload: ESP32 status heartbeat (ESP32→Teensy, COMM_TYPE_ESP32_STATUS) ───
// Sent at 5 Hz independent of the Teensy link's own health, so the Teensy (and
// via TelemetryPayload, the GUI) can tell "ESP32 alive, Teensy silent" apart
// from "whole link down". Telemetry-only — never causes a fault or state change.
#define ESP32_STATUS_PAYLOAD_V1 1
typedef struct __attribute__((packed)) {
    uint32_t uptime_ms;
    uint8_t  wifi_ok;               // WiFi.status() == WL_CONNECTED
    uint8_t  tcp_client_connected;  // GUI TCP client attached
    int8_t   rssi_dbm;
    uint8_t  reserved;
    uint16_t uplink_drops;          // ESP32 g_uplink_drops (clamped u16)
    uint16_t uart_rx_drops;         // ESP32 g_teensy.rx_drops()    (Teensy→ESP32 dir)
    uint16_t uart_seq_gaps;         // ESP32 g_teensy.rx_seq_gaps() (Teensy→ESP32 dir)
} Esp32StatusPayload;               // 14 bytes

#ifdef __cplusplus
static_assert(sizeof(Esp32StatusPayload) == 14, "Esp32StatusPayload must be 14 bytes");
#endif

// ── Payload: telemetry ────────────────────────────────────────────────────────
//
// PROPAGATION CHECKLIST — touch ALL of these when adding/removing fields or bumping version:
//
//  1. shared/CommLink/CommLink.h
//       COMM_MAX_PAYLOAD must be > sizeof(TelemetryPayload) + 9 (frame overhead).
//       Failing this silently drops every telemetry packet on the ESP32 USB forward path
//       because Serial.write() may block and the ESP32 UART HW FIFO is only 128 bytes.
//       Current V5 payload = 244 bytes; COMM_MAX_PAYLOAD is currently 512.
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
//  5. software/gui/raw_data_tab.py  ← REQUIRED
//       Add new field rows to the matching tab (IMU / Hip Motors / Wheel Motors / Control /
//       RC-ToF) using _add_row(), and populate the value in _on_packet().
//       Also update imu_tab.py if the new field is IMU-related (orientation, rates, accel).
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
//   [126]  float×2  roll_rate_rads, yaw_rate_rads            ← V6 start (tof_age_ms removed)
//   [134]  float×3  accel_x_ms2, accel_y_ms2, accel_z_ms2
//   [146]  float×2  hip_l_vel_rads, hip_r_vel_rads
//   [154]  float×10 hip_l/r_cmd_pos, hip_l/r_cmd_vel, hip_l/r_cmd_kp, hip_l/r_cmd_kd, hip_l/r_cmd_tff
//   [194]  float×5  theta_ref, v_ref, omega_cmd_rds, tau_sym, tau_yaw        ← V7 change
//   [214]  float×2  ff1_out, ff2_out
//   [222]  uint16   health_flags
//   [224]  uint8    imu_packet_loss_pct
//   [225]  uint8    jump_state
//   [226]  uint32   loop_count
//   [230]  uint8    active_profile                                            ← V8 start
//   [231]  float    pitch_trim_rad
//   [235]  uint8    esp32_link_ok                                             ← V9 start
//   [236]  uint16   esp32_status_age_ms
//   [238]  uint16   uart_rx_drops
//   [240]  uint16   uart_seq_gaps
//   [242]  float    gain_sched_alpha                                          ← V10 start
//   [246]  uint8    standup_state                                             ← V11 start
//   [247]  ← end, sizeof = 247 bytes
//
#define TELEM_VERSION  11  // bump when adding/removing struct fields; triggers mismatch errors

typedef struct __attribute__((packed)) {
    uint32_t timestamp_ms;
    float    pitch_rad;
    float    pitch_rate_rads;
    float    wheel_vel_avg_ms;
    float    hip_l_pos_rad;
    float    hip_r_pos_rad;
    float    whl_tau_l;      // final left  wheel torque command [N·m] (0 when LQR disabled; hip ramp target during calibration)
    float    whl_tau_r;      // final right wheel torque command [N·m]
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
    // V6: tof_age_ms removed — GUI timestamps arrival instead (gain_sched_alpha added
    // back separately, as a V10 field at the end of the struct — see below)
    // V5 additions — IMU rates + accel (20 bytes)
    float    roll_rate_rads;     // roll  angular rate [rad/s] (gyro X)
    float    yaw_rate_rads;      // yaw   angular rate [rad/s] (gyro Z) — needed for Yaw PI
    float    accel_x_ms2;        // linear accel X (forward)  [m/s²]
    float    accel_y_ms2;        // linear accel Y (left)     [m/s²]
    float    accel_z_ms2;        // linear accel Z (up)       [m/s²] — freefall/landing detection
    // V5 additions — hip motor velocity feedback (8 bytes)
    float    hip_l_vel_rads;     // hip L rotor velocity [rad/s]
    float    hip_r_vel_rads;     // hip R rotor velocity [rad/s]
    // V5 additions — hip motor MIT setpoints — cmd vs feedback enables tracking quality plot (40 bytes)
    float    hip_l_cmd_pos_rad;  // commanded position  L [rad] (hm_sp_L.p)
    float    hip_r_cmd_pos_rad;  // commanded position  R [rad] (hm_sp_R.p)
    float    hip_l_cmd_vel_rads; // commanded velocity  L [rad/s] (hm_sp_L.v)
    float    hip_r_cmd_vel_rads; // commanded velocity  R [rad/s] (hm_sp_R.v)
    float    hip_l_cmd_kp;       // position gain       L [N·m/rad] (hm_sp_L.kp)
    float    hip_r_cmd_kp;       // position gain       R (hm_sp_R.kp)
    float    hip_l_cmd_kd;       // damping gain        L [N·m·s/rad] (hm_sp_L.kd)
    float    hip_r_cmd_kd;       // damping gain        R (hm_sp_R.kd)
    float    hip_l_cmd_tff;      // torque feedforward  L [N·m] (hm_sp_L.tff)
    float    hip_r_cmd_tff;      // torque feedforward  R [N·m] (hm_sp_R.tff)
    // V7: balance controller internals — setpoints, controlled vars, effort signals only
    float    theta_ref;          // vel PI effort / LQR setpoint [rad] — lean angle reference
    float    v_ref;              // vel PI setpoint [m/s] — velocity reference
    float    omega_cmd_rds;      // yaw PI setpoint [rad/s] — desired yaw rate (CH4 radio)
    float    tau_sym;            // LQR symmetric wheel torque [N·m] — balance effort
    float    tau_yaw;            // yaw PI differential torque [N·m] — yaw effort
    // V7: feedforward outputs (ff4 removed — always 0 until centripetal FF is implemented)
    float    ff1_out;            // hip reaction torque cancellation [N·m]
    float    ff2_out;            // gravity compensation [N·m]
    // V5 additions — diagnostics (8 bytes)
    uint16_t health_flags;       // packed health bits — see HEALTH_FLAG_* below
    uint8_t  imu_packet_loss_pct;// IMU link loss 0-100% (imu_packet_loss() × 100)
    uint8_t  jump_state;         // Jump FSM phase (0=BALANCE/inactive) — Phase 7
    uint32_t loop_count;         // control loop counter; GUI uses delta to detect dropped frames
    // V8 additions
    uint8_t  active_profile;     // speed profile selected by CH9 (0=slow, 1=normal, 2=fast)
    float    pitch_trim_rad;     // pitch equilibrium trim from CH7 [rad]; hook only until LQR wired
    // V9 additions — ESP32↔Teensy link supervision (7 bytes)
    uint8_t  esp32_link_ok;       // 1 = ESP32_STATUS heard < 1000 ms ago
    uint16_t esp32_status_age_ms; // ms since last ESP32_STATUS (clamped 65535)
    uint16_t uart_rx_drops;       // Teensy g_comm.rx_drops()    (ESP32→Teensy dir, clamped)
    uint16_t uart_seq_gaps;       // Teensy g_comm.rx_seq_gaps() (ESP32→Teensy dir, clamped)
    // V10 addition — leg gain-schedule blend factor (4 bytes)
    float    gain_sched_alpha;    // control_loop.cpp hip gain interpolation [0=retracted,1=extended];
                                   // mirrors g_state.gain_sched_alpha (Phase 5); 0.5 until calibrated
    // V11 addition — standing-up recovery FSM phase (1 byte)
    uint8_t  standup_state;       // Standup FSM phase (0=CROUCH/inactive) — mirrors g_state.standup_state
} TelemetryPayload;  // 247 bytes — TELEM_VERSION 11

// Split offsets for two-packet telemetry (TELEM_A + TELEM_B)
// Each frame ≤130 bytes — both drain safely with UART FIFO threshold=32
#define TELEM_A_LEN  118u  // bytes   0-117: IMU, hip pos, RC, wheel motors
#define TELEM_B_LEN  129u  // bytes 118-246: ToF, rates, accel, hip cmd, control internals, profile,
                           //                pitch trim, ESP32 link supervision, gain-schedule alpha,
                           //                standup state

#ifdef __cplusplus
static_assert(sizeof(TelemetryPayload) == 247,
    "TelemetryPayload size changed — bump TELEM_VERSION, update COMM_MAX_PAYLOAD, and see PROPAGATION CHECKLIST");
static_assert(TELEM_A_LEN + TELEM_B_LEN == 247, "TELEM_A_LEN + TELEM_B_LEN must equal sizeof(TelemetryPayload)");
#endif

// ── High-datarate SD log (.wlog) ──────────────────────────────────────────────
//
// The Teensy logs one LogRecord per 500 Hz control tick to a preallocated .wlog
// file on the built-in microSD. A LogRecord WRAPS the unchanged TelemetryPayload
// (so the live 50 Hz wire format is untouched) and prepends a micros() timestamp.
// The PC reads t_micros, then decodes the embedded 247-byte telem blob with the
// SAME split-telemetry decoder used for live data. See software/gui/log_playback.py.
//
#define WLOG_FORMAT_V1  1
#define WLOG_SAMPLE_HZ  500      // control/log tick rate [Hz]
#define LOG_CHUNK_DATA  480      // max file bytes per COMM_TYPE_LOG_DATA frame

typedef struct __attribute__((packed)) {
    char     magic[8];         // "WLRLOG\0" (7 used + 1 pad)
    uint8_t  format_version;   // WLOG_FORMAT_V1
    uint8_t  telem_version;    // == TELEM_VERSION at capture time (decode key)
    uint16_t record_size;      // == sizeof(LogRecord)
    uint16_t sample_rate_hz;   // WLOG_SAMPLE_HZ
    uint32_t start_millis;     // millis() at sd_logger_start()
    uint8_t  reserved[14];
} WlogHeader;                  // 32 bytes — file header

typedef struct __attribute__((packed)) {
    uint32_t         t_micros; // micros() at capture — sub-ms inter-tick timing
    TelemetryPayload telem;    // the exact 247-byte struct, TELEM_VERSION 11
} LogRecord;                   // 251 bytes — one per control tick

// ── Payload: COMM_TYPE_LOG_INFO (Teensy→PC) ──────────────────────────────────
#define LOG_INFO_PAYLOAD_V1  1
#define LOG_INFO_ENTRY       0x01  // one directory entry (reply to LIST): file_index, file_size
#define LOG_INFO_LIST_END    0x02  // end of directory listing
#define LOG_INFO_XFER_BEGIN  0x03  // start of a GET: file_index, file_size, total_chunks
#define LOG_INFO_XFER_END    0x04  // end of a GET: file_index, total_chunks, crc32
#define LOG_INFO_STATUS      0x05  // START/STOP/DELETE ack: file_index, status (0=ok)

typedef struct __attribute__((packed)) {
    uint8_t  info_type;    // LOG_INFO_*
    uint16_t file_index;   // LOGxxxx index (0xFFFF = n/a)
    uint32_t file_size;    // bytes (ENTRY / XFER_BEGIN)
    uint32_t total_chunks; // XFER_BEGIN / XFER_END
    uint32_t crc32;        // XFER_END — CRC32 of the whole file
    uint8_t  status;       // STATUS: 0=ok, non-zero=error code
} LogInfoPayload;          // 16 bytes

// ── Payload: COMM_TYPE_LOG_DATA (Teensy→PC) ──────────────────────────────────
// Frame payload = LogDataHeader (8 B) + data_len raw file bytes (≤ LOG_CHUNK_DATA).
#define LOG_DATA_PAYLOAD_V1  1
typedef struct __attribute__((packed)) {
    uint16_t file_index;
    uint32_t chunk_index;  // 0-based; file byte offset = chunk_index * LOG_CHUNK_DATA
    uint16_t data_len;     // raw file bytes following this header (≤ LOG_CHUNK_DATA)
} LogDataHeader;           // 8 bytes

#ifdef __cplusplus
static_assert(sizeof(WlogHeader) == 32, "WlogHeader must be 32 bytes");
static_assert(sizeof(LogRecord) == 251, "LogRecord must be sizeof(uint32)+sizeof(TelemetryPayload)");
static_assert(sizeof(LogInfoPayload) == 16, "LogInfoPayload must be 16 bytes");
static_assert(sizeof(LogDataHeader) == 8, "LogDataHeader must be 8 bytes");
// LOG_CHUNK_DATA + sizeof(LogDataHeader) <= COMM_MAX_PAYLOAD is checked in CommLink.h,
// since COMM_MAX_PAYLOAD is not yet defined at this point when comm_protocol.h is
// included directly (without CommLink.h first, as several .cpp files do).
#endif

// ── Health flag bits (TelemetryPayload::health_flags) ────────────────────────
#define HEALTH_HIP_L_OK          (1u << 0)  // hm_L.ok — CAN feedback fresh
#define HEALTH_HIP_R_OK          (1u << 1)  // hm_R.ok
#define HEALTH_WM_L_OK           (1u << 2)  // wm_L.ok — encoder feedback fresh
#define HEALTH_WM_R_OK           (1u << 3)  // wm_R.ok
#define HEALTH_HIP_LIMITS_VALID  (1u << 4)  // hm_limits_L.valid && hm_limits_R.valid
#define HEALTH_IMU_NOMINAL       (1u << 5)  // imu_state() == ImuState::NOMINAL
#define HEALTH_LQR_ACTIVE        (1u << 6)  // PARAM_LQR_ENABLE && STATE_RUNNING
#define HEALTH_VEL_PI_SAT        (1u << 7)  // theta_ref clamped at theta_max this tick
#define HEALTH_YAW_PI_SAT        (1u << 8)  // tau_yaw clamped at torque_max this tick
#define HEALTH_WM_L_VEL_LIMITED  (1u << 9)  // soft governor clamping left wheel
#define HEALTH_WM_R_VEL_LIMITED  (1u << 10) // soft governor clamping right wheel
#define HEALTH_LOOP_OVERRUN      (1u << 11) // control-loop work exceeded the 2 ms budget within the last second

// ── Payload: WiFi diagnostics (COMM_TYPE_WIFI_DIAG, ESP32→PC only) ───────────
#define WIFI_DIAG_PAYLOAD_V1  1
#define WIFI_DIAG_PAYLOAD_V2  2  // adds Teensy-link status (the "ESP32 alive" carrier — see UARTplat.md)

// build_variant_flags bits (WifiDiagPayload::build_variant_flags)
#define WIFI_DIAG_FLAG_UNICAST          (1u << 0)
#define WIFI_DIAG_FLAG_COMBINED         (1u << 1)
#define WIFI_DIAG_FLAG_TX_POWER_MAX     (1u << 2)
#define WIFI_DIAG_FLAG_NEO_DISABLED     (1u << 3)
#define WIFI_DIAG_FLAG_DISPLAY_DISABLED (1u << 4)
#define WIFI_DIAG_FLAG_TRANSPORT_GATING (1u << 5)

// active_telem_transport values (WifiDiagPayload::active_telem_transport)
#define WIFI_DIAG_TRANSPORT_BOTH_LEGACY 0  // gating disabled or no source announced
#define WIFI_DIAG_TRANSPORT_USB_ONLY    1  // GUI announced USB active — WiFi telemetry suppressed
#define WIFI_DIAG_TRANSPORT_WIFI_ONLY   2  // GUI announced WiFi active

typedef struct __attribute__((packed)) {
    uint32_t esp_uptime_ms;
    int8_t   rssi_dbm;              // WiFi.RSSI()
    uint8_t  wifi_channel;
    uint8_t  wifi_status;
    uint8_t  tx_power_raw;
    uint32_t free_heap;
    uint32_t min_free_heap;
    uint16_t loop_max_us;           // max loop() duration since last diag tick, then reset
    uint16_t udp_send_max_us;       // max time inside a single UDPStream::write() since last diag tick, then reset
    uint16_t wifi_reconnect_count;
    uint16_t udp_send_fail_count;
    uint8_t  build_variant_flags;   // WIFI_DIAG_FLAG_*
    uint8_t  active_telem_transport;// WIFI_DIAG_TRANSPORT_*
    // V2 additions — Teensy-link status, so the GUI can tell "ESP32 alive, Teensy
    // silent" apart from "whole link down" (12 bytes)
    uint8_t  teensy_link_up;        // (millis() - g_last_teensy_ms) < TEENSY_LINK_TIMEOUT_MS
    uint8_t  reserved2;
    uint16_t ms_since_teensy;       // ms since last Teensy packet (clamped 65535)
    uint16_t uart_crc_drops;        // g_teensy.rx_drops()    (Teensy→ESP32 dir)
    uint16_t uart_seq_gaps;         // g_teensy.rx_seq_gaps() (Teensy→ESP32 dir)
    uint16_t uplink_queue_drops;    // g_uplink_drops
    uint16_t tcp_send_max_us;       // max blocking TCP write since last diag tick, then reset
} WifiDiagPayload;  // 38 bytes — WIFI_DIAG_PAYLOAD_V2

#ifdef __cplusplus
static_assert(sizeof(WifiDiagPayload) == 38, "WifiDiagPayload must be 38 bytes");
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
#define CMD_ID_PING       0x02  // payload: none — GUI heartbeat; feeds the MANUAL GUI watchdog only
#define CMD_ID_HIP        0x05  // payload: uint8_t motor_id, uint8_t sub_cmd [, 5×float]
#define CMD_ID_REBOOT     0x06  // payload: none — triggers a full MCU reset (reruns setup())
#define CMD_ID_WHEEL      0x07  // payload: uint8_t sub_cmd [, data]
#define CMD_ID_SET_TELEM_TRANSPORT 0x08  // PC→ESP32 only (intercepted, never forwarded to Teensy):
                                          // payload: uint8_t suppress_wifi (0=send WiFi telemetry
                                          // when connected [default], 1=suppress — GUI is on USB)

// Wheel sub-commands (CMD_ID_WHEEL payload byte 1)
#define WHEEL_SUB_SET_MODE     0x01  // payload: uint8_t mode (WheelMode)
#define WHEEL_SUB_SEND         0x02  // payload: float L, float R
#define WHEEL_SUB_CLEAR_ERRORS 0x03  // no payload
#define CMD_ID_PARAM_SET  0x10  // payload: uint16_t param_id, float value  (6 bytes after cmd_id)
#define CMD_ID_PARAM_GET  0x11  // payload: uint16_t param_id  (0xFFFF = dump all)
#define CMD_ID_LOG        0x12  // payload: uint8_t sub_cmd [, args] — high-datarate SD logging
#define CMD_ID_PARAM_RESET_DEFAULTS  0x13  // payload: none — reverts all writable params to
                                            // compile-time defaults, replies with a full PARAM_REPORT dump
#define CMD_ID_TEST_INJECT_CORRUPT   0x14  // TEST ONLY (Phase 9, UARTplat.md stress testing):
                                            // payload: uint8_t count, uint8_t target, uint8_t mode
                                            // target 0 = UART (Teensy's next `count` outgoing
                                            //   TELEM_A frames to the ESP32 get corrupted per
                                            //   `mode`; forwarded through by the ESP32 like any
                                            //   command)
                                            // target 1 = WiFi (ESP32's next `count` outgoing
                                            //   TELEM_A UDP datagrams to the GUI get corrupted
                                            //   per `mode`; intercepted by the ESP32, never
                                            //   forwarded to the Teensy — see forward_to_teensy())
                                            // mode 1 = flipped CRC-8 byte (default; matches
                                            //   CommLink::send()'s corrupt_mode_for_test values)
                                            // mode 2 = flipped END byte
                                            // mode 3 = oversized on-wire length field (exercises
                                            //   the receiver's length-guard / resync path)
                                            // Verifies the receiving side's parser defenses
                                            // actually detect and drop a bad frame, not just
                                            // "hasn't seen corruption yet". target byte omitted
                                            // (len==2) is treated as target 0; mode byte omitted
                                            // (len<4) is treated as mode 1, both for backward
                                            // compatibility.

// Log sub-commands (CMD_ID_LOG payload byte 1)
#define LOG_SUB_START     0x01  // + uint32_t duration_ms (0 = log until STOP)
#define LOG_SUB_STOP      0x02  // no args — close the active log file
#define LOG_SUB_LIST      0x03  // no args — reply: one LOG_INFO ENTRY per file, then LIST_END
#define LOG_SUB_GET       0x04  // + uint16_t file_index, uint32_t start_chunk — stream LOG_DATA
#define LOG_SUB_DELETE    0x05  // + uint16_t file_index — erase one .wlog file
#define LOG_SUB_CHUNK_ACK 0x06  // ESP32→Teensy internal: + uint16_t file_index, uint32_t chunk_index

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
