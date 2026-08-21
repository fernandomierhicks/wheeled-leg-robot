#include <Arduino.h>
#include <stdarg.h>
#include "config.h"
#include "robot_state.h"
#include "state_machine.h"
#include "control_loop.h"
#include "CommLink.h"
#include "comm_protocol.h"
#include "command_validation.h"
#include "IMU.h"
#include "hip_motors.h"
#include "wheel_motors.h"
#include "RgbLed.h"
#include "Buzzer.h"
#include "param_registry.h"
#include "Crsf.h"
#include "CrsfTelem.h"
#include "sd_logger.h"
#include "limit_switches.h"
#include "live_tune.h"
#include "lean_turn.h"

CommLink g_comm(Serial5, COMM_SRC_TEENSY);     // ESP32 UART bridge
CommLink g_comm_usb(Serial, COMM_SRC_TEENSY);  // direct PC USB
Crsf     g_rc(Serial4);                        // ELRS/CRSF RC link, RX pin 16 / TX pin 17
CrsfTelemetry g_rc_telem;                      // return-path telemetry scheduler
RgbLed   g_led(PIN_LED_R, PIN_LED_G, PIN_LED_B);
Buzzer   g_buzzer(PIN_BUZZER);

HipCmd   g_hip_cmd = {};

enum class ArmAuthority : uint8_t { NONE, GUI, RADIO };
static ArmAuthority s_arm_authority = ArmAuthority::NONE;

static const char* arm_authority_name() {
    switch (s_arm_authority) {
        case ArmAuthority::GUI:   return "GUI";
        case ArmAuthority::RADIO: return "RADIO";
        default:                  return "NONE";
    }
}

// USB Serial's bool conversion only reports that the host configured the
// device; it does not guarantee the host is currently draining its endpoint.
// A write to an abandoned/full endpoint can block the 500 Hz loop for >100 ms.
static bool usb_send_if_ready(uint8_t type, uint8_t version,
                              const void* payload, uint16_t len) {
    if (!Serial || Serial.availableForWrite() < (int)(10u + len)) return false;
    return g_comm_usb.send(type, version, payload, len);
}

// Latest ToF packet received from ESP32 (updated in on_command, read in state machine and telemetry)
TofPayload g_tof         = {{0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF}, 0xFFFF, 0xFFFF};
uint32_t   g_tof_last_ms = 0;

// Latest ESP32_STATUS heartbeat (updated in on_command, read in fill_telemetry).
// Telemetry only — never causes a fault or state change. See Phase 3, UARTplat.md.
Esp32StatusPayload g_esp32_status      = {};
uint32_t           s_last_esp32_status_ms = 0;

// ── Logging ───────────────────────────────────────────────────────────────────

void comm_log(uint8_t level, const char* fmt, ...) {
    char msg[120];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    // vsnprintf returns the would-be length; the buffer holds at most
    // sizeof-1 chars (audit W6 — the old clamp sent the NUL terminator).
    if (n > (int)sizeof(msg) - 1) n = (int)sizeof(msg) - 1;
    uint8_t buf[1 + sizeof(msg)];
    buf[0] = level;
    memcpy(buf + 1, msg, n);
    g_comm.send(COMM_TYPE_LOG, LOG_PAYLOAD_V1, buf, 1 + n);
    usb_send_if_ready(COMM_TYPE_LOG, LOG_PAYLOAD_V1, buf, 1 + n);
}

// Which link requested the active SD-log GET (audit W5): bulk LOG_DATA chunks
// go only to that link — duplicating ~490 B frames onto the other link wastes
// its bandwidth for a stream nobody is reading (the GUI listens on one source).
static bool s_log_get_via_usb = false;

static void sd_logger_send(uint8_t type, uint8_t version, const void* payload, uint16_t len) {
    if (type == COMM_TYPE_LOG_DATA) {
        if (s_log_get_via_usb) { if (Serial) g_comm_usb.send(type, version, payload, len); }
        else                   g_comm.send(type, version, payload, len);
        return;
    }
    g_comm.send(type, version, payload, len);
    usb_send_if_ready(type, version, payload, len);
}

void comm_send_calib_event(uint8_t axis, uint8_t event,
                            float pos_rad, float min_rad, float max_rad) {
    CalibEventPayload p;
    p.axis    = axis;
    p.event   = event;
    p.pos_rad = pos_rad;
    p.min_rad = min_rad;
    p.max_rad = max_rad;
    g_comm.send(COMM_TYPE_CALIB_EVENT, CALIB_EVENT_PAYLOAD_V1, &p, sizeof(p));
    usb_send_if_ready(COMM_TYPE_CALIB_EVENT, CALIB_EVENT_PAYLOAD_V1, &p, sizeof(p));
}

// ── Param report helper ───────────────────────────────────────────────────────

static void send_param_report(uint16_t idx) {
    Param p;
    if (!param_by_index(idx, &p)) return;
    ParamReportPayload rpt;
    rpt.param_id = p.id;
    rpt.value    = p.value;
    rpt.min_val  = p.min_val;
    rpt.max_val  = p.max_val;
    rpt.flags    = p.flags;
    static_assert(sizeof(rpt.name) == 20, "name size mismatch");
    memcpy(rpt.name, p.name, 20);
    g_comm.send(COMM_TYPE_PARAM_REPORT, PARAM_REPORT_PAYLOAD_V1, &rpt, sizeof(rpt));
    usb_send_if_ready(COMM_TYPE_PARAM_REPORT, PARAM_REPORT_PAYLOAD_V1, &rpt, sizeof(rpt));
}

// TEST ONLY (Phase 9, UARTplat.md stress testing): armed by CMD_ID_TEST_INJECT_CORRUPT,
// consumed one-at-a-time by send_telemetry() to deliberately send malformed frames to
// the ESP32 and confirm its CommLink::update() actually detects and drops them.
static volatile uint8_t g_test_corrupt_remaining = 0;
static volatile uint8_t g_test_corrupt_mode = 1;  // CommLink::send() corrupt_mode_for_test value (1=CRC,2=END,3=length)

// ── Paced param dump (PARAM_GET 0xFFFF / PARAM_RESET_DEFAULTS reply) ─────────
// Sending all ~60 PARAM_REPORTs in one burst can overflow the 512 B Serial5 TX
// buffer and blocks the 500 Hz loop for several ms. Instead, the command
// handlers below just arm this cursor; service_param_dump() — called once per
// loop() tick, alongside receive_commands() — sends a few at a time.
// 4 x 45 B = 180 B/tick fits the TX buffer without blocking; a ~60-param dump
// finishes in ~30 ms. See Phase 2, UARTplat.md.
static uint16_t s_param_dump_cursor = 0xFFFF;  // >= param_count() => idle

static void service_param_dump() {
    uint16_t sent = 0;
    while (sent < 4 && s_param_dump_cursor < param_count()) {
        send_param_report(s_param_dump_cursor);
        s_param_dump_cursor++;
        sent++;
    }
}

// ── Command permission matrix ─────────────────────────────────────────────────
// Single place that decides which commands are accepted in which state.
// Commands are requests: anything rejected here is logged and dropped, so it
// can never sit latched and fire on a later state change (e.g. a hip MIT
// command sent while in STANDBY executing the instant MANUAL is entered).
static bool cmd_allowed(uint8_t cmd_id, RobotStateEnum s) {
    switch (cmd_id) {
        case CMD_ID_HIP:                // direct motor commands: MANUAL only
        case CMD_ID_WHEEL:
            return s == STATE_MANUAL;
        case CMD_ID_REBOOT:             // never reset the MCU with torque active
            return s == STATE_STARTUP || s == STATE_STANDBY || s == STATE_ESTOP;
        default:                        // SET_MODE / PING / PARAM_* / LOG: any state
            return true;
    }
}

// Very short chirp for a param write, in any state — distinct from RADIO_ACQ_MELODY
// (main.cpp's radio_update()) so the two aren't confused on the bench.
static const BuzzerNote PARAM_SET_CHIRP[] = {{84, 30, 0}};  // C6, 30 ms

// Very short chirps for SD-log lifecycle events (bench audibility while
// running tuning.md's Test Protocol without watching the screen) — distinct
// pitches from each other and from PARAM_SET_CHIRP/RADIO_ACQ_MELODY so all
// are tellable apart by ear alone.
static const BuzzerNote LOG_START_CHIRP[] = {{79, 40, 0}};               // G5 — recording started
static const BuzzerNote LOG_GET_CHIRP[]   = {{74, 40, 0}};               // D5 — download started
static const BuzzerNote LOG_DONE_CHIRP[]  = {{79, 40, 15}, {86, 60, 0}}; // G5→D6 — download complete

static constexpr uint8_t COMMAND_RESULT_CACHE_SIZE = 8;
static CommandResultPayload s_command_result_cache[COMMAND_RESULT_CACHE_SIZE] = {};
static bool s_command_result_cache_valid[COMMAND_RESULT_CACHE_SIZE] = {};
static uint8_t s_command_result_cache_next = 0;

static void emit_command_result(const CommandResultPayload& result) {
    g_comm.send(COMM_TYPE_COMMAND_RESULT, COMMAND_RESULT_PAYLOAD_V1, &result, sizeof(result));
    usb_send_if_ready(COMM_TYPE_COMMAND_RESULT, COMMAND_RESULT_PAYLOAD_V1, &result, sizeof(result));
}

static void send_command_result(uint32_t request_id, uint8_t cmd_id,
                                uint8_t status, uint8_t reason) {
    CommandResultPayload result = {
        request_id, cmd_id, status, reason, (uint8_t)g_state.state
    };
    s_command_result_cache[s_command_result_cache_next] = result;
    s_command_result_cache_valid[s_command_result_cache_next] = true;
    s_command_result_cache_next = (s_command_result_cache_next + 1) % COMMAND_RESULT_CACHE_SIZE;
    emit_command_result(result);
}

static bool resend_cached_command_result(uint32_t request_id) {
    for (uint8_t i = 0; i < COMMAND_RESULT_CACHE_SIZE; ++i) {
        if (s_command_result_cache_valid[i] && s_command_result_cache[i].request_id == request_id) {
            emit_command_result(s_command_result_cache[i]);
            return true;
        }
    }
    return false;
}

// ── Command handler ───────────────────────────────────────────────────────────

// Defined below (near run_control_loop); forward-declared here for the SD-log
// command path. See its definition for the self-inflicted-stall rationale.
static void forgive_sd_blocking_stall();

static void on_command(uint8_t type, uint8_t version, uint8_t source,
                       const uint8_t* payload, uint16_t len) {
    (void)version;

    // ToF packet from ESP32: store latest distances for telemetry and obstacle avoidance
    if (type == COMM_TYPE_TOF && len >= (uint16_t)sizeof(TofPayload)) {
        memcpy(&g_tof, payload, sizeof(TofPayload));
        g_tof_last_ms = millis();
        return;
    }

    // ESP32 status heartbeat: telemetry only, no fault/state change (Phase 3, UARTplat.md)
    if (type == COMM_TYPE_ESP32_STATUS && len >= (uint16_t)sizeof(Esp32StatusPayload)) {
        memcpy(&g_esp32_status, payload, sizeof(Esp32StatusPayload));
        s_last_esp32_status_ms = millis();
        return;
    }

    if (type != COMM_TYPE_COMMAND) return;

    ValidatedCommand validated{};
    if (!validate_command_payload(version, payload, len, &validated)) {
        uint8_t rejected_id = validated.len ? validated.bytes[0] : 0;
        comm_log(LOG_LEVEL_WARN, "CMD rejected: version=%u len=%u reason=%u",
                 version, len, validated.reason);
        if (version == CMD_PAYLOAD_V2)
            send_command_result(validated.request_id, rejected_id,
                                CMD_RESULT_REJECTED, validated.reason);
        return;
    }
    payload = validated.bytes;
    len = validated.len;
    const bool wants_result = version == CMD_PAYLOAD_V2;
    if (wants_result && resend_cached_command_result(validated.request_id)) return;
    auto reply = [&](uint8_t status, uint8_t reason = CMD_REASON_NONE) {
        if (wants_result)
            send_command_result(validated.request_id, payload[0], status, reason);
    };

    stateMachine_ping_gui_watchdog();  // feed MANUAL-mode GUI watchdog (any command proves GUI alive)
    uint8_t cmd_id = payload[0];

    if (!cmd_allowed(cmd_id, g_state.state)) {
        comm_log(LOG_LEVEL_WARN, "CMD 0x%02X rejected in state %d", cmd_id, (int)g_state.state);
        reply(CMD_RESULT_REJECTED, CMD_REASON_WRONG_STATE);
        return;
    }

    // ── Mode change: signal the state machine ─────────────────────────────────
    if (cmd_id == CMD_ID_SET_MODE) {
        uint8_t target = payload[1];
        comm_log(LOG_LEVEL_INFO, "CMD set_mode -> %d", target);
        bool accepted = false;
        if (target == g_state.state) accepted = true;
        else if (target == STATE_MANUAL) accepted = stateMachine_request_manual();
        if (target == STATE_STANDBY) {
            // From ESTOP: attempt soft-clear (ESTOP→STANDBY directly for SOFT faults).
            // Active states enter explicit DISARMING; manual/calibration exit directly.
            if (g_state.state == STATE_ESTOP) accepted = stateMachine_request_soft_clear();
            else if (g_state.state == STATE_RUNNING || g_state.state == STATE_JUMPING ||
                     g_state.state == STATE_STANDING_UP) {
                comm_log(LOG_LEVEL_INFO,
                         "DISARM reason=GUI_REQUEST: authority=%s state=%d -> DISARMING",
                         arm_authority_name(), (int)g_state.state);
                accepted = stateMachine_disarm_running();
            }
            else if (g_state.state == STATE_MANUAL || g_state.state == STATE_CALIBRATION)
                accepted = stateMachine_exit_manual();
        }
        if (target == STATE_STARTUP)     accepted = stateMachine_request_reset();
        if (target == STATE_CALIBRATION) accepted = stateMachine_request_calibration();
        if (target == STATE_ESTOP)       accepted = stateMachine_request_estop();
        // STATE_RUNNING: previously radio-only (CH10 arm switch, main.cpp radio_update()).
        // Routed through the identical req_running() gate in state_machine.cpp — same
        // IMU/calibration/motor-enable checks apply, no separate/weaker path.
        if (target == STATE_RUNNING) {
            if (g_rc.alive()) {
                comm_log(LOG_LEVEL_WARN,
                         "ARM rejected: GUI requested RUNNING while radio ONLINE (CH10=%u); radio owns arming",
                         (unsigned)g_rc.channel(10));
                accepted = false;
            } else {
                accepted = stateMachine_request_running();
                if (accepted) {
                    s_arm_authority = ArmAuthority::GUI;
                    comm_log(LOG_LEVEL_INFO,
                             "ARM authority=GUI: radio absent; GUI RUNNING request accepted");
                }
            }
        }
        if (target == STATE_JUMPING)     accepted = stateMachine_request_jump();
        // STANDING_UP, CMD_REJECT, and DISARMING are internal states, not
        // direct operator targets.
        if (target == STATE_CMD_REJECT || target == STATE_STANDING_UP || target == STATE_DISARMING) {
            reply(CMD_RESULT_REJECTED, CMD_REASON_INVALID_TARGET);
            return;
        }
        reply(accepted ? CMD_RESULT_ACCEPTED : CMD_RESULT_REJECTED,
              accepted ? CMD_REASON_NONE : CMD_REASON_GUARD_REJECTED);
        return;
    }

    if (cmd_id == CMD_ID_PING) {
        reply(CMD_RESULT_APPLIED);
        return;
    }

    if (cmd_id == CMD_ID_SET_TELEM_TRANSPORT) {
        // This command is ESP32-local. Seeing it here means the GUI selected a
        // direct Teensy link, where no telemetry transport can be gated.
        reply(CMD_RESULT_REJECTED, CMD_REASON_OPERATION_FAILED);
        return;
    }

    // ── Reboot: full MCU reset — re-runs setup() from scratch ──────────────────
    // Only reachable in STARTUP/STANDBY/ESTOP (cmd_allowed); still put the
    // motors in a safe state first — MIT keepalive is active in STANDBY.
    if (cmd_id == CMD_ID_REBOOT) {
        comm_log(LOG_LEVEL_WARN, "Reboot requested");
        hip_motors_exit_mit();
        wheel_motors_set_mode(WheelMode::IDLE);
        reply(CMD_RESULT_ACCEPTED);
        Serial.flush();
        Serial5.flush();
        delay(50);
        SCB_AIRCR = 0x05FA0004;  // Cortex-M7 system reset request
        return;
    }

    // ── Hip command: queue for execution by the MANUAL state action ───────────
    // MANUAL-only (cmd_allowed); on_manual() also clears pending on entry.
    if (cmd_id == CMD_ID_HIP && len >= 3) {
        comm_log(LOG_LEVEL_INFO, "CMD hip motor=0x%02X sub=0x%02X", payload[1], payload[2]);
        g_hip_cmd.motor_id = payload[1];
        g_hip_cmd.sub_cmd  = payload[2];
        if (payload[2] == HIP_SUB_MIT && len >= 3 + 5 * 4) {
            memcpy(&g_hip_cmd.p,   payload + 3,  4);
            memcpy(&g_hip_cmd.v,   payload + 7,  4);
            memcpy(&g_hip_cmd.kp,  payload + 11, 4);
            memcpy(&g_hip_cmd.kd,  payload + 15, 4);
            memcpy(&g_hip_cmd.tff, payload + 19, 4);
        }
        g_hip_cmd.pending = true;
        reply(CMD_RESULT_ACCEPTED);
        return;
    }

    // ── Wheel command: set mode / send setpoint / clear errors ───────────────
    // MANUAL-only (cmd_allowed); LQR (STATE_RUNNING) commands wheels directly.
    if (cmd_id == CMD_ID_WHEEL && len >= 2) {
        uint8_t sub = payload[1];
        if (sub == WHEEL_SUB_SET_MODE && len >= 3) {
            wheel_motors_set_mode((WheelMode)payload[2]);
        } else if (sub == WHEEL_SUB_SEND && len >= 10) {
            float L, R;
            memcpy(&L, payload + 2, 4);
            memcpy(&R, payload + 6, 4);
            wheel_motors_send(L, R);
        } else if (sub == WHEEL_SUB_CLEAR_ERRORS) {
            wheel_motors_clear_errors();
        }
        reply(CMD_RESULT_APPLIED);
        return;
    }

    // ── Param write: set one parameter value ──────────────────────────────────
    if (cmd_id == CMD_ID_PARAM_SET && len >= 7) {
        uint16_t id;  float val;
        memcpy(&id,  payload + 1, 2);
        memcpy(&val, payload + 3, 4);
        comm_log(LOG_LEVEL_INFO, "CMD param_set 0x%04X = %.4f", id, val);
        float old_val = param_get(id);
        ParamSetResult set_result = param_set(id, val);  // out-of-range values clamp to [min, max]
        if (set_result == ParamSetResult::NOT_FOUND || set_result == ParamSetResult::READONLY ||
            set_result == ParamSetResult::NONFINITE) {
            uint8_t reason = set_result == ParamSetResult::NOT_FOUND ? CMD_REASON_NOT_FOUND :
                             set_result == ParamSetResult::READONLY ? CMD_REASON_READONLY : CMD_REASON_NONFINITE;
            reply(CMD_RESULT_REJECTED, reason);
            return;
        }
        bool changed = (set_result == ParamSetResult::OK || set_result == ParamSetResult::CLAMPED) &&
                       param_get(id) != old_val;
        // Motor presence flags are normally boot-time configuration, but the
        // GUI's No Motors preset must be a safe live transition. Once a 1->0
        // write lands, explicitly idle that physical axis before the regular
        // polling path starts ignoring it. In particular, an AK45 otherwise
        // remains in MIT mode holding its last received command when polling
        // stops; merely changing the parameter is not an output cutoff.
        if (changed && old_val >= 0.5f && param_get(id) < 0.5f) {
            if      (id == PARAM_HIP_L_ENABLE)   hip_motor_disable_L();
            else if (id == PARAM_HIP_R_ENABLE)   hip_motor_disable_R();
            else if (id == PARAM_WHEEL_L_ENABLE) wheel_motor_disable_L();
            else if (id == PARAM_WHEEL_R_ENABLE) wheel_motor_disable_R();
        }
        // Buzzer volume normally only applies at boot (param_init() ran before
        // g_buzzer.set_volume() in setup()) — re-apply live so a volume change
        // (including muting to 0) takes effect on every chirp immediately,
        // not just after a reboot.
        if (changed && id == PARAM_BUZZER_VOLUME) {
            g_buzzer.set_volume(param_get(id));
        }
        // send_reliable() (GUI comm_commands.py) retries the identical (id, val) up to
        // 3x every 250 ms until it sees the PARAM_REPORT echo, so a slow link can deliver
        // the same value more than once — only chirp when it actually changed.
        if (changed) {
            g_buzzer.play(PARAM_SET_CHIRP, 1);
        }
        // Echo back actual (possibly clamped) value
        Param p; uint16_t idx = 0;
        while (param_by_index(idx, &p)) { if (p.id == id) { send_param_report(idx); break; } idx++; }
        reply(CMD_RESULT_APPLIED);
        return;
    }

    // ── Param read: report one param, or dump all ─────────────────────────────
    if (cmd_id == CMD_ID_PARAM_GET && len >= 3) {
        uint16_t id;
        memcpy(&id, payload + 1, 2);
        if (id == 0xFFFF) {
            s_param_dump_cursor = 0;  // paced by service_param_dump() in loop()
        } else {
            if (!param_exists(id)) {
                reply(CMD_RESULT_REJECTED, CMD_REASON_NOT_FOUND);
                return;
            }
            Param p; uint16_t idx = 0;
            while (param_by_index(idx, &p)) { if (p.id == id) { send_param_report(idx); break; } idx++; }
        }
        reply(CMD_RESULT_APPLIED);
        return;
    }

    // ── Param reset: revert all writable params to compile-time defaults ──────
    if (cmd_id == CMD_ID_PARAM_RESET_DEFAULTS) {
        comm_log(LOG_LEVEL_WARN, "CMD param_reset_defaults");
        param_reset_defaults();
        s_param_dump_cursor = 0;  // paced refresh of the GUI table (service_param_dump() in loop())
        reply(CMD_RESULT_APPLIED);
        return;
    }

    // ── TEST ONLY: arm N deliberately-corrupted frames to the ESP32 ────────────
    // target 1 (WiFi) is intercepted by the ESP32 itself (forward_to_teensy())
    // and never reaches here; only act on target 0 (UART) or an omitted byte.
    if (cmd_id == CMD_ID_TEST_INJECT_CORRUPT && len >= 2 &&
        (len < 3 || payload[2] == 0)) {
        g_test_corrupt_remaining = payload[1];
        g_test_corrupt_mode = (len >= 4) ? payload[3] : 1;
        comm_log(LOG_LEVEL_WARN, "TEST: injecting %u corrupt frame(s) mode=%u to ESP32",
                 payload[1], g_test_corrupt_mode);
        reply(CMD_RESULT_ACCEPTED);
        return;
    }

    // ── SD log control: start/stop/list/get/delete ─────────────────────────────
    if (cmd_id == CMD_ID_LOG && len >= 2) {
        uint8_t sub = payload[1];
        if (sub == LOG_SUB_CHUNK_ACK && len >= 8) {
            // Internal ESP32 relay acknowledgement. Do not log this high-rate
            // control message or echo any response back onto either transport.
            uint16_t idx;
            uint32_t chunk;
            memcpy(&idx, payload + 2, 2);
            memcpy(&chunk, payload + 4, 4);
            sd_logger_ack_chunk(idx, chunk);
            // Internal relay traffic is v1 and therefore intentionally has no result.
        } else if (sub == LOG_SUB_START) {
            uint32_t dur = 0; if (len >= 6) memcpy(&dur, payload + 2, 4);
            comm_log(LOG_LEVEL_INFO, "CMD log start dur_ms=%lu", (unsigned long)dur);
            if (g_state.state != STATE_STANDBY && g_state.state != STATE_ESTOP) {
                comm_log(LOG_LEVEL_WARN, "CMD log start denied: start before arming");
                reply(CMD_RESULT_REJECTED, CMD_REASON_OPERATION_FAILED);
                return;
            }
            bool started = sd_logger_start(dur);
            forgive_sd_blocking_stall();  // sd_logger_start() can block the loop ~96 ms
            if (started) g_buzzer.play(LOG_START_CHIRP, 1);
            if (!started) {
                reply(CMD_RESULT_REJECTED, CMD_REASON_OPERATION_FAILED);
                return;
            }
        } else if (sub == LOG_SUB_STOP) {
            comm_log(LOG_LEVEL_INFO, "CMD log stop");
            sd_logger_stop();
        } else if (sub == LOG_SUB_LIST) {
            comm_log(LOG_LEVEL_INFO, "CMD log list");
            sd_logger_list();
        } else if (sub == LOG_SUB_GET && len >= 8) {
            uint16_t idx; uint32_t start;
            memcpy(&idx, payload + 2, 2); memcpy(&start, payload + 4, 4);
            uint8_t kind = (len >= 9) ? payload[8] : LOG_FILE_KIND_WLOG;
            // Bulk streaming only when the robot is inert (audit W5) — a GET
            // during RUNNING adds blocking-write jitter to the control loop.
            if (g_state.state != STATE_STANDBY && g_state.state != STATE_ESTOP) {
                comm_log(LOG_LEVEL_WARN, "CMD log get denied: only in STANDBY/ESTOP");
                LogInfoPayload p{};
                p.info_type  = LOG_INFO_STATUS;
                p.file_index = idx;
                p.status     = 1;
                sd_logger_send(COMM_TYPE_LOG_INFO, LOG_INFO_PAYLOAD_V1, &p, sizeof(p));
                reply(CMD_RESULT_REJECTED, CMD_REASON_WRONG_STATE);
                return;
            }
            comm_log(LOG_LEVEL_INFO, "CMD log get idx=%u start_chunk=%lu", idx, (unsigned long)start);
            // Pace to the requesting transport: direct Teensy USB is high-speed;
            // the ESP32-relayed path is flow-controlled one chunk at a time.
            s_log_get_via_usb = (source == COMM_SRC_PC);
            sd_logger_set_get_ack_required(!s_log_get_via_usb);
            if (s_log_get_via_usb) sd_logger_set_get_pacing(0, 2);  // unthrottled direct USB
            else                   sd_logger_set_get_pacing(0, 1);  // ACK provides relay backpressure
            sd_logger_begin_get(idx, start, kind);
            if (sd_logger_transfer_active()) g_buzzer.play(LOG_GET_CHIRP, 1);
            else {
                reply(CMD_RESULT_REJECTED, CMD_REASON_OPERATION_FAILED);
                return;
            }
        } else if (sub == LOG_SUB_DELETE && len >= 4) {
            uint16_t idx; memcpy(&idx, payload + 2, 2);
            comm_log(LOG_LEVEL_INFO, "CMD log delete idx=%u", idx);
            sd_logger_delete(idx);
        }
        reply(CMD_RESULT_APPLIED);
        return;
    }
}

// ── Hardware watchdog ─────────────────────────────────────────────────────────
// Resets the MCU if it's not petted for this long. Needed because a stuck
// driver call can otherwise hang forever with no recovery — e.g. the BNO08x
// sh2 library's getProdIds request has no timeout of its own (sh2.c
// opProcess(): a zero timeout_us disables its bail-out entirely), so a
// non-responding IMU (loose SPI connector) can freeze the whole control loop
// with no crash and no reboot. Gated on PARAM_WATCHDOG_ENABLE (default off);
// when on, armed partway through setup() (see the param_init() block below)
// and petted through every long step from that point on (IMU wait, etc.),
// then petted once per tick in loop().
static constexpr uint32_t WATCHDOG_TIMEOUT_MS = 2000;  // WDOG1 granularity: 0.5 s steps

static void watchdog_enable(uint32_t timeout_ms) {
    uint16_t wt = (uint16_t)(timeout_ms / 500) - 1;  // timeout = (WT+1) * 0.5 s
    // SRS and WDA both read 1 at reset and are negative-edge triggered: writing
    // a 0 to either (i.e. leaving it out of this value) fires an immediate
    // reset / WDOG_B assertion right here. Must explicitly write 1 to both.
    WDOG1_WCR = WDOG_WCR_WDZST | WDOG_WCR_WDE | WDOG_WCR_SRS | WDOG_WCR_WDA | WDOG_WCR_WT(wt);
}

static void watchdog_pet() {
    WDOG1_WSR = 0x5555;
    WDOG1_WSR = 0xAAAA;
}

// ── Setup ─────────────────────────────────────────────────────────────────────

static void check_imu_state();  // defined below; logs IMU state transitions (incl. init failure/retry)

void setup() {
    Serial.begin(115200);
    // 512-byte TX buffer: write(253-byte frame) returns immediately at 1.2 Mbaud
    // instead of blocking ~1.6 ms and adding jitter to the 2 ms control loop.
    static uint8_t s_esp32_tx_buf[512];
    Serial5.addMemoryForWrite(s_esp32_tx_buf, sizeof(s_esp32_tx_buf));
    // 2048-byte RX ring (default is 64 B, ~160 us at 4 Mbaud): the loop only
    // drains it every 2 ms, so any inbound burst > 64 B between services was
    // silently lost. See Phase 2, UARTplat.md.
    static uint8_t s_esp32_rx_buf[2048];
    Serial5.addMemoryForRead(s_esp32_rx_buf, sizeof(s_esp32_rx_buf));
    Serial5.begin(ESP32_BAUD);
    // Fix 5: flush any boot-noise that arrived before the parser was ready
    delay(10);
    while (Serial5.available()) Serial5.read();
    g_comm.onPacket(on_command);
    g_comm_usb.onPacket(on_command);

    // param_init() (and thus the persisted PARAM_BUZZER_VOLUME) must be loaded
    // before the boot chime below plays, or that chime always plays at the
    // compiled-in 1.0 default regardless of what the user saved — silently
    // bypassing volume=0. Safe to run this early: everything param_init()/
    // comm_log() need (Serial, Serial5, comm packet handlers) is already up.
    comm_log(LOG_LEVEL_INFO, "Firmware starting");
    param_init();

    // alpha_force_ret_en became persistent on 2026-08-09 so a tuning session at
    // one leg height survives a power cycle. It used to always boot to 0
    // precisely so a forgotten bench override could not silently follow the
    // robot into a real run -- with retracted gains applied at full extension
    // and nothing on screen to say why. This warning is what replaces that
    // guarantee, so it is deliberately loud and unconditional.
    if (param_get(PARAM_ALPHA_FORCE_RETRACTED_EN) >= 0.5f) {
        comm_log(LOG_LEVEL_WARN,
                 "alpha_force_ret_en=1 restored from flash: gain schedule PINNED to "
                 "retracted (alpha=0) at every leg height, and telemetry alpha reads 0. "
                 "Clear it before extending the legs.");
    }

    // run_wheel_bypass_en became persistent on 2026-08-09 for the opposite
    // reason to the two above: booting to 0 every time meant arming silently
    // broke after every power cycle, and on_running_guard() reports it as
    // "a motor is disabled" rather than naming the bypass that went missing --
    // easy to chase for a long time. This warning is the replacement guarantee.
    if (param_get(PARAM_RUNNING_WHEEL_BYPASS_EN) >= 0.5f) {
        comm_log(LOG_LEVEL_WARN,
                 "run_wheel_bypass_en=1 restored from flash: RUNNING will arm with the "
                 "wheel motors DISABLED. Clear it before a real run.");
    }

    // Same bargain as above: wheel_runaway_en is persistent so a bench session
    // that legitimately overspeeds the wheels survives a power cycle, and this
    // warning is what stands in for the boot-to-safe guarantee that gives up.
    if (param_get(PARAM_WHEEL_RUNAWAY_EN) < 0.5f) {
        comm_log(LOG_LEVEL_WARN,
                 "wheel_runaway_en=0 restored from flash: the FAULT_WHEEL_RUNAWAY "
                 "backup behind the wheel velocity governor is DISABLED. "
                 "Set it before running.");
    }

    g_led.begin();
    g_buzzer.begin();
    g_buzzer.set_volume(param_get(PARAM_BUZZER_VOLUME));
    limit_switches_begin();
    comm_log(LOG_LEVEL_INFO, "Limit switches ready: L(pin %u)=%s R(pin %u)=%s",
             PIN_LIMIT_LEFT, limit_switch_left_active() ? "PRESSED" : "RELEASED",
             PIN_LIMIT_RIGHT, limit_switch_right_active() ? "PRESSED" : "RELEASED");

    // Boot indicator: quick rainbow flash + low-volume chime, ~1.3 s total.
    static const uint8_t BOOT_RAINBOW[][3] = {
        {255,   0,   0},  // red
        {255, 127,   0},  // orange
        {255, 255,   0},  // yellow
        {0,   255,   0},  // green
        {0,   255, 255},  // cyan
        {0,     0, 255},  // blue
        {255,   0, 255},  // magenta
    };
    static const BuzzerNote BOOT_CHIME[] = {
        {72, 40, 0},  // C5
        {76, 40, 0},  // E5
        {79, 40, 0},  // G5
    };
    g_buzzer.play(BOOT_CHIME, sizeof(BOOT_CHIME) / sizeof(BOOT_CHIME[0]));
    for (auto& c : BOOT_RAINBOW) {
        g_led.solid(c[0], c[1], c[2]);
        uint32_t step_start = millis();
        while (millis() - step_start < 180) {
            watchdog_pet();
            g_buzzer.update();
            delay(2);
        }
    }
    g_buzzer.off();
    g_led.off();
    watchdog_pet();

    g_led.pulse(255, 255, 255, 2000);  // STARTUP: white breathe

    // Peripheral enable flags — bench-test without full hardware connected.
    // See PARAM_*_ENABLE in param_ids.h. Takes effect at boot; toggling live
    // requires CMD_ID_REBOOT to re-run setup(). The hip/wheel CAN subsystems
    // come up iff at least one of their two per-motor flags is set.
    bool imu_en     = param_get(PARAM_IMU_ENABLE)     >= 0.5f;
    bool hip_l_en   = param_get(PARAM_HIP_L_ENABLE)   >= 0.5f;
    bool hip_r_en   = param_get(PARAM_HIP_R_ENABLE)   >= 0.5f;
    bool wheel_l_en = param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f;
    bool wheel_r_en = param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f;
    bool hip_en     = hip_l_en   || hip_r_en;
    bool wheel_en   = wheel_l_en || wheel_r_en;
    g_led.set_enabled(param_get(PARAM_LED_ENABLE) >= 0.5f);

    // Hardware watchdog: gated on PARAM_WATCHDOG_ENABLE (default off). Enabled
    // here (after param_init(), rather than at the very top of setup()) so a
    // persisted on/off choice is actually honored — WDOG1's enable bit is
    // write-once, so this is the only point where the decision can be made.
    // Boot steps before this point (rainbow, buzzer, SD/IBus init) therefore
    // run without watchdog coverage; the documented risk this guards against
    // (a stuck IMU/CAN driver call) only arises later, in loop().
    if (param_get(PARAM_WATCHDOG_ENABLE) >= 0.5f) watchdog_enable(WATCHDOG_TIMEOUT_MS);

    // Flag any single motor disabled for bench testing — distinct from the
    // "whole subsystem off" logs below, so a single-leg setup is obvious at boot.
    if (!hip_l_en)   comm_log(LOG_LEVEL_WARN, "Bench-test mode: hip_l disabled (hip_l_enable=0)");
    if (!hip_r_en)   comm_log(LOG_LEVEL_WARN, "Bench-test mode: hip_r disabled (hip_r_enable=0)");
    if (!wheel_l_en) comm_log(LOG_LEVEL_WARN, "Bench-test mode: wheel_l disabled (wheel_l_enable=0)");
    if (!wheel_r_en) comm_log(LOG_LEVEL_WARN, "Bench-test mode: wheel_r disabled (wheel_r_enable=0)");

    sd_logger_set_sender(sd_logger_send);
    if (sd_logger_begin()) comm_log(LOG_LEVEL_INFO, "SD logger ready");
    else                   comm_log(LOG_LEVEL_WARN, "SD logger: no card detected");
    g_rc.begin(CRSF_BAUD);
    comm_log(LOG_LEVEL_INFO, "CRSF ready (Serial4 @ %lu, RX pin %d / TX pin 17)",
             (unsigned long)CRSF_BAUD, PIN_IBUS_RX);

    if (imu_en) {
        // Blocking: STARTUP is the one phase that doesn't need to hold the
        // 500 Hz tick budget (no torque is commanded until STANDBY+), so we
        // wait right here for IMU health to resolve one way or the other
        // instead of smearing attempt_init()'s ~1 s SPI/SH2 handshake across
        // loop() ticks — that used to show up as ~875000 us "Loop overrun"
        // warnings and let "Setup complete" print before IMU health was
        // actually known. Reuses the same MAX_INIT_ATTEMPTS budget a runtime
        // reconnect gets (see IMU.cpp); the outcome (NOMINAL/ERROR) is logged
        // by check_imu_state() on the first loop() tick below.
        imu_init();
        comm_log(LOG_LEVEL_INFO, "IMU initializing...");
        while (imu_state() == ImuState::INITIALIZING) {
            watchdog_pet();
            imu_update();
        }
    } else {
        comm_log(LOG_LEVEL_WARN, "IMU disabled (imu_enable=0)");
    }

    if (hip_en) {
        hip_motors_init();
        comm_log(LOG_LEVEL_INFO, "Hip CAN init OK");
        hip_motors_enter_mit();
        comm_log(LOG_LEVEL_INFO, "Hip MIT mode enabled");
    } else {
        comm_log(LOG_LEVEL_WARN, "Hip motors disabled (hip_l_enable=0, hip_r_enable=0)");
    }

    if (wheel_en) {
        wheel_motors_init();
        wheel_motors_set_mode(WheelMode::IDLE);
        wheel_motors_clear_errors();
        // Keep petting IMU's silence watchdog during this wait — a blind
        // delay(300) here starves imu_update() for longer than IMU.cpp's
        // 100 ms TIMEOUT_MS, which falsely declares a healthy, already-
        // connected sensor ERROR (its liveness clock goes stale even though
        // the sensor itself is fine). See IMU.cpp imu_update() silence check.
        uint32_t t_wheel_wait = millis();
        while (millis() - t_wheel_wait < 300) {
            watchdog_pet();
            if (imu_en) imu_update();
        }
        comm_log(LOG_LEVEL_INFO, "Wheel CAN init OK");
    } else {
        comm_log(LOG_LEVEL_WARN, "Wheel motors disabled (wheel_l_enable=0, wheel_r_enable=0)");
    }

    controlLoop_init();
    stateMachine_init();
    watchdog_pet();
    // "Startup complete" is logged from the state machine (on_standby/on_estop
    // in state_machine.cpp) once STARTUP actually resolves — not here, since
    // IMU health is still pending at this point (imu_init() is non-blocking;
    // see the imu_en block above).
}

// ── Loop overrun tracking ─────────────────────────────────────────────────────
// Work time above the 2000 µs tick budget. This is also how flash-write or
// SD-transfer stalls become visible (see param_flush_service / R1).
static uint32_t s_overrun_count   = 0;
static uint32_t s_last_overrun_ms = 0;

// ── Loop section profiler (PARAM_LOOP_PROFILE_ENABLE) ────────────────────────
// Rolling max per section since the last print, reset every print. Max (not
// average) on purpose — an intermittent CAN retry or flash stall is exactly
// what "Loop overrun" is chasing, and an average would wash it out.
struct LoopProfile {
    uint32_t recv, imu, hip, wheel, ibus, sens_total, imu_chk, radio, ctrl, led, buz, sd, flash, telem;
    uint32_t telem_fill, telem_esp, telem_usb;
};
static LoopProfile s_prof_max        = {};
static uint32_t    s_prof_last_ms    = 0;

static inline void prof_mark(uint32_t& slot, uint32_t t0) {
    uint32_t dt = micros() - t0;
    if (dt > slot) slot = dt;
}

// ── Telemetry ─────────────────────────────────────────────────────────────────

static uint16_t build_health_flags() {
    // The limit actually governing this tick, so the flags don't report the
    // wheels as velocity-limited during a STANDING_UP catch that is running
    // under the higher standup_vel_limit override.
    float soft_lim = controlLoop_wheel_vel_limit();
    uint16_t f = 0;
    if (hm_L.ok)                                    f |= HEALTH_HIP_L_OK;
    if (hm_R.ok)                                    f |= HEALTH_HIP_R_OK;
    if (wm_L.ok)                                    f |= HEALTH_WM_L_OK;
    if (wm_R.ok)                                    f |= HEALTH_WM_R_OK;
    if (hm_limits_L.valid && hm_limits_R.valid)     f |= HEALTH_HIP_LIMITS_VALID;
    if (imu_state() == ImuState::NOMINAL)            f |= HEALTH_IMU_NOMINAL;
    if (param_get(PARAM_LQR_ENABLE) >= 0.5f
        && g_state.state == STATE_RUNNING)           f |= HEALTH_LQR_ACTIVE;
    if (fabsf(wm_L.vel_turns_s) > soft_lim)         f |= HEALTH_WM_L_VEL_LIMITED;
    if (fabsf(wm_R.vel_turns_s) > soft_lim)         f |= HEALTH_WM_R_VEL_LIMITED;
    // Compares against the same effective (asymmetric, gain-scheduled) bounds
    // control_loop.cpp actually clamped theta_ref to this tick, not a fixed
    // symmetric param -- see g_state.theta_max_fwd/bwd.
    if (param_get(PARAM_VEL_PI_EN) >= 0.5f
        && (g_state.theta_ref >= g_state.theta_max_fwd
            || g_state.theta_ref <= -g_state.theta_max_bwd))
        f |= HEALTH_VEL_PI_SAT;
    if (param_get(PARAM_YAW_PI_EN) >= 0.5f
        && fabsf(g_state.tau_yaw) >= param_get(PARAM_YAW_PI_TORQUE_MAX))
        f |= HEALTH_YAW_PI_SAT;
    if (s_overrun_count && millis() - s_last_overrun_ms < 1000)
        f |= HEALTH_LOOP_OVERRUN;
    return f;
}

static void fill_telemetry(TelemetryPayload& t) {
    t.timestamp_ms     = millis();
    t.pitch_rad        = g_state.pitch_rad;
    t.pitch_rate_rads  = g_state.pitch_rate_rads;
    t.wheel_vel_avg_ms = g_state.wheel_vel_avg_ms;
    t.hip_l_pos_rad    = g_state.hip_l_pos_rad;
    t.hip_r_pos_rad    = g_state.hip_r_pos_rad;
    t.whl_tau_l        = g_state.whl_tau_l;
    t.whl_tau_r        = g_state.whl_tau_r;
    // Sim-pitch bench-test injects a fake pitch in place of the real IMU reading
    // (see read_sensors()) — zero roll/yaw alongside it so the GUI/3D model shows
    // pure pitch rather than mixing injected pitch with whatever roll/yaw the
    // bench happens to be sitting at.
    bool sim_pitch_active = param_get(PARAM_ENABLE_SIM_PITCH_RAD) >= 0.5f;
    t.roll_rad         = sim_pitch_active ? 0.0f : imu_roll();
    t.yaw_rad          = sim_pitch_active ? 0.0f : imu_yaw();
    t.robot_state      = (uint8_t)g_state.state;
    t.fault_code       = g_state.fault_code;
    t.test_val         = sinf(2.0f * (float)M_PI * 2.0f * millis() / 1000.0f);
    t.hip_l_torque_nm  = g_state.hip_l_torque_nm;
    t.hip_r_torque_nm  = g_state.hip_r_torque_nm;
    // ibus_ch/ibus_alive keep their wire names: they are the protocol-agnostic
    // RC mirror, and renaming them would churn schema.json, the GUI and the twin
    // for no behavioural gain. Bound by the payload (14), not by CRSF's 16.
    constexpr uint8_t RC_MIRROR_CH = (uint8_t)(sizeof(t.ibus_ch) / sizeof(t.ibus_ch[0]));
    for (uint8_t i = 1; i <= RC_MIRROR_CH; i++) t.ibus_ch[i - 1] = g_rc.channel(i);
    t.ibus_alive       = g_rc.alive() ? 1 : 0;
    t.wm_l_vel_turns_s = wm_L.vel_turns_s;
    t.wm_r_vel_turns_s = wm_R.vel_turns_s;
    t.wm_l_pos_turns   = wm_L.pos_turns;
    t.wm_r_pos_turns   = wm_R.pos_turns;
    t.wm_l_vbus        = wm_L.vbus;
    t.wm_r_vbus        = wm_R.vbus;
    t.wm_l_error       = wm_L.error;
    t.wm_r_error       = wm_R.error;
    t.wm_l_state       = wm_L.axis_state;
    t.wm_r_state       = wm_R.axis_state;
    t.wm_mode          = (uint8_t)wm_mode;
    for (int i = 0; i < 4; i++) t.tof_dist_mm[i] = g_tof.dist_mm[i];
    // V5 — IMU rates + linear acceleration
    t.roll_rate_rads        = sim_pitch_active ? 0.0f : imu_roll_rate();
    t.yaw_rate_rads         = sim_pitch_active ? 0.0f : imu_yaw_rate();
    t.accel_x_ms2           = imu_accel_x();
    t.accel_y_ms2           = imu_accel_y();
    t.accel_z_ms2           = imu_accel_z();
    // V5 — hip velocity feedback
    t.hip_l_vel_rads        = hm_L.vel_rad_s;
    t.hip_r_vel_rads        = hm_R.vel_rad_s;
    // V5 — hip motor MIT setpoints (cmd vs feedback enables tracking quality plots)
    t.hip_l_cmd_pos_rad     = hm_sp_L.p;
    t.hip_r_cmd_pos_rad     = hm_sp_R.p;
    t.hip_l_cmd_vel_rads    = hm_sp_L.v;
    t.hip_r_cmd_vel_rads    = hm_sp_R.v;
    t.hip_l_cmd_kp          = hm_sp_L.kp;
    t.hip_r_cmd_kp          = hm_sp_R.kp;
    t.hip_l_cmd_kd          = hm_sp_L.kd;
    t.hip_r_cmd_kd          = hm_sp_R.kd;
    t.hip_l_cmd_tff         = hm_sp_L.tff;
    t.hip_r_cmd_tff         = hm_sp_R.tff;
    // V7 — balance controller internals: setpoints + effort signals only
    t.theta_ref             = g_state.theta_ref;
    t.v_ref                 = g_state.v_ref;
    t.omega_cmd_rds         = param_get(PARAM_OMEGA_CMD_RDS);
    t.tau_sym               = g_state.tau_sym;
    t.tau_yaw               = g_state.tau_yaw;
    t.ff1_out               = g_state.ff1_out;
    t.ff2_out               = g_state.ff2_out;
    // V5 — diagnostics
    t.health_flags          = build_health_flags();
    t.imu_packet_loss_pct   = (uint8_t)(imu_packet_loss() * 100.0f + 0.5f);
    t.jump_state            = g_state.jump_state;
    t.loop_count            = g_state.loop_count;
    // V8 — radio channel assignments
    t.active_profile        = (uint8_t)param_get(PARAM_ACTIVE_PROFILE);
    // Applied balance-point trim actually used in the LQR error (live knob while
    // live-trimming, otherwise the scheduled ret/ext value) — the number the
    // operator reads off to record/latch.
    t.pitch_trim_rad        = g_state.applied_pitch_trim;
    // V9 — ESP32<->Teensy link supervision (telemetry only, see Phase 3, UARTplat.md)
    uint32_t esp32_status_age = millis() - s_last_esp32_status_ms;
    uint32_t uart_rx_drops    = g_comm.rx_drops();
    uint32_t uart_seq_gaps    = g_comm.rx_seq_gaps();
    t.esp32_link_ok       = (s_last_esp32_status_ms != 0 && esp32_status_age < 1000) ? 1 : 0;
    t.esp32_status_age_ms = (uint16_t)(esp32_status_age > 65535 ? 65535 : esp32_status_age);
    t.uart_rx_drops       = (uint16_t)(uart_rx_drops   > 65535 ? 65535 : uart_rx_drops);
    t.uart_seq_gaps       = (uint16_t)(uart_seq_gaps   > 65535 ? 65535 : uart_seq_gaps);
    // V10 — leg gain-schedule blend factor (§1a, tuning.md)
    t.gain_sched_alpha    = g_state.gain_sched_alpha;
    // V11 — standing-up recovery FSM phase
    t.standup_state       = g_state.standup_state;
}

static void send_telemetry(bool prof) {
    uint32_t t0 = micros();
    TelemetryPayload telem;
    fill_telemetry(telem);
    if (prof) prof_mark(s_prof_max.telem_fill, t0);

    const uint8_t* tp = (const uint8_t*)&telem;

    t0 = micros();
    // TEST ONLY (Phase 9, UARTplat.md): corrupt just the A half so the ESP32's
    // parser has exactly one bad frame per armed count to detect and drop.
    uint8_t corrupt_mode = (g_test_corrupt_remaining > 0) ? g_test_corrupt_mode : 0;
    if (g_test_corrupt_remaining > 0) g_test_corrupt_remaining--;
    g_comm.send(COMM_TYPE_TELEM_A, TELEM_VERSION, tp,               TELEM_A_LEN, corrupt_mode);
    g_comm.send(COMM_TYPE_TELEM_B, TELEM_VERSION, tp + TELEM_A_LEN, TELEM_B_LEN);
    if (prof) prof_mark(s_prof_max.telem_esp, t0);

    t0 = micros();
    if (Serial.availableForWrite() >= (int)(20u + TELEM_A_LEN + TELEM_B_LEN)) {
        g_comm_usb.send(COMM_TYPE_TELEM_A, TELEM_VERSION, tp,               TELEM_A_LEN);
        g_comm_usb.send(COMM_TYPE_TELEM_B, TELEM_VERSION, tp + TELEM_A_LEN, TELEM_B_LEN);
    }
    if (prof) prof_mark(s_prof_max.telem_usb, t0);
}

// ── Radio (CRSF) telemetry ────────────────────────────────────────────────────
// The return half of the link. Gathers the same values fill_telemetry() reads,
// but only the handful worth a slice of radio bandwidth — see CrsfTelem.h.
//
// Bus current and used capacity are sent as CRSF's all-0xFF "no data" rather
// than as zero. Nothing on this robot measures either, and EdgeTX skips a
// field whose bytes are all 0xFF, so no sensor is created at all. A Curr
// sensor sitting at a confident 0.0 A is exactly what someone would later
// build a power alarm on. ODrive's Get_Iq (CAN 0x014) would give phase
// current, which is not bus current, and would add periodic traffic to the
// control-critical CAN3 bus — so it is a deliberate omission, not an oversight.
//
// Pack voltage is real: it is the ODrive bus voltage, which is the pack.
static void crsf_telemetry_tick() {
    CrsfTelemSources s;

    bool sim_pitch_active = param_get(PARAM_ENABLE_SIM_PITCH_RAD) >= 0.5f;
    s.pitch_rad = imu_pitch();
    s.roll_rad  = sim_pitch_active ? 0.0f : imu_roll();
    s.yaw_rad   = sim_pitch_active ? 0.0f : imu_yaw();

    // Whichever wheel controller is actually reporting; they share the pack.
    float vbus = wm_L.ok ? wm_L.vbus : (wm_R.ok ? wm_R.vbus : 0.0f);
    s.pack_volts = (vbus > 1.0f) ? vbus : CRSF_BATT_NO_DATA;
    s.pack_amps  = CRSF_BATT_NO_DATA;          // not instrumented — see above
    // 6S: 25.2 V full, 19.8 V empty. V_nom for this robot is the 24.0 V
    // fully-charged working assumption, not the 22.2 V LiPo textbook nominal.
    float pct = (vbus - 19.8f) / (25.2f - 19.8f) * 100.0f;
    s.pack_pct = (uint8_t)(vbus < 1.0f ? 0 : (pct < 0 ? 0 : (pct > 100 ? 100 : pct)));

    s.robot_state      = (uint8_t)g_state.state;
    s.fault_code       = g_state.fault_code;
    s.jump_state       = g_state.jump_state;
    s.standup_state    = g_state.standup_state;
    s.gain_sched_alpha = g_state.gain_sched_alpha;
    s.active_profile   = (uint8_t)param_get(PARAM_ACTIVE_PROFILE);
    s.health_flags     = build_health_flags();
    s.hip_l_torque_nm  = g_state.hip_l_torque_nm;
    s.hip_r_torque_nm  = g_state.hip_r_torque_nm;
    s.wheel_vel_avg_ms = g_state.wheel_vel_avg_ms;

    uint32_t esp32_age = millis() - s_last_esp32_status_ms;
    s.esp32_link_ok    = (s_last_esp32_status_ms != 0 && esp32_age < 1000) ? 1 : 0;
    // Both axes summed. The bench log that motivated the glitch filter showed
    // this climbing to 14.6% of samples before a spurious runaway trip, so
    // hearing it build is the whole point of putting it on the radio.
    s.vel_glitch_count = wm_L.vel_glitch_count + wm_R.vel_glitch_count;

    g_rc_telem.tick(g_rc, millis(), s);
}

// ── LED ───────────────────────────────────────────────────────────────────────

// Profile-change flash: set by radio_update(), consumed by update_led().
static uint32_t s_profile_flash_until_ms = 0;
static uint8_t  s_profile_flash_rgb[3]   = {};

static void update_buzzer() {
    g_buzzer.update();
}

static void update_led() {
    static RobotStateEnum prev = (RobotStateEnum)0xFF;
    static uint32_t s_imu_alert_next_ms = 0;

    // IMU-fault alert: brief red flash overlaid on STANDBY's amber pulse, repeating
    // every 2 s. STANDBY has no automatic ESTOP on IMU loss (only STARTUP/RUNNING
    // do — see startup_fail()/running_imu_fault() in state_machine.cpp), so without
    // this a dead/disconnected IMU is otherwise silent until you try to arm.
    bool imu_fault = (g_state.state == STATE_STANDBY) &&
                      (param_get(PARAM_IMU_ENABLE) >= 0.5f) &&
                      (imu_state() != ImuState::NOMINAL);
    if (imu_fault && !s_profile_flash_until_ms) {
        uint32_t now = millis();
        if (now >= s_imu_alert_next_ms) {
            s_profile_flash_rgb[0] = 255; s_profile_flash_rgb[1] = 0; s_profile_flash_rgb[2] = 0;
            s_profile_flash_until_ms = now + 150;
            s_imu_alert_next_ms     = now + 2000;
        }
    }

    // Profile-change flash takes priority for its duration, then restores state LED.
    if (s_profile_flash_until_ms) {
        if (millis() < s_profile_flash_until_ms) {
            g_led.solid(s_profile_flash_rgb[0], s_profile_flash_rgb[1], s_profile_flash_rgb[2]);
            g_led.update();
            return;
        }
        s_profile_flash_until_ms = 0;
        prev = (RobotStateEnum)0xFF;  // force state animation re-apply
    }

    RobotStateEnum cur = g_state.state;
    if (cur != prev) {
        bool was_estop = (prev == STATE_ESTOP);
        prev = cur;
        switch (cur) {
            case STATE_STARTUP:
                if (was_estop) g_led.flash_then_pulse(255, 255, 255, 1000, 2000);
                else           g_led.pulse(255, 255, 255, 2000);
                break;
            case STATE_CALIBRATION: g_led.pulse(0,   0,   255, 2000);      break;
            case STATE_STANDBY:     g_led.pulse(255, 200,   0, 2000);      break;
            case STATE_RUNNING:     g_led.blink(0,   255,   0,  167, 167); break;
            case STATE_JUMPING:     g_led.blink(255, 100,   0,   80,  80); break;  // fast orange: "launching"
            case STATE_STANDING_UP: g_led.blink(255,  60,   0,   60,  60); break;  // fast red-orange strobe: "recovering"
            case STATE_DISARMING:    g_led.blink(255, 180,   0,  200, 200); break;  // amber: normal torque ramp-down
            case STATE_MANUAL:      g_led.pulse(0,   200, 255, 2000);      break;
            case STATE_ESTOP:       g_led.blink(255,   0,   0,  100, 100); break;
            case STATE_CMD_REJECT:  g_led.blink(255,   0,   0,  300, 300); break;
        }
    }
    g_led.update();
}

// ── Per-loop tasks ────────────────────────────────────────────────────────────

static void receive_commands() {
    g_comm.update();
    g_comm_usb.update();
}

static void read_sensors(bool prof) {
    uint32_t t0;

    if (param_get(PARAM_IMU_ENABLE) >= 0.5f) {
        t0 = micros();
        imu_update();
        if (prof) prof_mark(s_prof_max.imu, t0);
        g_state.pitch_rad       = imu_pitch();        // raw IMU pitch already matches robot +X=forward convention as mounted
        g_state.pitch_rate_rads = imu_pitch_rate();
    }

    // ── Sim-pitch injection (bench-test, no arming required) ───────────────────
    // Overrides IMU pitch/pitch-rate with fake values so telemetry (GUI/3D model)
    // and, once armed, the LQR all see the same injected pitch. Applied every
    // loop regardless of state so it's visible in STANDBY without arming.
    if (param_get(PARAM_ENABLE_SIM_PITCH_RAD) >= 0.5f)
        g_state.pitch_rad = param_get(PARAM_SIM_PITCH_RAD);
    if (param_get(PARAM_ENABLE_SIM_PITCH_RATE) >= 0.5f)
        g_state.pitch_rate_rads = param_get(PARAM_SIM_PITCH_RATE_RAD_S);

    if (param_get(PARAM_HIP_L_ENABLE) >= 0.5f || param_get(PARAM_HIP_R_ENABLE) >= 0.5f) {
        t0 = micros();
        hip_motors_poll();
        if (prof) prof_mark(s_prof_max.hip, t0);
        g_state.hip_l_pos_rad   = hm_L.pos_rad;
        g_state.hip_r_pos_rad   = hm_R.pos_rad;
        g_state.hip_l_torque_nm = hm_L.torque_nm;
        g_state.hip_r_torque_nm = hm_R.torque_nm;
    }

    if (param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f || param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f) {
        t0 = micros();
        wheel_motors_poll();
        wheel_motors_pet_watchdog();
        if (prof) prof_mark(s_prof_max.wheel, t0);
    }

    t0 = micros();
    g_rc.update();
    if (prof) prof_mark(s_prof_max.ibus, t0);
}

// ── Radio melodies ────────────────────────────────────────────────────────────
static const BuzzerNote RADIO_ACQ_MELODY[]    = {{76, 80, 0}};             // E5 — single: "link up"
static const BuzzerNote RADIO_LOST_MELODY[]   = {{64, 100, 30}, {60, 150, 0}}; // E4→C4 desc: "link lost"
static const BuzzerNote ARM_IGNORED_MELODY[]  = {{69, 60, 40}, {69, 60, 0}};   // A4-A4 double-tap: "not ready, try again"
// Rescue stick combo (radio_update()). A fast alternating two-note siren, and a
// long descending fall for the reboot — deliberately unlike every other cue on
// the robot so neither can be mistaken for an arm/profile/link event.
static const BuzzerNote RESCUE_CLEAR_MELODY[]  = {{81, 60, 10}, {88, 60, 10}, {81, 60, 10}, {88, 140, 0}}; // A5⇄E6 siren: "rescue"
static const BuzzerNote RESCUE_REBOOT_MELODY[] = {{88, 70, 10}, {81, 70, 10}, {74, 70, 10}, {67, 220, 0}}; // E6→A5→D5→G4: "rebooting"
static constexpr uint32_t RESCUE_REBOOT_CHIME_MS = 500;  // blocking pump so the fall is audible before reset

// ── Live parameter tuning (live_tune.h) ────────────────────────────────────────
// CH7/CH8 knob -> param mapping, grouped in threes by the CH5/CH6 switch
// select bits. Reachable only with PARAM_LIVE_TUNE_MULTI_EN = 1; in
// the default SIMPLE mode CH5/CH6 are the SD-log and jump switches and live
// tuning is inactive. See "gain-group select" in radio_update().
//   group 0: CH5 down, CH6 up   -> LQR pitch/rate (retracted)
//   group 1: CH5 up,   CH6 down -> vel_pi KP/KI
//   group 2: CH5 down, CH6 down -> roll KP/KD
//   (CH5 up, CH6 up = no group selected, tuning inactive)
// Repoint a knob at a different param here and reflash; no other code changes
// needed as long as the target's control-loop read site uses live_tune_value().
//
// range_min/range_max is the value at the knob's raw-low/raw-high end, NOT a
// sorted (min,max) pair -- knob_val below interpolates raw-low -> range_min and
// raw-high -> range_max, independent of the target param's registry min/max.
// IMPORTANT: always set range_min to the value at knob-zero and range_max to
// the value at knob-max such that turning the knob UP increases the gain's
// MAGNITUDE (strength of action), never its raw signed value. Group 0's gains
// are negative, so range_min there is the least-negative (weakest) value and
// range_max is the most-negative (strongest) value, so the knob still reads
// as "more knob = more gain" even though the number it writes is decreasing.
// Do not flip this to make the raw value increase with the knob.
struct LiveTuneSlot {
    uint8_t  group;                // 0-2, selected by CH5/CH6 (see radio_update())
    uint8_t  ibus_channel;         // 1-indexed CH number (7 or 8)
    uint16_t live_param_id;        // readonly+command shadow, telemetry-visible
    uint16_t persist_param_id;     // real, persistent, control-loop-facing param
    float    range_min;            // value at knob-zero (weakest gain)
    float    range_max;            // value at knob-max (strongest gain)
};
static const LiveTuneSlot LIVE_TUNE_SLOTS[] = {
    // Group 0: CH5 down, CH6 up -- LQR pitch/rate (retracted)
    // Widened 2026-07-26: the 2026-07-25 bench run pegged CH7 at the old -0.5
    // end stop for over half the run, so k_pitch_ret was clamped by the knob,
    // not tuned by it.
    { 0, 7, PARAM_LIVE_TUNE_CH7_VAL, PARAM_LQR_K_PITCH_RET, -0.1f,  -2.0f },
    { 0, 8, PARAM_LIVE_TUNE_CH8_VAL, PARAM_LQR_K_RATE_RET,  -0.01f, -1.0f },
    // Group 1: CH5 up, CH6 down -- vel_pi KP/KI
    { 1, 7, PARAM_LIVE_TUNE_CH7_VAL, PARAM_VEL_PI_KP, 0.05f, 1.0f  },
    { 1, 8, PARAM_LIVE_TUNE_CH8_VAL, PARAM_VEL_PI_KI, 0.02f, 0.5f  },
    // Group 2: CH5 down, CH6 down -- roll KP/KD
    { 2, 7, PARAM_LIVE_TUNE_CH7_VAL, PARAM_ROLL_KP, 0.3f,  4.0f  },
    { 2, 8, PARAM_LIVE_TUNE_CH8_VAL, PARAM_ROLL_KD, 0.02f, 0.5f  },
};
static constexpr uint8_t NUM_LIVE_TUNE_SLOTS = sizeof(LIVE_TUNE_SLOTS) / sizeof(LIVE_TUNE_SLOTS[0]);
static bool s_live_tune_picked_up[NUM_LIVE_TUNE_SLOTS] = {};
static bool s_live_tune_mode_active = false;

// See live_tune.h. Only a picked-up slot's live shadow overrides the real
// param; everything else (including a not-yet-picked-up slot) falls through
// to the normal persisted value.
float live_tune_value(uint16_t persist_param_id) {
    if (s_live_tune_mode_active) {
        for (uint8_t i = 0; i < NUM_LIVE_TUNE_SLOTS; i++) {
            if (LIVE_TUNE_SLOTS[i].persist_param_id == persist_param_id && s_live_tune_picked_up[i]) {
                return param_get(LIVE_TUNE_SLOTS[i].live_param_id);
            }
        }
    }
    return param_get(persist_param_id);
}

// ── Radio interpretation ───────────────────────────────────────────────────────
// CH10 > 1990: arm into RUNNING (requires prior calibration).
// CH10 drop:   disarm back to STANDBY.
// Calibration stick combo (CH1/CH4 full up, CH2/CH3 full down): trigger
//               CALIBRATION from STANDBY; re-enter it to cancel.
// CH5  > 1990: start SD logging (SIMPLE mode); drop stops it.
// CH11 > 1990: request CALIBRATION from STANDBY; re-trigger during a
//               radio-owned CALIBRATION cancels it through DISARMING.
//               Edge-triggered, release-and-retry, 1 s lockout. The
//               calibration stick combo remains as the fallback.
// CH12 > 1990: reset a fault / clear an ESTOP -> STARTUP, clearing fault_code
//               regardless of severity. Armed only in STANDBY/ESTOP. Same
//               effect as the rescue combo's rising edge, minus its
//               hold-to-reboot -- a latching switch left down is not the
//               deliberate gesture that makes hold-to-reboot safe.
// CH13: live-tune gain group, encoded as levels from three mutually
//               exclusive RGB buttons: ~1500 none, 1660 g0, 1830 g1, 2000 g2.
//               Gated by PARAM_LIVE_TUNE_MULTI_EN.
// CH14 > 1990: commit the tuned gains (same one-shot as PARAM_LIVE_TUNE_LATCH).
// CH15 > 1990: coordinated-turn lean enable. Also needs PARAM_LEAN_TURN_EN and
//               a non-zero PARAM_LEAN_GAIN -- three gates, because a feature
//               that can put the robot on its side should not self-enable.
// CH16: spare.
// CH6  > 1990: trigger a JUMP from RUNNING (SIMPLE mode), one per rising edge.
// CH5/CH6 (while RUNNING, LEGACY mode only): debounced switch combination
//               selects which live-tune gain group CH7/CH8 drive -- see
//               LIVE_TUNE_SLOTS. In LEGACY mode CH5/CH6 drive nothing else.
// CH3 1000–2000: maps to PARAM_RADIO_HIP_CMD as t ∈ [0,1].
//   Left stale when radio is dead.

static void radio_update() {
    bool alive = g_rc.alive();
    uint16_t ch10 = g_rc.channel(10);
    uint16_t ch5  = g_rc.channel(5);
    uint16_t ch6  = g_rc.channel(6);

    bool energetic = g_state.state == STATE_RUNNING || g_state.state == STATE_JUMPING ||
                     g_state.state == STATE_STANDING_UP;
    static RobotStateEnum s_authority_prev_state = (RobotStateEnum)0xFF;
    bool entered_standby = g_state.state == STATE_STANDBY &&
                           s_authority_prev_state != STATE_STANDBY;
    if (entered_standby || g_state.state == STATE_ESTOP ||
        g_state.state == STATE_STARTUP || g_state.state == STATE_CMD_REJECT) {
        s_arm_authority = ArmAuthority::NONE;
    }
    s_authority_prev_state = g_state.state;

    static bool s_was_alive = false;
    static bool s_disarm_req_sent = false;
    bool radio_acquired = alive && !s_was_alive;
    bool radio_lost = !alive && s_was_alive;
    bool authority_disarm_requested = false;
    if (radio_acquired) {
        comm_log(LOG_LEVEL_INFO,
                 "Radio: signal acquired CH10=%u state=%d authority=%s",
                 (unsigned)ch10, (int)g_state.state, arm_authority_name());
        g_buzzer.play(RADIO_ACQ_MELODY, 1);
        if (energetic && s_arm_authority == ArmAuthority::GUI) {
            comm_log(LOG_LEVEL_WARN,
                     "DISARM reason=RADIO_TAKEOVER: radio acquired during GUI run CH10=%u -> DISARMING",
                     (unsigned)ch10);
            comm_log(LOG_LEVEL_INFO,
                     "Radio handover: wait for STANDBY, then raise CH10 to arm with radio");
            stateMachine_disarm_running();
            authority_disarm_requested = true;
            s_disarm_req_sent = true;
        }
    }
    if (radio_lost) {
        comm_log(LOG_LEVEL_WARN,
                 "Radio: signal lost state=%d authority=%s",
                 (int)g_state.state, arm_authority_name());
        g_buzzer.play(RADIO_LOST_MELODY, 2);
        if (energetic && s_arm_authority == ArmAuthority::RADIO) {
            comm_log(LOG_LEVEL_ERROR,
                     "DISARM reason=RADIO_SIGNAL_LOST: radio-owned run lost link -> DISARMING");
            stateMachine_disarm_running();
            authority_disarm_requested = true;
            s_disarm_req_sent = true;
        }
    }
    s_was_alive = alive;

    // ── Rescue stick combo: clear ESTOP, or hold 3 s to reboot ────────────────
    // CH3 + CH2 pinned full up, CH1 + CH4 pinned full down — both sticks jammed
    // into opposite corners. No driving input produces that, and it is awkward
    // enough to hold deliberately that it can't be stumbled into.
    //
    // Rising edge: full reset (ESTOP -> STARTUP), which clears fault_code
    // regardless of severity and re-runs the startup checks. In STANDBY there
    // is nothing to clear, so the edge is just the beep.
    // Held RESCUE_REBOOT_HOLD_MS: full MCU reset, identical to CMD_ID_REBOOT.
    //
    // Armed only in STANDBY/ESTOP — never with torque live. The hold timer
    // additionally survives STARTUP because that is exactly where the reset
    // above lands: a fault that can't actually be cleared re-faults out of
    // STARTUP back into ESTOP, and dropping the timer in that gap would make
    // the 3 s fallback unreachable in the one case it exists for. STARTUP is
    // torque-free and already an accepted CMD_ID_REBOOT state (cmd_allowed()),
    // so this widens nothing.
    //
    // Every term is guarded by `alive`: channel() returns 0 on signal loss, so
    // without it a dead radio would satisfy both stick-low tests for free.
    static constexpr uint8_t  RESCUE_DEBOUNCE_TICKS = 3;     // ~6 ms @ 500 Hz
    static constexpr uint32_t RESCUE_REBOOT_HOLD_MS = 3000;
    static uint8_t  s_rescue_ticks    = 0;
    static bool     s_rescue_held     = false;
    static uint32_t s_rescue_since_ms = 0;

    const bool rescue_arm_state  = (g_state.state == STATE_STANDBY || g_state.state == STATE_ESTOP);
    const bool rescue_hold_state = rescue_arm_state || (g_state.state == STATE_STARTUP);
    const bool rescue_raw = alive &&
                            g_rc.channel(3) > 1990 && g_rc.channel(2) > 1990 &&
                            g_rc.channel(1) < 1010 && g_rc.channel(4) < 1010;
    if (rescue_raw && (s_rescue_held ? rescue_hold_state : rescue_arm_state)) {
        if (s_rescue_ticks < RESCUE_DEBOUNCE_TICKS) s_rescue_ticks++;
    } else {
        // Release (or leaving the permitted states) re-arms the combo, so a
        // second attempt starts a fresh 3 s countdown rather than resuming one.
        s_rescue_ticks = 0;
        s_rescue_held  = false;
    }

    if (s_rescue_ticks >= RESCUE_DEBOUNCE_TICKS && !s_rescue_held) {
        s_rescue_held     = true;
        s_rescue_since_ms = millis();
        g_buzzer.play(RESCUE_CLEAR_MELODY, sizeof(RESCUE_CLEAR_MELODY) / sizeof(RESCUE_CLEAR_MELODY[0]));
        s_profile_flash_rgb[0] = 255; s_profile_flash_rgb[1] = 255; s_profile_flash_rgb[2] = 255;
        s_profile_flash_until_ms = millis() + 300;
        if (g_state.state == STATE_ESTOP) {
            comm_log(LOG_LEVEL_WARN,
                     "Radio: rescue combo — reset ESTOP [fault 0x%02X] -> STARTUP; hold %lu ms to reboot",
                     g_state.fault_code, (unsigned long)RESCUE_REBOOT_HOLD_MS);
            stateMachine_request_reset();
        } else {
            comm_log(LOG_LEVEL_WARN,
                     "Radio: rescue combo — no ESTOP to clear (state=%d); hold %lu ms to reboot",
                     (int)g_state.state, (unsigned long)RESCUE_REBOOT_HOLD_MS);
        }
    }

    if (s_rescue_held && (millis() - s_rescue_since_ms) >= RESCUE_REBOOT_HOLD_MS) {
        comm_log(LOG_LEVEL_WARN, "Radio: rescue combo held %lu ms -> REBOOT (state=%d)",
                 (unsigned long)RESCUE_REBOOT_HOLD_MS, (int)g_state.state);
        // Same safe-shutdown order as the CMD_ID_REBOOT handler: MIT keepalive
        // is live in STANDBY, so drop it before pulling the MCU out from under it.
        hip_motors_exit_mit();
        wheel_motors_set_mode(WheelMode::IDLE);
        // The buzzer is non-blocking and driven by update(), so pump it here —
        // the reset would otherwise silence it before the first note finished.
        // Well inside WATCHDOG_TIMEOUT_MS, and we are resetting regardless.
        g_buzzer.play(RESCUE_REBOOT_MELODY, sizeof(RESCUE_REBOOT_MELODY) / sizeof(RESCUE_REBOOT_MELODY[0]));
        const uint32_t chime_until = millis() + RESCUE_REBOOT_CHIME_MS;
        while ((int32_t)(millis() - chime_until) < 0) g_buzzer.update();
        Serial.flush();
        Serial5.flush();
        delay(50);
        SCB_AIRCR = 0x05FA0004;  // Cortex-M7 system reset request
    }

    // ── CH11 calibration request / CH12 fault reset ──────────────────────────
    // Both are switch equivalents of the two stick combos, which remain as the
    // fallback for when the transmitter is not the arming authority.
    //
    // Both are EDGE-triggered off a LATCHING control, so each needs the same
    // release-and-retry gate the combos use: the switch must be seen low again
    // before another edge counts. Without that, powering up or reconnecting
    // with the switch already down would fire the action immediately.
    //
    // Neither has a hold-to-reboot: the combos can afford that because holding
    // two sticks in opposite corners is unmistakably deliberate, whereas a
    // latching switch left down is the normal resting state of a switch
    // somebody flicked and forgot.
    //
    // Both guarded by `alive`, so a dead link cannot fire either.
    // Shared by the CH11 switch and the calibration stick combo: either can
    // start a radio-owned calibration, and either must be able to cancel one.
    static bool s_radio_calib_owned = false;
    static bool s_radio_calib_entered = false;

    struct RadioEdge {
        uint8_t  hi_ticks;
        uint8_t  lo_ticks;
        bool     held;
        bool     low_seen;
        uint32_t lockout_until_ms;
    };
    static constexpr uint8_t  SWITCH_DEBOUNCE_TICKS = 3;     // ~6 ms @ 500 Hz
    static constexpr uint32_t SWITCH_LOCKOUT_MS     = 1000;

    auto edge_update = [](RadioEdge& e, bool raw_hi, bool link_alive) -> bool {
        if (link_alive && raw_hi) {
            if (e.hi_ticks < SWITCH_DEBOUNCE_TICKS) e.hi_ticks++;
            e.lo_ticks = 0;
        } else if (link_alive) {
            if (e.lo_ticks < SWITCH_DEBOUNCE_TICKS) e.lo_ticks++;
            e.hi_ticks = 0;
        } else {
            // Link loss reads as neither. It must not look like a release,
            // or a dropout would re-arm the edge for free.
            e.hi_ticks = 0;
            e.lo_ticks = 0;
        }
        if (e.lo_ticks >= SWITCH_DEBOUNCE_TICKS) {
            e.low_seen = true;
            e.held = false;
        }
        if (e.hi_ticks >= SWITCH_DEBOUNCE_TICKS && !e.held) {
            e.held = true;
            const bool valid = e.low_seen &&
                               (int32_t)(millis() - e.lockout_until_ms) >= 0;
            e.low_seen = false;   // consume on every edge, valid or not
            return valid;
        }
        return false;
    };

    // CH11 — calibration. Same contract as the calibration combo: STANDBY
    // starts it, re-triggering during a radio-owned CALIBRATION cancels it
    // through DISARMING so the hip gains taper rather than snapping to zero.
    static RadioEdge s_calib_sw = {};
    if (edge_update(s_calib_sw, g_rc.channel(11) > 1990, alive)) {
        if (g_state.state == STATE_STANDBY) {
            comm_log(LOG_LEVEL_INFO, "Radio: CH11 -> CALIBRATION");
            if (stateMachine_request_calibration()) {
                s_radio_calib_owned = true;
                s_radio_calib_entered = false;
                s_calib_sw.lockout_until_ms = millis() + SWITCH_LOCKOUT_MS;
            }
        } else if (g_state.state == STATE_CALIBRATION && s_radio_calib_owned) {
            comm_log(LOG_LEVEL_WARN, "Radio: CH11 re-triggered -> DISARMING");
            stateMachine_disarm_calibration();
            s_radio_calib_owned = false;
            s_radio_calib_entered = false;
            s_calib_sw.lockout_until_ms = millis() + SWITCH_LOCKOUT_MS;
        } else {
            comm_log(LOG_LEVEL_WARN,
                     "Radio: CH11 calibration ignored in state %d", (int)g_state.state);
        }
    }

    // CH12 — reset a fault / clear an ESTOP. Same effect as the rescue combo's
    // rising edge: a full reset to STARTUP, which clears fault_code regardless
    // of severity and re-runs the startup checks. A fault that genuinely cannot
    // be cleared re-faults straight back to ESTOP, which is the honest answer.
    //
    // Armed only in STANDBY/ESTOP, never with torque live.
    static RadioEdge s_reset_sw = {};
    if (edge_update(s_reset_sw, g_rc.channel(12) > 1990, alive)) {
        if (g_state.state == STATE_ESTOP) {
            comm_log(LOG_LEVEL_WARN,
                     "Radio: CH12 -> reset ESTOP [fault 0x%02X] -> STARTUP",
                     g_state.fault_code);
            stateMachine_request_reset();
            g_buzzer.play(RESCUE_CLEAR_MELODY,
                          sizeof(RESCUE_CLEAR_MELODY) / sizeof(RESCUE_CLEAR_MELODY[0]));
            s_reset_sw.lockout_until_ms = millis() + SWITCH_LOCKOUT_MS;
        } else if (g_state.state == STATE_STANDBY) {
            comm_log(LOG_LEVEL_INFO, "Radio: CH12 -> nothing to clear (STANDBY)");
            g_buzzer.play(RADIO_ACQ_MELODY, 1);
        } else {
            comm_log(LOG_LEVEL_WARN,
                     "Radio: CH12 reset ignored in state %d", (int)g_state.state);
        }
    }

    // Debounce CH10: the raw armed level must hold for ARM_DEBOUNCE_TICKS
    // consecutive ticks before an arm is requested. Filters a single bad
    // tick of ch10/alive right at the RUNNING-entry race (radio_update()
    // reads g_state.state one tick stale vs. when on_running() actually
    // sets it — see README/radio arm gotcha). Disarm reacts faster
    // (DISARM_DEBOUNCE_TICKS) since dropping the switch should cut power
    // promptly; both are still well under human reaction time at 500 Hz.
    static constexpr uint8_t ARM_DEBOUNCE_TICKS    = 3;  // ~6 ms @ 500 Hz
    static constexpr uint8_t DISARM_DEBOUNCE_TICKS = 2;  // ~4 ms @ 500 Hz
    static uint8_t s_armed_ticks   = 0;
    static uint8_t s_unarmed_ticks = 0;
    bool armed_raw = alive && (ch10 > 1990);
    bool disarmed_raw = alive && (ch10 <= 1990);
    if (armed_raw) {
        if (s_armed_ticks < ARM_DEBOUNCE_TICKS) s_armed_ticks++;
        s_unarmed_ticks = 0;
    } else if (disarmed_raw) {
        if (s_unarmed_ticks < DISARM_DEBOUNCE_TICKS) s_unarmed_ticks++;
        s_armed_ticks = 0;
    } else {
        // No radio link is neutral. It neither arms nor disarms; GUI/API
        // control remains usable with the transmitter completely off.
        s_armed_ticks = 0;
        s_unarmed_ticks = 0;
    }
    bool armed = s_armed_ticks >= ARM_DEBOUNCE_TICKS;

    static bool s_was_armed = false;
    if (armed && !s_was_armed) {
        if (g_state.state == STATE_ESTOP &&
            fault_severity(g_state.fault_code) == FAULT_SEVERITY_SOFT) {
            comm_log(LOG_LEVEL_INFO, "Radio: soft-clear ESTOP [0x%02X]", g_state.fault_code);
            stateMachine_request_soft_clear();
        } else if (g_state.state == STATE_STANDBY) {
            // No CH5 interlock here any more: CH5 is the SD-log switch, and
            // arming with a log already running is exactly the intended order
            // ("start the log before arming").
            bool accepted = stateMachine_request_running();
            if (accepted) {
                s_arm_authority = ArmAuthority::RADIO;
                comm_log(LOG_LEVEL_INFO,
                         "ARM authority=RADIO: live CH10 HIGH -> RUNNING");
            }
        } else {
            // stateMachine_request_running() only latches from STANDBY, so a flip
            // that lands mid-transition (e.g. tail end of CALIBRATION, or still in
            // STARTUP) is otherwise silently dropped with no operator feedback.
            comm_log(LOG_LEVEL_WARN, "Radio: arm ignored, not in STANDBY (state=%d)", (int)g_state.state);
            g_buzzer.play(ARM_IGNORED_MELODY, sizeof(ARM_IGNORED_MELODY) / sizeof(ARM_IGNORED_MELODY[0]));
            s_profile_flash_rgb[0] = 255; s_profile_flash_rgb[1] = 0; s_profile_flash_rgb[2] = 255;
            s_profile_flash_until_ms = millis() + 200;
        }
    }
    s_was_armed = armed;

    // Disarm is level-based, not edge-based, and applies to every energetic
    // state, but only while the radio link is live. A live CH10-low aborts
    // JUMPING/STANDING_UP immediately; a missing/off radio is neutral so a
    // GUI/API-armed session is not immediately undone.
    //
    // s_disarm_req_sent latches the request to a single send per active
    // session. Without it, this level check double-fires: g_state.state is
    // only updated when on_standby() actually runs, one tick after the
    // StateMachine library (see State::execute()) has already evaluated the
    // RUNNING->STANDBY transition, so this code still reads state==RUNNING
    // on the following tick and re-requests. That second request lands after
    // the FSM has already moved on to STANDBY, so it's never consumed there —
    // it sits stale until the next arm, where it fires instantly and kicks
    // RUNNING straight back to STANDBY (the "have to arm twice" symptom).
    if (!energetic) s_disarm_req_sent = false;
    if (s_unarmed_ticks >= DISARM_DEBOUNCE_TICKS && energetic &&
        !s_disarm_req_sent && !authority_disarm_requested) {
        comm_log(LOG_LEVEL_WARN,
                 "DISARM reason=RADIO_SWITCH_LOW: live CH10=%u authority=%s state=%d -> DISARMING",
                 (unsigned)ch10, arm_authority_name(), (int)g_state.state);
        stateMachine_disarm_running();
        s_disarm_req_sent = true;
    }

    // ── Calibration stick combo: STANDBY -> CALIBRATION, re-enter to cancel ───
    // CH1 + CH4 pinned full up, CH2 + CH3 pinned full down — the exact mirror of
    // the rescue combo above, so the two stick positions are mutually exclusive
    // and can never be satisfied on the same tick. This replaces the old CH5
    // switch, which is now the SD-log switch.
    //
    // Armed only in STANDBY (to start) and CALIBRATION (to cancel) —
    // deliberately NOT in ESTOP, so in a fault state the rescue combo is the
    // only live stick gesture and the two can't be confused under pressure.
    //
    // One-shot on the debounced rising edge. Re-entering the combo while a
    // radio-started calibration is still running cancels it through DISARMING:
    // a momentary combo has no "switch low" to fall back on, so this is the
    // only radio-side abort. CALIB_COMBO_LOCKOUT_MS then blocks a second action
    // for a second, so a stick glitch part-way through a deliberate hold can't
    // start a calibration and immediately cancel it again.
    //
    // Every term is guarded by `alive`: channel() returns 0 on signal loss, so
    // without it a dead radio would satisfy both stick-low tests for free.
    // A dead radio reads as "combo released", which is inert — only rising
    // edges act, so link loss can never imitate an operator request.
    static constexpr uint8_t  CALIB_DEBOUNCE_TICKS  = 3;     // ~6 ms @ 500 Hz
    static constexpr uint32_t CALIB_COMBO_LOCKOUT_MS = 1000;
    static uint8_t  s_calib_hi_ticks = 0;
    static uint8_t  s_calib_lo_ticks = 0;
    static bool     s_calib_combo_held = false;
    // Startup/link acquisition is an unknown stick position, not a released
    // one. Require a debounced release before accepting the next rising edge
    // so powering up or reconnecting with the sticks already in the combo
    // cannot start a calibration.
    static bool     s_calib_low_seen = false;
    static uint32_t s_calib_lockout_until_ms = 0;

    const bool calib_raw = alive &&
                           g_rc.channel(1) > 1990 && g_rc.channel(4) > 1990 &&
                           g_rc.channel(2) < 1010 && g_rc.channel(3) < 1010;
    if (calib_raw) {
        if (s_calib_hi_ticks < CALIB_DEBOUNCE_TICKS) s_calib_hi_ticks++;
        s_calib_lo_ticks = 0;
    } else {
        if (s_calib_lo_ticks < CALIB_DEBOUNCE_TICKS) s_calib_lo_ticks++;
        s_calib_hi_ticks = 0;
    }

    if (s_radio_calib_owned && g_state.state == STATE_CALIBRATION)
        s_radio_calib_entered = true;
    if (s_radio_calib_owned && s_radio_calib_entered &&
        g_state.state != STATE_CALIBRATION) {
        s_radio_calib_owned = false;
        s_radio_calib_entered = false;
    }

    if (s_calib_lo_ticks >= CALIB_DEBOUNCE_TICKS) {
        s_calib_low_seen   = true;
        s_calib_combo_held = false;
    }

    if (s_calib_hi_ticks >= CALIB_DEBOUNCE_TICKS && !s_calib_combo_held) {
        s_calib_combo_held = true;
        const bool valid_edge = s_calib_low_seen;
        // Consume the gate on every rising edge, so an edge taken outside
        // STANDBY/CALIBRATION still costs the operator a release-and-retry.
        s_calib_low_seen = false;
        const bool locked_out = (int32_t)(millis() - s_calib_lockout_until_ms) < 0;
        if (valid_edge && !locked_out && g_state.state == STATE_STANDBY) {
            comm_log(LOG_LEVEL_INFO, "Radio: calibration combo -> CALIBRATION");
            if (stateMachine_request_calibration()) {
                s_radio_calib_owned = true;
                s_radio_calib_entered = false;
                s_calib_lockout_until_ms = millis() + CALIB_COMBO_LOCKOUT_MS;
            }
        } else if (valid_edge && !locked_out &&
                   g_state.state == STATE_CALIBRATION && s_radio_calib_owned) {
            comm_log(LOG_LEVEL_WARN,
                     "Radio: calibration combo re-entered -> DISARMING");
            stateMachine_disarm_calibration();
            s_radio_calib_owned = false;
            s_radio_calib_entered = false;
            s_calib_lockout_until_ms = millis() + CALIB_COMBO_LOCKOUT_MS;
        } else if (!valid_edge && g_state.state == STATE_STANDBY) {
            comm_log(LOG_LEVEL_WARN,
                     "Radio: calibration combo ignored; release the sticks and re-enter");
        }
    }

    // CH5: debounced level. Always the SD-log switch -- gain-group select moved
    // to CH13/CH14, so CH5 no longer changes meaning with the tuning mode.
    static constexpr uint8_t CH5_DEBOUNCE_TICKS = 3;  // ~6 ms @ 500 Hz
    static uint8_t s_ch5_hi_ticks = 0;
    static uint8_t s_ch5_lo_ticks = 0;
    static bool s_ch5_switch_high = false;
    const bool ch5_hi_raw = alive && (ch5 > 1990);
    const bool ch5_lo_raw = alive && (ch5 <= 1990);
    if (ch5_hi_raw) {
        if (s_ch5_hi_ticks < CH5_DEBOUNCE_TICKS) s_ch5_hi_ticks++;
        s_ch5_lo_ticks = 0;
    } else if (ch5_lo_raw) {
        if (s_ch5_lo_ticks < CH5_DEBOUNCE_TICKS) s_ch5_lo_ticks++;
        s_ch5_hi_ticks = 0;
    } else {
        s_ch5_hi_ticks = 0;
        s_ch5_lo_ticks = 0;
    }
    if (s_ch5_hi_ticks >= CH5_DEBOUNCE_TICKS) s_ch5_switch_high = true;
    if (s_ch5_lo_ticks >= CH5_DEBOUNCE_TICKS) s_ch5_switch_high = false;

    // CH6: debounced level. Always the jump trigger, for the same reason.
    static constexpr uint8_t CH6_DEBOUNCE_TICKS = 3;  // ~6 ms @ 500 Hz
    static uint8_t s_ch6_hi_ticks = 0;
    static uint8_t s_ch6_lo_ticks = 0;
    static bool s_ch6_switch_high = true;  // neutral/unknown-at-boot default: no gain group selected
    const bool ch6_hi_raw = alive && (ch6 > 1990);
    const bool ch6_lo_raw = alive && (ch6 <= 1990);
    if (ch6_hi_raw) {
        if (s_ch6_hi_ticks < CH6_DEBOUNCE_TICKS) s_ch6_hi_ticks++;
        s_ch6_lo_ticks = 0;
    } else if (ch6_lo_raw) {
        if (s_ch6_lo_ticks < CH6_DEBOUNCE_TICKS) s_ch6_lo_ticks++;
        s_ch6_hi_ticks = 0;
    } else {
        s_ch6_hi_ticks = 0;
        s_ch6_lo_ticks = 0;
    }
    if (s_ch6_hi_ticks >= CH6_DEBOUNCE_TICKS) s_ch6_switch_high = true;
    if (s_ch6_lo_ticks >= CH6_DEBOUNCE_TICKS) s_ch6_switch_high = false;

    // ── CH5 as the SD-log switch (SIMPLE mode only) ───────────────────────────
    // Edge-triggered, not level-driven: re-issuing start every tick would hammer
    // a logger that is already running. s_ch5_log_prev starts UNKNOWN so the
    // first debounced reading only seeds it -- otherwise the init above would
    // look like an edge at boot and start/stop a log nobody asked for.
    //
    // Start is gated to STANDBY/ESTOP, matching CMD_ID_LOG's own gate: opening a
    // log preallocates and can block the loop ~96 ms, which is not something to
    // do with torque live. Once opened, the logger deliberately remains active
    // through RUNNING/JUMPING/STANDING_UP; only final close is deferred until
    // the robot returns to a non-energetic state.
    static int8_t s_ch5_log_prev = -1;              // -1 unknown, 0 low, 1 high
    const int8_t ch5_now = s_ch5_switch_high ? 1 : 0;
    if (s_ch5_log_prev < 0) {
        s_ch5_log_prev = ch5_now;                   // seed, never act on it
    } else if (ch5_now != s_ch5_log_prev) {
        s_ch5_log_prev = ch5_now;
        if (ch5_now == 1) {
            if (g_state.state == STATE_STANDBY || g_state.state == STATE_ESTOP) {
                const bool started = sd_logger_start(0);   // 0 = until stopped
                forgive_sd_blocking_stall();
                if (started) {
                    g_buzzer.play(LOG_START_CHIRP, 1);
                    comm_log(LOG_LEVEL_INFO, "Radio: CH5 up -> SD logging STARTED");
                } else {
                    comm_log(LOG_LEVEL_WARN, "Radio: CH5 up -> SD log start FAILED");
                }
            } else {
                comm_log(LOG_LEVEL_WARN,
                         "Radio: CH5 up ignored -- start the log before arming");
            }
        } else if (sd_logger_is_active()) {
            sd_logger_stop();
            forgive_sd_blocking_stall();
            comm_log(LOG_LEVEL_INFO, "Radio: CH5 down -> SD logging STOPPED");
        }
    }

    // ── CH6 as the jump trigger (SIMPLE mode only) ────────────────────────────
    // One jump per debounced low->high edge: the switch must be dropped and
    // raised again for the next one, so holding it high does not hop repeatedly.
    // stateMachine_request_jump() is the real gate -- it refuses unless
    // PARAM_JUMP_ENABLE is set and the state is exactly RUNNING, and logs its
    // own reason -- so all this has to do is not re-request while held.
    //
    // s_ch6_jump_prev seeds UNKNOWN for the same reason the log switch does:
    // the `true` init above would otherwise read as an edge at boot.
    static int8_t s_ch6_jump_prev = -1;             // -1 unknown, 0 low, 1 high
    const int8_t ch6_now = s_ch6_switch_high ? 1 : 0;
    if (s_ch6_jump_prev < 0) {
        s_ch6_jump_prev = ch6_now;                  // seed, never act on it
    } else if (ch6_now != s_ch6_jump_prev) {
        s_ch6_jump_prev = ch6_now;
        if (ch6_now == 1) {
            if (stateMachine_request_jump()) {
                comm_log(LOG_LEVEL_INFO, "Radio: CH6 up -> JUMP");
            } else {
                comm_log(LOG_LEVEL_WARN,
                         "Radio: CH6 up -> jump refused (state=%d)", (int)g_state.state);
            }
        }
    }

    // §1d (tuning.md): while PARAM_GUI_MOTION_CTRL_EN is set, the two radio-driven
    // writes to v_cmd_ms/omega_cmd_rds below are skipped so a GUI/CLI param_set()
    // on those two params stands instead. Never gates radio arming/disarming.
    // When the radio is absent, fresh GUI commands stand; the watchdog zeros
    // them and releases the override if packets stop arriving. Auto-reverts
    // after GUI_MOTION_CTRL_TIMEOUT_MS, mirroring
    // state_machine.cpp's MANUAL_GUI_TIMEOUT_MS watchdog pattern.
    static constexpr uint32_t GUI_MOTION_CTRL_TIMEOUT_MS = 300;
    bool gui_motion_ctrl = param_get(PARAM_GUI_MOTION_CTRL_EN) >= 0.5f;
    if (gui_motion_ctrl && stateMachine_ms_since_gui_packet() >= GUI_MOTION_CTRL_TIMEOUT_MS) {
        comm_log(LOG_LEVEL_WARN, "GUI motion ctrl: watchdog timeout -> reverting to radio");
        param_force_set(PARAM_V_CMD_MS, 0.0f);
        param_force_set(PARAM_OMEGA_CMD_RDS, 0.0f);
        param_force_set(PARAM_GUI_MOTION_CTRL_EN, 0.0f);
        gui_motion_ctrl = false;
    }

    if (alive) {
        float t = constrain((g_rc.channel(3) - 1000.0f) / 1000.0f, 0.0f, 1.0f);  // CH3 (1-indexed)
        param_force_set(PARAM_RADIO_HIP_CMD, t);

        // CH1: roll setpoint for the active-suspension roll controller. Ungated by
        // gui_motion_ctrl (that override only covers v/omega). Sign bench-verified.
        float roll_norm = constrain((g_rc.channel(1) - 1500.0f) / 500.0f, -1.0f, 1.0f);
        const float roll_stick = roll_norm * param_get(PARAM_RADIO_ROLL_MAX);

        // ── Coordinated-turn lean ────────────────────────────────────────────
        // Turning at speed throws the mass sideways; leaning into the turn puts
        // the resultant of gravity and centripetal acceleration back along the
        // robot's own vertical, so the hips carry compression instead of a side
        // load. See lean_turn.h for the physics and, more importantly, the sign
        // derivation.
        //
        // This is a SETPOINT GENERATOR, not a controller. Everything that makes
        // it safe already exists downstream in control_loop.cpp: the slew limit
        // on the setpoint, the PID with anti-windup, and the travel-headroom
        // clamp at min(t, 1-t)*span. It also inherits, for free, the roll
        // controller's exclusion during jump launch and flight.
        //
        // Measured velocity, COMMANDED yaw rate, deliberately. Forward speed is
        // well tracked so the measurement is accurate and slow-moving; yaw rate
        // is the fast term, and using the command means the robot leans AS the
        // turn is asked for rather than after the disturbance shows up in roll
        // error. That lead is the whole point of a feedforward.
        //
        // Three gates, all required: the persistent lean_turn_en, a non-zero
        // lean_gain, and the radio's lean button. A feature that can put the
        // robot on its side should not switch itself on.
        float lean = 0.0f;
        const bool lean_switch = (g_rc.channel(15) > 1990);
        if (lean_switch && param_get(PARAM_LEAN_TURN_EN) >= 0.5f) {
            lean = lean_turn_setpoint(g_state.wheel_vel_avg_ms,
                                      param_get(PARAM_OMEGA_CMD_RDS),
                                      param_get(PARAM_LEAN_GAIN),
                                      param_get(PARAM_LEAN_MAX_RAD),
                                      param_get(PARAM_LEAN_MIN_MS));
            // Asked to lean but the roll controller is off: the setpoint would
            // be written and silently ignored, which is exactly the kind of
            // "why is it not doing anything" that costs an evening.
            static uint32_t s_lean_warn_ms = 0;
            if (param_get(PARAM_ROLL_CTRL_EN) < 0.5f &&
                (int32_t)(millis() - s_lean_warn_ms) >= 0) {
                s_lean_warn_ms = millis() + 5000;
                comm_log(LOG_LEVEL_WARN,
                         "Lean-turn requested but roll_ctrl_en=0; the roll "
                         "controller delivers the lean, so nothing will happen");
            }
        }
        param_force_set(PARAM_LEAN_CMD_RAD, lean);
        param_force_set(PARAM_LEAN_AUTHORITY,
                        lean_turn_authority(param_get(PARAM_RADIO_HIP_CMD)));

        // The stick still adds on top, so manual counter-lean stays available
        // for recovery. Clamped to the profile's roll limit so the automatic
        // term can never exceed what the operator could have commanded by hand.
        const float roll_lim = param_get(PARAM_RADIO_ROLL_MAX);
        float roll_cmd = roll_stick + lean;
        if (roll_cmd >  roll_lim) roll_cmd =  roll_lim;
        if (roll_cmd < -roll_lim) roll_cmd = -roll_lim;
        param_force_set(PARAM_ROLL_CMD_RAD, roll_cmd);

        if (!gui_motion_ctrl) {
            float vel_norm = constrain((g_rc.channel(2) - 1500.0f) / 500.0f, -1.0f, 1.0f);
            param_force_set(PARAM_V_CMD_MS, vel_norm * param_get(PARAM_RADIO_VEL_MAX));

            float yaw_norm = -constrain((g_rc.channel(4) - 1500.0f) / 500.0f, -1.0f, 1.0f);  // inverted: stick left -> robot yaws left
            param_force_set(PARAM_OMEGA_CMD_RDS, yaw_norm * param_get(PARAM_RADIO_YAW_MAX));
        }

        // CH9: speed profile selector (3-position switch → profile 0/1/2)
        static const uint16_t PROFILE_VEL[]    = {PARAM_PROFILE_1_VEL_MAX,    PARAM_PROFILE_2_VEL_MAX,    PARAM_PROFILE_3_VEL_MAX};
        static const uint16_t PROFILE_YAW[]    = {PARAM_PROFILE_1_YAW_MAX,    PARAM_PROFILE_2_YAW_MAX,    PARAM_PROFILE_3_YAW_MAX};
        static const uint16_t PROFILE_TORQUE[] = {PARAM_PROFILE_1_TORQUE_LIM, PARAM_PROFILE_2_TORQUE_LIM, PARAM_PROFILE_3_TORQUE_LIM};
        static const uint16_t PROFILE_ROLL[]   = {PARAM_PROFILE_1_ROLL_MAX,   PARAM_PROFILE_2_ROLL_MAX,   PARAM_PROFILE_3_ROLL_MAX};
        static uint8_t s_last_profile = 255;   // force apply on first packet
        static float   s_trq_target   = -1.0f; // <0 = no pending slew
        uint16_t ch9 = g_rc.channel(9);
        uint8_t profile = (ch9 < 1333) ? 0 : (ch9 < 1667) ? 1 : 2;
        if (profile != s_last_profile) {
            s_last_profile = profile;
            param_force_set(PARAM_ACTIVE_PROFILE, (float)profile);
            param_force_set(PARAM_RADIO_VEL_MAX, param_get(PROFILE_VEL[profile]));
            param_force_set(PARAM_RADIO_YAW_MAX, param_get(PROFILE_YAW[profile]));
            param_force_set(PARAM_RADIO_ROLL_MAX, param_get(PROFILE_ROLL[profile]));
            s_trq_target = param_get(PROFILE_TORQUE[profile]); // applied via slew below
            comm_log(LOG_LEVEL_INFO, "Radio: speed profile %u", (unsigned)(profile + 1));

            // LED flash: green=slow, yellow=medium, red=fast
            static const uint8_t PROFILE_COLORS[][3] = {
                {0, 200, 0},    // P1 — green
                {255, 180, 0},  // P2 — yellow
                {255, 0, 0},    // P3 — red
            };
            memcpy(s_profile_flash_rgb, PROFILE_COLORS[profile], 3);
            s_profile_flash_until_ms = millis() + 400;

            // Buzzer: ascending CMaj arpeggio, note count = profile+1
            static const BuzzerNote PROFILE_MELODIES[][3] = {
                {{72, 120, 0},  {0,   0,  0}, {0,   0,  0}},  // P1: C5
                {{72,  80, 20}, {76, 120, 0}, {0,   0,  0}},  // P2: C5→E5
                {{72,  60, 15}, {76,  60, 15}, {79, 120, 0}}, // P3: C5→E5→G5
            };
            g_buzzer.play(PROFILE_MELODIES[profile], profile + 1);
        }

        // Slew PARAM_LQR_TORQUE_LIMIT toward the profile target at 5 N·m/s.
        // Upward steps apply immediately (safe); downward steps are ramped to
        // avoid destabilising the balancer mid-run.
        if (s_trq_target >= 0.0f) {
            static constexpr float TRQ_SLEW_STEP = 5.0f * 0.002f; // 5 N·m/s @ 500 Hz
            float cur = param_get(PARAM_LQR_TORQUE_LIMIT);
            if (s_trq_target >= cur) {
                param_force_set(PARAM_LQR_TORQUE_LIMIT, s_trq_target);
                s_trq_target = -1.0f;
            } else {
                float next = cur - TRQ_SLEW_STEP;
                if (next <= s_trq_target) { next = s_trq_target; s_trq_target = -1.0f; }
                param_force_set(PARAM_LQR_TORQUE_LIMIT, next);
            }
        }

    } else {
        // Radio link dead: never hold a lean. Roll isn't part of gui_motion_ctrl,
        // so zero it regardless; v/omega only when the GUI isn't driving them.
        param_force_set(PARAM_ROLL_CMD_RAD, 0.0f);
        param_force_set(PARAM_LEAN_CMD_RAD, 0.0f);
        if (!gui_motion_ctrl) {
            param_force_set(PARAM_V_CMD_MS, 0.0f);
            param_force_set(PARAM_OMEGA_CMD_RDS, 0.0f);
        }
    }

    // ── Live parameter tuning (CH7/CH8 knobs) ──────────────────────────────────
    // See LIVE_TUNE_SLOTS above for the group table and knob-direction
    // convention, and live_tune.h for the safety rationale (pickup, latch).
    //
    // The group is selected by ONE channel, CH13, carrying all three
    // mutually-exclusive buttons encoded as distinct levels. That selection
    // used to live on the CH5/CH6 combination, which cost a tuning session both
    // SD logging and the jump trigger -- making tuning bench-only by
    // construction, and the one session you most want a log of the one you
    // could not log.
    //
    // Encoded rather than one channel per group because all 16 channels are
    // spoken for and the buttons are already mutually exclusive, so three
    // channels was three ways to say one thing. Each button mixes in with
    // weight w and offset w, so OFF contributes 0 and ON contributes 2w:
    //
    //   none  ->    0%  -> 1500 us
    //   SG    ->   32%  -> 1660 us    group 0
    //   SH    ->   66%  -> 1830 us    group 1
    //   SI    ->  100%  -> 2000 us    group 2
    //
    // ~165 us between levels, decoded with +/-80 us bands. The exclusive switch
    // group makes any other sum impossible; if one ever appears the band decode
    // still lands somewhere deterministic rather than oscillating.
    //
    // "None = 1500 us = inactive" matters as much as the rest: the resting
    // state of the buttons has to be the state that does nothing.
    static constexpr uint16_t LT_BAND_NONE = 1580;   // below -> inactive
    static constexpr uint16_t LT_BAND_G0   = 1745;
    static constexpr uint16_t LT_BAND_G1   = 1915;

    static constexpr uint8_t LT_SEL_DEBOUNCE_TICKS = 3;
    static uint8_t s_lt_stable_ticks = 0;
    static int8_t  s_lt_group_raw = -1;
    static int8_t  s_lt_group_deb = -1;

    int8_t group_now = -1;
    if (alive) {
        const uint16_t us = g_rc.channel(13);
        if      (us < LT_BAND_NONE) group_now = -1;
        else if (us < LT_BAND_G0)   group_now = 0;
        else if (us < LT_BAND_G1)   group_now = 1;
        else                        group_now = 2;
    }
    // Link loss reads as inactive, which drops every slot's pickup next pass.
    if (group_now == s_lt_group_raw) {
        if (s_lt_stable_ticks < LT_SEL_DEBOUNCE_TICKS) s_lt_stable_ticks++;
    } else {
        s_lt_group_raw = group_now;
        s_lt_stable_ticks = 0;
    }
    if (s_lt_stable_ticks >= LT_SEL_DEBOUNCE_TICKS) s_lt_group_deb = s_lt_group_raw;

    int8_t live_tune_group = -1;
    if (param_get(PARAM_LIVE_TUNE_MULTI_EN) >= 0.5f) live_tune_group = s_lt_group_deb;

    static int8_t s_lt_group_prev = -1;
    if (live_tune_group != s_lt_group_prev) {
        s_lt_group_prev = live_tune_group;
        if (live_tune_group >= 0)
            comm_log(LOG_LEVEL_INFO, "Live-tune group %d selected (CH13)",
                     (int)live_tune_group);
        else
            comm_log(LOG_LEVEL_INFO, "Live-tune inactive (no group selected)");
    }
    bool live_tune_active = (g_state.state == STATE_RUNNING) && (live_tune_group >= 0);
    static constexpr float LIVE_TUNE_PICKUP_EPS = 0.01f;
    for (uint8_t i = 0; i < NUM_LIVE_TUNE_SLOTS; i++) {
        const LiveTuneSlot& slot = LIVE_TUNE_SLOTS[i];
        bool slot_active = live_tune_active && (slot.group == (uint8_t)live_tune_group);
        if (!slot_active) { s_live_tune_picked_up[i] = false; continue; }
        uint16_t raw = g_rc.channel(slot.ibus_channel);
        float knob_val = slot.range_min + constrain((raw - 1000.0f) / 1000.0f, 0.0f, 1.0f)
                                            * (slot.range_max - slot.range_min);
        param_force_set(slot.live_param_id, knob_val);  // telemetry mirror, always kept live
        float current = param_get(slot.persist_param_id);
        if (!s_live_tune_picked_up[i] && fabsf(knob_val - current) <= LIVE_TUNE_PICKUP_EPS) {
            s_live_tune_picked_up[i] = true;
            comm_log(LOG_LEVEL_INFO, "Live-tune slot %u picked up: CH%u -> param 0x%04X",
                     (unsigned)i, (unsigned)slot.ibus_channel, (unsigned)slot.persist_param_id);
        }
    }
    s_live_tune_mode_active = live_tune_active;  // read by live_tune_value()

    // Latch (one-shot): commit every currently-picked-up slot's shadow into its
    // real, persistent param. A slot that hasn't picked up yet is skipped, not
    // latched at whatever the knob happens to read. Serviced regardless of
    // live_tune_active so the command flag is always consumed and reset.
    // CH14 rising edge is a physical "commit these gains", so latching no
    // longer means walking back to the GUI mid-session. Edge-triggered with the
    // same release-and-retry gate as CH11/CH12: the button may rest either way
    // and a reconnect must not look like a fresh press. It simply sets the same
    // one-shot flag the GUI writes, so there is one latch path, not two.
    static RadioEdge s_latch_sw = {};
    if (edge_update(s_latch_sw, g_rc.channel(14) > 1990, alive)) {
        comm_log(LOG_LEVEL_INFO, "Radio: CH14 -> live-tune latch requested");
        param_force_set(PARAM_LIVE_TUNE_LATCH, 1.0f);
    }

    if (param_get(PARAM_LIVE_TUNE_LATCH) >= 0.5f) {
        if (live_tune_active) {
            bool any = false;
            for (uint8_t i = 0; i < NUM_LIVE_TUNE_SLOTS; i++) {
                if (!s_live_tune_picked_up[i]) continue;
                const LiveTuneSlot& slot = LIVE_TUNE_SLOTS[i];
                float v = param_get(slot.live_param_id);
                param_set(slot.persist_param_id, v);  // persistent write; flushed by param_flush_service()
                comm_log(LOG_LEVEL_INFO, "Live-tune latched: %.4f -> param 0x%04X (CH%u)",
                         v, (unsigned)slot.persist_param_id, (unsigned)slot.ibus_channel);
                any = true;
            }
            if (!any) comm_log(LOG_LEVEL_WARN, "Live-tune latch ignored: no slot has picked up yet");
        } else {
            comm_log(LOG_LEVEL_WARN,
                     "Live-tune latch ignored: enter live-tune mode first "
                     "(RUNNING, live_tune_multi_en=1, and a group button lit)");
        }
        param_force_set(PARAM_LIVE_TUNE_LATCH, 0.0f);
    }

    // Mirrors v_cmd_ms live regardless of FSM state (like w_cmd/omega_cmd_rds
    // above) so it's visible for troubleshooting before arming, not just
    // while controlLoop_run() is active in RUNNING/JUMPING.
    g_state.v_ref = (param_get(PARAM_VEL_PI_EN) >= 0.5f) ? param_get(PARAM_V_CMD_MS) : 0.0f;
}

static void run_control_loop() {
    const RobotStateEnum state_before_update = g_state.state;
    stateMachine_update();

    // One transition-time diagnostic makes the recording invariant visible in
    // the HOST log: an SD file opened in STANDBY must still be active after the
    // state machine enters its first torque-producing state. Do not auto-open
    // here -- preAllocate() blocks for ~96 ms -- but make a missed pre-arm start
    // explicit instead of discovering it only after downloading an empty log.
    const bool was_armed = state_before_update == STATE_RUNNING ||
                           state_before_update == STATE_JUMPING ||
                           state_before_update == STATE_STANDING_UP;
    const bool is_armed  = g_state.state == STATE_RUNNING ||
                           g_state.state == STATE_JUMPING ||
                           g_state.state == STATE_STANDING_UP;
    if (!was_armed && is_armed) {
        if (sd_logger_is_active()) {
            comm_log(LOG_LEVEL_INFO, "ARM SD logging ACTIVE index=%u",
                     (unsigned)sd_logger_active_index());
        } else {
            comm_log(LOG_LEVEL_WARN, "ARM entered without active SD logging");
        }
    }
}

// SD-log open/finalize are known-blocking main-loop operations that can freeze
// the 500 Hz tick for tens of ms (measured ~96 ms on open, ~56 ms on finalize).
// During that freeze no motor command goes out, so request-response feedback
// (AK45 hips) goes stale and the feedback watchdog would spuriously ESTOP on
// the next poll(). Call this right after any such op — logs only ever open or
// finalize in non-energetic states (STANDBY/ESTOP/STARTUP/CMD_REJECT), so a
// one-interval grace here can't hide a dropout that matters to balance.
static void forgive_sd_blocking_stall() {
    hip_motors_forgive_feedback_stall();
    wheel_motors_forgive_feedback_stall();
}

// Watch for IMU state transitions and emit a log packet on each change.
// Never logs in steady-state NOMINAL — no noise on a healthy system.
static void check_imu_state() {
    static ImuState prev = ImuState::NOT_READY;
    ImuState cur = imu_state();
    if (cur == prev) return;
    prev = cur;
    switch (cur) {
        case ImuState::INITIALIZING:
            comm_log(LOG_LEVEL_INFO,  "IMU: initializing...");           break;
        case ImuState::NOMINAL:
            comm_log(LOG_LEVEL_INFO,  "IMU: NOMINAL");                   break;
        case ImuState::DEGRADED:
            comm_log(LOG_LEVEL_WARN,  "IMU: degraded/recovering — %.0f%% loss",
                     imu_packet_loss() * 100.0f);                        break;
        case ImuState::ERROR:
            comm_log(LOG_LEVEL_ERROR, "IMU: recovery failed — automatic retry pending");  break;
        default: break;
    }
}

static void check_limit_switches() {
    const uint8_t changes = limit_switches_update();
    const bool left_pressed =
        (changes & LIMIT_SWITCH_LEFT_CHANGED) && limit_switch_left_active();
    const bool right_pressed =
        (changes & LIMIT_SWITCH_RIGHT_CHANGED) && limit_switch_right_active();

    if (changes & LIMIT_SWITCH_LEFT_CHANGED) {
        comm_log(LOG_LEVEL_INFO, "Limit switch LEFT (pin %u): %s",
                 PIN_LIMIT_LEFT,
                 limit_switch_left_active() ? "PRESSED" : "RELEASED");
    }
    if (changes & LIMIT_SWITCH_RIGHT_CHANGED) {
        comm_log(LOG_LEVEL_INFO, "Limit switch RIGHT (pin %u): %s",
                 PIN_LIMIT_RIGHT,
                 limit_switch_right_active() ? "PRESSED" : "RELEASED");
    }

    // Preserve both audible events if the two debounced edges land in the
    // same 500 Hz tick; separate midi() calls would let the second overwrite
    // the first in the single-channel buzzer driver.
    static const BuzzerNote BOTH_SWITCH_CHIRPS[] = {
        {84, 50, 15}, {88, 50, 0}
    };
    if (left_pressed && right_pressed)
        g_buzzer.play(BOTH_SWITCH_CHIRPS, 2);
    else if (left_pressed)
        g_buzzer.midi(84, 255, 60);
    else if (right_pressed)
        g_buzzer.midi(88, 255, 60);
}

// ── Main loop ─────────────────────────────────────────────────────────────────

void loop() {
    uint32_t t_start = micros();
    // One increment per 500 Hz tick, before any early-out below can skip it.
    // Telemetry ships this so the GUI can (a) spot dropped telemetry frames
    // from a delta != 10 between consecutive 50 Hz packets, and (b) confirm a
    // reflash actually landed by seeing it restart near 0. It was declared and
    // sent from the start but never incremented anywhere, so it always read 0.
    g_state.loop_count++;
    watchdog_pet();

    bool prof = param_get(PARAM_LOOP_PROFILE_ENABLE) >= 0.5f;
    uint32_t t0;

    t0 = micros(); receive_commands();     if (prof) prof_mark(s_prof_max.recv,       t0);
    service_param_dump();
    t0 = micros(); read_sensors(prof);     if (prof) prof_mark(s_prof_max.sens_total, t0);
    check_limit_switches();
    t0 = micros(); check_imu_state();      if (prof) prof_mark(s_prof_max.imu_chk,    t0);
    t0 = micros(); radio_update();         if (prof) prof_mark(s_prof_max.radio,      t0);
    t0 = micros(); run_control_loop();     if (prof) prof_mark(s_prof_max.ctrl,       t0);
    t0 = micros(); update_led();           if (prof) prof_mark(s_prof_max.led,        t0);
    t0 = micros(); update_buzzer();        if (prof) prof_mark(s_prof_max.buz,        t0);

    t0 = micros();
    bool sd_recording_safe = g_state.state == STATE_STARTUP ||
                             g_state.state == STATE_STANDBY ||
                             g_state.state == STATE_ESTOP ||
                             g_state.state == STATE_CMD_REJECT;
    // Recording deliberately CONTINUES through RUNNING/JUMPING/STANDING_UP.
    // There used to be a "stopping before energetic state" stop here (added in
    // 39a5040 alongside the deferred finalize below); it was removed 2026-08-09
    // because it made the 500 Hz path useless for exactly the states worth
    // logging, and it was never what protected the control loop:
    //
    //   - sd_logger_write() only copies into a 32 KB DMAMEM RingBuf. Never
    //     blocks; on overflow it warns and drops samples.
    //   - sd_logger_service() flushes at most ONE 512-byte sector per tick, and
    //     only when !isBusy() -- it skips rather than waits. That is SdFat's own
    //     TeensySdioLogger streaming pattern, designed to run continuously.
    //   - The two genuinely blocking operations are already handled elsewhere:
    //     start/preAllocate (~96 ms) is gated to STANDBY/ESTOP at its call
    //     sites, and close (~56 ms) is deferred by the
    //     sd_logger_finalize_service(sd_recording_safe) call below.
    //
    // At 251 B/record and 500 Hz the buffer fills at 123 KB/s against a
    // 250 KB/s drain, so it absorbs ~261 ms of total card stall before dropping
    // samples -- a data-quality risk that announces itself in the log, not a
    // loop-timing one. sd_recording_safe survives purely to gate the finalize.
    if (sd_logger_is_active()) {
        static LogRecord rec;
        fill_telemetry(rec.telem);
        rec.t_micros = micros();
        sd_logger_write(&rec);
    }
    sd_logger_service();            // 1 sector/tick + auto-stop
    sd_logger_service_transfer();   // paced chunk streaming during a GET
    // Closing a preallocated log may sync/truncate for tens of milliseconds.
    // Defer that work until no energetic state needs the 500 Hz controller.
    if (sd_logger_finalize_service(sd_recording_safe))
        forgive_sd_blocking_stall();  // finalize blocked the loop ~56 ms
    // Download-complete chirp: sd_logger_service_transfer() emits XFER_END
    // and clears its own active flag internally (sd_logger.cpp) with no
    // callback out to here, so detect the same thing via the falling edge
    // of sd_logger_transfer_active() instead of touching that file.
    {
        static bool s_xfer_was_active = false;
        bool xfer_active = sd_logger_transfer_active();
        if (s_xfer_was_active && !xfer_active) g_buzzer.play(LOG_DONE_CHIRP, 2);
        s_xfer_was_active = xfer_active;
    }
    if (prof) prof_mark(s_prof_max.sd, t0);

    // Deferred param flash flush — a LittleFS rewrite stalls the loop for
    // several ms, so never while balancing (RUNNING/JUMPING).
    t0 = micros();
    // LittleFS save/verify takes several milliseconds. Keep it out of every
    // energetic or ramp-down state, not just RUNNING/JUMPING: a flush during
    // DISARMING was measured at 4.6 ms and starved the IMU badly enough to
    // trigger IMU_LOST. sd_recording_safe names the same inert-state set.
    param_flush_service(sd_recording_safe);
    if (prof) prof_mark(s_prof_max.flash, t0);

    // Send telemetry at 50 Hz (every 10th tick of the 500 Hz loop) — skipped
    // while a log GET is streaming (sd_logger_transfer_active()) so 100% of
    // the paced Teensy->ESP32 link bandwidth goes to the log chunks instead
    // of competing with telemetry; the ESP32/GUI have their own "downloading"
    // indicator that doesn't depend on telemetry for this window (see
    // esp32/src/main.cpp g_log_xfer_active).
    t0 = micros();
    static uint8_t telem_div = 0;
    if (++telem_div >= 10) {
        telem_div = 0;
        if (!sd_logger_transfer_active()) {
            if (param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f || param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f) wheel_motors_request_vbus();
            send_telemetry(prof);
        }
    }
    if (prof) prof_mark(s_prof_max.telem, t0);

    // Radio telemetry, the return half of the CRSF link. Rate-limits itself to
    // one frame every 25 ms; see CrsfTelem.h for the budget and why it is not
    // simply a mirror of TelemetryPayload.
    crsf_telemetry_tick();

    if (prof && millis() - s_prof_last_ms >= 1000) {
        s_prof_last_ms = millis();
        comm_log(LOG_LEVEL_INFO,
            "LoopProf(us) recv=%lu sens=%lu imu_chk=%lu radio=%lu ctrl=%lu led=%lu buz=%lu sd=%lu flash=%lu telem=%lu",
            (unsigned long)s_prof_max.recv,   (unsigned long)s_prof_max.sens_total,
            (unsigned long)s_prof_max.imu_chk,(unsigned long)s_prof_max.radio,
            (unsigned long)s_prof_max.ctrl,   (unsigned long)s_prof_max.led,
            (unsigned long)s_prof_max.buz,    (unsigned long)s_prof_max.sd,
            (unsigned long)s_prof_max.flash,  (unsigned long)s_prof_max.telem);
        comm_log(LOG_LEVEL_INFO,
            "LoopProf/sens(us) imu=%lu hip=%lu wheel=%lu ibus=%lu",
            (unsigned long)s_prof_max.imu, (unsigned long)s_prof_max.hip,
            (unsigned long)s_prof_max.wheel, (unsigned long)s_prof_max.ibus);
        comm_log(LOG_LEVEL_INFO,
            "LoopProf/telem(us) fill=%lu esp=%lu usb=%lu",
            (unsigned long)s_prof_max.telem_fill, (unsigned long)s_prof_max.telem_esp,
            (unsigned long)s_prof_max.telem_usb);
        s_prof_max = {};
    }

    // Loop-overrun detection: count ticks whose work time blew the 2 ms
    // budget; surfaced via HEALTH_LOOP_OVERRUN + a rate-limited WARN log.
    // Skipped during STARTUP — the 500 Hz budget isn't required there (no
    // torque is commanded yet), so a slow tick while e.g. hip motors are
    // still coming up isn't a real problem worth counting/warning about.
    uint32_t work_us = micros() - t_start;
    if (work_us > 2000 && g_state.state != STATE_STARTUP) {
        s_overrun_count++;
        s_last_overrun_ms = millis();
        static uint32_t s_last_log_ms = 0;
        if (millis() - s_last_log_ms >= 1000) {
            s_last_log_ms = millis();
            comm_log(LOG_LEVEL_WARN, "Loop overrun: %lu us (count %lu)",
                     (unsigned long)work_us, (unsigned long)s_overrun_count);
        }
    }

    while (micros() - t_start < 2000) {}
}
