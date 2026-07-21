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
#include "IBus.h"
#include "sd_logger.h"

CommLink g_comm(Serial5, COMM_SRC_TEENSY);     // ESP32 UART bridge
CommLink g_comm_usb(Serial, COMM_SRC_TEENSY);  // direct PC USB
IBus     g_ibus(Serial4);                       // FlySky RC receiver, pin 16
RgbLed   g_led(PIN_LED_R, PIN_LED_G, PIN_LED_B);
Buzzer   g_buzzer(PIN_BUZZER);

HipCmd   g_hip_cmd = {};

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
    if (Serial) g_comm_usb.send(COMM_TYPE_LOG, LOG_PAYLOAD_V1, buf, 1 + n);
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
    if (Serial) g_comm_usb.send(type, version, payload, len);
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
    if (Serial) g_comm_usb.send(COMM_TYPE_CALIB_EVENT, CALIB_EVENT_PAYLOAD_V1, &p, sizeof(p));
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
    if (Serial) g_comm_usb.send(COMM_TYPE_PARAM_REPORT, PARAM_REPORT_PAYLOAD_V1, &rpt, sizeof(rpt));
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

// Very short chirp for a param write while STANDBY — distinct from RADIO_ACQ_MELODY
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
    if (Serial) g_comm_usb.send(COMM_TYPE_COMMAND_RESULT, COMMAND_RESULT_PAYLOAD_V1, &result, sizeof(result));
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
                     g_state.state == STATE_STANDING_UP)
                accepted = stateMachine_disarm_running();
            else if (g_state.state == STATE_MANUAL || g_state.state == STATE_CALIBRATION)
                accepted = stateMachine_exit_manual();
        }
        if (target == STATE_STARTUP)     accepted = stateMachine_request_reset();
        if (target == STATE_CALIBRATION) accepted = stateMachine_request_calibration();
        if (target == STATE_ESTOP)       accepted = stateMachine_request_estop();
        // STATE_RUNNING: previously radio-only (CH10 arm switch, main.cpp radio_update()).
        // Routed through the identical req_running() gate in state_machine.cpp — same
        // IMU/calibration/motor-enable checks apply, no separate/weaker path.
        if (target == STATE_RUNNING)     accepted = stateMachine_request_running();
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
        // send_reliable() (GUI comm_commands.py) retries the identical (id, val) up to
        // 3x every 250 ms until it sees the PARAM_REPORT echo, so a slow link can deliver
        // the same value more than once — only chirp when it actually changed.
        if (g_state.state == STATE_STANDBY &&
            (set_result == ParamSetResult::OK || set_result == ParamSetResult::CLAMPED) &&
            param_get(id) != old_val) {
            g_buzzer.play(PARAM_SET_CHIRP, 1, 120);
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
            bool started = sd_logger_start(dur);
            if (started) g_buzzer.play(LOG_START_CHIRP, 1, 150);
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
            sd_logger_begin_get(idx, start);
            if (sd_logger_transfer_active()) g_buzzer.play(LOG_GET_CHIRP, 1, 150);
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

    g_led.begin();
    g_buzzer.begin();
    g_buzzer.set_volume(param_get(PARAM_BUZZER_VOLUME));

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
    g_buzzer.play(BOOT_CHIME, sizeof(BOOT_CHIME) / sizeof(BOOT_CHIME[0]), 30);  // quiet arpeggio
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
    g_ibus.begin();
    comm_log(LOG_LEVEL_INFO, "IBus RX ready (Serial4)");

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
    float soft_lim = param_get(PARAM_WHEEL_VEL_LIMIT_TURNS_S);
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
    if (param_get(PARAM_VEL_PI_EN) >= 0.5f
        && fabsf(g_state.theta_ref) >= param_get(PARAM_VEL_PI_THETA_MAX))
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
    t.hip_l_current_a  = g_state.hip_l_current_a;
    t.hip_r_current_a  = g_state.hip_r_current_a;
    for (uint8_t i = 1; i <= IBUS_NUM_CH; i++) t.ibus_ch[i - 1] = g_ibus.channel(i);
    t.ibus_alive       = g_ibus.alive() ? 1 : 0;
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
    t.pitch_trim_rad        = param_get(PARAM_RADIO_PITCH_TRIM);
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
    if (Serial) {
        g_comm_usb.send(COMM_TYPE_TELEM_A, TELEM_VERSION, tp,               TELEM_A_LEN);
        g_comm_usb.send(COMM_TYPE_TELEM_B, TELEM_VERSION, tp + TELEM_A_LEN, TELEM_B_LEN);
    }
    if (prof) prof_mark(s_prof_max.telem_usb, t0);
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
        g_state.hip_l_current_a = hm_L.current_A;
        g_state.hip_r_current_a = hm_R.current_A;
    }

    if (param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f || param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f) {
        t0 = micros();
        wheel_motors_poll();
        wheel_motors_pet_watchdog();
        if (prof) prof_mark(s_prof_max.wheel, t0);
    }

    t0 = micros();
    g_ibus.update();
    for (uint8_t i = 1; i <= IBUS_NUM_CH; i++)
        param_force_set(PARAM_IBUS_CH0 + (i - 1), (float)g_ibus.channel(i));
    param_force_set(PARAM_IBUS_ALIVE, g_ibus.alive() ? 1.0f : 0.0f);
    if (prof) prof_mark(s_prof_max.ibus, t0);
}

// ── Radio melodies ────────────────────────────────────────────────────────────
static const BuzzerNote RADIO_ACQ_MELODY[]    = {{76, 80, 0}};             // E5 — single: "link up"
static const BuzzerNote RADIO_LOST_MELODY[]   = {{64, 100, 30}, {60, 150, 0}}; // E4→C4 desc: "link lost"
static const BuzzerNote ARM_IGNORED_MELODY[]  = {{69, 60, 40}, {69, 60, 0}};   // A4-A4 double-tap: "not ready, try again"

// ── Radio interpretation ───────────────────────────────────────────────────────
// CH10 > 1990: arm into RUNNING (requires prior calibration).
// CH10 drop:   disarm back to STANDBY.
// CH5  > 1990: trigger CALIBRATION (only from STANDBY, rising edge).
// CH3 1000–2000: maps to PARAM_RADIO_HIP_CMD as t ∈ [0,1].
//   Left stale when radio is dead.

static void radio_update() {
    bool alive = g_ibus.alive();
    uint16_t ch10 = g_ibus.channel(10);
    uint16_t ch5  = g_ibus.channel(5);
    uint16_t ch6  = g_ibus.channel(6);

    static bool s_was_alive = false;
    if (alive && !s_was_alive) {
        comm_log(LOG_LEVEL_INFO, "Radio: signal OK");
        g_buzzer.play(RADIO_ACQ_MELODY, 1, 120);
    }
    if (!alive && s_was_alive) {
        comm_log(LOG_LEVEL_WARN, "Radio: signal lost");
        g_buzzer.play(RADIO_LOST_MELODY, 2, 120);
    }
    s_was_alive = alive;

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
    if (armed_raw) {
        if (s_armed_ticks < ARM_DEBOUNCE_TICKS) s_armed_ticks++;
        s_unarmed_ticks = 0;
    } else {
        if (s_unarmed_ticks < DISARM_DEBOUNCE_TICKS) s_unarmed_ticks++;
        s_armed_ticks = 0;
    }
    bool armed = s_armed_ticks >= ARM_DEBOUNCE_TICKS;

    static bool s_was_armed = false;
    if (armed && !s_was_armed) {
        if (g_state.state == STATE_ESTOP &&
            fault_severity(g_state.fault_code) == FAULT_SEVERITY_SOFT) {
            comm_log(LOG_LEVEL_INFO, "Radio: soft-clear ESTOP [0x%02X]", g_state.fault_code);
            stateMachine_request_soft_clear();
        } else if (g_state.state == STATE_STANDBY) {
            comm_log(LOG_LEVEL_INFO, "Radio: armed -> RUNNING");
            stateMachine_request_running();
        } else {
            // stateMachine_request_running() only latches from STANDBY, so a flip
            // that lands mid-transition (e.g. tail end of CALIBRATION, or still in
            // STARTUP) is otherwise silently dropped with no operator feedback.
            comm_log(LOG_LEVEL_WARN, "Radio: arm ignored, not in STANDBY (state=%d)", (int)g_state.state);
            g_buzzer.play(ARM_IGNORED_MELODY, sizeof(ARM_IGNORED_MELODY) / sizeof(ARM_IGNORED_MELODY[0]), 120);
            s_profile_flash_rgb[0] = 255; s_profile_flash_rgb[1] = 0; s_profile_flash_rgb[2] = 255;
            s_profile_flash_until_ms = millis() + 200;
        }
    }
    s_was_armed = armed;

    // Disarm is level-based, not edge-based, and applies to every energetic
    // state. CH10 loss aborts JUMPING/STANDING_UP immediately through the same
    // DISARMING path as an ordinary RUNNING exit.
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
    static bool s_disarm_req_sent = false;
    bool energetic = g_state.state == STATE_RUNNING || g_state.state == STATE_JUMPING ||
                     g_state.state == STATE_STANDING_UP;
    if (!energetic) s_disarm_req_sent = false;
    if (s_unarmed_ticks >= DISARM_DEBOUNCE_TICKS && energetic && !s_disarm_req_sent) {
        comm_log(LOG_LEVEL_INFO, "Radio: disarmed -> DISARMING");
        stateMachine_disarm_running();
        s_disarm_req_sent = true;
    }

    static bool s_was_calib = false;
    bool calib = alive && (ch5 > 1990);
    if (calib && !s_was_calib && g_state.state == STATE_STANDBY) {
        comm_log(LOG_LEVEL_INFO, "Radio: calib trigger");
        stateMachine_request_calibration();
    }
    s_was_calib = calib;

    static bool s_was_jump = false;
    bool jump_sw = alive && (ch6 > 1990);
    if (jump_sw && !s_was_jump && g_state.state == STATE_RUNNING) {
        comm_log(LOG_LEVEL_INFO, "Radio: CH6 -> JUMPING");
        stateMachine_request_jump();
    }
    s_was_jump = jump_sw;

    // RC log start/stop stub — CH5/6/7/9/10 already assigned (calib/jump/trim/profile/arm).
    // TODO(user): assign a spare iBUS channel for log start/stop.
    // Set LOG_SWITCH_CH to a real channel index (1-based) to enable; 0 keeps this disabled.
    static constexpr uint8_t LOG_SWITCH_CH = 0;   // 0 = unassigned (stub)
    if (LOG_SWITCH_CH != 0) {
        bool on = g_ibus.channel(LOG_SWITCH_CH) > 1500;
        static bool prev = false;
        if (on && !prev) { if (!sd_logger_is_active()) sd_logger_start(0); }
        if (!on && prev) { sd_logger_stop(); }
        prev = on;
    }

    // §1d (tuning.md): while PARAM_GUI_MOTION_CTRL_EN is set, the two radio-driven
    // writes to v_cmd_ms/omega_cmd_rds below are skipped so a GUI/CLI param_set()
    // on those two params stands instead. Never gates arming (CH10 stays
    // unconditional below) and never suppresses the !alive branch's zeroing —
    // a radio dropout always wins. Auto-reverts to radio control if no GUI
    // command packet arrives for GUI_MOTION_CTRL_TIMEOUT_MS, mirroring
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
        float t = constrain((g_ibus.channel(3) - 1000.0f) / 1000.0f, 0.0f, 1.0f);  // CH3 (1-indexed)
        param_force_set(PARAM_RADIO_HIP_CMD, t);

        if (!gui_motion_ctrl) {
            float vel_norm = constrain((g_ibus.channel(2) - 1500.0f) / 500.0f, -1.0f, 1.0f);
            param_force_set(PARAM_V_CMD_MS, vel_norm * param_get(PARAM_RADIO_VEL_MAX));

            float yaw_norm = -constrain((g_ibus.channel(4) - 1500.0f) / 500.0f, -1.0f, 1.0f);  // inverted: stick left -> robot yaws left
            param_force_set(PARAM_OMEGA_CMD_RDS, yaw_norm * param_get(PARAM_RADIO_YAW_MAX));
        }

        // CH9: speed profile selector (3-position switch → profile 0/1/2)
        static const uint16_t PROFILE_VEL[]    = {PARAM_PROFILE_1_VEL_MAX,    PARAM_PROFILE_2_VEL_MAX,    PARAM_PROFILE_3_VEL_MAX};
        static const uint16_t PROFILE_YAW[]    = {PARAM_PROFILE_1_YAW_MAX,    PARAM_PROFILE_2_YAW_MAX,    PARAM_PROFILE_3_YAW_MAX};
        static const uint16_t PROFILE_TORQUE[] = {PARAM_PROFILE_1_TORQUE_LIM, PARAM_PROFILE_2_TORQUE_LIM, PARAM_PROFILE_3_TORQUE_LIM};
        static uint8_t s_last_profile = 255;   // force apply on first packet
        static float   s_trq_target   = -1.0f; // <0 = no pending slew
        uint16_t ch9 = g_ibus.channel(9);
        uint8_t profile = (ch9 < 1333) ? 0 : (ch9 < 1667) ? 1 : 2;
        if (profile != s_last_profile) {
            s_last_profile = profile;
            param_force_set(PARAM_ACTIVE_PROFILE, (float)profile);
            param_force_set(PARAM_RADIO_VEL_MAX, param_get(PROFILE_VEL[profile]));
            param_force_set(PARAM_RADIO_YAW_MAX, param_get(PROFILE_YAW[profile]));
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
            g_buzzer.play(PROFILE_MELODIES[profile], profile + 1, 150);
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

        // CH7: pitch trim hook — reads knob, writes param; LQR wiring deferred (see control_loop.cpp TODO)
        float pitch_trim = constrain((g_ibus.channel(7) - 1500.0f) / 500.0f * 0.08727f, -0.08727f, 0.08727f);
        param_force_set(PARAM_RADIO_PITCH_TRIM, pitch_trim);
    } else {
        param_force_set(PARAM_V_CMD_MS, 0.0f);
        param_force_set(PARAM_OMEGA_CMD_RDS, 0.0f);
    }

    // Mirrors v_cmd_ms live regardless of FSM state (like w_cmd/omega_cmd_rds
    // above) so it's visible for troubleshooting before arming, not just
    // while controlLoop_run() is active in RUNNING/JUMPING.
    g_state.v_ref = (param_get(PARAM_VEL_PI_EN) >= 0.5f) ? param_get(PARAM_V_CMD_MS) : 0.0f;
}

static void run_control_loop() {
    stateMachine_update();
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
            comm_log(LOG_LEVEL_WARN,  "IMU: degraded — %.0f%% loss",
                     imu_packet_loss() * 100.0f);                        break;
        case ImuState::ERROR:
            comm_log(LOG_LEVEL_ERROR, "IMU: failed to connect — giving up until reboot");  break;
        default: break;
    }
}

// ── Main loop ─────────────────────────────────────────────────────────────────

void loop() {
    uint32_t t_start = micros();
    watchdog_pet();

    bool prof = param_get(PARAM_LOOP_PROFILE_ENABLE) >= 0.5f;
    uint32_t t0;

    t0 = micros(); receive_commands();     if (prof) prof_mark(s_prof_max.recv,       t0);
    service_param_dump();
    t0 = micros(); read_sensors(prof);     if (prof) prof_mark(s_prof_max.sens_total, t0);
    t0 = micros(); check_imu_state();      if (prof) prof_mark(s_prof_max.imu_chk,    t0);
    t0 = micros(); radio_update();         if (prof) prof_mark(s_prof_max.radio,      t0);
    t0 = micros(); run_control_loop();     if (prof) prof_mark(s_prof_max.ctrl,       t0);
    t0 = micros(); update_led();           if (prof) prof_mark(s_prof_max.led,        t0);
    t0 = micros(); update_buzzer();        if (prof) prof_mark(s_prof_max.buz,        t0);

    t0 = micros();
    if (sd_logger_is_active()) {
        static LogRecord rec;
        fill_telemetry(rec.telem);
        rec.t_micros = micros();
        sd_logger_write(&rec);
    }
    sd_logger_service();            // 1 sector/tick + auto-stop
    sd_logger_service_transfer();   // paced chunk streaming during a GET
    // Download-complete chirp: sd_logger_service_transfer() emits XFER_END
    // and clears its own active flag internally (sd_logger.cpp) with no
    // callback out to here, so detect the same thing via the falling edge
    // of sd_logger_transfer_active() instead of touching that file.
    {
        static bool s_xfer_was_active = false;
        bool xfer_active = sd_logger_transfer_active();
        if (s_xfer_was_active && !xfer_active) g_buzzer.play(LOG_DONE_CHIRP, 2, 150);
        s_xfer_was_active = xfer_active;
    }
    if (prof) prof_mark(s_prof_max.sd, t0);

    // Deferred param flash flush — a LittleFS rewrite stalls the loop for
    // several ms, so never while balancing (RUNNING/JUMPING).
    t0 = micros();
    param_flush_service(g_state.state != STATE_RUNNING && g_state.state != STATE_JUMPING);
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
