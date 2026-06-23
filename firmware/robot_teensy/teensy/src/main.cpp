#include <Arduino.h>
#include <stdarg.h>
#include "config.h"
#include "robot_state.h"
#include "state_machine.h"
#include "control_loop.h"
#include "CommLink.h"
#include "comm_protocol.h"
#include "IMU.h"
#include "hip_motors.h"
#include "wheel_motors.h"
#include "RgbLed.h"
#include "Buzzer.h"
#include "param_registry.h"
#include "IBus.h"

CommLink g_comm(Serial5, COMM_SRC_TEENSY);     // ESP32 UART bridge
CommLink g_comm_usb(Serial, COMM_SRC_TEENSY);  // direct PC USB
IBus     g_ibus(Serial4);                       // FlySky RC receiver, pin 16
RgbLed   g_led(PIN_LED_R, PIN_LED_G, PIN_LED_B);
Buzzer   g_buzzer(PIN_BUZZER);

HipCmd   g_hip_cmd = {};

// Latest ToF packet received from ESP32 (updated in on_command, read in state machine and telemetry)
TofPayload g_tof         = {{0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF}, 0xFFFF, 0xFFFF};
uint32_t   g_tof_last_ms = 0;

// ── Logging ───────────────────────────────────────────────────────────────────

void comm_log(uint8_t level, const char* fmt, ...) {
    char msg[62];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if (n > (int)sizeof(msg)) n = sizeof(msg);
    uint8_t buf[63];
    buf[0] = level;
    memcpy(buf + 1, msg, n);
    g_comm.send(COMM_TYPE_LOG, LOG_PAYLOAD_V1, buf, 1 + n);
    if (Serial) g_comm_usb.send(COMM_TYPE_LOG, LOG_PAYLOAD_V1, buf, 1 + n);
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

// ── Command handler ───────────────────────────────────────────────────────────

static void on_command(uint8_t type, uint8_t version, uint8_t source,
                       const uint8_t* payload, uint16_t len) {
    (void)version; (void)source;

    // ToF packet from ESP32: store latest distances for telemetry and obstacle avoidance
    if (type == COMM_TYPE_TOF && len >= (uint16_t)sizeof(TofPayload)) {
        memcpy(&g_tof, payload, sizeof(TofPayload));
        g_tof_last_ms = millis();
        return;
    }

    if (type != COMM_TYPE_COMMAND || len < 1) return;

    stateMachine_ping_gui_watchdog();  // feed MANUAL-mode GUI watchdog
    uint8_t cmd_id = payload[0];

    // ── Mode change: signal the state machine ─────────────────────────────────
    if (cmd_id == CMD_ID_SET_MODE && len >= 2) {
        uint8_t target = payload[1];
        comm_log(LOG_LEVEL_INFO, "CMD set_mode -> %d", target);
        if (target == STATE_MANUAL)      stateMachine_request_manual();
        if (target == STATE_STANDBY) {
            // From ESTOP: attempt soft-clear (ESTOP→STANDBY directly for SOFT faults).
            // From any other state: exit MANUAL back to STANDBY.
            if (g_state.state == STATE_ESTOP) stateMachine_request_soft_clear();
            else                              stateMachine_exit_manual();
        }
        if (target == STATE_STARTUP)     stateMachine_request_reset();
        if (target == STATE_CALIBRATION) stateMachine_request_calibration();
        if (target == STATE_ESTOP)       stateMachine_request_estop();
        return;
    }

    // ── Reboot: full MCU reset — re-runs setup() from scratch ──────────────────
    if (cmd_id == CMD_ID_REBOOT) {
        comm_log(LOG_LEVEL_WARN, "Reboot requested");
        Serial.flush();
        Serial5.flush();
        delay(50);
        SCB_AIRCR = 0x05FA0004;  // Cortex-M7 system reset request
        return;
    }

    // ── Hip command: queue for execution by the MANUAL state action ───────────
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
        return;
    }

    // ── Wheel command: set mode / send setpoint / clear errors ───────────────
    // Only accepted in MANUAL mode; LQR (STATE_RUNNING) commands wheels directly.
    if (cmd_id == CMD_ID_WHEEL && len >= 2) {
        if (g_state.state != STATE_MANUAL) return;
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
        return;
    }

    // ── Param write: set one parameter value ──────────────────────────────────
    if (cmd_id == CMD_ID_PARAM_SET && len >= 7) {
        uint16_t id;  float val;
        memcpy(&id,  payload + 1, 2);
        memcpy(&val, payload + 3, 4);
        comm_log(LOG_LEVEL_INFO, "CMD param_set 0x%04X = %.4f", id, val);
        ParamSetResult res = param_set(id, val);
        if (res == ParamSetResult::FAULT) {
            comm_log(LOG_LEVEL_ERROR, "Param 0x%04X out of bounds — ESTOP", id);
            g_state.fault_code = FAULT_PARAM_OUT_OF_BOUNDS;
            stateMachine_request_estop();
        }
        // Echo back actual (possibly clamped) value
        Param p; uint16_t idx = 0;
        while (param_by_index(idx, &p)) { if (p.id == id) { send_param_report(idx); break; } idx++; }
        return;
    }

    // ── Param read: report one param, or dump all ─────────────────────────────
    if (cmd_id == CMD_ID_PARAM_GET && len >= 3) {
        uint16_t id;
        memcpy(&id, payload + 1, 2);
        if (id == 0xFFFF) {
            for (uint16_t i = 0; i < param_count(); i++) send_param_report(i);
        } else {
            Param p; uint16_t idx = 0;
            while (param_by_index(idx, &p)) { if (p.id == id) { send_param_report(idx); break; } idx++; }
        }
        return;
    }
}

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    // 512-byte TX buffer: write(253-byte frame) returns immediately at 1.2 Mbaud
    // instead of blocking ~1.6 ms and adding jitter to the 2 ms control loop.
    static uint8_t s_esp32_tx_buf[512];
    Serial5.addMemoryForWrite(s_esp32_tx_buf, sizeof(s_esp32_tx_buf));
    Serial5.begin(ESP32_BAUD);
    // Fix 5: flush any boot-noise that arrived before the parser was ready
    delay(10);
    while (Serial5.available()) Serial5.read();
    g_comm.onPacket(on_command);
    g_comm_usb.onPacket(on_command);

    g_led.begin();
    g_buzzer.begin();

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
            g_buzzer.update();
            delay(2);
        }
    }
    g_buzzer.off();
    g_led.off();

    g_led.pulse(255, 255, 255, 2000);  // STARTUP: white breathe

    comm_log(LOG_LEVEL_INFO, "Firmware starting");
    param_init();
    g_ibus.begin();
    comm_log(LOG_LEVEL_INFO, "IBus RX ready (Serial4)");
    imu_init();
    comm_log(LOG_LEVEL_INFO, "IMU initializing...");
    hip_motors_init();
    comm_log(LOG_LEVEL_INFO, "Hip CAN init OK");
    hip_motors_enter_mit();
    wheel_motors_init();
    wheel_motors_set_mode(WheelMode::IDLE);
    wheel_motors_clear_errors();
    delay(300);  // wait for ODrive to send a fresh heartbeat with cleared error state
    comm_log(LOG_LEVEL_INFO, "Wheel CAN init OK");
    comm_log(LOG_LEVEL_INFO, "Hip MIT mode enabled");
    controlLoop_init();
    stateMachine_init();
    comm_log(LOG_LEVEL_INFO, "Setup complete");
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
    // HEALTH_VEL_PI_SAT and HEALTH_YAW_PI_SAT set by their controllers in Phase 3/4
    return f;
}

static void send_telemetry() {
    TelemetryPayload telem;
    telem.timestamp_ms     = millis();
    telem.pitch_rad        = g_state.pitch_rad;
    telem.pitch_rate_rads  = g_state.pitch_rate_rads;
    telem.wheel_vel_avg_ms = g_state.wheel_vel_avg_ms;
    telem.hip_l_pos_rad    = g_state.hip_l_pos_rad;
    telem.hip_r_pos_rad    = g_state.hip_r_pos_rad;
    telem.whl_tau_l        = g_state.whl_tau_l;
    telem.whl_tau_r        = g_state.whl_tau_r;
    telem.roll_rad         = imu_roll();
    telem.yaw_rad          = imu_yaw();
    telem.robot_state      = (uint8_t)g_state.state;
    telem.fault_code       = g_state.fault_code;
    telem.test_val         = sinf(2.0f * (float)M_PI * 2.0f * millis() / 1000.0f);
    telem.hip_l_current_a  = g_state.hip_l_current_a;
    telem.hip_r_current_a  = g_state.hip_r_current_a;
    for (uint8_t i = 1; i <= IBUS_NUM_CH; i++) telem.ibus_ch[i - 1] = g_ibus.channel(i);
    telem.ibus_alive       = g_ibus.alive() ? 1 : 0;
    telem.wm_l_vel_turns_s = wm_L.vel_turns_s;
    telem.wm_r_vel_turns_s = wm_R.vel_turns_s;
    telem.wm_l_pos_turns   = wm_L.pos_turns;
    telem.wm_r_pos_turns   = wm_R.pos_turns;
    telem.wm_l_vbus        = wm_L.vbus;
    telem.wm_r_vbus        = wm_R.vbus;
    telem.wm_l_error       = wm_L.error;
    telem.wm_r_error       = wm_R.error;
    telem.wm_l_state       = wm_L.axis_state;
    telem.wm_r_state       = wm_R.axis_state;
    telem.wm_mode          = (uint8_t)wm_mode;
    for (int i = 0; i < 4; i++) telem.tof_dist_mm[i] = g_tof.dist_mm[i];
    // V5 — IMU rates + linear acceleration
    telem.roll_rate_rads        = imu_roll_rate();
    telem.yaw_rate_rads         = imu_yaw_rate();
    telem.accel_x_ms2           = imu_accel_x();
    telem.accel_y_ms2           = imu_accel_y();
    telem.accel_z_ms2           = imu_accel_z();
    // V5 — hip velocity feedback
    telem.hip_l_vel_rads        = hm_L.vel_rad_s;
    telem.hip_r_vel_rads        = hm_R.vel_rad_s;
    // V5 — hip motor MIT setpoints (cmd vs feedback enables tracking quality plots)
    telem.hip_l_cmd_pos_rad     = hm_sp_L.p;
    telem.hip_r_cmd_pos_rad     = hm_sp_R.p;
    telem.hip_l_cmd_vel_rads    = hm_sp_L.v;
    telem.hip_r_cmd_vel_rads    = hm_sp_R.v;
    telem.hip_l_cmd_kp          = hm_sp_L.kp;
    telem.hip_r_cmd_kp          = hm_sp_R.kp;
    telem.hip_l_cmd_kd          = hm_sp_L.kd;
    telem.hip_r_cmd_kd          = hm_sp_R.kd;
    telem.hip_l_cmd_tff         = hm_sp_L.tff;
    telem.hip_r_cmd_tff         = hm_sp_R.tff;
    // V5 — balance controller internals (0 until respective phase is implemented)
    telem.theta_ref             = g_state.theta_ref;
    telem.v_ref                 = g_state.v_ref;
    telem.tau_sym               = g_state.tau_sym;
    telem.tau_yaw               = g_state.tau_yaw;
    telem.vel_err_integral      = g_state.vel_err_integral;
    telem.yaw_err_integral      = g_state.yaw_err_integral;
    telem.ff1_out               = g_state.ff1_out;
    telem.ff2_out               = g_state.ff2_out;
    telem.ff4_out               = g_state.ff4_out;
    // V5 — diagnostics
    telem.health_flags          = build_health_flags();
    telem.imu_packet_loss_pct   = (uint8_t)(imu_packet_loss() * 100.0f + 0.5f);
    telem.jump_state            = g_state.jump_state;
    telem.loop_count            = g_state.loop_count;
    const uint8_t* tp = (const uint8_t*)&telem;
    g_comm.send(COMM_TYPE_TELEM_A, TELEM_VERSION, tp,               TELEM_A_LEN);
    g_comm.send(COMM_TYPE_TELEM_B, TELEM_VERSION, tp + TELEM_A_LEN, TELEM_B_LEN);
    if (Serial) {
        g_comm_usb.send(COMM_TYPE_TELEM_A, TELEM_VERSION, tp,               TELEM_A_LEN);
        g_comm_usb.send(COMM_TYPE_TELEM_B, TELEM_VERSION, tp + TELEM_A_LEN, TELEM_B_LEN);
    }
}

// ── LED ───────────────────────────────────────────────────────────────────────

static void update_buzzer() {
    g_buzzer.update();
}

static void update_led() {
    static RobotStateEnum prev = (RobotStateEnum)0xFF;

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

static void read_sensors() {
    imu_update();
    g_state.pitch_rad       = -imu_pitch();       // IMU mounted inverted: negate to match robot +X=forward convention
    g_state.pitch_rate_rads = -imu_pitch_rate();

    hip_motors_poll();
    g_state.hip_l_pos_rad   = hm_L.pos_rad;
    g_state.hip_r_pos_rad   = hm_R.pos_rad;
    g_state.hip_l_current_a = hm_L.current_A;
    g_state.hip_r_current_a = hm_R.current_A;

    wheel_motors_poll();
    wheel_motors_pet_watchdog();

    g_ibus.update();
    for (uint8_t i = 1; i <= IBUS_NUM_CH; i++)
        param_force_set(PARAM_IBUS_CH0 + (i - 1), (float)g_ibus.channel(i));
    param_force_set(PARAM_IBUS_ALIVE, g_ibus.alive() ? 1.0f : 0.0f);
}

// ── Radio melodies ────────────────────────────────────────────────────────────
static const BuzzerNote RADIO_ACQ_MELODY[]  = {{76, 80, 0}};             // E5 — single: "link up"
static const BuzzerNote RADIO_LOST_MELODY[] = {{64, 100, 30}, {60, 150, 0}}; // E4→C4 desc: "link lost"

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

    static bool s_was_armed = false;
    bool armed = alive && (ch10 > 1990);
    if (armed && !s_was_armed) {
        if (g_state.state == STATE_ESTOP &&
            fault_severity(g_state.fault_code) == FAULT_SEVERITY_SOFT) {
            comm_log(LOG_LEVEL_INFO, "Radio: soft-clear ESTOP [0x%02X]", g_state.fault_code);
            stateMachine_request_soft_clear();
        } else {
            comm_log(LOG_LEVEL_INFO, "Radio: armed -> RUNNING");
            stateMachine_request_running();
        }
    } else if (!armed && s_was_armed && g_state.state == STATE_RUNNING) {
        comm_log(LOG_LEVEL_INFO, "Radio: disarmed -> STANDBY");
        stateMachine_disarm_running();
    }
    s_was_armed = armed;

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

    if (alive) {
        float t = constrain((g_ibus.channel(3) - 1000.0f) / 1000.0f, 0.0f, 1.0f);  // CH3 (1-indexed)
        param_force_set(PARAM_RADIO_HIP_CMD, t);
    }
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
            comm_log(LOG_LEVEL_ERROR, "IMU: error — retrying in 1 s");  break;
        default: break;
    }
}

// ── Main loop ─────────────────────────────────────────────────────────────────

void loop() {
    uint32_t t_start = micros();

    receive_commands();
    read_sensors();
    check_imu_state();
    radio_update();
    run_control_loop();
    update_led();
    update_buzzer();

    // Send telemetry at 50 Hz (every 10th tick of the 500 Hz loop)
    static uint8_t telem_div = 0;
    if (++telem_div >= 10) {
        telem_div = 0;
        send_telemetry();
    }

    while (micros() - t_start < 2000) {}
}
