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
#include "RgbLed.h"
#include "Buzzer.h"

CommLink g_comm(Serial5, COMM_SRC_TEENSY);     // ESP32 UART bridge
CommLink g_comm_usb(Serial, COMM_SRC_TEENSY);  // direct PC USB
RgbLed   g_led(PIN_LED_R, PIN_LED_G, PIN_LED_B);
Buzzer   g_buzzer(PIN_BUZZER);

HipCmd g_hip_cmd = {};

// ── Logging ───────────────────────────────────────────────────────────────────

static void comm_log(uint8_t level, const char* fmt, ...) {
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

// ── Command handler ───────────────────────────────────────────────────────────

static void on_command(uint8_t type, uint8_t version, uint8_t source,
                       const uint8_t* payload, uint16_t len) {
    (void)version; (void)source;
    if (type != COMM_TYPE_COMMAND || len < 1) return;

    uint8_t cmd_id = payload[0];

    // ── Mode change: signal the state machine ─────────────────────────────────
    if (cmd_id == CMD_ID_SET_MODE && len >= 2) {
        uint8_t target = payload[1];
        if (target == STATE_MANUAL)  stateMachine_request_manual();
        if (target == STATE_STANDBY) stateMachine_exit_manual();
        if (target == STATE_STARTUP) stateMachine_request_reset();
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
    }
}

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    Serial5.begin(ESP32_BAUD);
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
    imu_init();
    comm_log(LOG_LEVEL_INFO, "IMU initializing...");
    hip_motors_init();
    hip_motors_enter_mit();
    controlLoop_init();
    stateMachine_init();
    comm_log(LOG_LEVEL_INFO, "Setup complete");
}

// ── Telemetry ─────────────────────────────────────────────────────────────────

static void send_telemetry() {
    TelemetryPayload telem;
    telem.timestamp_ms     = millis();
    telem.pitch_rad        = g_state.pitch_rad;
    telem.pitch_rate_rads  = g_state.pitch_rate_rads;
    telem.wheel_vel_avg_ms = g_state.wheel_vel_avg_ms;
    telem.hip_l_pos_rad    = g_state.hip_l_pos_rad;
    telem.hip_r_pos_rad    = g_state.hip_r_pos_rad;
    telem.cmd_l            = g_state.cmd_l;
    telem.cmd_r            = g_state.cmd_r;
    telem.roll_rad         = imu_roll();
    telem.yaw_rad          = imu_yaw();
    telem.robot_state      = (uint8_t)g_state.state;
    telem.fault_code       = g_state.fault_code;
    telem.test_val         = sinf(2.0f * (float)M_PI * 2.0f * millis() / 1000.0f);
    g_comm.send(COMM_TYPE_TELEMETRY, TELEM_PAYLOAD_V1, &telem, sizeof(telem));
    if (Serial) g_comm_usb.send(COMM_TYPE_TELEMETRY, TELEM_PAYLOAD_V1, &telem, sizeof(telem));
}

// ── LED ───────────────────────────────────────────────────────────────────────

static void update_led() {
    static RobotStateEnum prev = (RobotStateEnum)0xFF;
    RobotStateEnum cur = g_state.state;
    if (cur != prev) {
        bool was_estop = (prev == STATE_ESTOP);
        prev = cur;
        switch (cur) {
            case STATE_STARTUP:
                // Reset accepted from ESTOP: flash white for 1 s, then breathe white.
                if (was_estop) g_led.flash_then_pulse(255, 255, 255, 1000, 2000);
                else           g_led.pulse(255, 255, 255, 2000);
                break;
            case STATE_CALIBRATION: g_led.pulse(0,   0,   255, 2000); break;
            case STATE_STANDBY:     g_led.pulse(255, 200,   0, 2000); break;
            case STATE_RUNNING:     g_led.pulse(0,   255,   0, 2000); break;
            case STATE_MANUAL:      g_led.pulse(0,   200, 255, 2000); break;
            case STATE_ESTOP:       g_led.blink(255,   0,   0,  100, 100); break;
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
    g_state.pitch_rad       = imu_pitch();
    g_state.pitch_rate_rads = imu_pitch_rate();

    hip_motors_poll();
    g_state.hip_l_pos_rad = hm_L.pos_rad;
    g_state.hip_r_pos_rad = hm_R.pos_rad;
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
    run_control_loop();
    update_led();

    // Send telemetry at 50 Hz (every 10th tick of the 500 Hz loop)
    static uint8_t telem_div = 0;
    if (++telem_div >= 10) {
        telem_div = 0;
        send_telemetry();
    }

    while (micros() - t_start < 2000) {}
}
