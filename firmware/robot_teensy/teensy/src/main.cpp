#include <Arduino.h>
#include <stdarg.h>
#include "config.h"
#include "robot_state.h"
#include "state_machine.h"
#include "control_loop.h"
#include "CommLink.h"
#include "comm_protocol.h"
#include "IMU.h"

CommLink g_comm(Serial, COMM_SRC_TEENSY);

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
}

// ── Command handler ───────────────────────────────────────────────────────────

static void on_command(uint8_t type, uint8_t version, uint8_t source,
                       const uint8_t* payload, uint16_t len) {
    // TODO: dispatch on type / cmd_id
    (void)type; (void)version; (void)source; (void)payload; (void)len;
}

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    g_comm.onPacket(on_command);

    comm_log(LOG_LEVEL_INFO, "Firmware starting");
    imu_init();
    comm_log(LOG_LEVEL_INFO, "IMU initializing...");
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
    g_comm.send(COMM_TYPE_TELEMETRY, TELEM_PAYLOAD_V1, &telem, sizeof(telem));
}

// ── Per-loop tasks ────────────────────────────────────────────────────────────

static void receive_commands() {
    g_comm.update();
}

static void read_sensors() {
    imu_update();
    g_state.pitch_rad       = imu_pitch();
    g_state.pitch_rate_rads = imu_pitch_rate();
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
    send_telemetry();

    while (micros() - t_start < 2000) {}
}
