#include <Arduino.h>
#include "state_machine.h"
#include "robot_state.h"
#include "IMU.h"
#include "hip_motors.h"
#include "calibration.h"
#include "comm_protocol.h"
#include "param_registry.h"
#include "control_loop.h"
#include "Buzzer.h"
#include <StateMachine.h>

extern Buzzer g_buzzer;

static const BuzzerNote ARMED_MELODY[] = {
    {79, 80, 20},   // G5
    {86, 120,  0},  // D6 — rising fifth: "armed"
};
static const BuzzerNote DISARMED_MELODY[] = {
    {86, 80, 20},   // D6
    {79, 120,  0},  // G5 — falling fifth: "safe"
};
static const BuzzerNote REJECT_MELODY[] = {
    {60, 100, 40},  // C4
    {55, 180,  0},  // G3 — low descending: "denied"
};
static const BuzzerNote ESTOP_MELODY[] = {
    {72, 80,  20},  // C5
    {72, 80,  20},  // C5
    {72, 80,  20},  // C5
    {60, 400,  0},  // C4 — three short blasts then low long: "danger"
};

// ── State machine ─────────────────────────────────────────────────────────────

static StateMachine sm;

static State* S_STARTUP;
static State* S_STANDBY;
static State* S_MANUAL;
static State* S_CALIBRATION;
static State* S_RUNNING;
static State* S_ESTOP;
static State* S_CMD_REJECT;

// ── ESTOP hip-disable tracking ────────────────────────────────────────────────
static bool s_estop_hip_disabled = false;  // set when MIT was killed on ESTOP entry

// ── Pending mode-change requests (set by command handler) ─────────────────────
static volatile bool s_req_manual        = false;
static volatile bool s_req_standby       = false;
static volatile bool s_req_reset         = false;
static volatile bool s_req_calibration   = false;
static volatile bool s_req_running       = false;
static volatile bool s_req_disarm_running = false;
static volatile bool s_req_estop         = false;
static volatile bool s_req_cmd_reject    = false;

// ── CMD_REJECT auto-exit timer ────────────────────────────────────────────────
static uint32_t s_cmd_reject_deadline_ms = 0;

// ── MANUAL GUI watchdog ───────────────────────────────────────────────────────
static uint32_t s_last_gui_packet_ms  = 0;
static const uint32_t MANUAL_GUI_TIMEOUT_MS = 500;

// ── State actions ─────────────────────────────────────────────────────────────

static void on_startup()  {
    bool entering = (g_state.state != STATE_STARTUP);
    g_state.state = STATE_STARTUP;
    if (entering) {
        g_state.fault_code = FAULT_NONE;
        comm_log(LOG_LEVEL_INFO, "-> STARTUP");
    }
    if (s_estop_hip_disabled) {
        s_estop_hip_disabled = false;
        if (param_get(PARAM_ESTOP_HIP_DISABLE) >= 0.5f) hip_motors_enter_mit();
    }
}
static void on_standby()  {
    bool entering      = (g_state.state != STATE_STANDBY);
    bool from_running  = (g_state.state == STATE_RUNNING);
    bool from_calib    = (g_state.state == STATE_CALIBRATION);
    g_state.state = STATE_STANDBY;
    hip_motors_clear_setpoints();  // revert to zero-torque ping (e.g. after CALIBRATION)
    g_state.cmd_l = 0.0f;          // clear calibration ramp echo from cmd_l/cmd_r
    g_state.cmd_r = 0.0f;
    if (from_calib) calibration_abort();
    if (entering) comm_log(LOG_LEVEL_INFO, "-> STANDBY");
    if (from_running) g_buzzer.play(DISARMED_MELODY, sizeof(DISARMED_MELODY) / sizeof(DISARMED_MELODY[0]), 200);
}
static void on_manual() {
    bool entering = (g_state.state != STATE_MANUAL);
    g_state.state = STATE_MANUAL;
    if (entering) {
        s_last_gui_packet_ms = millis();  // start watchdog fresh; don't inherit stale timestamp
        comm_log(LOG_LEVEL_INFO, "-> MANUAL");
    }
    if (!g_hip_cmd.pending) return;
    g_hip_cmd.pending = false;
    switch (g_hip_cmd.sub_cmd) {
        case HIP_SUB_ENABLE:  hip_motors_enter_mit(); break;
        case HIP_SUB_DISABLE: hip_motors_exit_mit();  break;
        case HIP_SUB_ZERO:    hip_motors_zero();       break;
        case HIP_SUB_MIT:
            if      (g_hip_cmd.motor_id == HIP_MOTOR_L)
                hip_motors_set_setpoint_L(g_hip_cmd.p, g_hip_cmd.v, g_hip_cmd.kp, g_hip_cmd.kd, g_hip_cmd.tff);
            else if (g_hip_cmd.motor_id == HIP_MOTOR_R)
                hip_motors_set_setpoint_R(g_hip_cmd.p, g_hip_cmd.v, g_hip_cmd.kp, g_hip_cmd.kd, g_hip_cmd.tff);
            else {
                hip_motors_set_setpoint_L(g_hip_cmd.p, g_hip_cmd.v, g_hip_cmd.kp, g_hip_cmd.kd, g_hip_cmd.tff);
                hip_motors_set_setpoint_R(g_hip_cmd.p, g_hip_cmd.v, g_hip_cmd.kp, g_hip_cmd.kd, g_hip_cmd.tff);
            }
            break;
        default: break;
    }
}
static void on_calibration() {
    bool entering = (g_state.state != STATE_CALIBRATION);
    g_state.state = STATE_CALIBRATION;
    if (entering) {
        comm_log(LOG_LEVEL_INFO, "-> CALIBRATION");
        calibration_start();
    }
    calibration_update();
}
static void on_running() {
    bool entering = (g_state.state != STATE_RUNNING);
    g_state.state = STATE_RUNNING;
    if (entering) {
        comm_log(LOG_LEVEL_INFO, "-> RUNNING (armed)");
        g_buzzer.play(ARMED_MELODY, sizeof(ARMED_MELODY) / sizeof(ARMED_MELODY[0]), 200);
    }
    controlLoop_run();
}
static void on_cmd_reject() {
    bool entering = (g_state.state != STATE_CMD_REJECT);
    g_state.state = STATE_CMD_REJECT;
    if (entering) {
        comm_log(LOG_LEVEL_WARN, "-> CMD_REJECT");
        g_buzzer.play(REJECT_MELODY, sizeof(REJECT_MELODY) / sizeof(REJECT_MELODY[0]), 200);
        s_cmd_reject_deadline_ms = millis() + 1000;
    }
}
static void on_estop() {
    bool entering = (g_state.state != STATE_ESTOP);
    g_state.state = STATE_ESTOP;
    if (entering) {
        // Discard any queued mode requests so they don't fire unexpectedly
        // after a future ESTOP → STARTUP → STANDBY sequence.
        s_req_cmd_reject  = false;
        s_req_manual      = false;
        s_req_running     = false;
        s_req_calibration = false;
        comm_log(LOG_LEVEL_ERROR, "-> ESTOP [fault 0x%02X]", g_state.fault_code);
        g_buzzer.play(ESTOP_MELODY, sizeof(ESTOP_MELODY) / sizeof(ESTOP_MELODY[0]), 200);
        if (param_get(PARAM_ESTOP_HIP_DISABLE) >= 0.5f) {
            hip_motors_exit_mit();
            s_estop_hip_disabled = true;
        }
    }
}

// ── Transition conditions ─────────────────────────────────────────────────────

static bool startup_ok() {
    return imu_state() == ImuState::NOMINAL && hip_motors_ok();
}
static bool startup_fail() {
    if (imu_state() == ImuState::ERROR) {
        comm_log(LOG_LEVEL_ERROR, "FAULT: IMU init failed");
        g_state.fault_code = FAULT_IMU_ERROR;
        return true;
    }
    if (millis() > 2000 && (!hm_L.ever_heard || !hm_R.ever_heard)) {
        comm_log(LOG_LEVEL_ERROR, "FAULT: hip init timeout (L=%d R=%d)", (int)hm_L.ever_heard, (int)hm_R.ever_heard);
        g_state.fault_code = FAULT_HIP_INIT_TIMEOUT;
        return true;
    }
    return false;
}
static bool standby_hip_fault() {
    if (!hip_motors_ok()) {
        comm_log(LOG_LEVEL_ERROR, "FAULT: hip feedback lost");
        g_state.fault_code = FAULT_HIP_FEEDBACK_LOST;
        return true;
    }
    return false;
}
static bool req_manual()        { bool v = s_req_manual;  s_req_manual  = false; return v; }
static bool req_standby()       { bool v = s_req_standby; s_req_standby = false; return v; }
static bool req_reset()         { bool v = s_req_reset;   s_req_reset   = false; return v; }
static bool req_calibration()   { bool v = s_req_calibration; s_req_calibration = false; return v; }
static bool req_cmd_reject() { bool v = s_req_cmd_reject; s_req_cmd_reject = false; return v; }
static bool cmd_reject_done() { return (millis() >= s_cmd_reject_deadline_ms); }
static bool manual_gui_timeout() {
    if (millis() - s_last_gui_packet_ms < MANUAL_GUI_TIMEOUT_MS) return false;
    comm_log(LOG_LEVEL_WARN, "MANUAL: GUI timeout -> STANDBY");
    return true;
}
static bool req_running() {
    if (!s_req_running) return false;
    s_req_running = false;
    if (!hm_limits_L.valid || !hm_limits_R.valid) {
        comm_log(LOG_LEVEL_WARN, "Running mode denied: calibrate first (limits not valid).");
        stateMachine_request_cmd_reject();
        return false;
    }
    return true;
}
static bool req_disarm_running() { bool v = s_req_disarm_running; s_req_disarm_running = false; return v; }
static bool req_estop() {
    if (!s_req_estop) return false;
    s_req_estop = false;
    if (g_state.fault_code == FAULT_NONE) g_state.fault_code = FAULT_HUMAN_ESTOP;
    return true;
}

static bool calibration_done_fn() {
    return calibration_done();
}
static bool calibration_failed_fn() {
    if (!calibration_failed()) return false;
    g_state.fault_code = FAULT_CALIBRATION_TIMEOUT;
    return true;
}

// ── Init / update ─────────────────────────────────────────────────────────────

void stateMachine_init() {
    S_STARTUP     = sm.addState(on_startup);
    S_STANDBY     = sm.addState(on_standby);
    S_MANUAL      = sm.addState(on_manual);
    S_CALIBRATION = sm.addState(on_calibration);
    S_RUNNING     = sm.addState(on_running);
    S_ESTOP       = sm.addState(on_estop);
    S_CMD_REJECT  = sm.addState(on_cmd_reject);

    S_STARTUP->addTransition(req_estop,    S_ESTOP);
    S_STARTUP->addTransition(startup_ok,   S_STANDBY);
    S_STARTUP->addTransition(startup_fail, S_ESTOP);

    S_STANDBY->addTransition(req_estop,         S_ESTOP);
    S_STANDBY->addTransition(standby_hip_fault, S_ESTOP);
    S_STANDBY->addTransition(req_manual,        S_MANUAL);
    S_STANDBY->addTransition(req_calibration,   S_CALIBRATION);
    S_STANDBY->addTransition(req_running,       S_RUNNING);
    S_STANDBY->addTransition(req_cmd_reject,    S_CMD_REJECT);

    S_MANUAL ->addTransition(req_estop,         S_ESTOP);
    S_MANUAL ->addTransition(standby_hip_fault, S_ESTOP);
    S_MANUAL ->addTransition(req_standby,       S_STANDBY);
    S_MANUAL ->addTransition(manual_gui_timeout, S_STANDBY);

    S_CALIBRATION->addTransition(req_estop,             S_ESTOP);
    S_CALIBRATION->addTransition(standby_hip_fault,     S_ESTOP);
    S_CALIBRATION->addTransition(calibration_failed_fn, S_ESTOP);
    S_CALIBRATION->addTransition(calibration_done_fn,   S_STANDBY);
    S_CALIBRATION->addTransition(req_standby,           S_STANDBY);

    S_RUNNING->addTransition(req_estop,          S_ESTOP);
    S_RUNNING->addTransition(standby_hip_fault,  S_ESTOP);
    S_RUNNING->addTransition(req_disarm_running, S_STANDBY);

    S_ESTOP  ->addTransition(req_reset,         S_STARTUP);

    S_CMD_REJECT->addTransition(cmd_reject_done, S_STANDBY);

    g_state.state = STATE_STARTUP;
}

void stateMachine_update() {
    sm.run();
}

// ── Public request API ────────────────────────────────────────────────────────

void stateMachine_request_manual()      { s_req_manual         = true; }
void stateMachine_exit_manual()         { s_req_standby        = true; }
void stateMachine_request_reset()       { s_req_reset          = true; }
void stateMachine_request_calibration() { s_req_calibration    = true; }
void stateMachine_request_running()     { s_req_running        = true; }
void stateMachine_disarm_running()      { s_req_disarm_running = true; }
void stateMachine_request_estop()       { s_req_estop          = true; }
void stateMachine_request_cmd_reject()  { s_req_cmd_reject     = true; }
void stateMachine_ping_gui_watchdog()   { s_last_gui_packet_ms = millis(); }
