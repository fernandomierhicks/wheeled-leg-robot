#include <Arduino.h>
#include "state_machine.h"
#include "robot_state.h"
#include "IMU.h"
#include "hip_motors.h"
#include "calibration.h"
#include "comm_protocol.h"
#include "param_registry.h"
#include <StateMachine.h>

// ── State machine ─────────────────────────────────────────────────────────────

static StateMachine sm;

static State* S_STARTUP;
static State* S_STANDBY;
static State* S_MANUAL;
static State* S_CALIBRATION;
static State* S_RUNNING;
static State* S_ESTOP;

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

// ── State actions ─────────────────────────────────────────────────────────────

static void on_startup()  {
    g_state.state = STATE_STARTUP;
    g_state.fault_code = FAULT_NONE;
    if (s_estop_hip_disabled) {
        s_estop_hip_disabled = false;
        if (param_get(PARAM_ESTOP_HIP_DISABLE) >= 0.5f) hip_motors_enter_mit();
    }
}
static void on_standby()  {
    g_state.state = STATE_STANDBY;
    hip_motors_clear_setpoints();  // revert to zero-torque ping (e.g. after CALIBRATION)
    g_state.cmd_l = 0.0f;          // clear calibration ramp echo from cmd_l/cmd_r
    g_state.cmd_r = 0.0f;
}
static void on_manual() {
    g_state.state = STATE_MANUAL;
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
    if (entering) calibration_start();
    calibration_update();
}
static void on_running() {
    g_state.state = STATE_RUNNING;
}
static void on_estop() {
    bool entering = (g_state.state != STATE_ESTOP);
    g_state.state = STATE_ESTOP;
    if (entering && param_get(PARAM_ESTOP_HIP_DISABLE) >= 0.5f) {
        hip_motors_exit_mit();
        s_estop_hip_disabled = true;
    }
}

// ── Transition conditions ─────────────────────────────────────────────────────

static bool startup_ok() {
    return imu_state() == ImuState::NOMINAL && hip_motors_ok();
}
static bool startup_fail() {
    if (imu_state() == ImuState::ERROR) {
        g_state.fault_code = FAULT_IMU_ERROR;
        return true;
    }
    if (millis() > 2000 && (!hm_L.ever_heard || !hm_R.ever_heard)) {
        g_state.fault_code = FAULT_HIP_INIT_TIMEOUT;
        return true;
    }
    return false;
}
static bool standby_hip_fault() {
    if (!hip_motors_ok()) {
        g_state.fault_code = FAULT_HIP_FEEDBACK_LOST;
        return true;
    }
    return false;
}
static bool req_manual()        { bool v = s_req_manual;  s_req_manual  = false; return v; }
static bool req_standby()       { bool v = s_req_standby; s_req_standby = false; return v; }
static bool req_reset()         { bool v = s_req_reset;   s_req_reset   = false; return v; }
static bool req_calibration()   { bool v = s_req_calibration; s_req_calibration = false; return v; }
static bool req_running() {
    if (!s_req_running) return false;
    s_req_running = false;
    if (param_get(PARAM_CALIB_DONE) < 0.5f) {
        comm_log(LOG_LEVEL_WARN, "RUNNING rejected: calibrate first");
        return false;
    }
    return true;
}
static bool req_disarm_running() { bool v = s_req_disarm_running; s_req_disarm_running = false; return v; }
static bool req_estop() {
    if (!s_req_estop) return false;
    s_req_estop = false;
    g_state.fault_code = FAULT_HUMAN_ESTOP;
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

    S_STARTUP->addTransition(req_estop,    S_ESTOP);
    S_STARTUP->addTransition(startup_ok,   S_STANDBY);
    S_STARTUP->addTransition(startup_fail, S_ESTOP);

    S_STANDBY->addTransition(req_estop,         S_ESTOP);
    S_STANDBY->addTransition(standby_hip_fault, S_ESTOP);
    S_STANDBY->addTransition(req_manual,        S_MANUAL);
    S_STANDBY->addTransition(req_calibration,   S_CALIBRATION);
    S_STANDBY->addTransition(req_running,       S_RUNNING);

    S_MANUAL ->addTransition(req_estop,         S_ESTOP);
    S_MANUAL ->addTransition(standby_hip_fault, S_ESTOP);
    S_MANUAL ->addTransition(req_standby,       S_STANDBY);

    S_CALIBRATION->addTransition(req_estop,             S_ESTOP);
    S_CALIBRATION->addTransition(standby_hip_fault,     S_ESTOP);
    S_CALIBRATION->addTransition(calibration_failed_fn, S_ESTOP);
    S_CALIBRATION->addTransition(calibration_done_fn,   S_STANDBY);
    S_CALIBRATION->addTransition(req_standby,           S_STANDBY);

    S_RUNNING->addTransition(req_estop,          S_ESTOP);
    S_RUNNING->addTransition(standby_hip_fault,  S_ESTOP);
    S_RUNNING->addTransition(req_disarm_running, S_STANDBY);

    S_ESTOP  ->addTransition(req_reset,         S_STARTUP);

    g_state.state = STATE_STARTUP;
}

void stateMachine_update() {
    sm.run();
}

// ── Public request API ────────────────────────────────────────────────────────

void stateMachine_request_manual()      { s_req_manual        = true; }
void stateMachine_exit_manual()         { s_req_standby       = true; }
void stateMachine_request_reset()       { s_req_reset         = true; }
void stateMachine_request_calibration() { s_req_calibration   = true; }
void stateMachine_request_running()     { s_req_running       = true; }
void stateMachine_disarm_running()      { s_req_disarm_running = true; }
void stateMachine_request_estop()       { s_req_estop         = true; }
