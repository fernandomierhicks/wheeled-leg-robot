#include <Arduino.h>
#include "state_machine.h"
#include "robot_state.h"
#include "IMU.h"
#include "hip_motors.h"
#include "comm_protocol.h"
#include <StateMachine.h>

// ── State machine ─────────────────────────────────────────────────────────────

static StateMachine sm;

static State* S_STARTUP;
static State* S_STANDBY;
static State* S_MANUAL;
static State* S_ESTOP;

// ── Pending mode-change requests (set by command handler) ─────────────────────
static volatile bool s_req_manual  = false;
static volatile bool s_req_standby = false;

// ── State actions ─────────────────────────────────────────────────────────────

static void on_startup()  { g_state.state = STATE_STARTUP; }
static void on_standby()  { g_state.state = STATE_STANDBY; }
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
                hip_motor_send_L(g_hip_cmd.p, g_hip_cmd.v, g_hip_cmd.kp, g_hip_cmd.kd, g_hip_cmd.tff);
            else if (g_hip_cmd.motor_id == HIP_MOTOR_R)
                hip_motor_send_R(g_hip_cmd.p, g_hip_cmd.v, g_hip_cmd.kp, g_hip_cmd.kd, g_hip_cmd.tff);
            else
                hip_motors_send(g_hip_cmd.p, g_hip_cmd.v, g_hip_cmd.kp, g_hip_cmd.kd, g_hip_cmd.tff,
                                g_hip_cmd.p, g_hip_cmd.v, g_hip_cmd.kp, g_hip_cmd.kd, g_hip_cmd.tff);
            break;
        default: break;
    }
}
static void on_estop()    { g_state.state = STATE_ESTOP;   }

// ── Transition conditions ─────────────────────────────────────────────────────

static bool startup_ok() {
    return imu_state() == ImuState::NOMINAL && hm_L.ever_heard && hm_R.ever_heard;
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

// ── Init / update ─────────────────────────────────────────────────────────────

void stateMachine_init() {
    S_STARTUP = sm.addState(on_startup);
    S_STANDBY = sm.addState(on_standby);
    S_MANUAL  = sm.addState(on_manual);
    S_ESTOP   = sm.addState(on_estop);

    S_STARTUP->addTransition(startup_ok,   S_STANDBY);
    S_STARTUP->addTransition(startup_fail, S_ESTOP);

    S_STANDBY->addTransition(standby_hip_fault, S_ESTOP);
    S_STANDBY->addTransition(req_manual,        S_MANUAL);
    S_MANUAL ->addTransition(standby_hip_fault, S_ESTOP);
    S_MANUAL ->addTransition(req_standby,       S_STANDBY);

    g_state.state = STATE_STARTUP;
}

void stateMachine_update() {
    sm.run();
}

// ── Public request API ────────────────────────────────────────────────────────

void stateMachine_request_manual() { s_req_manual  = true; }
void stateMachine_exit_manual()    { s_req_standby = true; }
