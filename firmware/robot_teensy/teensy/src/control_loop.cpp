#include "control_loop.h"
#include "config.h"
#include "robot_state.h"
#include "comm_protocol.h"
#include "state_machine.h"
#include "hip_motors.h"
#include "wheel_motors.h"
#include "param_registry.h"
#include <math.h>
#include <Arduino.h>

RobotState g_state = {};

// ── Hip gains for RUNNING mode (soft, initial testing) ───────────────────────
static constexpr float RUNNING_KP  = 5.0f;
static constexpr float RUNNING_KD  = 0.5f;
static constexpr float RUNNING_TFF = 0.0f;

// ── Pitch watchdog ────────────────────────────────────────────────────────────
static constexpr float    PITCH_WATCHDOG_RAD = 0.8727f;  // 50 degrees
static constexpr uint32_t PITCH_WATCHDOG_MS  = 200;
static uint32_t s_pitch_fault_start_ms = 0;

// ── LQR gains — nominal hip position (Q_NOM), from params.py baseline ────────
// K = [-9.77113533, -1.88054364, -7.13051190e-03]
// tau_sym = -(K_PITCH*x0 + K_PITCH_RATE*x1 + K_VEL*x2)
// Phase 5 adds gain scheduling across Q_RET/Q_NOM/Q_EXT.
static constexpr float K_PITCH      = -9.77113533f;
static constexpr float K_PITCH_RATE = -1.88054364f;
static constexpr float K_VEL        = -7.13051190e-03f;

static constexpr float WHEEL_R      = 0.075f;  // [m] wheel radius (150 mm OD)

// ── Public API ────────────────────────────────────────────────────────────────

void controlLoop_init() {}

void controlLoop_run() {
    // ── Hip setpoints from radio ──────────────────────────────────────────────
    float pos_L, pos_R;
    hip_cmd_to_setpoints(param_get(PARAM_RADIO_HIP_CMD), &pos_L, &pos_R);
    hip_motors_set_setpoint_L(pos_L, 0.0f, RUNNING_KP, RUNNING_KD, RUNNING_TFF);
    hip_motors_set_setpoint_R(pos_R, 0.0f, RUNNING_KP, RUNNING_KD, RUNNING_TFF);

    // ── Wheel velocity average [m/s] ──────────────────────────────────────────
    float vel_avg_ms = (wm_L.vel_turns_s + wm_R.vel_turns_s) * 0.5f
                       * (2.0f * (float)M_PI * WHEEL_R);
    g_state.wheel_vel_avg_ms = vel_avg_ms;

    // ── Effective pitch (real or injected) ────────────────────────────────────
    float sim = param_get(PARAM_SIM_PITCH_RAD);
    float pitch = (sim != 0.0f) ? sim : g_state.pitch_rad;

    // ── Pitch watchdog ────────────────────────────────────────────────────────
    if (fabsf(pitch) > PITCH_WATCHDOG_RAD) {
        if (s_pitch_fault_start_ms == 0) s_pitch_fault_start_ms = millis();
        if (millis() - s_pitch_fault_start_ms > PITCH_WATCHDOG_MS) {
            g_state.fault_code = FAULT_PITCH_WATCHDOG;
            stateMachine_request_estop();
            return;
        }
    } else {
        s_pitch_fault_start_ms = 0;
    }

    // ── Wheel runaway watchdog (hard backup) ──────────────────────────────────
    float hard_limit = param_get(PARAM_WHEEL_VEL_LIMIT_TURNS_S) * 2.0f;
    if (fabsf(wm_L.vel_turns_s) > hard_limit || fabsf(wm_R.vel_turns_s) > hard_limit) {
        g_state.fault_code = FAULT_WHEEL_RUNAWAY;
        stateMachine_request_estop();
        return;
    }

    // ── Balance LQR ──────────────────────────────────────────────────────────
    // theta_ref = 0, v_ref = 0 (Phase 3 adds velocity PI)
    float x0 = pitch;                    // pitch - theta_ref
    float x1 = g_state.pitch_rate_rads;
    float x2 = vel_avg_ms;              // wheel_vel_avg - v_ref

    float tau_sym = -(K_PITCH * x0 + K_PITCH_RATE * x1 + K_VEL * x2);

    // Clamp to adjustable test limit
    float torque_limit = param_get(PARAM_LQR_TORQUE_LIMIT);
    if (tau_sym >  torque_limit) tau_sym =  torque_limit;
    if (tau_sym < -torque_limit) tau_sym = -torque_limit;

    // Log computed torque regardless of enable flag
    g_state.cmd_l = tau_sym;
    g_state.cmd_r = tau_sym;

    // ── Wheel torque output ───────────────────────────────────────────────────
    if (param_get(PARAM_LQR_ENABLE) >= 0.5f) {
        float soft_limit = param_get(PARAM_WHEEL_VEL_LIMIT_TURNS_S);
        float tau_L = tau_sym;
        float tau_R = tau_sym;
        // Per-wheel soft governor: zero torque if spinning beyond limit in the commanded direction
        if ((wm_L.vel_turns_s >  soft_limit && tau_L > 0.0f) ||
            (wm_L.vel_turns_s < -soft_limit && tau_L < 0.0f)) tau_L = 0.0f;
        if ((wm_R.vel_turns_s >  soft_limit && tau_R > 0.0f) ||
            (wm_R.vel_turns_s < -soft_limit && tau_R < 0.0f)) tau_R = 0.0f;
        wheel_motors_send(tau_L, tau_R);
    }
}
