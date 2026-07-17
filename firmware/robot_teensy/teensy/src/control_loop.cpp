#include "control_loop.h"
#include "config.h"
#include "robot_state.h"
#include "comm_protocol.h"
#include "state_machine.h"
#include "hip_motors.h"
#include "wheel_motors.h"
#include "param_registry.h"
#include "IMU.h"
#include <math.h>
#include <Arduino.h>

RobotState g_state = {};

// ── Hip gains for RUNNING mode (soft, initial testing) ───────────────────────
static constexpr float RUNNING_KP  = 5.0f;
static constexpr float RUNNING_KD  = 0.5f;
static constexpr float RUNNING_TFF = 0.0f;

// ── Timing ────────────────────────────────────────────────────────────────────
static constexpr float DT = 0.002f;  // 500 Hz control tick [s]

// ── Pitch watchdog ────────────────────────────────────────────────────────────
static constexpr float    PITCH_WATCHDOG_RAD = 0.8727f;  // 50°
static constexpr uint32_t PITCH_WATCHDOG_MS  = 200;
static uint32_t s_pitch_fault_start_ms = 0;

// ── Phase 5: LQR gain table (computed by lqr.py self-test, Q_pitch=0.01,
//    Q_pitch_rate=0.1884, Q_vel=0.00508442, R=100.0) ─────────────────────────
// alpha=0 → retracted (Q_RET=-0.733 rad), alpha=1 → extended (Q_EXT=-1.432 rad)
static constexpr float K_PITCH_RET  = -13.0495742f;
static constexpr float K_RATE_RET   = -2.18083692f;
static constexpr float K_PITCH_EXT  = -7.92908352f;
static constexpr float K_RATE_EXT   = -1.69084204f;
static constexpr float K_VEL        = -7.13051190e-03f;  // invariant across leg height

// ── Phase 5: Effective pendulum length (IK at Q_RET and Q_EXT) ───────────────
static constexpr float L_EFF_RET    = 0.183117f;  // [m] fully retracted
static constexpr float L_EFF_EXT    = 0.295390f;  // [m] fully extended

// ── Phase 6: Body mass and gravity ───────────────────────────────────────────
// m_b = m_box + 2*(m_femur+m_tibia+m_coupler+m_bearing) + 2*motor_mass
static constexpr float M_BODY       = 1.6380f;    // [kg] body (excl. wheels)
static constexpr float GRAVITY      = 9.81f;      // [m/s²]
static constexpr float WHEEL_R      = 0.075f;     // [m] wheel radius (150 mm OD)
static constexpr float MOTOR_TRQ_MAX = 7.0f;      // [N·m] hard per-wheel clamp

// ── Controller state ──────────────────────────────────────────────────────────
// Phase 3 — Velocity PI
static float s_vel_integral    = 0.0f;
static float s_prev_v_desired  = 0.0f;
static float s_theta_ref_rlt   = 0.0f;  // rate-limited theta_ref

// Phase 4 — Yaw PI
static float s_yaw_integral    = 0.0f;

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
    // Written back into g_state so telemetry/GUI see exactly what the LQR sees.
    float pitch = (param_get(PARAM_ENABLE_SIM_PITCH_RAD) >= 0.5f)
                  ? param_get(PARAM_SIM_PITCH_RAD)
                  : g_state.pitch_rad;
    g_state.pitch_rad = pitch;

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

    // ── Phase 5: Hip gain scheduling ─────────────────────────────────────────
    // alpha ∈ [0,1]: 0 = fully retracted (high gains), 1 = fully extended (low gains).
    // Uses the calibrated position range so it is coordinate-system agnostic.
    float alpha = 0.5f;  // default to midpoint if calibration not done
    if (hm_limits_L.valid && hm_limits_R.valid) {
        float span_L = hm_limits_L.max_rad - hm_limits_L.min_rad;
        float span_R = hm_limits_R.max_rad - hm_limits_R.min_rad;
        float dir_L  = param_get(PARAM_CALIB_L_SEEK_DIR);
        float dir_R  = param_get(PARAM_CALIB_R_SEEK_DIR);
        float t_L = (dir_L > 0.0f) ? (hm_limits_L.max_rad - hm_L.pos_rad) / span_L
                                    : (hm_L.pos_rad - hm_limits_L.min_rad) / span_L;
        float t_R = (dir_R > 0.0f) ? (hm_limits_R.max_rad - hm_R.pos_rad) / span_R
                                    : (hm_R.pos_rad - hm_limits_R.min_rad) / span_R;
        alpha = 0.5f * (t_L + t_R);
        if (alpha < 0.0f) alpha = 0.0f;
        if (alpha > 1.0f) alpha = 1.0f;
    }
    g_state.gain_sched_alpha = alpha;

    // Interpolated LQR gains
    float k_pitch = K_PITCH_RET + alpha * (K_PITCH_EXT - K_PITCH_RET);
    float k_rate  = K_RATE_RET  + alpha * (K_RATE_EXT  - K_RATE_RET);

    // Effective pendulum length (linear interpolation from IK values)
    float l_eff = L_EFF_RET + alpha * (L_EFF_EXT - L_EFF_RET);

    // ── Phase 3: Velocity PI ──────────────────────────────────────────────────
    float v_desired = param_get(PARAM_V_CMD_MS);
    float theta_ref = 0.0f;

    if (param_get(PARAM_VEL_PI_EN) >= 0.5f) {
        float v_err = v_desired - vel_avg_ms;

        // Reset integrator on direction reversal to prevent windup carryover
        if (v_desired * s_prev_v_desired < 0.0f) s_vel_integral = 0.0f;

        float int_max = param_get(PARAM_VEL_PI_INT_MAX);
        s_vel_integral += v_err * DT;
        if (s_vel_integral >  int_max) s_vel_integral =  int_max;
        if (s_vel_integral < -int_max) s_vel_integral = -int_max;

        float dv_cmd_dt = (v_desired - s_prev_v_desired) / DT;
        float theta_raw = param_get(PARAM_VEL_PI_KP)  * v_err
                        + param_get(PARAM_VEL_PI_KI)  * s_vel_integral
                        + param_get(PARAM_VEL_PI_KFF) * dv_cmd_dt;

        float theta_max = param_get(PARAM_VEL_PI_THETA_MAX);
        if (theta_raw >  theta_max) theta_raw =  theta_max;
        if (theta_raw < -theta_max) theta_raw = -theta_max;

        // Rate limit: output tracks theta_raw but can only slew at rate_lim [rad/s]
        float d_max = param_get(PARAM_VEL_PI_RATE_LIM) * DT;
        float delta = theta_raw - s_theta_ref_rlt;
        if (delta >  d_max) delta =  d_max;
        if (delta < -d_max) delta = -d_max;
        s_theta_ref_rlt += delta;
        theta_ref = s_theta_ref_rlt;
    } else {
        // Reset state while disabled so enable is always a clean start
        s_vel_integral  = 0.0f;
        s_theta_ref_rlt = 0.0f;
    }
    s_prev_v_desired = v_desired;

    g_state.theta_ref        = theta_ref;
    g_state.v_ref            = (param_get(PARAM_VEL_PI_EN) >= 0.5f) ? v_desired : 0.0f;

    // ── Phase 4: Yaw PI ───────────────────────────────────────────────────────
    float tau_yaw = 0.0f;

    if (param_get(PARAM_YAW_PI_EN) >= 0.5f) {
        float omega_desired  = param_get(PARAM_OMEGA_CMD_RDS);
        float omega_measured = imu_yaw_rate();
        float err = omega_desired - omega_measured;

        float yaw_int_max = param_get(PARAM_YAW_PI_INT_MAX);
        s_yaw_integral += err * DT;
        if (s_yaw_integral >  yaw_int_max) s_yaw_integral =  yaw_int_max;
        if (s_yaw_integral < -yaw_int_max) s_yaw_integral = -yaw_int_max;

        tau_yaw = param_get(PARAM_YAW_PI_KP) * err
                + param_get(PARAM_YAW_PI_KI) * s_yaw_integral;

        float torque_max = param_get(PARAM_YAW_PI_TORQUE_MAX);
        if (tau_yaw >  torque_max) tau_yaw =  torque_max;
        if (tau_yaw < -torque_max) tau_yaw = -torque_max;
    } else {
        s_yaw_integral = 0.0f;
    }
    g_state.tau_yaw         = tau_yaw;

    // ── Balance LQR (Phase 5: gain-scheduled) ────────────────────────────────
    // TODO: apply PARAM_RADIO_PITCH_TRIM as offset to pitch_ref here before computing x0.
    //       Add: theta_ref += param_get(PARAM_RADIO_PITCH_TRIM);
    //       Decide whether trim should also offset the vel_PI setpoint or only the LQR error.
    float x0 = pitch - theta_ref;         // pitch error relative to lean setpoint
    float x1 = (param_get(PARAM_ENABLE_SIM_PITCH_RATE) >= 0.5f)
               ? param_get(PARAM_SIM_PITCH_RATE_RAD_S)
               : g_state.pitch_rate_rads;
    g_state.pitch_rate_rads = x1;
    float x2 = vel_avg_ms - g_state.v_ref;  // zero when at commanded speed

    g_state.tau_sym = -(k_pitch * x0 + k_rate * x1 + K_VEL * x2);

    // Clamp to adjustable test limit
    float torque_limit = param_get(PARAM_LQR_TORQUE_LIMIT);
    if (g_state.tau_sym >  torque_limit) g_state.tau_sym =  torque_limit;
    if (g_state.tau_sym < -torque_limit) g_state.tau_sym = -torque_limit;

    // ── Phase 6: Feedforward FF1 + FF2 ───────────────────────────────────────
    float tau_ff1 = 0.0f;
    float tau_ff2 = 0.0f;

    float ff2_alpha = param_get(PARAM_FF2_ALPHA);
    if (ff2_alpha > 0.0f) {
        tau_ff2 = ff2_alpha * M_BODY * GRAVITY * l_eff * sinf(pitch);
    }

    float ff1_alpha = param_get(PARAM_FF1_ALPHA);
    if (ff1_alpha > 0.0f) {
        float kt             = param_get(PARAM_FF1_KT_HIP);
        float tau_hip_total  = (hm_L.current_A + hm_R.current_A) * kt;
        tau_ff1 = -ff1_alpha * tau_hip_total * (WHEEL_R / l_eff);
    }

    g_state.ff1_out = tau_ff1;
    g_state.ff2_out = tau_ff2;

    // ── Wheel torque output ───────────────────────────────────────────────────
    float tau_L = 0.0f;
    float tau_R = 0.0f;

    if (param_get(PARAM_LQR_ENABLE) >= 0.5f) {
        float soft_limit = param_get(PARAM_WHEEL_VEL_LIMIT_TURNS_S);

        // Mix: symmetric + yaw differential + symmetric FF terms
        float tau_ff_sym = tau_ff1 + tau_ff2;
        tau_L = g_state.tau_sym + tau_yaw + tau_ff_sym;
        tau_R = g_state.tau_sym - tau_yaw + tau_ff_sym;

        // Hard clamp to motor limit after FF addition
        if (tau_L >  MOTOR_TRQ_MAX) tau_L =  MOTOR_TRQ_MAX;
        if (tau_L < -MOTOR_TRQ_MAX) tau_L = -MOTOR_TRQ_MAX;
        if (tau_R >  MOTOR_TRQ_MAX) tau_R =  MOTOR_TRQ_MAX;
        if (tau_R < -MOTOR_TRQ_MAX) tau_R = -MOTOR_TRQ_MAX;

        // Per-wheel soft governor: zero torque if spinning beyond limit in the commanded direction
        if ((wm_L.vel_turns_s >  soft_limit && tau_L > 0.0f) ||
            (wm_L.vel_turns_s < -soft_limit && tau_L < 0.0f)) tau_L = 0.0f;
        if ((wm_R.vel_turns_s >  soft_limit && tau_R > 0.0f) ||
            (wm_R.vel_turns_s < -soft_limit && tau_R < 0.0f)) tau_R = 0.0f;
    }
    // Send every tick unconditionally — tau_L/tau_R are 0 when LQR disabled.
    // This prevents stale ODrive torque on re-arm and acts as an implicit watchdog pet.
    wheel_motors_send(tau_L, tau_R);
    g_state.whl_tau_l = tau_L;
    g_state.whl_tau_r = tau_R;
}
