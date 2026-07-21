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

// ── Timing ────────────────────────────────────────────────────────────────────
static constexpr float DT = 0.002f;  // 500 Hz control tick [s]

// ── Pitch watchdog ────────────────────────────────────────────────────────────
static constexpr float    PITCH_WATCHDOG_RAD = 0.8727f;  // 50°
static constexpr uint32_t PITCH_WATCHDOG_MS  = 200;
static uint32_t s_pitch_fault_start_ms = 0;

// ── Phase 5: LQR gain table (computed by lqr.py self-test, Q_pitch=0.01,
//    Q_pitch_rate=0.1884, Q_vel=0.00508442, R=100.0) ─────────────────────────
// alpha=0 → retracted (Q_RET=-0.733 rad), alpha=1 → extended (Q_EXT=-1.432 rad)
// Runtime-tunable — see PARAM_LQR_K_* in param_ids.h/param_registry.cpp for defaults.

// ── Phase 5: Effective pendulum length (IK at Q_RET and Q_EXT) ───────────────
static constexpr float L_EFF_RET    = 0.183117f;  // [m] fully retracted
static constexpr float L_EFF_EXT    = 0.295390f;  // [m] fully extended

// ── Phase 6: Body mass and gravity ───────────────────────────────────────────
// m_b = m_box + 2*(m_femur+m_tibia+m_coupler+m_bearing) + 2*motor_mass
static constexpr float M_BODY       = 1.6380f;    // [kg] body (excl. wheels)
static constexpr float GRAVITY      = 9.81f;      // [m/s²]
static constexpr float WHEEL_R      = 0.075f;     // [m] wheel radius (150 mm OD)
// MOTOR_TRQ_MAX now lives in control_loop.h — shared with state_machine.cpp's standup recovery law

// ── Controller state ──────────────────────────────────────────────────────────
// Phase 3 — Velocity PI
static float s_vel_integral    = 0.0f;
static float s_prev_v_desired  = 0.0f;
static float s_theta_ref_rlt   = 0.0f;  // rate-limited theta_ref

// Phase 4 — Yaw PI
static float s_yaw_integral    = 0.0f;

// Hip torque ramp-in on RUNNING entry — kp/tff scale from 0 to full over
// PARAM_HIP_RUNNING_RAMP_TIME_S so arming doesn't snap the hip to the
// commanded position at full stiffness. kd runs at full value throughout.
static uint32_t s_hip_ramp_start_ms = 0;

// Hip torque ramp-DOWN on disarm (RUNNING -> STANDBY) — mirror of the above,
// same PARAM_HIP_RUNNING_RAMP_TIME_S rate, so releasing the hips doesn't snap
// stiffness straight to zero either. Starts from whatever kp/tff was actually
// in effect at the moment of disarm (not necessarily the full running value —
// e.g. if disarmed mid arm-in-ramp), so the two ramps compose smoothly.
static uint32_t s_hip_disarm_start_ms = 0;
static float    s_hip_disarm_kp0      = 0.0f;
static float    s_hip_disarm_tff0     = 0.0f;

// Gentle 1 s ESTOP ramp — see control_loop.h. Fixed duration (not a tunable
// param): this is a safety-timeout, not a tuning knob, so it isn't exposed to
// accidental GUI misconfiguration. Hip-only: wheels cut power immediately on
// ESTOP (state_machine.cpp) rather than ramping — no reason to keep driving a
// wheel through an emergency stop. Snapshots are per-axis: an axis with no
// active setpoint at ESTOP entry is simply left out of the ramp.
static constexpr float ESTOP_RAMP_TIME_S = 1.0f;
static uint32_t s_estop_ramp_start_ms = 0;
static bool     s_estop_ramp_hip_L    = false;
static bool     s_estop_ramp_hip_R    = false;
static float    s_estop_kp0_L = 0.0f, s_estop_kd0_L = 0.0f, s_estop_tff0_L = 0.0f, s_estop_pos0_L = 0.0f;
static float    s_estop_kp0_R = 0.0f, s_estop_kd0_R = 0.0f, s_estop_tff0_R = 0.0f, s_estop_pos0_R = 0.0f;

// ── Public API ────────────────────────────────────────────────────────────────

void controlLoop_init() {}

void controlLoop_reset() {
    s_vel_integral   = 0.0f;
    s_prev_v_desired = 0.0f;
    s_theta_ref_rlt  = 0.0f;
    s_yaw_integral   = 0.0f;
}

// Separate from controlLoop_reset(): called only when arming RUNNING fresh
// (not when returning to RUNNING from JUMPING), so a jump landing doesn't
// re-loosen the hips right when they need to hold the post-jump position.
void controlLoop_reset_hip_ramp() {
    s_hip_ramp_start_ms = millis();
}

void controlLoop_reset_hip_disarm_ramp() {
    s_hip_disarm_start_ms = millis();
    s_hip_disarm_kp0      = hm_sp_L.kp;
    s_hip_disarm_tff0     = hm_sp_L.tff;
}

bool controlLoop_run_hip_disarm_ramp() {
    float ramp_s  = param_get(PARAM_HIP_RUNNING_RAMP_TIME_S);
    float elapsed = (millis() - s_hip_disarm_start_ms) / 1000.0f;
    float t = (ramp_s > 0.0f) ? (elapsed / ramp_s) : 1.0f;
    if (t > 1.0f) t = 1.0f;
    float alpha = 1.0f - t;

    float pos_L, pos_R;
    hip_cmd_to_setpoints(param_get(PARAM_RADIO_HIP_CMD), &pos_L, &pos_R);
    float kp  = alpha * s_hip_disarm_kp0;
    float kd  = param_get(PARAM_HIP_RUNNING_KD);
    float tff = alpha * s_hip_disarm_tff0;
    hip_motors_set_setpoint_L(pos_L, 0.0f, kp, kd, tff);
    hip_motors_set_setpoint_R(pos_R, 0.0f, kp, kd, tff);

    return t < 1.0f;
}

// Snapshot whatever hip setpoint was actively commanded right before ESTOP so
// the ramp tapers from real values, not stale ones. hm_sp_L/R.active reflects
// any commanding state (RUNNING/JUMPING via the LQR/hip-ramp setpoints,
// CALIBRATION via update_axis(), MANUAL via a GUI MIT command).
void controlLoop_reset_estop_ramp() {
    s_estop_ramp_start_ms = millis();

    s_estop_ramp_hip_L = hm_sp_L.active;
    if (s_estop_ramp_hip_L) {
        s_estop_kp0_L  = hm_sp_L.kp;
        s_estop_kd0_L  = hm_sp_L.kd;
        s_estop_tff0_L = hm_sp_L.tff;
        s_estop_pos0_L = hm_L.pos_rad;
    }
    s_estop_ramp_hip_R = hm_sp_R.active;
    if (s_estop_ramp_hip_R) {
        s_estop_kp0_R  = hm_sp_R.kp;
        s_estop_kd0_R  = hm_sp_R.kd;
        s_estop_tff0_R = hm_sp_R.tff;
        s_estop_pos0_R = hm_R.pos_rad;
    }
}

bool controlLoop_estop_ramp_has_hip() { return s_estop_ramp_hip_L || s_estop_ramp_hip_R; }

bool controlLoop_run_estop_ramp() {
    float elapsed = (millis() - s_estop_ramp_start_ms) / 1000.0f;
    float t = elapsed / ESTOP_RAMP_TIME_S;
    if (t > 1.0f) t = 1.0f;
    float alpha = 1.0f - t;

    // Hold the frozen position (not tracking radio/live feedback) — an ESTOP
    // ramp must not depend on anything that could itself be the reason we're
    // ESTOPping (e.g. lost radio). vel target 0 throughout.
    if (s_estop_ramp_hip_L)
        hip_motors_set_setpoint_L(s_estop_pos0_L, 0.0f, alpha * s_estop_kp0_L, alpha * s_estop_kd0_L, alpha * s_estop_tff0_L);
    if (s_estop_ramp_hip_R)
        hip_motors_set_setpoint_R(s_estop_pos0_R, 0.0f, alpha * s_estop_kp0_R, alpha * s_estop_kd0_R, alpha * s_estop_tff0_R);

    return t < 1.0f;
}

void controlLoop_run() {
    // ── Hip setpoints from radio ──────────────────────────────────────────────
    float pos_L, pos_R;
    hip_cmd_to_setpoints(param_get(PARAM_RADIO_HIP_CMD), &pos_L, &pos_R);
    float ramp_s  = param_get(PARAM_HIP_RUNNING_RAMP_TIME_S);
    float elapsed = (millis() - s_hip_ramp_start_ms) / 1000.0f;
    float ramp_alpha = (ramp_s > 0.0f) ? (elapsed / ramp_s) : 1.0f;
    if (ramp_alpha > 1.0f) ramp_alpha = 1.0f;
    float running_kp  = ramp_alpha * param_get(PARAM_HIP_RUNNING_KP);
    float running_kd  = param_get(PARAM_HIP_RUNNING_KD);
    float running_tff = ramp_alpha * param_get(PARAM_HIP_RUNNING_TFF);
    hip_motors_set_setpoint_L(pos_L, 0.0f, running_kp, running_kd, running_tff);
    hip_motors_set_setpoint_R(pos_R, 0.0f, running_kp, running_kd, running_tff);

    // ── Wheel velocity average [m/s] ──────────────────────────────────────────
    float vel_avg_ms = (wm_L.vel_turns_s + wm_R.vel_turns_s) * 0.5f
                       * (2.0f * (float)M_PI * WHEEL_R);
    g_state.wheel_vel_avg_ms = vel_avg_ms;

    // ── Effective pitch (real or injected) ────────────────────────────────────
    // Blend (real IMU vs. sim override) already happened in read_sensors(), so
    // g_state.pitch_rad here is exactly what telemetry/GUI saw this tick too.
    float pitch = g_state.pitch_rad;

    // ── Pitch watchdog ────────────────────────────────────────────────────────
    if (param_get(PARAM_PITCH_WATCHDOG_ENABLE) >= 0.5f) {
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
    // §1c (tuning.md): hips zip-tied retracted and disabled, so real calibration
    // can never complete — force the retracted anchor directly, no encoder read,
    // no calibration dependency. Skips the block below entirely.
    if (param_get(PARAM_ALPHA_FORCE_RETRACTED_EN) >= 0.5f) {
        alpha = 0.0f;
    } else if (hm_limits_L.valid && hm_limits_R.valid) {
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
    float k_pitch_ret = param_get(PARAM_LQR_K_PITCH_RET);
    float k_rate_ret  = param_get(PARAM_LQR_K_RATE_RET);
    float k_pitch_ext = param_get(PARAM_LQR_K_PITCH_EXT);
    float k_rate_ext  = param_get(PARAM_LQR_K_RATE_EXT);
    float k_pitch = k_pitch_ret + alpha * (k_pitch_ext - k_pitch_ret);
    float k_rate  = k_rate_ret  + alpha * (k_rate_ext  - k_rate_ret);

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
    // g_state.v_ref is set every tick in main.cpp's radio_update(), not here —
    // it needs to stay live in STANDBY too (controlLoop_run() only runs in
    // RUNNING/JUMPING), same as v_cmd_ms/omega_cmd_rds.

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
    float x1 = g_state.pitch_rate_rads;   // blended in read_sensors() (real or injected)
    float x2 = vel_avg_ms - g_state.v_ref;  // zero when at commanded speed

    g_state.tau_sym = -(k_pitch * x0 + k_rate * x1 + param_get(PARAM_LQR_K_VEL) * x2);

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
        // Driving the left wheel harder than the right yaws the robot right (-Z),
        // so positive tau_yaw (commanding +yaw, CCW from above) must add to the
        // right wheel and subtract from the left.
        float tau_ff_sym = tau_ff1 + tau_ff2;
        tau_L = g_state.tau_sym - tau_yaw + tau_ff_sym;
        tau_R = g_state.tau_sym + tau_yaw + tau_ff_sym;

        // C2: clamp the mixed output (incl. FF) to the adjustable test limit —
        // FF terms are no longer exempt from PARAM_LQR_TORQUE_LIMIT.
        if (tau_L >  torque_limit) tau_L =  torque_limit;
        if (tau_L < -torque_limit) tau_L = -torque_limit;
        if (tau_R >  torque_limit) tau_R =  torque_limit;
        if (tau_R < -torque_limit) tau_R = -torque_limit;

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
