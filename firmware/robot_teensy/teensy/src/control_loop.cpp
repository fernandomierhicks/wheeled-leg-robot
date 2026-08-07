#include "control_loop.h"
#include "config.h"
#include "robot_state.h"
#include "comm_protocol.h"
#include "state_machine.h"
#include "hip_motors.h"
#include "wheel_motors.h"
#include "param_registry.h"
#include "live_tune.h"
#include "velocity_pi_anti_windup.h"
#include "control_safety.h"
#include "IMU.h"
#include <math.h>
#include <Arduino.h>

RobotState g_state = {};

// ── Timing ────────────────────────────────────────────────────────────────────
static constexpr float DT = 0.002f;  // 500 Hz control tick [s]

// ── Pitch watchdog ────────────────────────────────────────────────────────────
// Trip thresholds are asymmetric (PARAM_PITCH_WATCHDOG_FWD/BWD_RET/EXT, gain-
// scheduled by alpha) since the leg linkage collides with the ground at a
// different angle forward than backward. Duration is fixed.
static constexpr uint32_t PITCH_WATCHDOG_MS  = 200;
static uint32_t s_pitch_fault_start_ms = 0;

// ── Roll watchdog (lateral tip guard) ─────────────────────────────────────────
// |roll| threshold is PARAM_ROLL_WATCHDOG_LIMIT (runtime); duration is fixed.
static constexpr uint32_t ROLL_WATCHDOG_MS = 200;
static uint32_t s_roll_fault_start_ms = 0;

// ── Wheel runaway watchdog ────────────────────────────────────────────────────
// Debounce for the |wheel velocity| > 2x soft-limit trip. Short compared to
// the pitch/roll windows: a real runaway travels fast, so this only has to be
// long enough to outlast isolated encoder-frame corruption.
static constexpr uint32_t WHEEL_RUNAWAY_MS = 50;
static uint32_t s_wheel_fault_start_ms = 0;

// ── Backward target safety margin ────────────────────────────────────────────
// Fixed conservative gap (not a tunable param, same reasoning as
// ESTOP_RAMP_TIME_S below) kept between the code-level backward lean target
// and the pitch watchdog trip angle -- see safe_backward_theta_limit() in
// control_safety.h.
static constexpr float BACKWARD_TARGET_MARGIN_RAD = 0.0523599f;  // 3 deg

// ── Phase 5: LQR gain table (computed by lqr.py self-test, Q_pitch=0.01,
//    Q_pitch_rate=0.1884, Q_vel=0.00508442, R=100.0) ─────────────────────────
// alpha=0 → retracted, alpha=1 → extended. NOTE: alpha is NOT anchored to any
// fixed q_hip pair — it is the fraction of the *calibrated* hip span, which
// define_limits() sets to (calib_range_*_rad - calib_backoff_rad), currently
// 85° - 5° = 80°. The old "(Q_RET=-0.733, Q_EXT=-1.432)" annotation here was
// doubly wrong: those angles are for the superseded 5.29 mm F_Z geometry (see
// CLAUDE.md), and their 0.699 rad span is a nominal stance stroke, not the
// calibrated range. Reading them as the alpha endpoints understates the hip
// scale by ~2x.
//
// The mechanism's real travel is 66° (hard-stopped), so the configured 80°
// overruns the extended stop by 14°: alpha only reaches ~0.825 in practice and
// the extended end of this gain table is never actually used. Fix by setting
// calib_range_l/r_rad = 66° + backoff 5° = 71° (1.239 rad).
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

// Rate-limited RUNNING hip-height command, in the same normalized [0,1]
// extension space as PARAM_RADIO_HIP_CMD/hip_cmd_to_setpoints(). Invalid
// (no prior value to slew from) until controlLoop_reset_hip_ramp() seeds it
// on RUNNING entry; controlLoop_reset() invalidates it again on disarm.
static float s_hip_cmd_rlt       = 0.0f;
static bool  s_hip_cmd_rlt_valid = false;

// Roll controller — rate-limited roll setpoint [rad] + PI integral state.
// The integral nulls the steady-state droop the P-only law left against a
// standing CG/leg asymmetry; disabled (roll_ki=0) it contributes nothing.
static float s_roll_sp_rlt     = 0.0f;
static float s_roll_integral   = 0.0f;

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

// Reconstructs the measured hip extension fraction alpha ∈ [0,1] (0 =
// retracted, 1 = extended) from calibrated hip positions -- coordinate-
// system agnostic (uses CALIB_L_SEEK_DIR/CALIB_R_SEEK_DIR, same convention as
// hip_cmd_to_setpoints()'s t argument), so it's shared by gain scheduling
// and RUNNING-entry hip-command initialization instead of each recomputing
// their own version. *valid is false (return 0) whenever either axis'
// calibration hasn't been run this session.
static float measured_hip_alpha(bool* valid) {
    if (!(hm_limits_L.valid && hm_limits_R.valid)) {
        if (valid) *valid = false;
        return 0.0f;
    }
    float span_L = hm_limits_L.max_rad - hm_limits_L.min_rad;
    float span_R = hm_limits_R.max_rad - hm_limits_R.min_rad;
    float dir_L  = CALIB_L_SEEK_DIR;
    float dir_R  = CALIB_R_SEEK_DIR;
    float t_L = (dir_L > 0.0f) ? (hm_limits_L.max_rad - hm_L.pos_rad) / span_L
                                : (hm_L.pos_rad - hm_limits_L.min_rad) / span_L;
    float t_R = (dir_R > 0.0f) ? (hm_limits_R.max_rad - hm_R.pos_rad) / span_R
                                : (hm_R.pos_rad - hm_limits_R.min_rad) / span_R;
    float alpha = 0.5f * (t_L + t_R);
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    if (valid) *valid = true;
    return alpha;
}

// ── Public API ────────────────────────────────────────────────────────────────

void controlLoop_init() {}

void controlLoop_reset() {
    s_vel_integral   = 0.0f;
    s_prev_v_desired = 0.0f;
    s_theta_ref_rlt  = 0.0f;
    s_yaw_integral   = 0.0f;
    s_roll_sp_rlt         = 0.0f;
    s_roll_integral       = 0.0f;
    s_roll_fault_start_ms = 0;
    s_pitch_fault_start_ms = 0;
    s_wheel_fault_start_ms = 0;
    s_hip_cmd_rlt_valid   = false;
}

// Separate from controlLoop_reset(): called only when arming RUNNING fresh
// (not when returning to RUNNING from JUMPING), so a jump landing doesn't
// re-loosen the hips right when they need to hold the post-jump position.
void controlLoop_reset_hip_ramp() {
    s_hip_ramp_start_ms = millis();

    // Seed the rate-limited hip command from where the legs actually are (not
    // 0) so the new hip_cmd_rate_lim slew doesn't chase from a stale/zero
    // starting point on arm; falls back to the raw radio target when
    // calibration isn't valid, matching gain_sched_alpha's own default.
    bool  alpha_valid = false;
    float measured     = measured_hip_alpha(&alpha_valid);
    s_hip_cmd_rlt       = alpha_valid ? measured : param_get(PARAM_RADIO_HIP_CMD);
    s_hip_cmd_rlt_valid = true;
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
    float hip_cmd = s_hip_cmd_rlt_valid ? s_hip_cmd_rlt : param_get(PARAM_RADIO_HIP_CMD);
    hip_cmd_to_setpoints(hip_cmd, &pos_L, &pos_R);
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
    // ── Phase 5: Hip gain scheduling ─────────────────────────────────────────
    // alpha ∈ [0,1]: 0 = fully retracted (high gains), 1 = fully extended (low gains).
    // §1c (tuning.md): default to the retracted anchor whenever calibration is
    // invalid (e.g. not yet run this session), rather than an arbitrary
    // midpoint. alpha_force_ret_en stays available to force retracted even
    // once calibration is valid. Computed first thing in the tick because the
    // hip feedforward schedule, the pitch watchdog, the velocity PI and the
    // LQR gains all key off it. Derived from *measured* hip position, which
    // doesn't change within a tick, so hoisting it above the hip block below
    // doesn't change its value.
    float alpha = 0.0f;  // default: retracted, used whenever calibration is invalid
    if (param_get(PARAM_ALPHA_FORCE_RETRACTED_EN) >= 0.5f) {
        alpha = 0.0f;
    } else {
        bool  alpha_valid = false;
        float measured     = measured_hip_alpha(&alpha_valid);
        if (alpha_valid) alpha = measured;
    }
    g_state.gain_sched_alpha = alpha;

    // ── Hip setpoints from radio (rate-limited in normalized extension space) ──
    // Slew the raw CH3 target with PARAM_HIP_CMD_RATE_LIMIT_PER_S before
    // converting to motor positions, so a snapped stick can't step the hips
    // straight to a new height -- reduces reaction-current impulses and
    // linkage chatter. s_hip_cmd_rlt_valid is normally seeded by
    // controlLoop_reset_hip_ramp() on RUNNING entry; the fallback below only
    // matters if controlLoop_run() is ever reached without it (defensive).
    float hip_cmd_raw = param_get(PARAM_RADIO_HIP_CMD);
    if (!s_hip_cmd_rlt_valid) {
        s_hip_cmd_rlt       = hip_cmd_raw;
        s_hip_cmd_rlt_valid = true;
    }
    s_hip_cmd_rlt = slew_toward(s_hip_cmd_rlt, hip_cmd_raw,
                                param_get(PARAM_HIP_CMD_RATE_LIMIT_PER_S), DT);

    float pos_L, pos_R;
    hip_cmd_to_setpoints(s_hip_cmd_rlt, &pos_L, &pos_R);
    float ramp_s  = param_get(PARAM_HIP_RUNNING_RAMP_TIME_S);
    float elapsed = (millis() - s_hip_ramp_start_ms) / 1000.0f;
    float ramp_alpha = (ramp_s > 0.0f) ? (elapsed / ramp_s) : 1.0f;
    if (ramp_alpha > 1.0f) ramp_alpha = 1.0f;

    // ── Roll controller (active suspension) ───────────────────────────────────
    // PD on roll angle/rate → a differential hip position offset (+ on one leg,
    // − on the other), held with a soft (backdrivable) kp so obstacles can
    // back-drive the legs. RUNNING only; off by default, in which case the hips
    // are held symmetrically with the running gains exactly as before.
    float hip_kp = param_get(PARAM_HIP_RUNNING_KP);
    float hip_kd = param_get(PARAM_HIP_RUNNING_KD);
    if (g_state.state == STATE_RUNNING && param_get(PARAM_ROLL_CTRL_EN) >= 0.5f) {
        float roll      = imu_roll();
        float roll_rate = imu_roll_rate();

        // Slew-limit the setpoint so a snapped stick can't step-perturb pitch.
        float roll_sp_raw = param_get(PARAM_ROLL_CMD_RAD);
        float d_max = param_get(PARAM_ROLL_RATE_LIM) * DT;
        float dsp = roll_sp_raw - s_roll_sp_rlt;
        if (dsp >  d_max) dsp =  d_max;
        if (dsp < -d_max) dsp = -d_max;
        s_roll_sp_rlt += dsp;

        // PI + D on roll. The P and D terms are the original law; the integral
        // exists to null the steady-state droop P-only leaves against a
        // standing CG/leg-length asymmetry. roll_ki=0 (the default) makes the
        // integral term identically zero, so the response is byte-identical to
        // the previous PD controller.
        float roll_err  = s_roll_sp_rlt - roll;
        float roll_ki   = param_get(PARAM_ROLL_KI);
        float off_max   = param_get(PARAM_ROLL_OFFSET_MAX);
        float offset_pd = live_tune_value(PARAM_ROLL_KP) * roll_err
                        - live_tune_value(PARAM_ROLL_KD) * roll_rate;

        // Same conditional-integration anti-windup the velocity PI uses
        // (symmetric limits here): the integral freezes only when it would
        // drive an already-saturated offset further past roll_offset_max, and
        // is always free to unwind back out.
        s_roll_integral = velocity_pi_integral_step(
            s_roll_integral,
            roll_err,
            DT,
            param_get(PARAM_ROLL_INT_MAX),
            roll_ki,
            offset_pd,
            off_max,
            off_max
        );

        float offset = offset_pd + roll_ki * s_roll_integral;
        if (offset >  off_max) offset =  off_max;
        if (offset < -off_max) offset = -off_max;

        // Differential apply. Hip sign convention (README "Motor direction"):
        // increasing pos_L/pos_R retracts, decreasing extends. IMU convention
        // (quat_to_euler): positive roll = lean right (left side up). So a
        // positive offset (commanding more positive roll) must extend the
        // left leg (decrease pos_L) and retract the right leg (increase
        // pos_R).
        pos_L -= offset;
        pos_R += offset;

        // Clamp to calibrated hip travel so a bad gain can't overtravel a leg.
        if (hm_limits_L.valid) {
            if (pos_L < hm_limits_L.min_rad) pos_L = hm_limits_L.min_rad;
            if (pos_L > hm_limits_L.max_rad) pos_L = hm_limits_L.max_rad;
        }
        if (hm_limits_R.valid) {
            if (pos_R < hm_limits_R.min_rad) pos_R = hm_limits_R.min_rad;
            if (pos_R > hm_limits_R.max_rad) pos_R = hm_limits_R.max_rad;
        }

        // Soft, backdrivable hold while suspension is active.
        hip_kp = param_get(PARAM_HIP_ROLL_KP);
        hip_kd = param_get(PARAM_HIP_ROLL_KD);
    } else {
        // Clear both so re-enabling roll control is always a clean start and a
        // stale integral can't snap the legs apart on the first tick.
        s_roll_sp_rlt   = 0.0f;
        s_roll_integral = 0.0f;
    }

    // Hip feedforward, gain-scheduled by leg height on the same alpha as the
    // LQR gains. The hip hold is proportional-only, so its steady-state sag is
    // (holding torque − tff)/kp; the 4-bar's mechanical advantage makes the
    // holding torque vary with height, so one constant tff can only null the
    // sag at a single leg height. ramp_alpha is the separate arm-in stiffness
    // ramp and still scales the result.
    float tff_ret = param_get(PARAM_HIP_RUNNING_TFF_RET);
    float tff_ext = param_get(PARAM_HIP_RUNNING_TFF_EXT);
    float running_kp  = ramp_alpha * hip_kp;
    float running_kd  = hip_kd;
    float running_tff = ramp_alpha * (tff_ret + alpha * (tff_ext - tff_ret));
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

    // ── Scheduled watchdog / barrier thresholds (alpha known, compute once) ──
    // Asymmetric, gain-scheduled magnitudes (pitch positive = lean forward per
    // quat_to_euler()). Shared below by the pitch watchdog, the velocity PI's
    // safe backward lean clamp, the direct velocity-term guard, and the
    // backward soft-limit barrier so none of them re-derive their own alpha
    // interpolation of the same underlying params.
    float pw_fwd_ret   = param_get(PARAM_PITCH_WATCHDOG_FWD_RET);
    float pw_bwd_ret   = param_get(PARAM_PITCH_WATCHDOG_BWD_RET);
    float pw_fwd_ext   = param_get(PARAM_PITCH_WATCHDOG_FWD_EXT);
    float pw_bwd_ext   = param_get(PARAM_PITCH_WATCHDOG_BWD_EXT);
    float sched_pw_fwd = pw_fwd_ret + alpha * (pw_fwd_ext - pw_fwd_ret);
    float sched_pw_bwd = pw_bwd_ret + alpha * (pw_bwd_ext - pw_bwd_ret);

    float bar_th_ret    = param_get(PARAM_LQR_BARRIER_THRESH_RET);
    float bar_th_ext    = param_get(PARAM_LQR_BARRIER_THRESH_EXT);
    float sched_bar_th  = bar_th_ret + alpha * (bar_th_ext - bar_th_ret);

    // ── Balance-point pitch trim ─────────────────────────────────────────────
    // The lean that holds zero velocity isn't pitch=0 when the CG isn't exactly
    // over the wheel axle; it also shifts with leg height. Gain-schedule the
    // trim by the same alpha as the LQR gains (linear ret→ext for now).
    // trim_ret goes through live_tune_value() too (live_tune.h) but currently
    // has no LIVE_TUNE_SLOTS entry, so it always falls through to the latched
    // persisted value -- CH7/CH8 are presently assigned to the LQR gains below.
    // trim_ext has no knob and always reads its persisted value directly.
    // Offsets only the LQR pitch error, never the velocity setpoint. Computed
    // here (after alpha, before the velocity PI) because the velocity PI's
    // safe backward lean clamp below needs it.
    float trim_ret = live_tune_value(PARAM_LQR_PITCH_TRIM_RET);
    float trim_ext = param_get(PARAM_LQR_PITCH_TRIM_EXT);
    float pitch_trim = trim_ret + alpha * (trim_ext - trim_ret);
    g_state.applied_pitch_trim = pitch_trim;

    // ── Pitch watchdog ────────────────────────────────────────────────────────
    if (param_get(PARAM_PITCH_WATCHDOG_ENABLE) >= 0.5f) {
        if (pitch > sched_pw_fwd || pitch < -sched_pw_bwd) {
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

    // ── Roll watchdog (lateral tip guard) ─────────────────────────────────────
    // A commanded/external roll shifts the CG toward the low wheel; wheels make
    // no lateral force, so past a limit the robot tips sideways with no recovery.
    if (param_get(PARAM_ROLL_WATCHDOG_EN) >= 0.5f) {
        if (fabsf(imu_roll()) > param_get(PARAM_ROLL_WATCHDOG_LIMIT)) {
            if (s_roll_fault_start_ms == 0) s_roll_fault_start_ms = millis();
            if (millis() - s_roll_fault_start_ms > ROLL_WATCHDOG_MS) {
                g_state.fault_code = FAULT_ROLL_WATCHDOG;
                stateMachine_request_estop();
                return;
            }
        } else {
            s_roll_fault_start_ms = 0;
        }
    }

    // ── Wheel runaway watchdog (hard backup) ──────────────────────────────────
    // Debounced like the pitch/roll watchdogs (this one used to trip on a
    // single sample, and a corrupt encoder frame duly ESTOPped a run that was
    // otherwise balancing fine — 2026-07-28). A real runaway holds the wheels
    // over the limit continuously, so the debounce costs nothing against it;
    // WHEEL_RUNAWAY_MS is much shorter than the 200 ms pitch/roll windows
    // because a genuinely runaway wheel covers ground fast.
    float hard_limit = param_get(PARAM_WHEEL_VEL_LIMIT_TURNS_S) * 2.0f;
    if (fabsf(wm_L.vel_turns_s) > hard_limit || fabsf(wm_R.vel_turns_s) > hard_limit) {
        if (s_wheel_fault_start_ms == 0) s_wheel_fault_start_ms = millis();
        if (millis() - s_wheel_fault_start_ms > WHEEL_RUNAWAY_MS) {
            g_state.fault_code = FAULT_WHEEL_RUNAWAY;
            stateMachine_request_estop();
            return;
        }
    } else {
        s_wheel_fault_start_ms = 0;
    }

    // Interpolated LQR gains (alpha computed earlier, above the pitch watchdog).
    // k_pitch_ret/k_rate_ret are the CH7/CH8 live-tune targets (live_tune.h,
    // LIVE_TUNE_SLOTS in main.cpp) -- live_tune_value() returns the picked-up
    // knob shadow while bench-tuning, falling through to the latched persisted
    // value otherwise.
    float k_pitch_ret = live_tune_value(PARAM_LQR_K_PITCH_RET);
    float k_rate_ret  = live_tune_value(PARAM_LQR_K_RATE_RET);
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

        // Asymmetric, gain-scheduled lean clamp (both stored as positive
        // magnitudes) -- mirrors the pitch watchdog split: the leg linkage
        // clears the ground at a different forward angle than backward.
        float theta_max_fwd_ret = param_get(PARAM_VEL_PI_THETA_MAX_FWD_RET);
        float theta_max_bwd_ret = param_get(PARAM_VEL_PI_THETA_MAX_BWD_RET);
        float theta_max_fwd_ext = param_get(PARAM_VEL_PI_THETA_MAX_FWD_EXT);
        float theta_max_bwd_ext = param_get(PARAM_VEL_PI_THETA_MAX_BWD_EXT);
        float theta_max_fwd     = theta_max_fwd_ret + alpha * (theta_max_fwd_ext - theta_max_fwd_ret);
        float theta_max_bwd_cfg = theta_max_bwd_ret + alpha * (theta_max_bwd_ext - theta_max_bwd_ret);

        // Code-level backward-authority ceiling: never let the velocity loop
        // request a lean target beyond what the pitch watchdog would itself
        // trip on. See safe_backward_theta_limit() in control_safety.h --
        // invariant is theta_max_bwd <= pitch_watchdog_bwd + pitch_trim -
        // margin (pitch_trim is typically negative/backward-leaning here,
        // eating into the configured clamp on top of the fixed margin).
        float theta_max_bwd = safe_backward_theta_limit(
            theta_max_bwd_cfg, sched_pw_bwd, pitch_trim, BACKWARD_TARGET_MARGIN_RAD);

        float kp = live_tune_value(PARAM_VEL_PI_KP);
        float ki = live_tune_value(PARAM_VEL_PI_KI);
        float dv_cmd_dt = (v_desired - s_prev_v_desired) / DT;
        float theta_non_integral =
            kp * v_err + param_get(PARAM_VEL_PI_KFF) * dv_cmd_dt;

        // Conditional integration: freeze only when the proposed I update
        // would drive theta farther beyond the active lean clamp. Updates that
        // reduce the stored integral remain enabled so saturation can unwind.
        s_vel_integral = velocity_pi_integral_step(
            s_vel_integral,
            v_err,
            DT,
            param_get(PARAM_VEL_PI_INT_MAX),
            ki,
            theta_non_integral,
            theta_max_fwd,
            theta_max_bwd
        );

        float theta_raw = theta_non_integral + ki * s_vel_integral;
        if (theta_raw >  theta_max_fwd) theta_raw =  theta_max_fwd;
        if (theta_raw < -theta_max_bwd) theta_raw = -theta_max_bwd;
        // Exposed so main.cpp's HEALTH_VEL_PI_SAT check compares against the
        // same effective (gain-scheduled) bounds instead of recomputing alpha.
        g_state.theta_max_fwd = theta_max_fwd;
        g_state.theta_max_bwd = theta_max_bwd;

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
    // pitch_trim was computed earlier (with alpha, before the velocity PI);
    // reused here unchanged so x0 keeps the exact same definition.
    float x0 = pitch - theta_ref - pitch_trim;  // pitch error vs. lean setpoint + balance trim
    float x1 = g_state.pitch_rate_rads;   // blended in read_sensors() (real or injected)
    float x2 = vel_avg_ms - g_state.v_ref;  // zero when at commanded speed

    float pitch_term    = k_pitch * x0;
    float rate_term     = k_rate * x1;
    float velocity_term = param_get(PARAM_LQR_K_VEL) * x2;

    // Balance-priority guard: past the backward barrier threshold, remove
    // only the part of the direct velocity term that opposes recovery,
    // fading it linearly to zero at the pitch watchdog. A full-reverse
    // velocity command was seen canceling much of the pitch/pitch-rate
    // recovery torque right as the watchdog was about to trip; this leaves
    // pitch/rate stabilization with full authority near the boundary while
    // never touching the term when it's inside the barrier or already
    // assisting recovery. See safe_backward_theta_limit()'s sibling helper,
    // backward_velocity_term_guard(), in control_safety.h.
    float guarded_velocity_term = backward_velocity_term_guard(
        velocity_term, pitch, sched_bar_th, sched_pw_bwd);

    g_state.tau_sym = -(pitch_term + rate_term + guarded_velocity_term);

    // ── Backward soft-limit barrier ──────────────────────────────────────────
    // Authority is asymmetric: the leg linkage hits the ground on a hard
    // backward lean well before forward pitch runs out of room (see "Control
    // algorithm" in README). Adds extra backward-recovery torque once pitch
    // swings past the (gain-scheduled) threshold computed above with the rest
    // of the scheduled thresholds; continuous at the boundary and exactly
    // zero everywhere inside it, so it never disturbs the tuned LQR response
    // in normal operation — a soft wall bolted on where the mechanism runs
    // out, not a change to the response inside normal range.
    float bar_over = -pitch - sched_bar_th;  // > 0 only past the backward soft limit
    if (bar_over > 0.0f) g_state.tau_sym -= param_get(PARAM_LQR_BARRIER_K) * bar_over;

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
        float tau_hip_total  = (hm_L.torque_nm + hm_R.torque_nm) * kt;
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
