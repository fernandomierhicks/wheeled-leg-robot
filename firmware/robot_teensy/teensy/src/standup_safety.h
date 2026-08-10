#pragma once

#include <math.h>
#include <stdint.h>

// Fixed admission gate between the zero-wheel hip positioning phase and the
// balance controller. These are safety criteria, not stand-up tuning knobs:
// the LQR must never engage merely because the commanded ramp timer elapsed.
static constexpr float STANDUP_HIP_POS_TOL_RAD       = 0.0349066f;  // 2 deg
static constexpr float STANDUP_HIP_VEL_TOL_RADS      = 0.20f;
static constexpr float STANDUP_CALIB_TARGET_TOL_RAD  = 0.001f;
static constexpr uint32_t STANDUP_HIP_SETTLE_HOLD_MS = 100;
static constexpr uint32_t STANDUP_HIP_SETTLE_TIMEOUT_MS = 2000;
// Ceiling on the feedforward allowance below. Past this the leg is simply not
// in the neighbourhood of the crouch pose and "settled" would stop meaning
// anything, so a badly scaled kp/tff pair fails the gate instead of disabling
// it. 0.2 rad = 11.5 deg, comfortably under a typical calib_backoff_rad.
static constexpr float STANDUP_HIP_POS_TOL_MAX_RAD   = 0.20f;

// Position tolerance for the settle gate, given the gains actually commanded.
//
// The hip hold is proportional plus a constant feedforward, so it comes to rest
// where kp*(target - pos) + tff balances the external load, i.e. offset by
// (tff - load)/kp. Both ends of that are legitimate here: on the bench the legs
// are near unloaded and the offset is the full tff/kp, while standing on the
// wheels the load is what tff was tuned to cancel and the offset is ~0. With
// hip_running_tff_ret = -2.5 N*m and hip_running_kp = 25 that spread is 0.1 rad
// — three times the 2 deg band, so a fixed 2 deg gate is unreachable off the
// ground however long you wait for it.
//
// Allowing |tff|/kp on top of the band admits exactly that range and nothing
// wider: a leg jammed part-way through the excursion is tens of degrees out and
// still fails.
inline float standup_hip_pos_tol(float kp, float tff) {
    if (kp <= 0.0f) return STANDUP_HIP_POS_TOL_RAD;  // no hold: only velocity means anything
    float tol = STANDUP_HIP_POS_TOL_RAD + fabsf(tff) / kp;
    return tol > STANDUP_HIP_POS_TOL_MAX_RAD ? STANDUP_HIP_POS_TOL_MAX_RAD : tol;
}

// Quintic minimum-jerk trajectory. Position, velocity and acceleration are all
// continuous, with zero velocity and acceleration at both endpoints. This
// avoids the old linear ramp's instantaneous 0 -> dq -> 0 velocity steps.
inline float standup_min_jerk_position(float u) {
    if (u <= 0.0f) return 0.0f;
    if (u >= 1.0f) return 1.0f;
    float u2 = u * u;
    float u3 = u2 * u;
    return u3 * (10.0f + u * (-15.0f + 6.0f * u));
}

inline float standup_min_jerk_rate(float u, float duration_s) {
    if (u <= 0.0f || u >= 1.0f || duration_s <= 0.0f) return 0.0f;
    float one_minus_u = 1.0f - u;
    return 30.0f * u * u * one_minus_u * one_minus_u / duration_s;
}

// Handoff precondition on wheel speed, checked with the pitch capture band.
//
// STANDING_UP may be running under standup_vel_limit, a higher governor
// threshold than RUNNING's. The moment it captures, the limit snaps back to
// wm_vel_limit: any wheel above that has its torque zeroed by the soft governor
// on the first RUNNING tick, and any wheel above 2x it trips the runaway
// watchdog ~50 ms later — a successful catch immediately followed by a fall or
// an ESTOP. So the robot is not "settled" until the wheels are inside the limit
// the next state will hold them to, whatever limit got it there.
inline bool standup_wheels_ready_for_handoff(float vel_l_turns_s,
                                              float vel_r_turns_s,
                                              float running_limit_turns_s) {
    return fabsf(vel_l_turns_s) <= running_limit_turns_s &&
           fabsf(vel_r_turns_s) <= running_limit_turns_s;
}

// End-of-excursion test for CROUCH: both hips have stopped moving. Position is
// deliberately NOT checked here — CROUCH runs at a fraction of the running hip
// stiffness, and a proportional hold at reduced kp sags by (hold torque - tff)
// / kp, which is far more than STANDUP_HIP_POS_TOL_RAD. The strict position
// band is checked at the end of STIFFEN instead, once the gains are at their
// full running values and the sag has been pulled out.
inline bool standup_hips_quiet(float vel_l, float vel_r) {
    return fabsf(vel_l) <= STANDUP_HIP_VEL_TOL_RADS &&
           fabsf(vel_r) <= STANDUP_HIP_VEL_TOL_RADS;
}

// Hip stiffness scale during STIFFEN: a linear ramp from the CROUCH fraction to
// 1.0 (full hip_running_kp/tff) over `duration_s`. Scales kp and the hip
// feedforward together, exactly like control_loop.cpp's arm-in ramp_alpha, so
// engaging the LQR at the end is a continuation of the same profile rather than
// a step in either quantity.
inline float standup_stiffen_scale(float elapsed_s, float duration_s, float from_scale) {
    if (duration_s <= 0.0f) return 1.0f;
    float u = elapsed_s / duration_s;
    if (u <= 0.0f) return from_scale;
    if (u >= 1.0f) return 1.0f;
    return from_scale + (1.0f - from_scale) * u;
}

// pos_tol comes from standup_hip_pos_tol() at the gains in force — see there
// for why it is not simply STANDUP_HIP_POS_TOL_RAD.
inline bool standup_hips_in_settle_band(float pos_l, float vel_l,
                                         float pos_r, float vel_r,
                                         float target_l, float target_r,
                                         float pos_tol) {
    return fabsf(pos_l - target_l) <= pos_tol &&
           fabsf(pos_r - target_r) <= pos_tol &&
           fabsf(vel_l) <= STANDUP_HIP_VEL_TOL_RADS &&
           fabsf(vel_r) <= STANDUP_HIP_VEL_TOL_RADS;
}

inline bool standup_target_matches_configured_backoff(
        float target_l, float target_r, float backoff_rad,
        float seek_dir_l, float seek_dir_r) {
    return fabsf(target_l - (-seek_dir_l * backoff_rad)) <=
               STANDUP_CALIB_TARGET_TOL_RAD &&
           fabsf(target_r - (-seek_dir_r * backoff_rad)) <=
               STANDUP_CALIB_TARGET_TOL_RAD;
}
