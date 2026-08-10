#pragma once

#include <math.h>

// Quadratic balance-trim schedule with fixed retracted/extended endpoints.
// A zero curve coefficient is exactly the legacy linear interpolation; at
// alpha=0.5 the curve contributes one quarter of its configured value.
inline float scheduled_pitch_trim(float trim_ret, float trim_ext,
                                  float trim_curve, float alpha) {
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    return trim_ret + alpha * (trim_ext - trim_ret)
         + trim_curve * alpha * (1.0f - alpha);
}

// Keep the velocity-loop backward pitch target inside the independently
// configured backward watchdog after accounting for the scheduled balance
// trim. For theta_ref = -theta_bwd, the absolute target is:
//
//     pitch_target = pitch_trim - theta_bwd
//
// Requiring pitch_target >= -watchdog + margin gives:
//
//     theta_bwd <= watchdog + pitch_trim - margin
//
// A badly configured trim/watchdog pair can leave no safe backward lean; zero
// is safer than honoring a configured clamp that asks the robot to cross its
// own watchdog.
inline float safe_backward_theta_limit(float configured_limit,
                                       float watchdog_bwd,
                                       float pitch_trim,
                                       float margin) {
    float safe_limit = watchdog_bwd + pitch_trim - margin;
    if (safe_limit < 0.0f) safe_limit = 0.0f;
    return (configured_limit < safe_limit) ? configured_limit : safe_limit;
}

// Near the backward mechanical boundary, remove only the direct velocity-LQR
// contribution that asks for torque opposite the barrier's recovery direction.
// The contribution remains unchanged inside the barrier and fades linearly to
// zero at the watchdog. Other velocity terms (including one that assists
// recovery) are untouched.
inline float backward_velocity_term_guard(float velocity_term,
                                          float pitch,
                                          float barrier_threshold,
                                          float watchdog_bwd) {
    if (velocity_term >= 0.0f || pitch >= -barrier_threshold) {
        return velocity_term;
    }
    float span = watchdog_bwd - barrier_threshold;
    if (span <= 0.0f) return 0.0f;
    float fade = (watchdog_bwd + pitch) / span;
    if (fade < 0.0f) fade = 0.0f;
    if (fade > 1.0f) fade = 1.0f;
    return velocity_term * fade;
}

inline float slew_toward(float current, float target,
                         float max_rate_per_s, float dt_s) {
    if (max_rate_per_s <= 0.0f || dt_s <= 0.0f) return target;
    float step = max_rate_per_s * dt_s;
    float delta = target - current;
    if (delta > step) delta = step;
    if (delta < -step) delta = -step;
    return current + delta;
}
