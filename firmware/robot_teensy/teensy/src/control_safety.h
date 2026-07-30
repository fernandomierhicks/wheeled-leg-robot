#pragma once

#include <math.h>

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
