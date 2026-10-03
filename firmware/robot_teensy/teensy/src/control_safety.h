#pragma once

#include <math.h>

// Balance-trim lookup table: n points evenly spaced over alpha [0,1]
// (point 0 at alpha=0, point n-1 at alpha=1), linearly interpolated. Finds
// the bracketing pair (*lo, *lo+1) and the weight of the upper one.
inline void pitch_trim_bracket(float alpha, int n, int* lo, float* w_hi) {
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    float x = alpha * (float)(n - 1);
    int i = (int)x;
    if (i > n - 2) i = n - 2;
    *lo = i;
    *w_hi = x - (float)i;
}

inline float pitch_trim_table(const float* points, int n, float alpha) {
    int i;
    float w;
    pitch_trim_bracket(alpha, n, &i, &w);
    return points[i] + w * (points[i + 1] - points[i]);
}

// Move the interpolated trim at alpha by exactly delta, changing the two
// bracketing points by the smallest amount that does it (each in proportion
// to its interpolation weight). A point at the current alpha takes all of it;
// midway between two points both move by delta.
inline void pitch_trim_table_nudge(float* points, int n, float alpha, float delta) {
    int i;
    float w;
    pitch_trim_bracket(alpha, n, &i, &w);
    float norm = (1.0f - w) * (1.0f - w) + w * w;  // >= 0.5, never zero
    points[i]     += delta * (1.0f - w) / norm;
    points[i + 1] += delta * w / norm;
}

// Body yaw rate seen by the wheels [rad/s], positive CCW from above (+Z):
// the right wheel rolling forward faster than the left turns the robot left.
// Inputs are firmware-frame wheel speeds (positive = forward on both sides).
inline float wheel_yaw_rate(float vel_l_turns_s, float vel_r_turns_s,
                            float wheel_r_m, float track_m) {
    return (vel_r_turns_s - vel_l_turns_s) * 2.0f * (float)M_PI * wheel_r_m / track_m;
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
