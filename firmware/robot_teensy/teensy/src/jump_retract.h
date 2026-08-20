#pragma once

#include <math.h>

// EXTEND ends with substantial outward hip velocity. Starting an ordinary
// minimum-jerk RETRACT at that instant commands zero velocity on its first
// sample, which is a velocity step even though position is continuous. This
// short blend preserves measured position/velocity at entry, smoothly brakes
// to zero, then hands a stationary turnaround point to the return trajectory.
// Fifteen milliseconds gives 7-8 control samples at 500 Hz. In the matched
// jump it retains the 2.5 cm clearance target; a 40 ms blend kept the wheels
// near the floor long enough to cut that clearance roughly in half.
static constexpr float JUMP_RETRACT_BRAKE_NOMINAL_S = 0.015f;

struct JumpRetractSample {
    float position;
    float velocity;
};

inline JumpRetractSample jump_retract_brake_sample(float start_position,
                                                    float start_velocity,
                                                    float elapsed_s,
                                                    float duration_s) {
    if (duration_s <= 0.0f) return {start_position, 0.0f};
    float u = elapsed_s / duration_s;
    if (u < 0.0f) u = 0.0f;
    if (u > 1.0f) u = 1.0f;

    // v/v0 = 1 - smoothstep(u). Its integral is
    // u - u^3 + 0.5*u^4, so acceleration is also zero at both endpoints.
    const float u2 = u * u;
    const float u3 = u2 * u;
    const float u4 = u3 * u;
    const float velocity_scale = 1.0f - 3.0f * u2 + 2.0f * u3;
    const float position_integral = u - u3 + 0.5f * u4;
    return {
        start_position + start_velocity * duration_s * position_integral,
        start_velocity * velocity_scale,
    };
}

inline float jump_retract_axis_brake_duration(float start_position,
                                              float start_velocity,
                                              float extended_limit,
                                              float seek_direction,
                                              float hardstop_margin,
                                              float nominal_duration_s) {
    // Only outward motion (opposite seek_direction) consumes extended-limit
    // margin. q_stop = q0 + 0.5*v0*T, which gives the largest safe duration.
    if (start_velocity * seek_direction >= 0.0f || fabsf(start_velocity) < 1e-6f) {
        return nominal_duration_s;
    }
    const float safe_extended = extended_limit + seek_direction * hardstop_margin;
    const float remaining = seek_direction * (start_position - safe_extended);
    if (remaining <= 0.0f) return 0.0f;
    const float available_duration = 2.0f * remaining / fabsf(start_velocity);
    return available_duration < nominal_duration_s
        ? available_duration : nominal_duration_s;
}

inline float jump_retract_feedback_gain_scale(float measured_position,
                                              float measured_velocity,
                                              float command_position,
                                              float command_velocity,
                                              float kp,
                                              float kd,
                                              float torque_ceiling_nm) {
    if (torque_ceiling_nm <= 0.0f) return 0.0f;
    const float predicted = kp * (command_position - measured_position)
                          + kd * (command_velocity - measured_velocity);
    const float magnitude = fabsf(predicted);
    if (magnitude <= torque_ceiling_nm || magnitude < 1e-6f) return 1.0f;
    return torque_ceiling_nm / magnitude;
}
