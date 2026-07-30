#pragma once

#include <math.h>
#include <stdint.h>

// Reject a wheel-velocity sample that moved further since the last accepted
// one than the motors could physically have produced.
//
// Motivation (2026-07-28 bench log): the ODrive encoder feed developed
// escalating single-sample corruption — jumps of up to 6.9 turns/s between
// consecutive 50 Hz telemetry samples, where the free-spinning wheel's own
// inertia at MOTOR_TRQ_MAX caps it near 1300 turns/s². Those spikes fed the
// runaway watchdog (and the balance loop's velocity term) directly, and one
// of them tripped FAULT_WHEEL_RUNAWAY mid-flight while the robot was
// otherwise balancing fine. This is defence-in-depth for that class of
// electrical/CAN noise, NOT a substitute for fixing the wiring.
//
// Deliberately fail-open: after `max_consecutive` rejections in a row the
// next sample is accepted regardless. A genuine hard acceleration (or a
// legitimately stale `last_good` after a feedback dropout) therefore always
// gets through within a bounded time, so the filter can never permanently
// blind the runaway watchdog to a real runaway.
//
// `dt_s` is the time since the last ACCEPTED sample, so the allowance grows
// across a gap in encoder frames instead of falsely rejecting the first
// sample after one.
inline float wheel_vel_glitch_filter(float new_vel, float last_good,
                                     float max_accel_turns_s2, float dt_s,
                                     uint8_t max_consecutive,
                                     uint8_t* consecutive_rejects) {
    // Disabled, or no usable time base to judge plausibility against.
    if (max_accel_turns_s2 <= 0.0f || dt_s <= 0.0f) {
        if (consecutive_rejects) *consecutive_rejects = 0;
        return new_vel;
    }
    const float max_delta = max_accel_turns_s2 * dt_s;
    if (fabsf(new_vel - last_good) <= max_delta) {
        if (consecutive_rejects) *consecutive_rejects = 0;
        return new_vel;
    }
    // Implausible. Hold the last good value, but only for a bounded run.
    if (consecutive_rejects && *consecutive_rejects < max_consecutive) {
        (*consecutive_rejects)++;
        return last_good;
    }
    if (consecutive_rejects) *consecutive_rejects = 0;
    return new_vel;
}
