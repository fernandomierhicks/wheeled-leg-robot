#pragma once

#include <math.h>

// Conditional-integration anti-windup for the velocity PI's lean command.
//
// The proportional and acceleration-feedforward terms are supplied together
// as theta_non_integral. The candidate integral is rejected only when its
// contribution would push an already-out-of-range lean request farther beyond
// the active asymmetric clamp. An update that reduces the stored integral is
// always accepted so the controller can unwind promptly after an overshoot or
// command release.
inline float velocity_pi_integral_step(
    float integral,
    float velocity_error,
    float dt,
    float integral_max,
    float ki,
    float theta_non_integral,
    float theta_max_fwd,
    float theta_max_bwd
) {
    float candidate = integral + velocity_error * dt;
    if (candidate >  integral_max) candidate =  integral_max;
    if (candidate < -integral_max) candidate = -integral_max;

    const float integral_theta_delta = ki * (candidate - integral);
    const float theta_candidate = theta_non_integral + ki * candidate;
    const bool unwinding = fabsf(candidate) < fabsf(integral);
    const bool pushes_further_fwd =
        theta_candidate > theta_max_fwd && integral_theta_delta > 0.0f;
    const bool pushes_further_bwd =
        theta_candidate < -theta_max_bwd && integral_theta_delta < 0.0f;

    if (!unwinding && (pushes_further_fwd || pushes_further_bwd)) {
        return integral;
    }
    return candidate;
}
