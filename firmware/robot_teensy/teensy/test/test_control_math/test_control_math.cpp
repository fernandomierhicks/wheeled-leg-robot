#include <unity.h>

#include "control_safety.h"
#include "jump_landing.h"
#include "jump_retract.h"
#include "standup_safety.h"
#include "wheel_safety.h"
#include "velocity_pi_anti_windup.h"

void setUp() {}
void tearDown() {}

static void test_integrates_inside_lean_limits() {
    const float next = velocity_pi_integral_step(
        0.10f, 0.50f, 0.02f, 1.0f, 0.20f, 0.05f, 0.30f, 0.20f);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.11f, next);
}

static void test_freezes_when_integral_pushes_farther_forward() {
    const float next = velocity_pi_integral_step(
        0.40f, 1.00f, 0.02f, 1.0f, 0.50f, 0.20f, 0.30f, 0.20f);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.40f, next);
}

static void test_freezes_when_integral_pushes_farther_backward() {
    const float next = velocity_pi_integral_step(
        -0.30f, -1.00f, 0.02f, 1.0f, 0.50f, -0.10f, 0.30f, 0.20f);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -0.30f, next);
}

static void test_allows_integral_to_unwind_while_output_is_saturated() {
    const float next = velocity_pi_integral_step(
        0.60f, -1.00f, 0.02f, 1.0f, 0.50f, 0.20f, 0.30f, 0.20f);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.58f, next);
}

static void test_integral_state_clamp_still_applies() {
    const float next = velocity_pi_integral_step(
        0.99f, 1.00f, 0.02f, 1.0f, 0.10f, 0.0f, 1.0f, 1.0f);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, next);
}

static void test_backward_theta_limit_accounts_for_negative_trim_and_margin() {
    const float safe = safe_backward_theta_limit(
        0.2617994f,  // configured 15 deg
        0.3490658f,  // watchdog 20 deg
       -0.1308997f,  // trim -7.5 deg
        0.0523599f); // margin 3 deg
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.1658062f, safe); // 9.5 deg
}

static void test_backward_theta_limit_preserves_already_safe_configuration() {
    const float safe = safe_backward_theta_limit(
        0.1396263f, 0.3490658f, -0.1308997f, 0.0523599f);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.1396263f, safe);
}

static void test_pitch_trim_curve_preserves_endpoints_and_linear_default() {
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, -0.14f, scheduled_pitch_trim(-0.14f, 0.02f, -0.08f, 0.0f));
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, 0.02f, scheduled_pitch_trim(-0.14f, 0.02f, -0.08f, 1.0f));
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, -0.10f, scheduled_pitch_trim(-0.14f, 0.02f, 0.0f, 0.25f));
}

static void test_pitch_trim_curve_contributes_one_quarter_at_midpoint() {
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, -0.08f, scheduled_pitch_trim(-0.14f, 0.02f, -0.08f, 0.5f));
}

static void test_backward_velocity_guard_fades_only_opposing_term() {
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, -0.05f,
        backward_velocity_term_guard(-0.05f, -0.20f, 0.21f, 0.35f));
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, -0.025f,
        backward_velocity_term_guard(-0.05f, -0.28f, 0.21f, 0.35f));
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, 0.04f,
        backward_velocity_term_guard(0.04f, -0.34f, 0.21f, 0.35f));
}

static void test_slew_toward_limits_both_directions_and_can_be_disabled() {
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0004f, slew_toward(0.0f, 1.0f, 0.2f, 0.002f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.9996f, slew_toward(1.0f, 0.0f, 0.2f, 0.002f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.7f, slew_toward(0.1f, 0.7f, 0.0f, 0.002f));
}

static void test_standup_hip_gate_requires_both_positions_and_velocities() {
    const float tol = STANDUP_HIP_POS_TOL_RAD;
    TEST_ASSERT_TRUE(standup_hips_in_settle_band(
        1.01f, 0.10f, -1.01f, -0.10f, 1.0f, -1.0f, tol));
    TEST_ASSERT_FALSE(standup_hips_in_settle_band(
        1.04f, 0.10f, -1.01f, -0.10f, 1.0f, -1.0f, tol));
    TEST_ASSERT_FALSE(standup_hips_in_settle_band(
        1.01f, 0.10f, -1.04f, -0.10f, 1.0f, -1.0f, tol));
    TEST_ASSERT_FALSE(standup_hips_in_settle_band(
        1.01f, 0.21f, -1.01f, -0.10f, 1.0f, -1.0f, tol));
    TEST_ASSERT_FALSE(standup_hips_in_settle_band(
        1.01f, 0.10f, -1.01f, -0.21f, 1.0f, -1.0f, tol));
}

// Regression: a P-hold with hip_running_tff_ret = -2.5 and hip_running_kp = 25
// rests 0.1 rad off target on unloaded legs, so a fixed 2 deg gate could never
// be met off the ground — STANDING_UP faulted every arm at the STIFFEN gate.
static void test_standup_pos_tol_allows_the_feedforward_offset() {
    const float tol = standup_hip_pos_tol(25.0f, -2.5f);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, STANDUP_HIP_POS_TOL_RAD + 0.1f, tol);
    TEST_ASSERT_TRUE(standup_hips_in_settle_band(
        -0.091f, -0.044f, -0.093f, 0.015f, 0.0f, 0.0f, tol));
    // Still catches a leg that never made it to the crouch pose.
    TEST_ASSERT_FALSE(standup_hips_in_settle_band(
        -0.40f, 0.0f, -0.40f, 0.0f, 0.0f, 0.0f, tol));
    // No feedforward, or no hold at all, leaves the strict band in place.
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, STANDUP_HIP_POS_TOL_RAD, standup_hip_pos_tol(25.0f, 0.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, STANDUP_HIP_POS_TOL_RAD, standup_hip_pos_tol(0.0f, -2.5f));
    // A badly scaled pair is capped, not allowed to disable the gate.
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, STANDUP_HIP_POS_TOL_MAX_RAD, standup_hip_pos_tol(1.0f, -2.5f));
}

// The CROUCH gate is velocity-only by design: the excursion runs at a fraction
// of the running hip stiffness, so the legs are still sagging out of the strict
// position band when the motion has finished.
// A catch running under the raised standup_vel_limit must not hand off with the
// wheels still above RUNNING's governor: RUNNING zeroes their torque on tick one
// and the runaway watchdog trips at 2x it.
static void test_standup_handoff_requires_wheels_inside_running_limit() {
    TEST_ASSERT_TRUE(standup_wheels_ready_for_handoff(2.9f, -2.9f, 3.0f));
    TEST_ASSERT_TRUE(standup_wheels_ready_for_handoff(3.0f, 3.0f, 3.0f));
    TEST_ASSERT_FALSE(standup_wheels_ready_for_handoff(3.1f, 0.0f, 3.0f));
    TEST_ASSERT_FALSE(standup_wheels_ready_for_handoff(0.0f, -6.5f, 3.0f));
}

static void test_standup_crouch_gate_is_velocity_only() {
    TEST_ASSERT_TRUE(standup_hips_quiet(0.10f, -0.10f));
    TEST_ASSERT_TRUE(standup_hips_quiet(0.20f, 0.20f));
    TEST_ASSERT_FALSE(standup_hips_quiet(0.21f, 0.0f));
    TEST_ASSERT_FALSE(standup_hips_quiet(0.0f, -0.21f));
}

static void test_standup_stiffen_ramp_spans_crouch_fraction_to_full() {
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.5f,  standup_stiffen_scale(0.0f, 2.0f, 0.5f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.75f, standup_stiffen_scale(1.0f, 2.0f, 0.5f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f,  standup_stiffen_scale(2.0f, 2.0f, 0.5f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f,  standup_stiffen_scale(9.0f, 2.0f, 0.5f));
    // Zero duration means "already stiff", not a divide by zero.
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f,  standup_stiffen_scale(0.0f, 0.0f, 0.5f));
}

static void test_standup_minimum_jerk_trajectory_has_smooth_endpoints() {
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, standup_min_jerk_position(0.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.5f, standup_min_jerk_position(0.5f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, standup_min_jerk_position(1.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, standup_min_jerk_rate(0.0f, 1.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.875f, standup_min_jerk_rate(0.5f, 1.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, standup_min_jerk_rate(1.0f, 1.0f));
}

static void test_standup_target_must_match_backoff_used_by_calibration() {
    TEST_ASSERT_TRUE(standup_target_matches_configured_backoff(
        -0.2617994f, 0.2617994f, 0.2617994f, 1.0f, -1.0f));
    TEST_ASSERT_FALSE(standup_target_matches_configured_backoff(
        -0.0872665f, 0.0872665f, 0.2617994f, 1.0f, -1.0f));
}

static void test_jump_landing_gyro_needs_two_fresh_events() {
    JumpLandingDetector d{};
    jump_landing_reset(&d, 2000, 2000, 0.0f, 0.0f, 0.0f);
    TEST_ASSERT_FALSE(jump_landing_update(
        &d, 2110, 2110, 0.0f, 1.6f, 0.0f, 0.10f, 2.5f));
    TEST_ASSERT_FALSE(d.landed);  // one large sample can be corruption

    // A held sensor value spans some 500 Hz control ticks. Even if callers
    // present different rates, an unchanged report timestamp cannot add an
    // event or make the one physical sample look like two.
    TEST_ASSERT_FALSE(jump_landing_update(
        &d, 2112, 2110, 0.8f, 2.8f, 0.0f, 0.10f, 2.5f));
    TEST_ASSERT_EQUAL_UINT8(1, d.gyro_event_count);

    TEST_ASSERT_TRUE(jump_landing_update(
        &d, 2116, 2116, 0.8f, 2.8f, 0.0f, 0.10f, 2.5f));
    TEST_ASSERT_TRUE(d.landed);
}

static void test_jump_landing_blanking_rejects_launch_impulse() {
    JumpLandingDetector d{};
    jump_landing_reset(&d, 3000, 3000, 0.0f, 0.0f, 0.0f);
    TEST_ASSERT_FALSE(jump_landing_update(
        &d, 3020, 3020, 0.0f, -2.0f, 0.0f, 0.10f, 2.5f));
    TEST_ASSERT_FALSE(jump_landing_update(
        &d, 3026, 3026, 1.0f, -3.5f, 0.0f, 0.10f, 2.5f));
    // By the time blanking expires, the launch events have aged out of the
    // 12 ms gyro window and cannot trigger a delayed false landing.
    TEST_ASSERT_FALSE(jump_landing_update(
        &d, 3100, 3100, 1.0f, -3.5f, 0.0f, 0.10f, 2.5f));
}

static void test_jump_retract_brake_preserves_entry_and_stops_smoothly() {
    const JumpRetractSample entry = jump_retract_brake_sample(
        -1.20f, -7.0f, 0.0f, 0.015f);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -1.20f, entry.position);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -7.0f, entry.velocity);

    const JumpRetractSample stop = jump_retract_brake_sample(
        -1.20f, -7.0f, 0.015f, 0.015f);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -1.2525f, stop.position);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, stop.velocity);
}

static void test_jump_retract_brake_shortens_before_extended_margin() {
    const float duration = jump_retract_axis_brake_duration(
        -1.37f, -7.0f, -1.4835f, 1.0f, 0.0873f, 0.015f);
    const JumpRetractSample stop = jump_retract_brake_sample(
        -1.37f, -7.0f, duration, duration);
    TEST_ASSERT_TRUE(duration < 0.015f);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -1.3962f, stop.position);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, stop.velocity);
}

static void test_jump_retract_feedback_scale_enforces_command_ceiling() {
    const float scale = jump_retract_feedback_gain_scale(
        -1.20f, -7.0f, -1.18f, -5.0f, 120.0f, 1.0f, 3.0f);
    const float predicted = scale * (120.0f * 0.02f + 1.0f * 2.0f);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 3.0f, predicted);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f,
        jump_retract_feedback_gain_scale(
            -1.20f, -7.0f, -1.20f, -7.0f, 120.0f, 1.0f, 3.0f));
}

static void test_wheel_glitch_filter_passes_plausible_change() {
    uint8_t run = 0;
    // 1500 turns/s^2 over one 2 ms tick allows 3.0 turns/s of change.
    const float v = wheel_vel_glitch_filter(2.5f, 0.0f, 1500.0f, 0.002f, 3, &run);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.5f, v);
    TEST_ASSERT_EQUAL_UINT8(0, run);
}

static void test_wheel_glitch_filter_rejects_impossible_jump() {
    uint8_t run = 0;
    const float v = wheel_vel_glitch_filter(-5.85f, -0.20f, 1500.0f, 0.002f, 3, &run);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -0.20f, v);  // held last-good
    TEST_ASSERT_EQUAL_UINT8(1, run);
}

static void test_wheel_glitch_filter_fails_open_after_max_consecutive() {
    uint8_t run = 0;
    float v = -0.20f;
    for (int i = 0; i < 3; ++i) {
        v = wheel_vel_glitch_filter(-8.0f, v, 1500.0f, 0.002f, 3, &run);
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, -0.20f, v);
    }
    TEST_ASSERT_EQUAL_UINT8(3, run);
    // 4th consecutive implausible sample: reality wins so a real runaway can
    // never be permanently suppressed.
    v = wheel_vel_glitch_filter(-8.0f, v, 1500.0f, 0.002f, 3, &run);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -8.0f, v);
    TEST_ASSERT_EQUAL_UINT8(0, run);
}

static void test_wheel_glitch_filter_allowance_scales_with_gap() {
    uint8_t run = 0;
    // Same 6 turns/s jump that is rejected across one 2 ms tick is accepted
    // after a 20 ms feedback gap (1500 * 0.020 = 30 turns/s allowance).
    const float v = wheel_vel_glitch_filter(6.0f, 0.0f, 1500.0f, 0.020f, 3, &run);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 6.0f, v);
    TEST_ASSERT_EQUAL_UINT8(0, run);
}

static void test_wheel_glitch_filter_disabled_passes_everything() {
    uint8_t run = 0;
    float v = wheel_vel_glitch_filter(99.0f, 0.0f, 0.0f, 0.002f, 3, &run);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 99.0f, v);
    // No usable dt (first sample after boot) must also pass through.
    v = wheel_vel_glitch_filter(99.0f, 0.0f, 1500.0f, 0.0f, 3, &run);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 99.0f, v);
    TEST_ASSERT_EQUAL_UINT8(0, run);
}

int main(int, char**) {
    UNITY_BEGIN();
    RUN_TEST(test_integrates_inside_lean_limits);
    RUN_TEST(test_freezes_when_integral_pushes_farther_forward);
    RUN_TEST(test_freezes_when_integral_pushes_farther_backward);
    RUN_TEST(test_allows_integral_to_unwind_while_output_is_saturated);
    RUN_TEST(test_integral_state_clamp_still_applies);
    RUN_TEST(test_backward_theta_limit_accounts_for_negative_trim_and_margin);
    RUN_TEST(test_backward_theta_limit_preserves_already_safe_configuration);
    RUN_TEST(test_pitch_trim_curve_preserves_endpoints_and_linear_default);
    RUN_TEST(test_pitch_trim_curve_contributes_one_quarter_at_midpoint);
    RUN_TEST(test_backward_velocity_guard_fades_only_opposing_term);
    RUN_TEST(test_slew_toward_limits_both_directions_and_can_be_disabled);
    RUN_TEST(test_standup_hip_gate_requires_both_positions_and_velocities);
    RUN_TEST(test_standup_pos_tol_allows_the_feedforward_offset);
    RUN_TEST(test_standup_handoff_requires_wheels_inside_running_limit);
    RUN_TEST(test_standup_crouch_gate_is_velocity_only);
    RUN_TEST(test_standup_stiffen_ramp_spans_crouch_fraction_to_full);
    RUN_TEST(test_standup_minimum_jerk_trajectory_has_smooth_endpoints);
    RUN_TEST(test_standup_target_must_match_backoff_used_by_calibration);
    RUN_TEST(test_jump_landing_gyro_needs_two_fresh_events);
    RUN_TEST(test_jump_landing_blanking_rejects_launch_impulse);
    RUN_TEST(test_jump_retract_brake_preserves_entry_and_stops_smoothly);
    RUN_TEST(test_jump_retract_brake_shortens_before_extended_margin);
    RUN_TEST(test_jump_retract_feedback_scale_enforces_command_ceiling);
    RUN_TEST(test_wheel_glitch_filter_passes_plausible_change);
    RUN_TEST(test_wheel_glitch_filter_rejects_impossible_jump);
    RUN_TEST(test_wheel_glitch_filter_fails_open_after_max_consecutive);
    RUN_TEST(test_wheel_glitch_filter_allowance_scales_with_gap);
    RUN_TEST(test_wheel_glitch_filter_disabled_passes_everything);
    return UNITY_END();
}
