#include <unity.h>

#include "control_safety.h"
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
    RUN_TEST(test_backward_velocity_guard_fades_only_opposing_term);
    RUN_TEST(test_slew_toward_limits_both_directions_and_can_be_disabled);
    RUN_TEST(test_wheel_glitch_filter_passes_plausible_change);
    RUN_TEST(test_wheel_glitch_filter_rejects_impossible_jump);
    RUN_TEST(test_wheel_glitch_filter_fails_open_after_max_consecutive);
    RUN_TEST(test_wheel_glitch_filter_allowance_scales_with_gap);
    RUN_TEST(test_wheel_glitch_filter_disabled_passes_everything);
    return UNITY_END();
}
