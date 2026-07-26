#include <unity.h>

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

int main(int, char**) {
    UNITY_BEGIN();
    RUN_TEST(test_integrates_inside_lean_limits);
    RUN_TEST(test_freezes_when_integral_pushes_farther_forward);
    RUN_TEST(test_freezes_when_integral_pushes_farther_backward);
    RUN_TEST(test_allows_integral_to_unwind_while_output_is_saturated);
    RUN_TEST(test_integral_state_clamp_still_applies);
    return UNITY_END();
}
