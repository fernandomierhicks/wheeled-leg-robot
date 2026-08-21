// Native tests for the coordinated-turn lean setpoint.
//
// The sign is the whole reason these exist. Lean the wrong way in a turn and
// the robot falls harder and faster than it would with no feature at all, and
// the failure only shows up at the one moment you cannot afford it. Everything
// else here -- clamps, deadband, authority -- is cheap to cover once the
// harness is standing.
//
//   pio test -e windows_lean     (Windows, PlatformIO's MinGW)
//   pio test -e native_lean      (Linux/macOS with a system g++)

#include <unity.h>
#include "lean_turn.h"

void setUp(void) {}
void tearDown(void) {}

static constexpr float GAIN = 1.0f;
static constexpr float MAXR = 0.35f;
static constexpr float MIN_V = 0.15f;

// ── the sign, from every direction ───────────────────────────────────────────

void test_left_turn_leans_left(void) {
    // omega > 0 is a LEFT turn (stick left -> yaws left, right-hand rule about
    // +Z). Leaning left is NEGATIVE roll, because positive roll lifts the left
    // side. So a left turn while moving forward must produce negative roll.
    const float lean = lean_turn_setpoint(1.0f, 2.0f, GAIN, MAXR, MIN_V);
    TEST_ASSERT_TRUE_MESSAGE(lean < 0.0f,
        "forward + left turn must command NEGATIVE roll (lean left)");
}

void test_right_turn_leans_right(void) {
    const float lean = lean_turn_setpoint(1.0f, -2.0f, GAIN, MAXR, MIN_V);
    TEST_ASSERT_TRUE_MESSAGE(lean > 0.0f,
        "forward + right turn must command POSITIVE roll (lean right)");
}

void test_reversing_flips_the_lean(void) {
    // Driving backwards through a left-hand yaw rate curves the other way, so
    // the centripetal force reverses and the lean must too. v * omega carries
    // this for free -- the test exists so a future "optimisation" to
    // fabsf(v) does not quietly break reverse.
    const float fwd = lean_turn_setpoint( 1.0f, 2.0f, GAIN, MAXR, MIN_V);
    const float rev = lean_turn_setpoint(-1.0f, 2.0f, GAIN, MAXR, MIN_V);
    TEST_ASSERT_TRUE(fwd < 0.0f);
    TEST_ASSERT_TRUE_MESSAGE(rev > 0.0f,
        "reversing through the same yaw rate must reverse the lean");
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -fwd, rev);
}

// ── magnitude ────────────────────────────────────────────────────────────────

void test_matches_the_coordinated_turn_angle(void) {
    // atan(v*w/g). 1.0 m/s at 1.5 rad/s -> atan(1.5/9.81) = 0.1517 rad = 8.7 deg.
    const float lean = lean_turn_setpoint(1.0f, 1.5f, GAIN, MAXR, MIN_V);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -0.1517f, lean);
}

void test_uses_atan_not_small_angle(void) {
    // At the robot's own limits (2 m/s, 4 rad/s) the argument is 0.8155, where
    // the small-angle approximation is ~12% high -- and high means over-lean,
    // in the regime where over-leaning is least recoverable.
    const float big = lean_turn_setpoint(2.0f, 4.0f, GAIN, 2.0f, MIN_V);
    TEST_ASSERT_TRUE_MESSAGE(fabsf(big) < 0.8155f,
        "must be atan(x), which is strictly less than x for x > 0");
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -0.6842f, big);
}

// ── gates and clamps ─────────────────────────────────────────────────────────

void test_gain_zero_is_inert(void) {
    // lean_gain defaults to 0 so the feature does nothing until deliberately
    // raised, the same pattern jump_enable uses.
    TEST_ASSERT_EQUAL_FLOAT(0.0f, lean_turn_setpoint(2.0f, 4.0f, 0.0f, MAXR, MIN_V));
}

void test_speed_deadband(void) {
    // Spinning in place must not command lean. v*omega would be near zero
    // anyway, but an explicit floor stops jitter at a standstill from dithering
    // the roll setpoint.
    TEST_ASSERT_EQUAL_FLOAT(0.0f, lean_turn_setpoint(0.0f,  4.0f, GAIN, MAXR, MIN_V));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, lean_turn_setpoint(0.1f,  4.0f, GAIN, MAXR, MIN_V));
    TEST_ASSERT_TRUE(lean_turn_setpoint(0.2f, 4.0f, GAIN, MAXR, MIN_V) != 0.0f);
}

void test_zero_yaw_rate_is_level(void) {
    TEST_ASSERT_EQUAL_FLOAT(0.0f, lean_turn_setpoint(2.0f, 0.0f, GAIN, MAXR, MIN_V));
}

void test_clamped_both_ways(void) {
    // roll_watchdog_limit is 0.35 rad. lean_max_rad has to keep the feature
    // well inside the watchdog it exists to avoid tripping.
    const float lim = 0.20f;
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -lim,
        lean_turn_setpoint( 2.0f,  4.0f, GAIN, lim, MIN_V));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f,  lim,
        lean_turn_setpoint( 2.0f, -4.0f, GAIN, lim, MIN_V));
}

void test_gain_scales(void) {
    const float full = lean_turn_setpoint(1.0f, 1.5f, 1.0f, MAXR, MIN_V);
    const float half = lean_turn_setpoint(1.0f, 1.5f, 0.5f, MAXR, MIN_V);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, full * 0.5f, half);
}

// ── leg authority ────────────────────────────────────────────────────────────

void test_authority_collapses_at_stroke_ends(void) {
    // The roll controller clamps the differential offset to min(t, 1-t)*span,
    // which is zero at full crouch and full extension. The clamp is safe but
    // silent: the robot simply will not lean there. This is what lets the HUD
    // say so.
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, lean_turn_authority(0.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, lean_turn_authority(1.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, lean_turn_authority(0.5f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.5f, lean_turn_authority(0.25f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.5f, lean_turn_authority(0.75f));
}

void test_authority_clamps_out_of_range(void) {
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, lean_turn_authority(-0.5f));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, lean_turn_authority( 1.5f));
}

// ── the numbers from the design note ─────────────────────────────────────────

void test_design_table(void) {
    // These are the figures the feature was justified with, so they are worth
    // pinning: if the maths drifts, the argument for the feature drifts too.
    struct { float v, w, deg; } cases[] = {
        {0.5f, 1.0f,  2.92f},
        {1.0f, 1.5f,  8.69f},
        {1.5f, 2.0f, 17.02f},
        {2.0f, 4.0f, 39.20f},
    };
    for (auto& c : cases) {
        const float lean = lean_turn_setpoint(c.v, c.w, 1.0f, 2.0f, MIN_V);
        const float deg = -lean * 57.29578f;
        TEST_ASSERT_FLOAT_WITHIN(0.05f, c.deg, deg);
    }
}

int main(int, char**) {
    UNITY_BEGIN();
    RUN_TEST(test_left_turn_leans_left);
    RUN_TEST(test_right_turn_leans_right);
    RUN_TEST(test_reversing_flips_the_lean);
    RUN_TEST(test_matches_the_coordinated_turn_angle);
    RUN_TEST(test_uses_atan_not_small_angle);
    RUN_TEST(test_gain_zero_is_inert);
    RUN_TEST(test_speed_deadband);
    RUN_TEST(test_zero_yaw_rate_is_level);
    RUN_TEST(test_clamped_both_ways);
    RUN_TEST(test_gain_scales);
    RUN_TEST(test_authority_collapses_at_stroke_ends);
    RUN_TEST(test_authority_clamps_out_of_range);
    RUN_TEST(test_design_table);
    return UNITY_END();
}
