#include <Arduino.h>
#include <unity.h>
#include "IMU.h"
#include "../test_led.h"

// ── Helpers ───────────────────────────────────────────────────────────────────

static void spin_imu(uint32_t duration_ms) {
    uint32_t end = millis() + duration_ms;
    while (millis() < end) {
        imu_update();
        delayMicroseconds(500);  // ~2 kHz poll — faster than 400 Hz report rate
    }
}

static const char* state_name(ImuState s) {
    switch (s) {
        case ImuState::NOT_READY:    return "NOT_READY";
        case ImuState::INITIALIZING: return "INITIALIZING";
        case ImuState::NOMINAL:      return "NOMINAL";
        case ImuState::DEGRADED:     return "DEGRADED";
        case ImuState::ERROR:        return "ERROR";
        default:                     return "UNKNOWN";
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// Spin up to 3 s for BNO086 to reach NOMINAL.
void test_imu_reaches_nominal(void) {
    imu_init();
    spin_imu(3000);
    ImuState s = imu_state();
    Serial.printf("[IMU] State after init: %s\n", state_name(s));
    TEST_ASSERT_EQUAL_INT((int)ImuState::NOMINAL, (int)s);
}

// After NOMINAL: pitch/roll/rate should not be NaN and should be plausible
// (board is sitting still on a desk — expect |pitch| < 1.57 rad).
void test_imu_values_plausible(void) {
    spin_imu(200);  // get a fresh batch of packets

    float pitch = imu_pitch();
    float roll  = imu_roll();
    float rate  = imu_pitch_rate();

    Serial.printf("[IMU] pitch=%.4f rad  roll=%.4f rad  pitch_rate=%.4f rad/s\n",
                  pitch, roll, rate);
    Serial.printf("[IMU] pitch_mag=%.4f  roll_mag=%.4f  yaw_mag=%.4f\n",
                  imu_pitch_mag(), imu_roll_mag(), imu_yaw_mag());

    TEST_ASSERT_FALSE_MESSAGE(isnan(pitch), "pitch is NaN");
    TEST_ASSERT_FALSE_MESSAGE(isnan(roll),  "roll is NaN");
    TEST_ASSERT_FALSE_MESSAGE(isnan(rate),  "pitch_rate is NaN");

    // Plausibility: sitting still the angles should be within ±π/2 of level
    TEST_ASSERT_TRUE_MESSAGE(fabsf(pitch) < 1.57f, "pitch out of range");
    TEST_ASSERT_TRUE_MESSAGE(fabsf(roll)  < 1.57f, "roll out of range");
    // Gyro should be near zero at rest (< 0.5 rad/s)
    TEST_ASSERT_TRUE_MESSAGE(fabsf(rate)  < 0.5f,  "pitch_rate not near zero at rest");
}

// Packet loss should be below 10% after a full 1.5 s window.
void test_imu_packet_loss_low(void) {
    spin_imu(1500);  // cover at least one loss-window cycle (window = 1 s)
    float loss = imu_packet_loss();
    Serial.printf("[IMU] Packet loss: %.1f%%\n", loss * 100.0f);
    TEST_ASSERT_TRUE_MESSAGE(loss < 0.10f, "packet loss >= 10%");
}

// last_update_ms must be recent (< 50 ms ago) — sensor is still alive.
void test_imu_update_fresh(void) {
    spin_imu(50);
    uint32_t age = millis() - imu_last_update_ms();
    Serial.printf("[IMU] Last update age: %u ms\n", age);
    TEST_ASSERT_LESS_THAN(50u, age);
}

// ── Entry ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    delay(1000);  // let serial port settle before Unity output starts
    test_led_begin();

    UNITY_BEGIN();
    RUN_TEST(test_imu_reaches_nominal);
    RUN_TEST(test_imu_values_plausible);
    RUN_TEST(test_imu_packet_loss_low);
    RUN_TEST(test_imu_update_fresh);
    test_led_done(UNITY_END());
}

void loop() {
    imu_update();

    static uint32_t last_print = 0;
    if (millis() - last_print >= 100) {
        last_print = millis();
        Serial.printf("state=%-10s  pitch=%+7.3f  roll=%+7.3f  yaw=%+7.3f  "
                      "dPitch=%+7.3f  loss=%.1f%%\n",
                      state_name(imu_state()),
                      imu_pitch(), imu_roll(), imu_yaw(),
                      imu_pitch_rate(), imu_packet_loss() * 100.0f);
    }
}
