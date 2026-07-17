// IMU test — plain setup()/loop() format (matches rgb_led_demo), not PlatformIO's
// Unity test harness. Prints progress live as it goes instead of staying silent
// until the end, then keeps streaming live telemetry after checks finish.

#include <Arduino.h>
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

static uint8_t g_failures = 0;

static void check(bool cond, const char* label) {
    Serial.printf("[%s] %s\n", cond ? "PASS" : "FAIL", label);
    if (!cond) g_failures++;
}

// ── Entry ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}  // wait up to 3 s for USB serial
    Serial.println("\n=== IMU Test ===");
    test_led_begin();

    Serial.println("[IMU] imu_init()...");
    imu_init();

    Serial.println("[IMU] waiting up to 3 s for NOMINAL...");
    spin_imu(3000);
    ImuState s = imu_state();
    Serial.printf("[IMU] State after init: %s\n", state_name(s));
    check(s == ImuState::NOMINAL, "reaches NOMINAL within 3 s");

    Serial.println("[IMU] sampling fresh packets...");
    spin_imu(200);
    float pitch = imu_pitch();
    float roll  = imu_roll();
    float rate  = imu_pitch_rate();
    Serial.printf("[IMU] pitch=%.4f rad  roll=%.4f rad  pitch_rate=%.4f rad/s\n",
                  pitch, roll, rate);
    Serial.printf("[IMU] pitch_mag=%.4f  roll_mag=%.4f  yaw_mag=%.4f\n",
                  imu_pitch_mag(), imu_roll_mag(), imu_yaw_mag());
    check(!isnan(pitch), "pitch is not NaN");
    check(!isnan(roll),  "roll is not NaN");
    check(!isnan(rate),  "pitch_rate is not NaN");
    check(fabsf(pitch) < 1.57f, "pitch within +/-90deg of level");
    check(fabsf(roll)  < 1.57f, "roll within +/-90deg of level");
    check(fabsf(rate)  < 0.5f,  "pitch_rate near zero at rest");

    Serial.println("[IMU] measuring packet loss over a 1.5 s window...");
    spin_imu(1500);
    float loss = imu_packet_loss();
    Serial.printf("[IMU] Packet loss: %.1f%%\n", loss * 100.0f);
    check(loss < 0.10f, "packet loss below 10%");

    spin_imu(50);
    uint32_t age = millis() - imu_last_update_ms();
    Serial.printf("[IMU] Last update age: %u ms\n", age);
    check(age < 50u, "last update fresher than 50 ms");

    Serial.printf("\n=== IMU test done: %u failure(s) ===\n\n", g_failures);
    test_led_done(g_failures);
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
