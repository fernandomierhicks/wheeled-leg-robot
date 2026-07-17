#include <Arduino.h>
#include <unity.h>
#include "wheel_motors.h"
#include "../test_led.h"
#include "../test_stubs.h"

void setUp(void) {}
void tearDown(void) {}

// ── Tests ─────────────────────────────────────────────────────────────────────

void test_can_init(void) {
    TEST_ASSERT_TRUE(wheel_motors_init());
}

void test_heartbeats_received(void) {
    uint32_t deadline = millis() + 2000;
    while (millis() < deadline) {
        if (wm_L.last_hb_ms && wm_R.last_hb_ms) break;
        delay(10);
    }
    TEST_ASSERT_MESSAGE(wm_L.last_hb_ms > 0, "No heartbeat from L (node 0) — check CAN wiring / ODrive power");
    TEST_ASSERT_MESSAGE(wm_R.last_hb_ms > 0, "No heartbeat from R (node 1) — check CAN wiring / ODrive power");
}

void test_no_errors(void) {
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(0, wm_L.error, "L ODrive has latched error");
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(0, wm_R.error, "R ODrive has latched error");
}

void test_encoder_feedback(void) {
    uint32_t deadline = millis() + 500;
    while (millis() < deadline) {
        if (wm_L.last_fb_ms && wm_R.last_fb_ms) break;
        delay(10);
    }
    TEST_ASSERT_MESSAGE(wm_L.last_fb_ms > 0, "No encoder feedback from L");
    TEST_ASSERT_MESSAGE(wm_R.last_fb_ms > 0, "No encoder feedback from R");
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static const char* axis_state_name(uint8_t s) {
    switch (s) {
        case 0:  return "UNDEFINED";
        case 1:  return "IDLE";
        case 2:  return "STARTUP_SEQ";
        case 4:  return "MOTOR_CALIB";
        case 6:  return "ENCODER_CALIB";
        case 7:  return "ENCODER_DIR";
        case 8:  return "CLOSED_LOOP";
        case 11: return "ENCODER_HALL";
        default: return "UNKNOWN";
    }
}

// ── Spin state machine ────────────────────────────────────────────────────────

static enum { STOPPED, SPINNING } spin_state = STOPPED;
static uint32_t spin_end_ms = 0;
static constexpr float SPIN_RAD_S = TWO_PI;   // 1 turn/s
static constexpr uint32_t SPIN_MS  = 3000;

static void spin_update() {
    // Drain serial — any character triggers a spin
    if (Serial.available()) {
        while (Serial.available()) Serial.read();
        if (spin_state == STOPPED) {
            wheel_motors_set_mode(WheelMode::VELOCITY);
            wheel_motors_send(SPIN_RAD_S, SPIN_RAD_S);
            spin_end_ms = millis() + SPIN_MS;
            spin_state  = SPINNING;
            Serial.println(">> SPIN START — 1 turn/s for 3 s");
        }
    }

    if (spin_state == SPINNING) {
        wheel_motors_send(SPIN_RAD_S, SPIN_RAD_S);
        if (millis() >= spin_end_ms) {
            wheel_motors_send(0.0f, 0.0f);   // back to zero, stay in closed loop
            spin_state = STOPPED;
            Serial.println(">> SPIN STOP — send any key to spin again");
        }
    }
}

// ── Entry ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    delay(500);
    test_led_begin();

    UNITY_BEGIN();
    RUN_TEST(test_can_init);
    RUN_TEST(test_heartbeats_received);

    // Force IDLE then clear any latched errors from previous runs or USB sessions.
    // Wait 300 ms so the ODrive sends a fresh heartbeat with updated error/state fields.
    wheel_motors_set_mode(WheelMode::IDLE);
    wheel_motors_clear_errors();
    delay(300);

    RUN_TEST(test_no_errors);
    RUN_TEST(test_encoder_feedback);
    test_led_done(UNITY_END());

    wheel_motors_request_vbus();

    // Enter velocity mode at zero — gets axes into CLOSED_LOOP so encoder estimates flow.
    wheel_motors_set_mode(WheelMode::VELOCITY);
    wheel_motors_send(0.0f, 0.0f);

    Serial.println();
    Serial.println("--- Axes in CLOSED_LOOP at 0 vel — encoder data now live ---");
    Serial.println("--- Send any key to spin at 1 turn/s for 3 s            ---");
}

void loop() {
    wheel_motors_poll();
    spin_update();

    static uint32_t last_vbus_req = 0;
    static uint32_t last_print    = 0;
    uint32_t now = millis();

    if (now - last_vbus_req >= 500) {
        wheel_motors_request_vbus();
        last_vbus_req = now;
    }

    if (now - last_print >= 200) {
        last_print = now;
        static uint8_t row = 0;
        if (row % 10 == 0) {
            Serial.println();
            Serial.println("L state          L pos(trn)  | R state          R pos(trn)  | spinning");
            Serial.println("------------------+-----------+------------------+-----------+---------");
        }
        row++;

        char spin_buf[12];
        if (spin_state == SPINNING)
            snprintf(spin_buf, sizeof(spin_buf), "YES (%lus)", (spin_end_ms - now) / 1000 + 1);
        else
            snprintf(spin_buf, sizeof(spin_buf), "no");

        Serial.printf("%-16s  %+10.4f  | %-16s  %+10.4f  | %s\n",
                      axis_state_name(wm_L.axis_state), wm_L.pos_turns,
                      axis_state_name(wm_R.axis_state), wm_R.pos_turns,
                      spin_buf);
    }
}
