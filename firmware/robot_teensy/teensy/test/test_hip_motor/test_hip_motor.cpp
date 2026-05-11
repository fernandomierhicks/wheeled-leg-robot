#include <Arduino.h>
#include <unity.h>
#include "hip_motors.h"
#include "../test_led.h"

void setUp(void) {}
void tearDown(void) {}

// ── Tests ─────────────────────────────────────────────────────────────────────

void test_can_init(void) {
    TEST_ASSERT_TRUE(hip_motors_init());
}

// Motor only replies when commanded, so pump zero-torque spring commands while
// waiting for first feedback from each axis.
// Startup sequence mirrors the proven springness sketch: 10 ms settle after
// enter_mit, then an immediate first command before the polling loop.
void test_feedback_received(void) {
    hip_motors_enter_mit();
    delay(20);  // 10 ms per motor settle, matching springness sketch
    hip_motors_send(0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    delay(5);

    uint32_t deadline = millis() + 2000;
    while (millis() < deadline) {
        hip_motors_send(0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
        delay(10);
        if (hm_L.last_fb_ms && hm_R.last_fb_ms) break;
    }
    TEST_ASSERT_MESSAGE(hm_L.last_fb_ms > 0, "No reply from L (id=11) — check CAN wiring / motor power / common ground");
    TEST_ASSERT_MESSAGE(hm_R.last_fb_ms > 0, "No reply from R (id=12) — check CAN wiring / motor power / common ground");
}

// ── Move state machine ────────────────────────────────────────────────────────

static enum { HOLDING, MOVING } move_state = HOLDING;
static uint32_t move_end_ms = 0;

static constexpr float MOVE_POS_RAD  = 0.3f;   // step target [rad]
static constexpr float HOLD_KP       = 5.0f;   // N·m/rad
static constexpr float HOLD_KD       = 0.5f;   // N·m·s/rad
static constexpr uint32_t MOVE_MS    = 3000;

static void move_update() {
    if (Serial.available()) {
        while (Serial.available()) Serial.read();
        if (move_state == HOLDING) {
            move_end_ms = millis() + MOVE_MS;
            move_state  = MOVING;
            Serial.println(">> MOVE START — stepping to +0.3 rad for 3 s");
        }
    }

    float target = (move_state == MOVING) ? MOVE_POS_RAD : 0.0f;

    if (move_state == MOVING && millis() >= move_end_ms) {
        move_state = HOLDING;
        Serial.println(">> MOVE STOP — holding at 0 rad — send any key to move again");
    }

    hip_motors_send(target, 0.0f, HOLD_KP, HOLD_KD, 0.0f,
                    target, 0.0f, HOLD_KP, HOLD_KD, 0.0f);
}

// ── Entry ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    delay(500);
    test_led_begin();

    UNITY_BEGIN();
    RUN_TEST(test_can_init);
    RUN_TEST(test_feedback_received);
    test_led_done(UNITY_END());

    Serial.println();
    Serial.println("--- MIT mode active — holding at 0 rad (kp=5, kd=0.5) ---");
    Serial.println("--- Send any key to step both motors to +0.3 rad for 3 s ---");
}

void loop() {
    static uint32_t last_send = 0;
    uint32_t now = millis();

    hip_motors_poll();

    if (now - last_send >= 10) {
        last_send = now;
        move_update();
    }

    static uint32_t last_print = 0;

    if (now - last_print >= 100) {
        last_print = now;
        static uint8_t row = 0;
        if (row % 15 == 0) {
            Serial.println();
            Serial.println("L pos(rad)  L vel(r/s)  L cur(A)  | R pos(rad)  R vel(r/s)  R cur(A)  | state");
            Serial.println("-----------+-----------+---------+--+-----------+-----------+---------+--------");
        }
        row++;

        const char* state_str = (move_state == MOVING) ? "MOVING" : "hold";
        Serial.printf("%+9.4f   %+9.4f  %+7.3f  | %+9.4f   %+9.4f  %+7.3f  | %s\n",
                      hm_L.pos_rad, hm_L.vel_rad_s, hm_L.current_A,
                      hm_R.pos_rad, hm_R.vel_rad_s, hm_R.current_A,
                      state_str);
    }
}
