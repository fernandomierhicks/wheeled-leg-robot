#include <Arduino.h>
#include <unity.h>
#include "../test_led.h"

void setUp(void) {}
void tearDown(void) {}

void test_placeholder(void) { TEST_ASSERT_TRUE(true); }

void setup() {
    Serial.begin(115200);
    test_led_begin();
    UNITY_BEGIN();
    RUN_TEST(test_placeholder);
    test_led_done(UNITY_END());
}
void loop() {}
