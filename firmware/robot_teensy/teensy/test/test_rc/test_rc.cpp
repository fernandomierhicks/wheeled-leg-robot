#include <Arduino.h>
#include <unity.h>

void setUp(void) {}
void tearDown(void) {}

void test_placeholder(void) { TEST_ASSERT_TRUE(true); }

void setup() { Serial.begin(115200); UNITY_BEGIN(); RUN_TEST(test_placeholder); UNITY_END(); }
void loop()  {}
