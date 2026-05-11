#pragma once
// Non-intrusive RGB LED status for unit tests.
// Uses IntervalTimer so animations run without needing update() in test loops.
//
//   test_led_begin()       — call once before UNITY_BEGIN()
//   test_led_done(n)       — call with UNITY_END() return value
//
// While tests run : fast white strobe  (proof of life)
// All pass        : slow green pulse
// Any failure     : fast red blink

#include <Arduino.h>
#include <IntervalTimer.h>
#include "config.h"
#include "RgbLed.h"

namespace _tled {
    inline RgbLed        led(PIN_LED_R, PIN_LED_G, PIN_LED_B);
    inline IntervalTimer timer;
    inline void tick() { led.update(); }
}

inline void test_led_begin() {
    _tled::led.begin();
    _tled::led.blink(255, 255, 255, 80, 80);       // fast white strobe
    _tled::timer.begin(_tled::tick, 20000);         // 50 Hz
}

inline void test_led_done(int failures) {
    if (failures == 0)
        _tled::led.pulse(0, 255, 0, 1500);          // slow green breathe
    else
        _tled::led.blink(255, 0, 0, 150, 150);      // fast red blink
}
