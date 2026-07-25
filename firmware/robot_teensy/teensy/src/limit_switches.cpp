#include <Arduino.h>

#include "config.h"
#include "limit_switches.h"

namespace {

constexpr uint32_t DEBOUNCE_MS = 20;

struct DebouncedInput {
    uint8_t pin;
    bool stable_pressed;
    bool candidate_pressed;
    uint32_t candidate_since_ms;
};

DebouncedInput s_left  = {PIN_LIMIT_LEFT,  false, false, 0};
DebouncedInput s_right = {PIN_LIMIT_RIGHT, false, false, 0};

bool read_pressed(uint8_t pin) {
    // NC contact to GND: released is closed/LOW; pressed is open/pulled HIGH.
    return digitalRead(pin) == HIGH;
}

bool update_input(DebouncedInput& input, uint32_t now_ms) {
    const bool pressed = read_pressed(input.pin);
    if (pressed != input.candidate_pressed) {
        input.candidate_pressed = pressed;
        input.candidate_since_ms = now_ms;
        return false;
    }
    if (pressed != input.stable_pressed &&
        (uint32_t)(now_ms - input.candidate_since_ms) >= DEBOUNCE_MS) {
        input.stable_pressed = pressed;
        return true;
    }
    return false;
}

}  // namespace

void limit_switches_begin() {
    pinMode(PIN_LIMIT_LEFT, INPUT_PULLUP);
    pinMode(PIN_LIMIT_RIGHT, INPUT_PULLUP);

    const uint32_t now_ms = millis();
    s_left.stable_pressed = s_left.candidate_pressed = read_pressed(s_left.pin);
    s_left.candidate_since_ms = now_ms;
    s_right.stable_pressed = s_right.candidate_pressed = read_pressed(s_right.pin);
    s_right.candidate_since_ms = now_ms;
}

uint8_t limit_switches_update() {
    const uint32_t now_ms = millis();
    uint8_t changes = LIMIT_SWITCH_NO_CHANGE;
    if (update_input(s_left, now_ms))  changes |= LIMIT_SWITCH_LEFT_CHANGED;
    if (update_input(s_right, now_ms)) changes |= LIMIT_SWITCH_RIGHT_CHANGED;
    return changes;
}

bool limit_switch_left_active() {
    return s_left.stable_pressed;
}

bool limit_switch_right_active() {
    return s_right.stable_pressed;
}
