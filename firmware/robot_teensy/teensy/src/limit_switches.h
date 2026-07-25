#pragma once

#include <stdint.h>

enum LimitSwitchChange : uint8_t {
    LIMIT_SWITCH_NO_CHANGE     = 0,
    LIMIT_SWITCH_LEFT_CHANGED  = 1u << 0,
    LIMIT_SWITCH_RIGHT_CHANGED = 1u << 1,
};

void limit_switches_begin();
uint8_t limit_switches_update();
bool limit_switch_left_active();
bool limit_switch_right_active();
