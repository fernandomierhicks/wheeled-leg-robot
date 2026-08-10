#pragma once
// control_loop.cpp only needs fault IDs from the full packed wire header.
// Keeping this stub tiny avoids making a desktop compiler understand the
// Teensy/GCC packed-struct declarations; the protocol layout has separate tests.
#define FAULT_PITCH_WATCHDOG 8
#define FAULT_WHEEL_RUNAWAY 9
#define FAULT_ROLL_WATCHDOG 14
