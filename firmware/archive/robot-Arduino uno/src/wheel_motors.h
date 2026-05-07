#pragma once
#include <stdint.h>

// Wheel motor control modes (apply to both axes simultaneously).
enum class WheelMode : uint8_t { IDLE = 0, VELOCITY = 1, POSITION = 2, TORQUE = 3 };

// Per-axis state populated from CAN callbacks every tick.
struct WheelAxisState {
    float    pos_rad;       // encoder position [rad]
    float    vel_rad_s;     // encoder velocity [rad/s]
    uint32_t error;         // ODrive Axis_Error word (0 = no fault)
    uint8_t  axis_state;    // ODrive Axis_State enum value
    uint32_t last_fb_ms;    // millis() of most recent encoder callback
    uint32_t last_hb_ms;    // millis() of most recent heartbeat
    bool     ok;            // true when encoder feedback is fresh (< CAN_TIMEOUT_MS)
};

extern WheelAxisState wm_L, wm_R;
extern WheelMode      wm_mode;

// Initialize CAN bus and register ODrive callbacks. Call once in setup().
// Returns false if CAN.begin() fails.
bool wheel_motors_init();

// Pump CAN RX callbacks; update wm_L/wm_R.ok; auto-transition to IDLE on any fault.
// Call once per tick, before controllers read wheel feedback.
void wheel_motors_poll();

// Transition both axes to the requested mode.
// IDLE:     sends AXIS_STATE_IDLE
// VELOCITY/POSITION/TORQUE: sets controller mode then CLOSED_LOOP_CONTROL
void wheel_motors_set_mode(WheelMode mode);

// Send L/R setpoints in the current mode.
// Units: rad/s (VELOCITY), rad (POSITION), N·m (TORQUE). No-op in IDLE.
void wheel_motors_send(float L, float R);

// Feed ODrive watchdog. Call every tick unconditionally at 500 Hz.
// In IDLE sends a zero-velocity keepalive so the ODrive watchdog doesn't trip.
void wheel_motors_pet_watchdog();

// Send CLEAR_ERRORS to both axes (clears latched ODrive faults).
void wheel_motors_clear_errors();
