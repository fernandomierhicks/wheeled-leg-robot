#pragma once
#include <stdint.h>

enum class WheelMode : uint8_t { IDLE = 0, VELOCITY = 1, POSITION = 2, TORQUE = 3 };

// Per-axis state updated from CAN callbacks.
struct WheelAxisState {
    float    pos_turns;    // encoder position  [turns]
    float    vel_turns_s;  // encoder velocity  [turns/s]
    float    vbus;         // bus voltage [V] — updated by wheel_motors_request_vbus()
    uint32_t error;        // ODrive Axis_Error word (0 = no fault)
    uint8_t  axis_state;   // ODrive Axis_State enum value
    uint32_t last_fb_ms;   // millis() of most recent encoder estimate callback
    uint32_t last_hb_ms;   // millis() of most recent heartbeat
    bool     ever_heard;   // true once any encoder estimate has been received
    bool     ok;           // true when encoder feedback is fresh (< CAN_TIMEOUT_MS)
};

extern WheelAxisState wm_L, wm_R;
extern WheelMode      wm_mode;

// Call once in setup(). Starts CAN2 at 1 Mbps and registers the RX callback.
bool wheel_motors_init();

// Update wm_L/wm_R.ok; auto-transition to IDLE on any fault.
// Call once per control tick, before reading wheel feedback.
void wheel_motors_poll();

// Transition both axes to the requested mode.
//   IDLE:                    sends AXIS_STATE_IDLE to both axes
//   VELOCITY/POSITION/TORQUE: sets controller mode then CLOSED_LOOP_CONTROL
void wheel_motors_set_mode(WheelMode mode);

// Send L/R setpoints. Units depend on mode:
//   VELOCITY : rad/s   (converted to turns/s internally)
//   POSITION : rad     (converted to turns internally)
//   TORQUE   : N·m
// No-op in IDLE.
void wheel_motors_send(float L, float R);

// Keep ODrive watchdog alive. Call every tick at CONTROL_HZ; internally
// divided to 50 Hz. In IDLE, sends a zero-velocity keepalive. TORQUE is
// covered by the control loop's wheel_motors_send(); VELOCITY/POSITION by
// the 50 Hz vbus poll (any axis-addressed CAN frame feeds the watchdog).
void wheel_motors_pet_watchdog();

// Send CLEAR_ERRORS to both axes (clears latched ODrive faults).
void wheel_motors_clear_errors();

// Request bus voltage from both axes (ODrive replies asynchronously via CAN).
// Call this, then read wm_L.vbus / wm_R.vbus a few ms later.
void wheel_motors_request_vbus();
