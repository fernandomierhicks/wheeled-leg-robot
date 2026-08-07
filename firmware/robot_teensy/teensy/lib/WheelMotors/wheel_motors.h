#pragma once
#include <stdint.h>

enum class WheelMode : uint8_t { IDLE = 0, VELOCITY = 1, POSITION = 2, TORQUE = 3 };

// Per-axis state updated from CAN callbacks.
struct WheelAxisState {
    float    pos_turns;    // encoder position  [turns]
    // Validated velocity — what the control loop, runaway watchdogs, health
    // flags and telemetry all read. Updated in wheel_motors_poll() from
    // vel_raw_turns_s through wheel_vel_glitch_filter() (see wheel_safety.h),
    // so a corrupt encoder frame can't reach the balance loop or trip a
    // runaway fault on its own.
    float    vel_turns_s;  // encoder velocity  [turns/s]
    // Unfiltered value straight off CAN, written by the RX ISR. Kept separate
    // so the filter has a stable "last good" to compare against and so the
    // raw feed stays inspectable.
    float    vel_raw_turns_s;
    uint32_t fb_seq;          // increments on every encoder-estimate frame (ISR)
    uint32_t fb_seq_seen;     // last fb_seq the filter processed (main loop)
    uint32_t vel_accept_ms;   // millis() of the most recent ACCEPTED sample
    uint32_t vel_glitch_count;// total samples rejected as implausible (diagnostic)
    uint8_t  vel_reject_run;  // current consecutive-rejection run length
    float    vbus;         // bus voltage [V] — updated by wheel_motors_request_vbus()
    uint32_t error;        // ODrive Axis_Error word (0 = no fault)
    uint8_t  axis_state;   // ODrive Axis_State enum value
    uint32_t last_fb_ms;   // millis() of most recent encoder estimate callback
    uint32_t last_hb_ms;   // millis() of most recent heartbeat
    bool     ever_heard;   // true once any encoder estimate has been received
    bool     hb_ever_heard;// true once any heartbeat has been received
    // true when the axis is usable: fresh encoder feedback, no latched CAN TX
    // stall, and — once heartbeats have been seen at all — a fresh heartbeat
    // reporting the axis state we asked for. See wheel_motors_poll().
    bool     ok;
};

extern WheelAxisState wm_L, wm_R;
extern WheelMode      wm_mode;

// Call once in setup(). Starts CAN2 at 1 Mbps and registers the RX callback.
bool wheel_motors_init();

// Update wm_L/wm_R.ok; auto-transition to IDLE on any fault.
// Call once per control tick, before reading wheel feedback.
void wheel_motors_poll();

// Reset the encoder-freshness clock on both axes after a deliberate,
// known-blocking main-loop operation (e.g. SD-log open/finalize) that froze
// the tick. Companion to hip_motors_forgive_feedback_stall(); a real ODrive
// error flag still faults. See definition for details.
void wheel_motors_forgive_feedback_stall();

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

// True when every enabled axis (PARAM_WHEEL_{L,R}_ENABLE) has fresh encoder
// feedback and no latched ODrive error. Mirrors hip_motors_ok().
bool wheel_motors_ok();

// Request bus voltage from both axes (ODrive replies asynchronously via CAN).
// Call this, then read wm_L.vbus / wm_R.vbus a few ms later.
void wheel_motors_request_vbus();

// Lifetime count of CAN3 frames that could not be placed directly into a
// hardware mailbox and were deferred to the software TX queue. Diagnostic only
// — a nonzero value is normal under burst load; see wheel_motors.cpp for when a
// sustained run of these latches a TX stall and clears wm_*.ok.
uint32_t wheel_motors_tx_defer_count();
