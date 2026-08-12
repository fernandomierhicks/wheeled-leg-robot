#pragma once
#include <stdint.h>

// pos_rad/vel_rad_s/torque_nm are in the firmware-frame convention: L's raw
// CAN feedback is negated in rx_callback() so positive means the same
// physical direction on both L and R (they're physically mirrored motors).
// Commands (pos/vel/torque args below) use the same firmware-frame
// convention — pack_and_send() applies the matching negation for L on TX.
struct HipAxisState {
    float    pos_rad;      // rotor position [rad]
    float    vel_rad_s;    // rotor velocity [rad/s]
    float    torque_nm;    // shaft torque   [N·m]  (NOT amps — see hip_motors.cpp)
    uint32_t last_fb_ms;   // millis() of most recent reply
    uint32_t feedback_seq; // increments for every decoded CAN feedback frame
    bool     ever_heard;   // true once any CAN reply has been received
    bool     ok;           // true when feedback is fresh (< HIP_CAN_TIMEOUT_MS) AND ever_heard
    bool     mit_active;   // true after enter_mit sent, false after exit_mit
    // Set when an encoder-zero command has been sent but the motor has not yet
    // replied in the new frame. Commands are already in the zeroed frame while
    // feedback is still in the old one, so the two are not comparable — see
    // hip_motors.cpp for why the position-jump guard must stand down until the
    // motor confirms. Cleared by hip_motors_poll().
    bool     zero_pending;
    uint32_t zero_cmd_ms;  // millis() when the zero command went out
    uint32_t zero_fb_seq;  // feedback_seq at that moment (to spot in-flight frames)
};

// Cached MIT setpoint, re-sent every poll() tick while active. Falls back to
// the safe zero-torque ping if cleared, stale (> HIP_SETPOINT_TIMEOUT_MS), or
// the robot enters ESTOP.
struct HipSetpoint {
    float    p, v, kp, kd, tff;
    bool     active;
    uint32_t last_cmd_ms;
};

// Software position limits, populated by calibration (RAM-only — lost on
// reboot). While `valid` is false, setpoints are not clamped.
struct HipLimits {
    float min_rad, max_rad;
    bool  valid;
};

extern HipAxisState hm_L, hm_R;
extern HipSetpoint  hm_sp_L, hm_sp_R;
extern HipLimits    hm_limits_L, hm_limits_R;

// Call once in setup(). Starts CAN1 at 1 Mbps and registers the RX callback.
bool hip_motors_init();

// Check feedback timeouts, then send this tick's frame to each MIT-active axis
// (the cached setpoint, or a zero-torque ping when none is active).
// Call once per control tick before reading hip feedback.
void hip_motors_poll();

// Returns true when both motors have replied at least once and feedback is fresh.
bool hip_motors_ok();

// Lifetime count of CAN2 frames deferred to the software TX queue instead of
// going straight into a hardware mailbox. Diagnostic only — see hip_motors.cpp
// for when a sustained run of these latches a TX stall and clears hm_*.ok.
uint32_t hip_motors_tx_defer_count();

// Reset the feedback-freshness clock on both axes after a deliberate,
// known-blocking main-loop operation (e.g. SD-log open/finalize) that froze
// the tick long enough for last_fb_ms to go stale. Prevents a self-inflicted
// loop stall from tripping the feedback watchdog; a truly dead motor still
// faults on the next interval. See definition for details.
void hip_motors_forgive_feedback_stall();

// Send the MIT-mode enable command to both motors.
// Called automatically by hip_motors_poll(); call explicitly on startup.
void hip_motors_enter_mit();

// Send the MIT-mode disable command to both motors (motors go idle / safe).
void hip_motors_exit_mit();

// Immediately disable one axis, independent of its just-updated enable param.
// Used by the live parameter path so changing hip_*_enable from 1 to 0 sends
// the motor's MIT-exit frame before polling for that axis stops.
void hip_motor_disable_L();
void hip_motor_disable_R();

// Zero the encoder on both motors at the current shaft position.
void hip_motors_zero();

// Zero the encoder on a single motor at its current shaft position.
// Used by calibration so one axis can be zeroed while the other is still
// seeking. Zeroing persists across power cycles (written to motor flash).
void hip_motor_zero_L();
void hip_motor_zero_R();

// Send a MIT Cheetah torque-control command to both motors.
//   pos    : target position   [rad]    (-12.5 … +12.5)
//   vel    : feedforward vel   [rad/s]  (-20 … +20)
//   kp     : position gain     [N·m/rad] (0 … 500)
//   kd     : damping gain      [N·m·s/rad] (0 … 5)
//   torque : feedforward torque [N·m]   (-8 … +8)
// No-op if MIT mode is not active on either axis.
void hip_motors_send(float pos_L, float vel_L, float kp_L, float kd_L, float trq_L,
                     float pos_R, float vel_R, float kp_R, float kd_R, float trq_R);

// Single-motor MIT commands — leave the other motor's CAN mailbox untouched.
void hip_motor_send_L(float pos, float vel, float kp, float kd, float torque);
void hip_motor_send_R(float pos, float vel, float kp, float kd, float torque);

// Cache a MIT setpoint that hip_motors_poll() will re-send every tick until
// it's cleared, goes stale (> HIP_SETPOINT_TIMEOUT_MS without a refresh), or
// the robot enters ESTOP — at which point poll() falls back to the safe
// zero-torque ping.
void hip_motors_set_setpoint_L(float pos, float vel, float kp, float kd, float torque);
void hip_motors_set_setpoint_R(float pos, float vel, float kp, float kd, float torque);

// Drop any cached setpoints on both axes, reverting poll() to the safe
// zero-torque ping. Call on disable, or when leaving the commanding mode.
void hip_motors_clear_setpoints();

// Clamp a position [rad] to a calibrated software limit pair (returns it
// unchanged while `valid` is false). Exposed because every command sent below
// is clamped this way, so a caller that ramps *from* the measured pose has to
// start the ramp from the clamped pose — otherwise the first command silently
// steps to the limit instead of tracking the ramp. See on_standing_up().
float hip_motors_clamp_to_limits(float pos, const HipLimits& lim);

// Convert a normalised hip extension command t ∈ [0,1] (0 = retracted,
// 1 = fully extended) into per-motor position setpoints [rad], accounting
// for each motor's calibrated span and seek direction.
void hip_cmd_to_setpoints(float t, float* pos_L, float* pos_R);

// Absolute-angle counterpart of hip_cmd_to_setpoints(), in "extension angle":
// radians of hip travel away from the retract switch, positive = extended.
// Zero is the switch itself, so the calibrated retracted endpoint sits at
// calib_backoff_rad and the extended endpoint at calib_range_*_rad.
//
// Absolute angle, not normalised t, is what the jump phases want. t is a
// fraction of each leg's own calibrated span, so the same t is a *different*
// physical angle on two legs whose spans differ; an angle from each leg's own
// switch is the same physical pose on both. It is also the frame a person can
// read off the machine, which normalised height is not.
//
// Results are clamped to the calibrated limits, so an angle past either end
// saturates rather than commanding through a stop.
void hip_ext_angle_to_setpoints(float ext_rad, float* pos_L, float* pos_R);

// Inverse of the above for one axis: measured extension angle [rad] away from
// the retract switch, positive = extended. Returns 0 when uncalibrated.
float hip_measured_ext_angle(const HipAxisState& hm, const HipLimits& lim,
                             float seek_dir);
