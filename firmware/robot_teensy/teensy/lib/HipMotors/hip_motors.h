#pragma once
#include <stdint.h>

struct HipAxisState {
    float    pos_rad;      // rotor position [rad]
    float    vel_rad_s;    // rotor velocity [rad/s]
    float    current_A;    // phase current  [A]
    uint32_t last_fb_ms;   // millis() of most recent reply
    bool     ok;           // true when feedback is fresh (< CAN_TIMEOUT_MS)
    bool     mit_active;   // true after enter_mit sent, false after exit_mit
};

extern HipAxisState hm_L, hm_R;

// Call once in setup(). Starts CAN1 at 1 Mbps and registers the RX callback.
bool hip_motors_init();

// Check feedback timeouts; re-enter MIT mode on both axes if due (every 3 s).
// Call once per control tick before reading hip feedback.
void hip_motors_poll();

// Send the MIT-mode enable command to both motors.
// Called automatically by hip_motors_poll(); call explicitly on startup.
void hip_motors_enter_mit();

// Send the MIT-mode disable command to both motors (motors go idle / safe).
void hip_motors_exit_mit();

// Zero the encoder on both motors at the current shaft position.
void hip_motors_zero();

// Send a MIT Cheetah torque-control command to both motors.
//   pos    : target position   [rad]    (-12.5 … +12.5)
//   vel    : feedforward vel   [rad/s]  (-65 … +65)
//   kp     : position gain     [N·m/rad] (0 … 500)
//   kd     : damping gain      [N·m·s/rad] (0 … 5)
//   torque : feedforward torque [N·m]   (-18 … +18)
// No-op if MIT mode is not active on either axis.
void hip_motors_send(float pos_L, float vel_L, float kp_L, float kd_L, float trq_L,
                     float pos_R, float vel_R, float kp_R, float kd_R, float trq_R);
