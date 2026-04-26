#pragma once
// robot_state.h — Central mutable state struct (~120 bytes).
// Passed by pointer to every module's update function.

#include <stdint.h>

// ── Robot operating modes ───────────────────────────────────────────────────
enum class Mode : uint8_t {
    IDLE = 0,
    BALANCE,
    DRIVE,
    JUMP,       // future
    STAND_UP,   // future
    FAULT,
};

// ── Central state ───────────────────────────────────────────────────────────
struct RobotState {
    // Timing
    uint32_t tick;              // monotonic tick counter (wraps at ~2^32)
    uint32_t dt_us;             // measured loop period [µs]

    // Mode
    Mode mode;

    // Connectivity
    bool host_connected;            // true if host PC is sending commands

    // IMU (BNO086 fused output)
    float pitch;                // [rad] body pitch (positive = leaning forward)
    float pitch_rate;           // [rad/s]
    float roll;                 // [rad] body roll
    float roll_rate;            // [rad/s]
    float yaw;                  // [rad] body yaw (mag-fused, from Rotation Vector)
    bool  imu_ok;               // true if data is fresh
    uint32_t imu_last_ms;       // millis() of last good read

    // Wheel motors (ODESC via CAN)
    float wheel_vel_L;          // [rad/s] left wheel velocity
    float wheel_vel_R;          // [rad/s] right wheel velocity
    float wheel_pos_L;          // [rad] left wheel position
    float wheel_pos_R;          // [rad] right wheel position
    bool  wheel_ok;             // CAN feedback fresh
    uint32_t wheel_last_ms;

    // Hip motors (AK45-10 via CAN)
    float hip_q_L;              // [rad] left hip position
    float hip_q_R;              // [rad] right hip position
    float hip_dq_L;             // [rad/s] left hip velocity
    float hip_dq_R;             // [rad/s] right hip velocity
    bool  hip_ok;               // CAN feedback fresh
    uint32_t hip_last_ms;
    bool    hip_enabled;        // true = send MIT commands each tick
    float   hip_temp;           // [°C] motor temperature (last RX)
    uint8_t hip_error;          // motor error code (last RX)

    // Computed averages
    float wheel_vel_avg;        // (L + R) / 2
    float hip_q_avg;            // (L + R) / 2

    // Controller outputs
    float v_cmd;                // [m/s] velocity command
    float omega_cmd;            // [rad/s] yaw rate command
    float hip_q_target;         // [rad] hip position target
    float theta_ref;            // [rad] lean angle reference (from VelocityPI)
    float tau_sym;              // [N·m] symmetric wheel torque (from LQR)
    float tau_yaw;              // [N·m] differential yaw torque

    // Final motor commands
    float tau_wheel_L;          // [N·m] left wheel torque command
    float tau_wheel_R;          // [N·m] right wheel torque command
    float tau_hip_L;            // [N·m] left hip torque command
    float tau_hip_R;            // [N·m] right hip torque command

    // Direct ODrive control (bench / troubleshooting mode)
    // When odrive_ctrl_mode != 0, these override balance controller CAN output.
    uint8_t  odrive_ctrl_mode;   // 0=off(torque from balance), 1=torque, 2=velocity, 3=position
    float    odrive_vel_L;       // [turns/s] velocity setpoint — left  axis
    float    odrive_vel_R;       // [turns/s] velocity setpoint — right axis
    float    odrive_pos_L;       // [turns]   position setpoint — left  axis
    float    odrive_pos_R;       // [turns]   position setpoint — right axis
    float    odrive_tau_L;       // [N·m]     torque setpoint  — left  axis
    float    odrive_tau_R;       // [N·m]     torque setpoint  — right axis
    uint8_t  odrive_axis_state;    // axis0 state from last heartbeat (1=IDLE, 8=CLOSED_LOOP)
    uint32_t odrive_axis_error;    // axis0 error from last heartbeat (0=none)
    uint8_t  odrive_axis_state_R;  // axis1 state from last heartbeat
    uint32_t odrive_axis_error_R;  // axis1 error from last heartbeat (0=none)

    // Overrun tracking
    uint8_t overrun_flash;      // >0 = flash fault bar (decremented each LED update)

    // WiFi profiling (USE_WIFI=1 only, zero otherwise)
    uint32_t wifi_send_us;      // last UDP telemetry send duration [µs]
    uint32_t wifi_recv_us;      // last command receive duration [µs]
    uint32_t wifi_skips;        // telemetry packets skipped (insufficient slack)

    // Debug
    float debug_sine;           // noisy sine for telemetry rate check
};
