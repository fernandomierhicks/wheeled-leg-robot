#pragma once
// telemetry.h — UDP + Serial telemetry sender (50 Hz, 69-byte binary packet).

#include "robot_state.h"

// Packed telemetry struct sent over UDP/Serial (82 bytes, little-endian).
struct __attribute__((packed)) TelemetryPacket {
    uint32_t timestamp_ms;    // millis()
    uint8_t  mode;            // Mode enum value
    float    pitch;           // [rad]
    float    pitch_rate;      // [rad/s]
    float    roll;            // [rad]
    float    yaw;             // [rad] mag-fused heading
    float    wheel_vel_avg;   // [rad/s]
    float    v_cmd;           // [rad/s]
    float    theta_ref;       // [rad]
    float    tau_sym;         // [N·m]
    float    tau_yaw;         // [N·m]
    float    tau_wheel_L;     // [N·m]
    float    tau_wheel_R;     // [N·m]
    float    hip_q_L;         // [rad] left hip position
    float    tau_hip_L;       // [N·m]
    float    tau_hip_R;       // [N·m]
    float    hip_q_R;         // [rad] right hip position
    float    dt_us;           // loop dt [µs]
    float    debug_sine;      // noisy sine for rate check
    // --- encoder feedback (added for ODrive bench testing) ---
    float    wheel_pos_L;     // [rad] left wheel encoder position
    float    wheel_vel_L;     // [rad/s] left wheel encoder velocity
    float    wheel_pos_R;     // [rad] right wheel encoder position
    float    wheel_vel_R;     // [rad/s] right wheel encoder velocity
    uint8_t  status_flags;    // bit0=wheel_ok, bit1=imu_ok, bits[5:2]=axis0_state, bit6=axis0_err
    uint8_t  odrive_flags_R;  // bits[3:0]=axis1_state, bit4=axis1_has_error
};
static_assert(sizeof(TelemetryPacket) == 91, "TelemetryPacket must be 91 bytes");

// Initialise the telemetry UDP socket.  Call once in setup() after wifi_init().
void telemetry_init();

// Pack RobotState into a TelemetryPacket and send via UDP + Serial.
// Call every TELEMETRY_DIV ticks from the main loop.
// Serial framing: [0xAA][0x55][65-byte packet][1-byte XOR checksum]
void telemetry_send(const RobotState& state);
