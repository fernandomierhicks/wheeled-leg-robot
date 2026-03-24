#pragma once
// telemetry.h — UDP + Serial telemetry sender (50 Hz, 69-byte binary packet).

#include "robot_state.h"

// Packed telemetry struct sent over UDP/Serial (69 bytes, little-endian).
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
    float    hip_q_avg;       // [rad]
    float    tau_hip_L;       // [N·m]
    float    tau_hip_R;       // [N·m]
    float    dt_us;           // loop dt [µs]
    float    debug_sine;     // noisy sine for rate check
};
static_assert(sizeof(TelemetryPacket) == 69, "TelemetryPacket must be 69 bytes");

// Initialise the telemetry UDP socket.  Call once in setup() after wifi_init().
void telemetry_init();

// Pack RobotState into a TelemetryPacket and send via UDP + Serial.
// Call every TELEMETRY_DIV ticks from the main loop.
// Serial framing: [0xAA][0x55][65-byte packet][1-byte XOR checksum]
void telemetry_send(const RobotState& state);
