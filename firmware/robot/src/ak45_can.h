#pragma once
// ak45_can.h — CubeMars AK45-10 MIT CAN driver for hip motors.
//
// Protocol: CubeMars MIT CAN (manual v1.0.18 §5.3)
// TX ID = motor MIT ID, RX ID = motor MIT ID, standard 11-bit frame, 1 Mbps.
// Motor IDs configured via CubeMars PC software; see config.h for values.

#include <stdint.h>
#include "robot_state.h"

// ── Public API ──────────────────────────────────────────────────────────────

/// No-op init (CAN bus already started by odesc_can_init). Call in setup().
void ak45_init();

/// Send MIT Enable command to one motor (motor stiffens).
void ak45_enable(uint8_t id);

/// Send MIT Disable command to one motor (motor goes limp).
void ak45_disable(uint8_t id);

/// Send Set Zero command — sets current position as zero reference.
void ak45_set_zero(uint8_t id);

/// Send MIT position/velocity/torque command.
/// p_des [rad], v_des [rad/s], kp [N·m/rad], kd [N·m·s/rad], t_ff [N·m]
void ak45_send_cmd(uint8_t id, float p_des, float v_des,
                   float kp, float kd, float t_ff);

/// Parse an 8-byte RX frame from a hip motor into state.
/// Call from odesc_can_poll() when std_id matches CAN_ID_HIP_L or _R.
void ak45_parse_rx(uint32_t can_id, const uint8_t* data, RobotState& state);

/// Returns true if both hip motors sent feedback within CAN_TIMEOUT_MS.
bool ak45_hip_ok();
