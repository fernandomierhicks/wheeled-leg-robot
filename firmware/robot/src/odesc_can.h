#pragma once
// odesc_can.h — ODESC (ODrive v3.x) CAN driver for wheel motors.
//
// ODrive CAN protocol: arb_id = (node_id << 5) | cmd_id
// Each ODESC axis is a separate CAN node.  Configure node IDs on the
// ODESC via odrivetool to match ODESC_NODE_L / ODESC_NODE_R in config.h.
//
// Required ODESC configuration (use ODrive GUI CAN tab):
//   odrv0.can.config.baud_rate               = 1000000
//   odrv0.axis0.config.can.node_id           = 0   # must match ODESC_NODE_L
//   odrv0.axis1.config.can.node_id           = 1   # must match ODESC_NODE_R
//   odrv0.axis0.config.can.encoder_rate_ms   = 10  # 100 Hz — must be < CAN_TIMEOUT_MS(20)
//   odrv0.axis1.config.can.encoder_rate_ms   = 10
//   odrv0.save_configuration()

#include "robot_state.h"

// ── Public API ──────────────────────────────────────────────────────────────

/// Initialise CAN peripheral at 1 Mbps.  Call once in setup().
bool odesc_can_init();

/// Drain CAN RX buffer, parse encoder feedback + heartbeats into state.
/// Call once per tick, before controllers.
void odesc_can_poll(RobotState& state);

/// Send torque commands to both wheel axes (TORQUE control mode).
void odesc_can_send_torque(float tau_L, float tau_R);

/// Send velocity setpoints to both wheel axes (VELOCITY control mode).
/// Units: turns/s (ODrive native).  Call once per tick.
void odesc_can_send_velocity(float vel_L_turns_s, float vel_R_turns_s);

/// Send position setpoints to both wheel axes (POSITION control mode).
/// Units: turns (ODrive native).  Call once per tick.
void odesc_can_send_position(float pos_L_turns, float pos_R_turns);

/// Set ODrive control mode on both axes.
/// ctrl_mode: 1=torque, 2=velocity, 3=position.
/// input_mode: 1=passthrough (recommended for direct commands).
void odesc_can_set_control_mode(uint32_t ctrl_mode, uint32_t input_mode);

/// Enable both axes in VELOCITY control mode (safe default for bench testing).
void odesc_can_enable_velocity();

/// Enable both axes in POSITION control mode.
void odesc_can_enable_position();

/// Enable both axes in TORQUE control mode (balance controller use).
void odesc_can_enable_torque();

/// Command both axes to IDLE (zero torque, coast).
void odesc_can_disable();

/// Emergency stop — sends estop to both axes.
void odesc_can_estop();
