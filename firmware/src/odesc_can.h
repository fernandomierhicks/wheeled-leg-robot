#pragma once
// odesc_can.h — ODESC (ODrive v3.x) CAN driver for wheel motors.
//
// ODrive CAN protocol: arb_id = (node_id << 5) | cmd_id
// Each ODESC axis is a separate CAN node.  Configure node IDs on the
// ODESC via odrivetool to match ODESC_NODE_L / ODESC_NODE_R in config.h.
//
// Required ODESC configuration (run once via odrivetool):
//   odrv0.axis0.config.can.node_id = 0          # must match ODESC_NODE_L
//   odrv0.axis1.config.can.node_id = 1          # must match ODESC_NODE_R
//   odrv0.axis0.config.can.encoder_rate_ms = 0  # stream encoder at bus rate
//   odrv0.axis1.config.can.encoder_rate_ms = 0  # stream encoder at bus rate
//   odrv0.save_configuration()

#include "robot_state.h"

// ── Public API ──────────────────────────────────────────────────────────────

/// Initialise CAN peripheral at 1 Mbps.  Call once in setup().
bool odesc_can_init();

/// Drain CAN RX buffer, parse encoder feedback + heartbeats into state.
/// Call once per tick, before controllers.
void odesc_can_poll(RobotState& state);

/// Send torque commands to both wheel axes.
/// Call once per tick, after controllers.
void odesc_can_send_torque(float tau_L, float tau_R);

/// Command both axes into CLOSED_LOOP_CONTROL.
/// Typically called on transition to BALANCE mode.
void odesc_can_enable();

/// Command both axes to IDLE (zero torque, coast).
/// Called on FAULT or IDLE transition.
void odesc_can_disable();

/// Emergency stop — sends estop to both axes.
void odesc_can_estop();
