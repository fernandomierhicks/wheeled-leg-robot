#pragma once
#include <stdint.h>

// MIRROR: software/gui/flash_monitor.py  _STATE_NAMES dict (add STATE_JUMPING=7, STATE_STANDING_UP=8) and
//         software/gui/hip_motors.py  _STATE_LABELS + _STATE_STANDBY/_STATE_MANUAL must stay in sync
typedef enum : uint8_t {
    STATE_STARTUP      = 0,
    STATE_CALIBRATION  = 1,
    STATE_STANDBY      = 2,
    STATE_RUNNING      = 3,
    STATE_ESTOP        = 4,
    STATE_MANUAL       = 5,
    STATE_CMD_REJECT   = 6,  // ~1 s transient: plays REJECT_MELODY, red blink, then returns to originating state
    STATE_JUMPING      = 7,  // ~3 s transient from RUNNING: plays jump melody, then returns to RUNNING
    STATE_STANDING_UP  = 8,  // transient from STANDBY on arm: retract legs, energetic wheel recovery, then RUNNING
    STATE_DISARMING    = 9,  // normal torque ramp-down; re-arm/manual/calibration blocked until safe
} RobotStateEnum;

typedef struct {
    // ── Sensors ───────────────────────────────────────────────────────────────
    float          pitch_rad;
    float          pitch_rate_rads;
    float          wheel_vel_avg_ms;
    float          hip_l_pos_rad;
    float          hip_r_pos_rad;
    float          hip_l_current_a;
    float          hip_r_current_a;
    // ── Robot meta ────────────────────────────────────────────────────────────
    RobotStateEnum state;
    uint8_t        fault_code;        // FAULT_* from comm_protocol.h; set before entering ESTOP
    uint32_t       loop_count;
    // ── Controller outputs (written by control_loop.cpp / future phases) ──────
    float          whl_tau_l;         // final left  wheel torque command [N·m] = tau_sym + tau_yaw
    float          whl_tau_r;         // final right wheel torque command [N·m] = tau_sym - tau_yaw
    float          tau_sym;           // unclamped LQR symmetric torque [N·m]
    float          tau_yaw;           // yaw PI differential torque [N·m] — Phase 4; 0 until then
    float          theta_ref;         // velocity PI lean setpoint [rad]  — Phase 3; 0 until then
    float          v_ref;             // velocity reference [m/s]          — Phase 3; 0 until then
    float          gain_sched_alpha;  // hip gain interpolation [0-1]      — Phase 5; 0 until then
    float          ff1_out;           // hip reaction FF torque [N·m]      — Phase 6; 0 until then
    float          ff2_out;           // gravity compensation FF [N·m]     — Phase 6; 0 until then
    uint8_t        jump_state;        // Jump FSM phase                    — Phase 7; 0 until then
    uint8_t        standup_state;     // Standup FSM phase                 — 0 until then
} RobotState;

// Pending hip command queued by the comm handler; consumed by on_manual().
typedef struct {
    bool    pending;
    uint8_t motor_id;
    uint8_t sub_cmd;
    float   p, v, kp, kd, tff;  // used only for HIP_SUB_MIT
} HipCmd;

extern RobotState g_state;
extern HipCmd     g_hip_cmd;
