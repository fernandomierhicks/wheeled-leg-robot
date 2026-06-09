#pragma once
#include <stdint.h>

// MIRROR: software/gui/flash_monitor.py  _STATE_NAMES dict and
//         software/gui/hip_motors.py  _STATE_LABELS + _STATE_STANDBY/_STATE_MANUAL must stay in sync
typedef enum : uint8_t {
    STATE_STARTUP      = 0,
    STATE_CALIBRATION  = 1,
    STATE_STANDBY      = 2,
    STATE_RUNNING      = 3,
    STATE_ESTOP        = 4,
    STATE_MANUAL       = 5,
} RobotStateEnum;

typedef struct {
    float          pitch_rad;
    float          pitch_rate_rads;
    float          wheel_vel_avg_ms;
    float          hip_l_pos_rad;
    float          hip_r_pos_rad;
    float          cmd_l;
    float          cmd_r;
    RobotStateEnum state;
    uint8_t        fault_code;   // FAULT_* from comm_protocol.h; set before entering ESTOP
    uint32_t       loop_count;
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
