#pragma once
#include <stdint.h>

typedef enum : uint8_t {
    STATE_STARTUP      = 0,
    STATE_CALIBRATION  = 1,
    STATE_RUNNING      = 2,
    STATE_ESTOP        = 3,
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
    uint32_t       loop_count;
} RobotState;

extern RobotState g_state;
