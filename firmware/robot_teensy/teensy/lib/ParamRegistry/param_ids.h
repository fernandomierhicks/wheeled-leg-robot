#pragma once
#include <stdint.h>

// ── Group IDs (high byte of param ID) ────────────────────────────────────────
#define GROUP_SYSTEM   0x00
#define GROUP_CALIB    0x01
#define GROUP_HIP      0x02
#define GROUP_WHEEL    0x03
#define GROUP_CONTROL  0x04
#define GROUP_COMMAND  0x05
#define GROUP_IBUS     0x06

// ── Param IDs ─────────────────────────────────────────────────────────────────
// Convention: high byte = group, low byte = index within group.
// IMPORTANT: After adding a param here and in param_registry.cpp, update
//            software/gui/params_tab.py (_SUBGROUPS) so the GUI shows it
//            in the correct section.

// GROUP_HIP — hip motor behaviour
#define PARAM_ESTOP_HIP_DISABLE   0x0200  // 1=exit MIT on ESTOP entry and re-enter on reset, 0=leave MIT running
// TODO: add PARAM_HIP_RUNNING_KP (0x0201), PARAM_HIP_RUNNING_KD (0x0202), PARAM_HIP_RUNNING_TFF (0x0203)
//       once hip-stiffness tuning experiments begin (currently hardcoded in state_machine.cpp:117-119)

// GROUP_WHEEL — wheel motor settings
// Note: ODrive PID gains (vel_gain, pos_gain, current_lim, etc.) are not exposed here because
// ODrive 0.5.x only allows property access over USB/UART, not CAN. Tune those once via the
// ODrive USB GUI and call save_configuration() to persist them in ODrive flash.
#define PARAM_WM_ENC_TIMEOUT_MS   0x0300  // encoder feedback watchdog [ms]; increase if CAN is flaky

// GROUP_CALIB — hardstop calibration tuning
#define PARAM_CALIB_SEEK_SPEED    0x0100  // ramp speed toward hardstop [rad/s]
#define PARAM_CALIB_KP            0x0101  // position gain while seeking
#define PARAM_CALIB_KD            0x0102  // damping while seeking
#define PARAM_CALIB_HOLD_KP       0x0103  // position gain while holding at home
#define PARAM_CALIB_HOLD_KD       0x0104  // damping while holding at home
#define PARAM_CALIB_STALL_CUR     0x0105  // current threshold to declare a hardstop [A]
#define PARAM_CALIB_STALL_DEADBAND 0x0106 // max pos movement per tick to count as stalled [rad]
#define PARAM_CALIB_STALL_TICKS   0x0107  // consecutive stalled ticks before declaring hardstop (int stored as float)
#define PARAM_CALIB_MARGIN        0x0108  // safety margin from each hardstop [rad]
#define PARAM_CALIB_SAFETY_BOUND  0x0109  // max seek distance before fault [rad]
#define PARAM_CALIB_L_SEEK_DIR    0x010A  // sign (+1/-1) toward left hip bottom hardstop
#define PARAM_CALIB_R_SEEK_DIR    0x010B  // sign (+1/-1) toward right hip bottom hardstop
#define PARAM_CALIB_DONE          0x010C  // 1.0 = calibration completed at least once (persisted across reboots)

// GROUP_CONTROL — LQR controller settings
#define PARAM_LQR_ENABLE              0x0400  // 1 = wheel torque output active; 0 = LQR runs but outputs zero
#define PARAM_SIM_PITCH_RAD           0x0401  // pitch value to inject when PARAM_ENABLE_SIM_PITCH_RAD=1
#define PARAM_ENABLE_SIM_PITCH_RAD    0x0420  // 1 = use sim_pitch_rad instead of real IMU pitch
#define PARAM_SIM_PITCH_RATE_RAD_S    0x0421  // pitch rate value to inject when PARAM_ENABLE_SIM_PITCH_RATE=1
#define PARAM_ENABLE_SIM_PITCH_RATE   0x0422  // 1 = use sim_pitch_rate_rad_s instead of real IMU pitch rate
#define PARAM_LQR_TORQUE_LIMIT        0x0402  // |tau_sym| clamp [N·m]; default 1.0, hard max 7.0
#define PARAM_WHEEL_VEL_LIMIT_TURNS_S 0x0403  // per-tick soft governor [turns/s]; ESTOP at 2×
// Velocity PI (Phase 3)
#define PARAM_VEL_PI_EN               0x0404  // 1 = velocity PI active; 0 = theta_ref fixed at 0
#define PARAM_VEL_PI_KP               0x0405  // proportional gain [rad/(m/s)]
#define PARAM_VEL_PI_KI               0x0406  // integral gain [rad/m]
#define PARAM_VEL_PI_KFF              0x0407  // acceleration feedforward gain [s²·rad/m] (≈ 1/g)
#define PARAM_VEL_PI_THETA_MAX        0x0408  // |theta_ref| hard clamp [rad]
#define PARAM_VEL_PI_RATE_LIM         0x0409  // theta_ref slew rate limit [rad/s]
#define PARAM_VEL_PI_INT_MAX          0x040A  // integrator anti-windup clamp [rad·s]
#define PARAM_V_CMD_MS                0x040B  // desired forward velocity setpoint [m/s]; GUI/Phase3 testing
// Yaw PI (Phase 4)
#define PARAM_YAW_PI_EN               0x040C  // 1 = yaw PI active; 0 = tau_yaw fixed at 0
#define PARAM_YAW_PI_KP               0x040D  // proportional gain [N·m/(rad/s)]
#define PARAM_YAW_PI_KI               0x040E  // integral gain [N·m/rad]
#define PARAM_YAW_PI_TORQUE_MAX       0x040F  // |tau_yaw| clamp [N·m]
#define PARAM_YAW_PI_INT_MAX          0x0410  // integrator anti-windup [N·m·s]
#define PARAM_OMEGA_CMD_RDS           0x0411  // desired yaw rate [rad/s]; positive = CCW from above
// Feedforward (Phase 6)
#define PARAM_FF1_ALPHA               0x0412  // hip reaction cancel gain [0–1]; start at 0, ramp up
#define PARAM_FF2_ALPHA               0x0413  // gravity compensation gain [0–1]; start at 0, ramp up
#define PARAM_FF1_KT_HIP              0x0414  // hip motor output torque constant [N·m/A]; default 1.2732
// Jump controller (Phase 7)
#define PARAM_JUMP_ENABLE             0x0415  // master gate: 0=no-op, 1=execute sequence; default 0
#define PARAM_JUMP_TORQUE_MAX         0x0416  // [N·m] max hip tff during EXTEND; default 0.0, ramp up from 0
#define PARAM_JUMP_CROUCH_TIME_S      0x0417  // [s] CROUCH phase duration; default 0.30
#define PARAM_JUMP_RAMP_UP_S          0x0418  // [s] EXTEND tff softstart; default 0.05
#define PARAM_JUMP_RAMP_DOWN_RAD      0x0419  // [rad] torque→0 zone near extended limit; default 0.08
#define PARAM_JUMP_OMEGA_MAX          0x041A  // [rad/s] hip velocity where tff→0; default 40.0
#define PARAM_JUMP_HARDSTOP_MARGIN    0x041B  // [rad] hard cutoff from calibrated limit; default 0.06
#define PARAM_JUMP_KP                 0x041C  // position gain for CROUCH/RETRACT; default 80.0
#define PARAM_JUMP_KD                 0x041D  // damping for CROUCH/RETRACT; default 1.0
#define PARAM_JUMP_EXTEND_KD          0x041E  // small kd during EXTEND (electrical damping); default 0.1
#define PARAM_JUMP_EXTEND_TIMEOUT_S   0x041F  // [s] max time in EXTEND before forced RETRACT; default 0.15

// GROUP_COMMAND — high-freq setpoints from radio/GUI
#define PARAM_RADIO_HIP_CMD       0x0500  // hip extension command from CH3 [0=retracted, 1=extended]; stale when radio dead
#define PARAM_RADIO_VEL_MAX       0x0501  // max forward speed mapped from full CH2 deflection [m/s]
#define PARAM_RADIO_YAW_MAX       0x0502  // max yaw rate mapped from full CH4 deflection [rad/s]

// GROUP_IBUS — RC receiver channel readings (live, READONLY, firmware-written)
#define PARAM_IBUS_CH0     0x0600  // RC channel 0  [1000–2000 µs]
#define PARAM_IBUS_CH1     0x0601  // RC channel 1
#define PARAM_IBUS_CH2     0x0602  // RC channel 2
#define PARAM_IBUS_CH3     0x0603  // RC channel 3
#define PARAM_IBUS_CH4     0x0604  // RC channel 4
#define PARAM_IBUS_CH5     0x0605  // RC channel 5
#define PARAM_IBUS_CH6     0x0606  // RC channel 6
#define PARAM_IBUS_CH7     0x0607  // RC channel 7
#define PARAM_IBUS_CH8     0x0608  // RC channel 8
#define PARAM_IBUS_CH9     0x0609  // RC channel 9
#define PARAM_IBUS_CH10    0x060A  // RC channel 10
#define PARAM_IBUS_CH11    0x060B  // RC channel 11
#define PARAM_IBUS_CH12    0x060C  // RC channel 12
#define PARAM_IBUS_CH13    0x060D  // RC channel 13
#define PARAM_IBUS_ALIVE   0x060E  // link alive: 1.0 = packet within 500 ms, 0.0 = lost
