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
//            software/gui/tabs/params_tab.py:
//              - _SUBGROUPS, so the GUI shows it in the correct section
//              - _PARAM_DEFS, adding this id's (name, description) entry —
//                name copied from param_registry.cpp, description from the
//                comment on this param's #define below.

// GROUP_SYSTEM — peripheral enable flags (bench-test without full hardware)
// 0 = peripheral not connected: init/poll is skipped, and the state machine
// no longer gates STARTUP/CALIBRATION/RUNNING on it. Persisted so it
// survives reboot during a bench-test session. Takes effect at boot —
// changing at runtime requires CMD_ID_REBOOT to re-run setup().
#define PARAM_IMU_ENABLE     0x0000  // BNO086 IMU (SPI). 0 also blocks RUNNING (arming) —
                                      // real pitch feedback is required to balance.
#define PARAM_BUZZER_ENABLE  0x0003  // buzzer
#define PARAM_LED_ENABLE     0x0004  // status RGB LED

// Per-motor presence flags — 0 = that motor physically disconnected/not
// present; CAN traffic to it is skipped and the state machine no longer
// gates STARTUP/CALIBRATION on it. The whole hip/wheel CAN subsystem is
// initialized at boot iff at least one of its two per-motor flags is set
// (main.cpp setup()). Default 1 (present).
// RUNNING/JUMPING are hard-blocked (state_machine.cpp req_running()) unless
// all four are enabled.
#define PARAM_HIP_L_ENABLE    0x0005  // left AK45 hip motor present
#define PARAM_HIP_R_ENABLE    0x0006  // right AK45 hip motor present
#define PARAM_WHEEL_L_ENABLE  0x0007  // left ODrive wheel motor present
#define PARAM_WHEEL_R_ENABLE  0x0008  // right ODrive wheel motor present

// Hardware watchdog (WDOG1) — auto-reboots the MCU if the main loop stops
// petting it (see main.cpp watchdog_enable()/watchdog_pet()). Guards against
// a stuck driver call (e.g. BNO08x sh2 getProdIds has no internal timeout)
// hanging the control loop with no crash and no recovery. Takes effect at
// boot; changing at runtime requires CMD_ID_REBOOT. Note: WDOG1's enable bit
// is write-once in hardware — once armed it cannot be turned off in software
// until the next reset, regardless of this param.
#define PARAM_WATCHDOG_ENABLE 0x0009  // 1 = enable hardware watchdog; default 0 (disabled)

// Loop section profiler — when 1, main.cpp loop() times each top-level section
// (and read_sensors()'s imu/hip/wheel/ibus sub-calls) and logs the rolling max
// per section once a second, then resets. Debug aid for chasing "Loop overrun"
// warnings; not persisted so it can't accidentally survive a reboot and spam logs.
#define PARAM_LOOP_PROFILE_ENABLE 0x000A

// GROUP_HIP — hip motor behaviour
#define PARAM_ESTOP_HIP_DISABLE   0x0200  // 1=exit MIT on ESTOP entry and re-enter on reset, 0=leave MIT running
#define PARAM_HIP_RUNNING_KP      0x0201  // MIT position gain used for hip setpoints while RUNNING (control_loop.cpp)
#define PARAM_HIP_RUNNING_KD      0x0202  // MIT damping gain used for hip setpoints while RUNNING (control_loop.cpp)
#define PARAM_HIP_RUNNING_TFF     0x0203  // MIT feedforward torque used for hip setpoints while RUNNING (control_loop.cpp)
// kp/tff ramp from 0 to their running value over this many seconds after
// entering RUNNING, so the hip eases into the commanded position instead of
// snapping there at full stiffness. kd is applied at full value throughout
// (damping only, no position pull). 0 = no ramp (snap immediately, old behavior).
#define PARAM_HIP_RUNNING_RAMP_TIME_S 0x0204

// GROUP_WHEEL — wheel motor settings
// Note: ODrive PID gains (vel_gain, pos_gain, current_lim, etc.) are not exposed here because
// ODrive 0.5.x only allows property access over USB/UART, not CAN. Tune those once via the
// ODrive USB GUI and call save_configuration() to persist them in ODrive flash.
#define PARAM_WM_ENC_TIMEOUT_MS   0x0300  // encoder feedback watchdog [ms]; increase if CAN is flaky

// GROUP_CALIB — hardstop calibration tuning
#define PARAM_CALIB_SEEK_SPEED    0x0100  // ramp speed toward hardstop [rad/s]
// CAL_SEEK_BOTTOM = t=0/"retracted" hardstop (robot weight assists the motor
// here — see PARAM_RADIO_HIP_CMD convention, confirmed via hip_cmd_to_setpoints()):
// less kp/strength needed, and a lower (more sensitive) stall current threshold
// since the assisted baseline current while moving is already low.
#define PARAM_CALIB_KP_BOTTOM     0x0101  // position gain while seeking (SEEK_BOTTOM/retract);
                                           // also reused by CAL_RETURN_HOME's traverse-back move
#define PARAM_CALIB_KD            0x0102  // damping while seeking (both phases)
#define PARAM_CALIB_HOLD_KP       0x0103  // position gain while holding at home
#define PARAM_CALIB_HOLD_KD       0x0104  // damping while holding at home
#define PARAM_CALIB_STALL_CUR_BOTTOM 0x0105  // current threshold to declare a hardstop [A] (SEEK_BOTTOM/retract)
#define PARAM_CALIB_STALL_DEADBAND 0x0106 // max pos movement per tick to count as stalled [rad]
#define PARAM_CALIB_STALL_TICKS   0x0107  // consecutive stalled ticks before declaring hardstop (int stored as float)
#define PARAM_CALIB_MARGIN        0x0108  // safety margin from each hardstop [rad]
#define PARAM_CALIB_SAFETY_BOUND  0x0109  // max CAL_SEEK_BOTTOM travel before fault [rad] —
                                           // worst case, an unknown start position needs up to
                                           // the full joint range to reach the first hardstop.
#define PARAM_CALIB_L_SEEK_DIR    0x010A  // sign (+1/-1) toward left hip bottom hardstop
#define PARAM_CALIB_R_SEEK_DIR    0x010B  // sign (+1/-1) toward right hip bottom hardstop
#define PARAM_CALIB_DONE          0x010C  // 1.0 = calibration completed at least once (persisted across reboots)
#define PARAM_CALIB_SAFETY_BOUND_TOP 0x010D  // max CAL_SEEK_TOP travel before fault [rad] —
                                              // measured from the just-zeroed bottom hardstop,
                                              // so this only needs to cover the joint range once.
// CAL_SEEK_TOP = t=1/"extended" hardstop — motor must fight robot weight here:
// more kp/strength needed, and a higher (less sensitive) stall current threshold
// so the elevated baseline current from fighting gravity doesn't false-trigger.
#define PARAM_CALIB_KP_TOP        0x010E  // position gain while seeking (SEEK_TOP/extend)
#define PARAM_CALIB_STALL_CUR_TOP 0x010F  // current threshold to declare a hardstop [A] (SEEK_TOP/extend)

// Per-direction calibration enable — bench-testing one hardstop direction at a
// time. READONLY: edit + reflash to change. SEEK_BOTTOM (retract) is the
// prerequisite phase (establishes the zero reference), so disabling it skips
// the whole axis, same as a disabled hip motor. Disabling SEEK_TOP (extend)
// lets SEEK_BOTTOM run and zero normally, then holds there (CAL_HOLD_RETRACT)
// instead of continuing — no limits are computed, calibration_done() never
// fires for that axis, so STATE_CALIBRATION won't auto-exit to STANDBY.
#define PARAM_CALIB_RETRACT_ENABLE 0x0110  // 1 = run SEEK_BOTTOM (retract); default 0
#define PARAM_CALIB_EXTEND_ENABLE  0x0111  // 1 = run SEEK_TOP (extend) after retract; default 0

// Once an axis reaches its held home position, kp/kd ramp from the hold
// values down to zero over this many seconds before calibration_done() fires
// — so exiting CALIBRATION (which drops the setpoint entirely) never yanks
// torque from a nonzero value to zero in a single tick. 0 = no ramp (snap to
// zero-torque immediately, old behavior).
#define PARAM_CALIB_RAMPDOWN_TIME_S 0x0112

// Bypass hardstop calibration requirement for RUNNING mode — lets req_running()
// (state_machine.cpp) arm the balance controller without a completed
// calibration (hm_limits_L/R.valid). For bench-testing only: with this on, hip
// position limits are unenforced, so the hips can travel to their physical
// hardstops uncontrolled. Persisted. Default 0 (bypass off).
#define PARAM_CALIB_BYPASS_EN 0x0113

// GROUP_CONTROL — LQR controller settings
#define PARAM_LQR_ENABLE              0x0400  // 1 = wheel torque output active; 0 = LQR runs but outputs zero
#define PARAM_SIM_PITCH_RAD           0x0401  // pitch value to inject when PARAM_ENABLE_SIM_PITCH_RAD=1
#define PARAM_ENABLE_SIM_PITCH_RAD    0x0420  // 1 = use sim_pitch_rad instead of real IMU pitch
#define PARAM_SIM_PITCH_RATE_RAD_S    0x0421  // pitch rate value to inject when PARAM_ENABLE_SIM_PITCH_RATE=1
#define PARAM_ENABLE_SIM_PITCH_RATE   0x0422  // 1 = use sim_pitch_rate_rad_s instead of real IMU pitch rate
// |pitch| > 50 deg for > 200 ms -> ESTOP (FAULT_PITCH_WATCHDOG). Default 1
// (enabled) and NOT persisted — always starts back on after reboot, so a
// bench-test disable never silently survives into a real run.
#define PARAM_PITCH_WATCHDOG_ENABLE   0x0423
#define PARAM_LQR_TORQUE_LIMIT        0x0402  // |tau_sym| clamp [N·m]; READONLY — slewed automatically
                                               // from the active CH9 speed profile's torque_lim (see
                                               // PARAM_PROFILE_*_TORQUE_LIM below); hard max 7.0
#define PARAM_WHEEL_VEL_LIMIT_TURNS_S 0x0403  // per-tick soft governor [turns/s]; ESTOP at 2×
// LQR gain table (Phase 5) — computed offline by lqr.py self-test (Q_pitch=0.01,
// Q_pitch_rate=0.1884, Q_vel=0.00508442, R=100.0). alpha=0 -> retracted (Q_RET),
// alpha=1 -> extended (Q_EXT); K_VEL is invariant across leg height (see
// control_loop.cpp interpolation). Exposed at runtime for bring-up bench
// testing — start near 0 and ramp toward the computed defaults.
#define PARAM_LQR_K_PITCH_RET         0x0424  // pitch gain, fully retracted; default -13.0495742
#define PARAM_LQR_K_RATE_RET          0x0425  // pitch-rate gain, fully retracted; default -2.18083692
#define PARAM_LQR_K_PITCH_EXT         0x0426  // pitch gain, fully extended; default -7.92908352
#define PARAM_LQR_K_RATE_EXT          0x0427  // pitch-rate gain, fully extended; default -1.69084204
#define PARAM_LQR_K_VEL               0x0428  // velocity-error gain (invariant); default -7.13051190e-03
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
#define PARAM_RADIO_HIP_CMD        0x0500  // hip extension command from CH3 [0=retracted, 1=extended]; stale when radio dead
#define PARAM_RADIO_VEL_MAX        0x0501  // max forward speed mapped from full CH2 deflection [m/s];
                                            // READONLY — copied from the active CH9 profile's vel_max
#define PARAM_RADIO_YAW_MAX        0x0502  // max yaw rate mapped from full CH4 deflection [rad/s];
                                            // READONLY — copied from the active CH9 profile's yaw_max
#define PARAM_RADIO_PITCH_TRIM     0x0503  // pitch equilibrium trim from CH7 [rad]; hook only, not yet applied to LQR
// Speed profiles (CH9 3-position switch selects profile 0/1/2)
#define PARAM_PROFILE_1_VEL_MAX    0x0510  // profile 1 (slow) max forward speed [m/s]
#define PARAM_PROFILE_1_YAW_MAX    0x0511  // profile 1 (slow) max yaw rate [rad/s]
#define PARAM_PROFILE_1_TORQUE_LIM 0x0512  // profile 1 (slow) LQR torque limit [N·m]
#define PARAM_PROFILE_2_VEL_MAX    0x0513  // profile 2 (normal) max forward speed [m/s]
#define PARAM_PROFILE_2_YAW_MAX    0x0514  // profile 2 (normal) max yaw rate [rad/s]
#define PARAM_PROFILE_2_TORQUE_LIM 0x0515  // profile 2 (normal) LQR torque limit [N·m]
#define PARAM_PROFILE_3_VEL_MAX    0x0516  // profile 3 (fast) max forward speed [m/s]
#define PARAM_PROFILE_3_YAW_MAX    0x0517  // profile 3 (fast) max yaw rate [rad/s]
#define PARAM_PROFILE_3_TORQUE_LIM 0x0518  // profile 3 (fast) LQR torque limit [N·m]
#define PARAM_ACTIVE_PROFILE       0x0519  // active speed profile index 0/1/2 (firmware-written, READONLY)

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
