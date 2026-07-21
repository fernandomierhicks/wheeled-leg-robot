#include "param_registry.h"
#include "comm_protocol.h"
#include <LittleFS.h>

// ── Flash filesystem ──────────────────────────────────────────────────────────
// This board has no onboard QSPI NOR/NAND chip (confirmed on hardware — begin()
// always failed), so params live in a carved-out slice of the Teensy's own
// program flash instead. No extra hardware needed.
// Must be >= 2 sectors — LittleFS stores its metadata as a redundant pair and
// won't format a 1-block volume. Teensy 4.1's LittleFS_Program sector size is
// 64 KiB (SECTOR_SIZE in the framework's LittleFS.cpp), so the minimum viable
// size is 128 KiB; 256 KiB (4 sectors) leaves headroom and is still negligible
// next to the 7.75 MB available.
static LittleFS_Program s_fs;
static const uint32_t FS_SIZE_BYTES = 256 * 1024;
static const char*   PARAMS_FILE = "/params.bin";
static const uint16_t MAGIC      = 0xB0B1;
static const uint8_t  VERSION    = 1;

// ── Deferred flash flush ──────────────────────────────────────────────────────
// save_to_flash() is a full LittleFS remove+rewrite — far too slow to run
// inside the 500 Hz loop (and the CH9 profile torque slew calls param_set()
// every tick while ramping). PERSISTENT writes therefore only mark a dirty
// flag; param_flush_service() performs the actual write once writes have been
// quiet for FLUSH_QUIET_MS and the caller allows it (not RUNNING/JUMPING).
static bool           s_dirty          = false;
static uint32_t       s_last_change_ms = 0;
static const uint32_t FLUSH_QUIET_MS   = 1000;

// ── Registry table ────────────────────────────────────────────────────────────
// Add new params here. Defaults are the compile-time values previously in config.h.
// IMPORTANT: After adding a param here and in param_ids.h, update
//            software/gui/tabs/params_tab.py:
//              - _SUBGROUPS, so the GUI shows it in the correct section
//              - _PARAM_DEFS, adding this id's (name, description) entry —
//                name copied from the "name" string below, description from
//                the comment on this param's #define in param_ids.h.
// clang-format off
static Param g_params[] = {
    // Designated initializers (.id = ..., .value = ...) on purpose: a dropped
    // comma in a plain positional {a, b, c, ...} list doesn't fail to parse,
    // it silently shifts every later value into the wrong field (bit us once —
    // see git history). Naming each field makes that a hard compile error instead.
    //
    // Rows are grouped/ordered to mirror software/gui/tabs/params_tab.py's
    // _PARAM_DEFS — keep both in sync when adding/moving a param.

    // GROUP_SYSTEM — peripheral enable flags (bench-test without full hardware)
    {.id = PARAM_IMU_ENABLE,              .group_id = GROUP_SYSTEM,  .name = "imu_enable",          .value = 1.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_BUZZER_VOLUME,           .group_id = GROUP_SYSTEM,  .name = "buzzer_volume",       .value = 1.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_LED_ENABLE,              .group_id = GROUP_SYSTEM,  .name = "led_enable",          .value = 1.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    // Per-motor presence flags — set to 0 to bench-test with that motor physically
    // disconnected. See param_ids.h for details.
    {.id = PARAM_HIP_L_ENABLE,            .group_id = GROUP_SYSTEM,  .name = "hip_l_enable",        .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_HIP_R_ENABLE,            .group_id = GROUP_SYSTEM,  .name = "hip_r_enable",        .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_WHEEL_L_ENABLE,          .group_id = GROUP_SYSTEM,  .name = "wheel_l_enable",      .value = 1.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_WHEEL_R_ENABLE,          .group_id = GROUP_SYSTEM,  .name = "wheel_r_enable",      .value = 1.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_WATCHDOG_ENABLE,         .group_id = GROUP_SYSTEM,  .name = "watchdog_enable",     .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    // Debug: 1 = log a per-second breakdown of loop() section timing (see
    // param_ids.h). Not persisted — always starts off.
    {.id = PARAM_LOOP_PROFILE_ENABLE,     .group_id = GROUP_SYSTEM,  .name = "loop_profile_enable", .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = 0,                                      .on_change = nullptr},

    // GROUP_HIP — hip motor RUNNING-mode gains
    // C5: previously hardcoded RUNNING_KP/KD/TFF constants in control_loop.cpp — soft, initial testing values kept as defaults
    {.id = PARAM_HIP_RUNNING_KP,          .group_id = GROUP_HIP,     .name = "hip_running_kp",      .value = 5.0f,     .min_val = 0.0f,     .max_val = 100.0f,   .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_HIP_RUNNING_KD,          .group_id = GROUP_HIP,     .name = "hip_running_kd",      .value = 0.5f,     .min_val = 0.0f,     .max_val = 5.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_HIP_RUNNING_TFF,         .group_id = GROUP_HIP,     .name = "hip_running_tff",     .value = 0.0f,     .min_val = -5.0f,    .max_val = 5.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_HIP_RUNNING_RAMP_TIME_S, .group_id = GROUP_HIP,     .name = "hip_running_ramp_s",  .value = 2.0f,     .min_val = 0.0f,     .max_val = 10.0f,    .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_ESTOP_HIP_DISABLE,       .group_id = GROUP_HIP,     .name = "estop_hip_disable",   .value = 1.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},

    // GROUP_CALIB — hip hardstop calibration/homing sequence
    {.id = PARAM_CALIB_SEEK_SPEED,        .group_id = GROUP_CALIB,   .name = "calib_seek_speed",    .value = 0.17453f, .min_val = 0.01f,    .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_CALIB_KP_BOTTOM,         .group_id = GROUP_CALIB,   .name = "calib_kp_bottom",     .value = 16.0f,    .min_val = 0.0f,     .max_val = 500.0f,   .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_CALIB_KD,                .group_id = GROUP_CALIB,   .name = "calib_kd",            .value = 0.05f,    .min_val = 0.0f,     .max_val = 5.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_CALIB_HOLD_KP,           .group_id = GROUP_CALIB,   .name = "calib_hold_kp",       .value = 1.0f,     .min_val = 0.0f,     .max_val = 500.0f,   .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_CALIB_HOLD_KD,           .group_id = GROUP_CALIB,   .name = "calib_hold_kd",       .value = 0.05f,    .min_val = 0.0f,     .max_val = 5.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_CALIB_STALL_CUR_BOTTOM,  .group_id = GROUP_CALIB,   .name = "calib_stall_cur_btm", .value = 0.75f,    .min_val = 0.1f,     .max_val = 10.0f,    .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_CALIB_STALL_DEADBAND,    .group_id = GROUP_CALIB,   .name = "calib_stall_db",      .value = 0.015f,   .min_val = 0.001f,   .max_val = 0.5f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_CALIB_STALL_TICKS,       .group_id = GROUP_CALIB,   .name = "calib_stall_ticks",   .value = 60.0f,    .min_val = 5.0f,     .max_val = 500.0f,   .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_CALIB_MARGIN,            .group_id = GROUP_CALIB,   .name = "calib_margin",        .value = 0.17453f, .min_val = 0.0f,     .max_val = 1.5708f,  .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    // Joint mechanical range is ~90 deg; start position within it is unknown, so
    // SEEK_BOTTOM budgets a full 90 deg (worst case) and SEEK_TOP budgets 180 deg
    // (2x range) from the just-zeroed hardstop for margin. Geometry-derived —
    // READONLY, edit + reflash to change.
    {.id = PARAM_CALIB_SAFETY_BOUND,      .group_id = GROUP_CALIB,   .name = "calib_safety_bound",  .value = 1.5708f,  .min_val = 0.5f,     .max_val = 6.28319f, .flags = PARAM_FLAG_READONLY,                    .on_change = nullptr},
    // Wiring-direction constants, not live tuning knobs — READONLY (no PERSISTENT)
    // so a stale flash-saved value can never shadow the compiled default; the only
    // way to change these is to edit this file and reflash.
    {.id = PARAM_CALIB_L_SEEK_DIR,        .group_id = GROUP_CALIB,   .name = "calib_l_seek_dir",    .value = 1.0f,     .min_val = -1.0f,    .max_val = 1.0f,     .flags = PARAM_FLAG_READONLY,                    .on_change = nullptr},
    // Flipped from -1.0: on hardware, SEEK_BOTTOM at -1.0 moved the right leg
    // toward extend instead of retract — confirmed backwards on the bench.
    {.id = PARAM_CALIB_R_SEEK_DIR,        .group_id = GROUP_CALIB,   .name = "calib_r_seek_dir",    .value = 1.0f,     .min_val = -1.0f,    .max_val = 1.0f,     .flags = PARAM_FLAG_READONLY,                    .on_change = nullptr},
    {.id = PARAM_CALIB_DONE,              .group_id = GROUP_CALIB,   .name = "calib_done",          .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_READONLY,                    .on_change = nullptr},
    {.id = PARAM_CALIB_SAFETY_BOUND_TOP,  .group_id = GROUP_CALIB,   .name = "calib_bound_top",     .value = 3.14159f, .min_val = 0.5f,     .max_val = 6.28319f, .flags = PARAM_FLAG_READONLY,                    .on_change = nullptr},
    // Direction-dependent seek tuning: retract (SEEK_BOTTOM, weight-assisted) uses
    // calib_kp/calib_stall_cur above; extend (SEEK_TOP, fights weight) uses these —
    // more strength, less sensitive threshold. Starting guesses — tune on hardware.
    {.id = PARAM_CALIB_KP_TOP,            .group_id = GROUP_CALIB,   .name = "calib_kp_top",        .value = 32.0f,    .min_val = 0.0f,     .max_val = 500.0f,   .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_CALIB_STALL_CUR_TOP,     .group_id = GROUP_CALIB,   .name = "calib_stall_cur_top", .value = 1.5f,     .min_val = 0.1f,     .max_val = 10.0f,    .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    // Retract and extend both enabled. READONLY, edit + reflash to change.
    {.id = PARAM_CALIB_RETRACT_ENABLE,    .group_id = GROUP_CALIB,   .name = "calib_retract_en",    .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_CALIB_EXTEND_ENABLE,     .group_id = GROUP_CALIB,   .name = "calib_extend_en",     .value = 1.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_CALIB_RAMPDOWN_TIME_S,   .group_id = GROUP_CALIB,   .name = "calib_rampdown_s",    .value = 2.0f,     .min_val = 0.0f,     .max_val = 10.0f,    .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_CALIB_BYPASS_EN,         .group_id = GROUP_CALIB,   .name = "calib_bypass_en",     .value = 1.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},

    // GROUP_WHEEL — wheel motor settings
    {.id = PARAM_WM_ENC_TIMEOUT_MS,       .group_id = GROUP_WHEEL,   .name = "wm_enc_timeout_ms",   .value = 20.0f,    .min_val = 5.0f,     .max_val = 500.0f,   .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},

    // GROUP_CONTROL — LQR core / limits
    {.id = PARAM_LQR_ENABLE,              .group_id = GROUP_CONTROL, .name = "lqr_enable",          .value = 1.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_SIM_PITCH_RAD,           .group_id = GROUP_CONTROL, .name = "sim_pitch_rad",       .value = 0.0f,     .min_val = -1.5708f, .max_val = 1.5708f,  .flags = 0,                                      .on_change = nullptr},
    // Firmware-slewed from the active CH9 profile's torque_lim (main.cpp radio_update());
    // not independently persisted — READONLY|COMMAND matches other profile/radio-derived params.
    {.id = PARAM_LQR_TORQUE_LIMIT,        .group_id = GROUP_CONTROL, .name = "lqr_torque_limit",    .value = 0.1f,     .min_val = 0.0f,     .max_val = 7.0f,     .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_WHEEL_VEL_LIMIT_TURNS_S, .group_id = GROUP_CONTROL, .name = "wm_vel_limit",        .value = 3.0f,     .min_val = 1.0f,     .max_val = 20.0f,    .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},

    // Velocity PI 
    {.id = PARAM_VEL_PI_EN,               .group_id = GROUP_CONTROL, .name = "vel_pi_en",           .value = 1.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_VEL_PI_KP,               .group_id = GROUP_CONTROL, .name = "vel_pi_kp",           .value = 0.2f,     .min_val = 0.0f,     .max_val = 5.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_VEL_PI_KI,               .group_id = GROUP_CONTROL, .name = "vel_pi_ki",           .value = 0.1f,     .min_val = 0.0f,     .max_val = 5.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_VEL_PI_KFF,              .group_id = GROUP_CONTROL, .name = "vel_pi_kff",          .value = 0.1049f,  .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_VEL_PI_THETA_MAX,        .group_id = GROUP_CONTROL, .name = "vel_pi_theta_max",    .value = 0.698f,   .min_val = 0.1f,     .max_val = 0.698f,   .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_VEL_PI_RATE_LIM,         .group_id = GROUP_CONTROL, .name = "vel_pi_rate_lim",     .value = 1.745f,   .min_val = 0.1f,     .max_val = 10.0f,    .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_VEL_PI_INT_MAX,          .group_id = GROUP_CONTROL, .name = "vel_pi_int_max",      .value = 1.0f,     .min_val = 0.1f,     .max_val = 5.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_V_CMD_MS,                .group_id = GROUP_CONTROL, .name = "v_cmd_ms",            .value = 0.0f,     .min_val = -2.0f,    .max_val = 2.0f,     .flags = 0,                                      .on_change = nullptr},

    // Yaw PI 
    {.id = PARAM_YAW_PI_EN,               .group_id = GROUP_CONTROL, .name = "yaw_pi_en",           .value = 1.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_YAW_PI_KP,               .group_id = GROUP_CONTROL, .name = "yaw_pi_kp",           .value = 0.2f,     .min_val = 0.0f,     .max_val = 5.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_YAW_PI_KI,               .group_id = GROUP_CONTROL, .name = "yaw_pi_ki",           .value = 0.1f,     .min_val = 0.0f,     .max_val = 5.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_YAW_PI_TORQUE_MAX,       .group_id = GROUP_CONTROL, .name = "yaw_pi_torque_max",   .value = 0.2f,     .min_val = 0.0f,     .max_val = 3.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_YAW_PI_INT_MAX,          .group_id = GROUP_CONTROL, .name = "yaw_pi_int_max",      .value = 0.5f,     .min_val = 0.0f,     .max_val = 3.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_OMEGA_CMD_RDS,           .group_id = GROUP_CONTROL, .name = "omega_cmd_rds",       .value = 0.0f,     .min_val = -4.0f,    .max_val = 4.0f,     .flags = 0,                                      .on_change = nullptr},

    // Feedforward 
    {.id = PARAM_FF1_ALPHA,               .group_id = GROUP_CONTROL, .name = "ff1_alpha",           .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = 0,                                      .on_change = nullptr},
    {.id = PARAM_FF2_ALPHA,               .group_id = GROUP_CONTROL, .name = "ff2_alpha",           .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = 0,                                      .on_change = nullptr},
    // AK45-10 motor torque constant [N·m/A] — hardware characteristic, not a
    // live tuning knob. READONLY, edit + reflash to change.
    {.id = PARAM_FF1_KT_HIP,              .group_id = GROUP_CONTROL, .name = "ff1_kt_hip",          .value = 1.2732f,  .min_val = 0.0f,     .max_val = 5.0f,     .flags = PARAM_FLAG_READONLY,                    .on_change = nullptr},

    // Jump controller (Phase 7)
    {.id = PARAM_JUMP_ENABLE,             .group_id = GROUP_CONTROL, .name = "jump_enable",         .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_JUMP_TORQUE_MAX,         .group_id = GROUP_CONTROL, .name = "jump_torque_max",     .value = 0.0f,     .min_val = 0.0f,     .max_val = 18.0f,    .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_JUMP_CROUCH_TIME_S,      .group_id = GROUP_CONTROL, .name = "jump_crouch_time",    .value = 0.30f,    .min_val = 0.05f,    .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_JUMP_RAMP_UP_S,          .group_id = GROUP_CONTROL, .name = "jump_ramp_up",        .value = 0.05f,    .min_val = 0.005f,   .max_val = 0.5f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_JUMP_RAMP_DOWN_RAD,      .group_id = GROUP_CONTROL, .name = "jump_ramp_down",      .value = 0.08f,    .min_val = 0.01f,    .max_val = 0.5f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_JUMP_OMEGA_MAX,          .group_id = GROUP_CONTROL, .name = "jump_omega_max",      .value = 40.0f,    .min_val = 5.0f,     .max_val = 200.0f,   .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_JUMP_HARDSTOP_MARGIN,    .group_id = GROUP_CONTROL, .name = "jump_hs_margin",      .value = 0.06f,    .min_val = 0.01f,    .max_val = 0.5f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_JUMP_KP,                 .group_id = GROUP_CONTROL, .name = "jump_kp",             .value = 80.0f,    .min_val = 0.0f,     .max_val = 500.0f,   .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_JUMP_KD,                 .group_id = GROUP_CONTROL, .name = "jump_kd",             .value = 1.0f,     .min_val = 0.0f,     .max_val = 5.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_JUMP_EXTEND_KD,          .group_id = GROUP_CONTROL, .name = "jump_ext_kd",         .value = 0.1f,     .min_val = 0.0f,     .max_val = 5.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_JUMP_EXTEND_TIMEOUT_S,   .group_id = GROUP_CONTROL, .name = "jump_ext_timeout",    .value = 0.15f,    .min_val = 0.05f,    .max_val = 1.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},

    // Sim injection
    {.id = PARAM_ENABLE_SIM_PITCH_RAD,    .group_id = GROUP_CONTROL, .name = "enable_sim_pitch",    .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = 0,                                      .on_change = nullptr},
    {.id = PARAM_SIM_PITCH_RATE_RAD_S,    .group_id = GROUP_CONTROL, .name = "sim_pitch_rate",      .value = 0.0f,     .min_val = -10.0f,   .max_val = 10.0f,    .flags = 0,                                      .on_change = nullptr},
    {.id = PARAM_ENABLE_SIM_PITCH_RATE,   .group_id = GROUP_CONTROL, .name = "enable_sim_prate",    .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = 0,                                      .on_change = nullptr},
    {.id = PARAM_PITCH_WATCHDOG_ENABLE,   .group_id = GROUP_CONTROL, .name = "pitch_watchdog_en",   .value = 1.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = 0,                                      .on_change = nullptr},
    // LQR gain table (Phase 5) — see param_ids.h. Range floor keeps sign fixed
    // (magnitude only) so bring-up ramps from near-zero toward the computed
    // default rather than risking a sign flip into positive feedback.
    {.id = PARAM_LQR_K_PITCH_RET,         .group_id = GROUP_CONTROL, .name = "lqr_k_pitch_ret",     .value = -0.3,     .min_val = -20.0f,   .max_val = 0.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_LQR_K_RATE_RET,          .group_id = GROUP_CONTROL, .name = "lqr_k_rate_ret",      .value = -0.1,    .min_val = -5.0f,    .max_val = 0.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_LQR_K_PITCH_EXT,         .group_id = GROUP_CONTROL, .name = "lqr_k_pitch_ext",     .value = -0.3,     .min_val = -15.0f,   .max_val = 0.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_LQR_K_RATE_EXT,          .group_id = GROUP_CONTROL, .name = "lqr_k_rate_ext",      .value = -0.1,    .min_val = -5.0f,    .max_val = 0.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_LQR_K_VEL,               .group_id = GROUP_CONTROL, .name = "lqr_k_vel",           .value = -0.007f,  .min_val = -0.05f,   .max_val = 0.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    // Not persisted (see param_ids.h) — always boots to 0 (bypass off).
    {.id = PARAM_RUNNING_WHEEL_BYPASS_EN, .group_id = GROUP_CONTROL, .name = "run_wheel_bypass_en", .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = 0,                                      .on_change = nullptr},
    // Not persisted (see param_ids.h) — always boots to 0 (real/uncalibrated alpha behavior).
    {.id = PARAM_ALPHA_FORCE_RETRACTED_EN,.group_id = GROUP_CONTROL, .name = "alpha_force_ret_en",  .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = 0,                                      .on_change = nullptr},
    // Not persisted (see param_ids.h) — always boots to 0 (radio-sourced motion).
    {.id = PARAM_GUI_MOTION_CTRL_EN,      .group_id = GROUP_CONTROL, .name = "gui_motion_ctrl_en",  .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = 0,                                      .on_change = nullptr},

    // Standing-up recovery controller — see standing_up.md. Physical-torque-output
    // gains start at 0 (must be tuned up on the bench), matching the PARAM_JUMP_*
    // "off/inert until tested" convention.
    {.id = PARAM_STANDUP_ENABLE,               .group_id = GROUP_CONTROL, .name = "standup_enable",     .value = 0.0f,    .min_val = 0.0f,   .max_val = 1.0f,   .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_MAX_PITCH_FWD_RAD,    .group_id = GROUP_CONTROL, .name = "standup_pitch_fwd",  .value = 0.6f,    .min_val = 0.0f,   .max_val = 1.4f,   .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_MAX_PITCH_BWD_RAD,    .group_id = GROUP_CONTROL, .name = "standup_pitch_bwd",  .value = 0.6f,    .min_val = 0.0f,   .max_val = 1.4f,   .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_CROUCH_KP,            .group_id = GROUP_CONTROL, .name = "standup_crouch_kp",  .value = 80.0f,   .min_val = 0.0f,   .max_val = 500.0f, .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_CROUCH_KD,            .group_id = GROUP_CONTROL, .name = "standup_crouch_kd",  .value = 1.0f,    .min_val = 0.0f,   .max_val = 5.0f,   .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_CROUCH_TIME_S,        .group_id = GROUP_CONTROL, .name = "standup_crouch_time",.value = 0.30f,   .min_val = 0.05f,  .max_val = 2.0f,   .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_K_PITCH,              .group_id = GROUP_CONTROL, .name = "standup_k_pitch",    .value = 0.0f,    .min_val = 0.0f,   .max_val = 60.0f,  .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_K_RATE,               .group_id = GROUP_CONTROL, .name = "standup_k_rate",     .value = 0.0f,    .min_val = 0.0f,   .max_val = 15.0f,  .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_TORQUE_LIMIT,         .group_id = GROUP_CONTROL, .name = "standup_torque_lim", .value = 0.0f,    .min_val = 0.0f,   .max_val = 7.0f,   .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_WHEEL_VEL_LIMIT_TURNS_S, .group_id = GROUP_CONTROL, .name = "standup_vel_limit", .value = 3.0f,  .min_val = 1.0f,   .max_val = 20.0f,  .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_CAPTURE_PITCH_RAD,    .group_id = GROUP_CONTROL, .name = "standup_cap_pitch",  .value = 0.12f,   .min_val = 0.02f,  .max_val = 0.4f,   .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_CAPTURE_RATE_RADS,    .group_id = GROUP_CONTROL, .name = "standup_cap_rate",   .value = 1.0f,    .min_val = 0.1f,   .max_val = 5.0f,   .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_CAPTURE_HOLD_S,       .group_id = GROUP_CONTROL, .name = "standup_cap_hold",   .value = 0.15f,   .min_val = 0.02f,  .max_val = 1.0f,   .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_ATTEMPT_TIMEOUT_S,    .group_id = GROUP_CONTROL, .name = "standup_timeout",    .value = 1.5f,    .min_val = 0.2f,   .max_val = 5.0f,   .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_MAX_RETRIES,          .group_id = GROUP_CONTROL, .name = "standup_max_retries",.value = 2.0f,    .min_val = 0.0f,   .max_val = 10.0f,  .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},
    {.id = PARAM_STANDUP_RETRY_PAUSE_S,        .group_id = GROUP_CONTROL, .name = "standup_retry_pause",.value = 0.3f,    .min_val = 0.0f,   .max_val = 3.0f,   .flags = PARAM_FLAG_PERSISTENT, .on_change = nullptr},

    // GROUP_COMMAND — radio-derived setpoints (firmware-written, never persisted)
    {.id = PARAM_RADIO_HIP_CMD,           .group_id = GROUP_COMMAND, .name = "radio_hip_cmd",       .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    // Firmware-copied from the active CH9 profile's vel_max/yaw_max (main.cpp radio_update());
    // not independently persisted — READONLY|COMMAND matches other profile/radio-derived params.
    {.id = PARAM_RADIO_VEL_MAX,           .group_id = GROUP_COMMAND, .name = "radio_vel_max",       .value = 0.5f,     .min_val = 0.0f,     .max_val = 2.0f,     .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_RADIO_YAW_MAX,           .group_id = GROUP_COMMAND, .name = "radio_yaw_max",       .value = 1.0f,     .min_val = 0.0f,     .max_val = 4.0f,     .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_RADIO_PITCH_TRIM,        .group_id = GROUP_COMMAND, .name = "radio_pitch_trim",    .value = 0.0f,     .min_val = -0.0873f, .max_val = 0.0873f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    // Speed profiles — each set of three is copied into the active params when CH9 switches
    {.id = PARAM_PROFILE_1_VEL_MAX,       .group_id = GROUP_COMMAND, .name = "profile1_vel_max",    .value = 0.2f,     .min_val = 0.0f,     .max_val = 2.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_PROFILE_1_YAW_MAX,       .group_id = GROUP_COMMAND, .name = "profile1_yaw_max",    .value = 0.5f,     .min_val = 0.0f,     .max_val = 4.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_PROFILE_1_TORQUE_LIM,    .group_id = GROUP_COMMAND, .name = "profile1_torque_lim", .value = 0.1f,     .min_val = 0.0f,     .max_val = 7.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_PROFILE_2_VEL_MAX,       .group_id = GROUP_COMMAND, .name = "profile2_vel_max",    .value = 0.5f,     .min_val = 0.0f,     .max_val = 2.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_PROFILE_2_YAW_MAX,       .group_id = GROUP_COMMAND, .name = "profile2_yaw_max",    .value = 1.0f,     .min_val = 0.0f,     .max_val = 4.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_PROFILE_2_TORQUE_LIM,    .group_id = GROUP_COMMAND, .name = "profile2_torque_lim", .value = 0.2f,     .min_val = 0.0f,     .max_val = 7.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_PROFILE_3_VEL_MAX,       .group_id = GROUP_COMMAND, .name = "profile3_vel_max",    .value = 1.0f,     .min_val = 0.0f,     .max_val = 2.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_PROFILE_3_YAW_MAX,       .group_id = GROUP_COMMAND, .name = "profile3_yaw_max",    .value = 2.0f,     .min_val = 0.0f,     .max_val = 4.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_PROFILE_3_TORQUE_LIM,    .group_id = GROUP_COMMAND, .name = "profile3_torque_lim", .value = 0.3f,     .min_val = 0.0f,     .max_val = 7.0f,     .flags = PARAM_FLAG_PERSISTENT,                  .on_change = nullptr},
    {.id = PARAM_ACTIVE_PROFILE,          .group_id = GROUP_COMMAND, .name = "active_profile",      .value = 0.0f,     .min_val = 0.0f,     .max_val = 2.0f,     .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},

    // GROUP_IBUS — RC receiver live channel readings (firmware-written via param_force_set)
    {.id = PARAM_IBUS_CH0,                .group_id = GROUP_IBUS,    .name = "ibus_ch0",            .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_CH1,                .group_id = GROUP_IBUS,    .name = "ibus_ch1",            .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_CH2,                .group_id = GROUP_IBUS,    .name = "ibus_ch2",            .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_CH3,                .group_id = GROUP_IBUS,    .name = "ibus_ch3",            .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_CH4,                .group_id = GROUP_IBUS,    .name = "ibus_ch4",            .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_CH5,                .group_id = GROUP_IBUS,    .name = "ibus_ch5",            .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_CH6,                .group_id = GROUP_IBUS,    .name = "ibus_ch6",            .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_CH7,                .group_id = GROUP_IBUS,    .name = "ibus_ch7",            .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_CH8,                .group_id = GROUP_IBUS,    .name = "ibus_ch8",            .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_CH9,                .group_id = GROUP_IBUS,    .name = "ibus_ch9",            .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_CH10,               .group_id = GROUP_IBUS,    .name = "ibus_ch10",           .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_CH11,               .group_id = GROUP_IBUS,    .name = "ibus_ch11",           .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_CH12,               .group_id = GROUP_IBUS,    .name = "ibus_ch12",           .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_CH13,               .group_id = GROUP_IBUS,    .name = "ibus_ch13",           .value = 1500.0f,  .min_val = 1000.0f,  .max_val = 2000.0f,  .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
    {.id = PARAM_IBUS_ALIVE,              .group_id = GROUP_IBUS,    .name = "ibus_alive",          .value = 0.0f,     .min_val = 0.0f,     .max_val = 1.0f,     .flags = PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, .on_change = nullptr},
};
// clang-format on

static const uint16_t PARAM_COUNT = sizeof(g_params) / sizeof(g_params[0]);

// Compile-time defaults, captured once in param_init() before load_from_flash()
// overwrites g_params[].value — the only record of "default" once flash restore
// or a GUI param_set() has touched a value. Sized/populated at init, not
// statically initialized, since it must mirror g_params[] exactly.
static float s_defaults[PARAM_COUNT];

// ── Internal helpers ──────────────────────────────────────────────────────────

static Param* find(uint16_t id) {
    for (uint16_t i = 0; i < PARAM_COUNT; i++) {
        if (g_params[i].id == id) return &g_params[i];
    }
    return nullptr;
}

static void load_from_flash() {
    File f = s_fs.open(PARAMS_FILE, FILE_READ);
    if (!f) {
        comm_log(LOG_LEVEL_WARN, "Param flash: no params.bin (first boot, or flash unmounted)");
        return;
    }

    uint16_t magic;
    uint8_t  ver;
    uint16_t count;
    if (f.read(&magic, 2) != 2 || magic != MAGIC) {
        comm_log(LOG_LEVEL_WARN, "Param flash: bad magic in params.bin, ignoring");
        f.close(); return;
    }
    if (f.read(&ver,   1) != 1 || ver   != VERSION) {
        comm_log(LOG_LEVEL_WARN, "Param flash: version mismatch in params.bin, ignoring");
        f.close(); return;
    }
    if (f.read(&count, 2) != 2) { f.close(); return; }

    uint16_t restored = 0;
    for (uint16_t i = 0; i < count; i++) {
        uint16_t id;
        float    val;
        if (f.read(&id,  2) != 2) break;
        if (f.read(&val, 4) != 4) break;
        Param* p = find(id);
        if (p && !(p->flags & PARAM_FLAG_READONLY)) {
            // clamp to current bounds before restoring
            if (val < p->min_val) val = p->min_val;
            if (val > p->max_val) val = p->max_val;
            p->value = val;
            restored++;
        }
    }
    f.close();
    comm_log(LOG_LEVEL_INFO, "Param flash: restored %u/%u params", restored, count);
}

static void save_to_flash() {
    s_fs.remove(PARAMS_FILE);
    File f = s_fs.open(PARAMS_FILE, FILE_WRITE);
    if (!f) {
        comm_log(LOG_LEVEL_ERROR, "Param flash: save FAILED (could not open params.bin for write)");
        return;
    }

    // Count persistent params
    uint16_t count = 0;
    for (uint16_t i = 0; i < PARAM_COUNT; i++) {
        if (g_params[i].flags & PARAM_FLAG_PERSISTENT) count++;
    }

    f.write((uint8_t*)&MAGIC,   2);
    f.write((uint8_t*)&VERSION, 1);
    f.write((uint8_t*)&count,   2);

    for (uint16_t i = 0; i < PARAM_COUNT; i++) {
        if (!(g_params[i].flags & PARAM_FLAG_PERSISTENT)) continue;
        f.write((uint8_t*)&g_params[i].id,    2);
        f.write((uint8_t*)&g_params[i].value, 4);
    }
    f.close();
    comm_log(LOG_LEVEL_INFO, "Param flash: saved %u params", count);
}

// ── Public API ────────────────────────────────────────────────────────────────

void param_init() {
    for (uint16_t i = 0; i < PARAM_COUNT; i++) s_defaults[i] = g_params[i].value;

    bool mounted = s_fs.begin(FS_SIZE_BYTES);
    if (!mounted) {
        s_fs.format();
        mounted = s_fs.begin(FS_SIZE_BYTES);
    }
    if (mounted) {
        comm_log(LOG_LEVEL_INFO, "Param flash: mounted (%s)", s_fs.getMediaName());
    } else {
        comm_log(LOG_LEVEL_ERROR,
                 "Param flash: mount FAILED — params will NOT persist across reboot");
    }
    load_from_flash();
}

ParamSetResult param_set(uint16_t id, float val) {
    Param* p = find(id);
    if (!p)                          return ParamSetResult::NOT_FOUND;
    if (p->flags & PARAM_FLAG_READONLY) return ParamSetResult::READONLY;

    ParamSetResult result = ParamSetResult::OK;

    if (val < p->min_val || val > p->max_val) {
        val    = (val < p->min_val) ? p->min_val : p->max_val;
        result = ParamSetResult::CLAMPED;
    }

    p->value = val;
    if (p->on_change) p->on_change(val);
    if (p->flags & PARAM_FLAG_PERSISTENT) {
        s_dirty          = true;
        s_last_change_ms = millis();
    }

    return result;
}

void param_flush_service(bool allow_flush) {
    if (!s_dirty || !allow_flush) return;
    if (millis() - s_last_change_ms < FLUSH_QUIET_MS) return;
    save_to_flash();
    s_dirty = false;
}

float param_get(uint16_t id) {
    const Param* p = find(id);
    return p ? p->value : 0.0f;
}

bool param_exists(uint16_t id) {
    return find(id) != nullptr;
}

uint16_t param_count() {
    return PARAM_COUNT;
}

uint16_t param_get_group(uint8_t group_id, Param* buf, uint16_t max) {
    uint16_t n = 0;
    for (uint16_t i = 0; i < PARAM_COUNT && n < max; i++) {
        if (g_params[i].group_id == group_id) buf[n++] = g_params[i];
    }
    return n;
}

bool param_by_index(uint16_t idx, Param* out) {
    if (idx >= PARAM_COUNT) return false;
    *out = g_params[idx];
    return true;
}

void param_save_all() {
    save_to_flash();
    s_dirty = false;
}

void param_reset_defaults() {
    for (uint16_t i = 0; i < PARAM_COUNT; i++) {
        if (g_params[i].flags & PARAM_FLAG_READONLY) continue;  // matches param_set()'s own write gate
        g_params[i].value = s_defaults[i];
        if (g_params[i].on_change) g_params[i].on_change(g_params[i].value);
    }
    save_to_flash();
    s_dirty = false;
    comm_log(LOG_LEVEL_WARN, "Param flash: all params reset to compile-time defaults");
}

void param_force_set(uint16_t id, float val) {
    Param* p = find(id);
    if (!p) return;
    p->value = val;
}
