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
    // id                        group          name                     value     min       max     flags                   on_change
    // GROUP_SYSTEM — peripheral enable flags (bench-test without full hardware)
    {PARAM_IMU_ENABLE,    GROUP_SYSTEM, "imu_enable",    1.0f, 0.0f, 1.0f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_BUZZER_ENABLE, GROUP_SYSTEM, "buzzer_enable", 1.0f, 0.0f, 1.0f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_LED_ENABLE,    GROUP_SYSTEM, "led_enable",    1.0f, 0.0f, 1.0f, PARAM_FLAG_PERSISTENT, nullptr},

    // Per-motor presence flags — set to 0 to bench-test with that motor physically
    // disconnected. See param_ids.h for details.
    {PARAM_HIP_L_ENABLE,   GROUP_SYSTEM, "hip_l_enable",   0.0f, 0.0f, 1.0f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_HIP_R_ENABLE,   GROUP_SYSTEM, "hip_r_enable",   0.0f, 0.0f, 1.0f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_WHEEL_L_ENABLE, GROUP_SYSTEM, "wheel_l_enable", 0.0f, 0.0f, 1.0f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_WHEEL_R_ENABLE, GROUP_SYSTEM, "wheel_r_enable", 0.0f, 0.0f, 1.0f, PARAM_FLAG_PERSISTENT, nullptr},

    {PARAM_WATCHDOG_ENABLE, GROUP_SYSTEM, "watchdog_enable", 0.0f, 0.0f, 1.0f, PARAM_FLAG_PERSISTENT, nullptr},

    // Debug: 1 = log a per-second breakdown of loop() section timing (see
    // param_ids.h). Not persisted — always starts off.
    {PARAM_LOOP_PROFILE_ENABLE, GROUP_SYSTEM, "loop_profile_enable", 0.0f, 0.0f, 1.0f, 0, nullptr},

    {PARAM_ESTOP_HIP_DISABLE, GROUP_HIP,   "estop_hip_disable",  1.0f,      0.0f,     1.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    // C5: previously hardcoded RUNNING_KP/KD/TFF constants in control_loop.cpp — soft, initial testing values kept as defaults
    {PARAM_HIP_RUNNING_KP,    GROUP_HIP,   "hip_running_kp",     5.0f,      0.0f,   100.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_HIP_RUNNING_KD,    GROUP_HIP,   "hip_running_kd",     0.5f,      0.0f,     5.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_HIP_RUNNING_TFF,   GROUP_HIP,   "hip_running_tff",    0.0f,     -5.0f,     5.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_HIP_RUNNING_RAMP_TIME_S, GROUP_HIP, "hip_running_ramp_s", 2.0f,  0.0f,     10.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_SEEK_SPEED,  GROUP_CALIB, "calib_seek_speed",  0.17453f,  0.01f,    1.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_KP_BOTTOM,   GROUP_CALIB, "calib_kp_bottom",   16.0f,     0.0f,   500.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_KD,          GROUP_CALIB, "calib_kd",           0.05f,    0.0f,     5.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_HOLD_KP,     GROUP_CALIB, "calib_hold_kp",      1.0f,     0.0f,   500.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_HOLD_KD,     GROUP_CALIB, "calib_hold_kd",      0.05f,    0.0f,     5.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_STALL_CUR_BOTTOM, GROUP_CALIB, "calib_stall_cur_btm", 0.75f, 0.1f,  10.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    // Direction-dependent seek tuning: retract (SEEK_BOTTOM, weight-assisted) uses
    // calib_kp/calib_stall_cur above; extend (SEEK_TOP, fights weight) uses these —
    // more strength, less sensitive threshold. Starting guesses — tune on hardware.
    {PARAM_CALIB_KP_TOP,        GROUP_CALIB, "calib_kp_top",        32.0f, 0.0f,  500.0f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_STALL_CUR_TOP, GROUP_CALIB, "calib_stall_cur_top",  1.5f, 0.1f,   10.0f, PARAM_FLAG_PERSISTENT, nullptr},
    // Retract and extend both enabled. READONLY, edit + reflash to change.
    {PARAM_CALIB_RETRACT_ENABLE, GROUP_CALIB, "calib_retract_en", 0.0f, 0.0f, 1.0f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_EXTEND_ENABLE,  GROUP_CALIB, "calib_extend_en",  1.0f, 0.0f, 1.0f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_RAMPDOWN_TIME_S, GROUP_CALIB, "calib_rampdown_s", 2.0f, 0.0f, 10.0f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_BYPASS_EN,   GROUP_CALIB, "calib_bypass_en",  1.0f, 0.0f, 1.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_STALL_DEADBAND, GROUP_CALIB, "calib_stall_db",  0.015f,   0.001f,  0.5f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_STALL_TICKS, GROUP_CALIB, "calib_stall_ticks",  60.0f,    5.0f,  500.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_MARGIN,      GROUP_CALIB, "calib_margin",       0.17453f, 0.0f,    1.5708f, PARAM_FLAG_PERSISTENT, nullptr},
    // Joint mechanical range is ~90 deg; start position within it is unknown, so
    // SEEK_BOTTOM budgets a full 90 deg (worst case) and SEEK_TOP budgets 180 deg
    // (2x range) from the just-zeroed hardstop for margin. Geometry-derived —
    // READONLY, edit + reflash to change.
    {PARAM_CALIB_SAFETY_BOUND,GROUP_CALIB, "calib_safety_bound", 1.5708f,  0.5f,   6.28319f, PARAM_FLAG_READONLY, nullptr},
    // Wiring-direction constants, not live tuning knobs — READONLY (no PERSISTENT)
    // so a stale flash-saved value can never shadow the compiled default; the only
    // way to change these is to edit this file and reflash.
    {PARAM_CALIB_L_SEEK_DIR,  GROUP_CALIB, "calib_l_seek_dir",   1.0f,    -1.0f,    1.0f,  PARAM_FLAG_READONLY, nullptr},
    // Flipped from -1.0: on hardware, SEEK_BOTTOM at -1.0 moved the right leg
    // toward extend instead of retract — confirmed backwards on the bench.
    {PARAM_CALIB_R_SEEK_DIR,  GROUP_CALIB, "calib_r_seek_dir",   1.0f,    -1.0f,    1.0f,  PARAM_FLAG_READONLY, nullptr},
    {PARAM_CALIB_DONE,        GROUP_CALIB, "calib_done",         0.0f,     0.0f,    1.0f,  PARAM_FLAG_READONLY, nullptr},
    {PARAM_CALIB_SAFETY_BOUND_TOP, GROUP_CALIB, "calib_bound_top", 3.14159f, 0.5f, 6.28319f, PARAM_FLAG_READONLY, nullptr},

    // GROUP_CONTROL — LQR controller settings
    {PARAM_LQR_ENABLE,              GROUP_CONTROL, "lqr_enable",          1.0f,    0.0f,    1.0f,  0,                     nullptr},
    {PARAM_SIM_PITCH_RAD,           GROUP_CONTROL, "sim_pitch_rad",       0.0f,   -1.5708f, 1.5708f, 0,                  nullptr},
    {PARAM_ENABLE_SIM_PITCH_RAD,    GROUP_CONTROL, "enable_sim_pitch",    0.0f,    0.0f,    1.0f,  0,                     nullptr},
    {PARAM_SIM_PITCH_RATE_RAD_S,    GROUP_CONTROL, "sim_pitch_rate",      0.0f,  -10.0f,   10.0f, 0,                     nullptr},
    {PARAM_ENABLE_SIM_PITCH_RATE,   GROUP_CONTROL, "enable_sim_prate",    0.0f,    0.0f,    1.0f,  0,                     nullptr},
    {PARAM_PITCH_WATCHDOG_ENABLE,   GROUP_CONTROL, "pitch_watchdog_en",    1.0f,    0.0f,    1.0f,  0,                     nullptr},
    // Firmware-slewed from the active CH9 profile's torque_lim (main.cpp radio_update());
    // not independently persisted — READONLY|COMMAND matches other profile/radio-derived params.
    {PARAM_LQR_TORQUE_LIMIT,        GROUP_CONTROL, "lqr_torque_limit",    0.1f,    0.0f,    7.0f,  PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_WHEEL_VEL_LIMIT_TURNS_S, GROUP_CONTROL, "wm_vel_limit",        3.0f,    1.0f,   20.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    // LQR gain table (Phase 5) — see param_ids.h. Range floor keeps sign fixed
    // (magnitude only) so bring-up ramps from near-zero toward the computed
    // default rather than risking a sign flip into positive feedback.
    {PARAM_LQR_K_PITCH_RET, GROUP_CONTROL, "lqr_k_pitch_ret", -13.0495742f, -20.0f, 0.0f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_LQR_K_RATE_RET,  GROUP_CONTROL, "lqr_k_rate_ret",   -2.18083692f, -5.0f, 0.0f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_LQR_K_PITCH_EXT, GROUP_CONTROL, "lqr_k_pitch_ext",  -7.92908352f,-15.0f, 0.0f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_LQR_K_RATE_EXT,  GROUP_CONTROL, "lqr_k_rate_ext",   -1.69084204f, -5.0f, 0.0f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_LQR_K_VEL,       GROUP_CONTROL, "lqr_k_vel",        -7.13051190e-03f, -0.05f, 0.0f, PARAM_FLAG_PERSISTENT, nullptr},
    // Not persisted (see param_ids.h) — always boots to 0 (bypass off).
    {PARAM_RUNNING_WHEEL_BYPASS_EN, GROUP_CONTROL, "run_wheel_bypass_en", 0.0f, 0.0f, 1.0f, 0, nullptr},
    // Velocity PI (Phase 3)
    {PARAM_VEL_PI_EN,               GROUP_CONTROL, "vel_pi_en",           0.0f,    0.0f,    1.0f,  0,                     nullptr},
    {PARAM_VEL_PI_KP,               GROUP_CONTROL, "vel_pi_kp",           0.2f,    0.0f,    5.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_VEL_PI_KI,               GROUP_CONTROL, "vel_pi_ki",           0.1f,    0.0f,    5.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_VEL_PI_KFF,              GROUP_CONTROL, "vel_pi_kff",          0.1049f, 0.0f,    1.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_VEL_PI_THETA_MAX,        GROUP_CONTROL, "vel_pi_theta_max",    0.698f,  0.1f,    0.698f,PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_VEL_PI_RATE_LIM,         GROUP_CONTROL, "vel_pi_rate_lim",     1.745f,  0.1f,   10.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_VEL_PI_INT_MAX,          GROUP_CONTROL, "vel_pi_int_max",      1.0f,    0.1f,    5.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_V_CMD_MS,                GROUP_CONTROL, "v_cmd_ms",            0.0f,   -2.0f,    2.0f,  0,                     nullptr},
    // Yaw PI (Phase 4)
    {PARAM_YAW_PI_EN,               GROUP_CONTROL, "yaw_pi_en",           0.0f,    0.0f,    1.0f,  0,                     nullptr},
    {PARAM_YAW_PI_KP,               GROUP_CONTROL, "yaw_pi_kp",           0.1f,    -1.0f,    5.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_YAW_PI_KI,               GROUP_CONTROL, "yaw_pi_ki",           0.2f,    -1.0f,    5.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_YAW_PI_TORQUE_MAX,       GROUP_CONTROL, "yaw_pi_torque_max",   0.5f,    0.0f,    3.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_YAW_PI_INT_MAX,          GROUP_CONTROL, "yaw_pi_int_max",      0.5f,    0.0f,    3.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_OMEGA_CMD_RDS,           GROUP_CONTROL, "omega_cmd_rds",       0.0f,   -4.0f,    4.0f,  0,                     nullptr},
    // Feedforward (Phase 6)
    {PARAM_FF1_ALPHA,               GROUP_CONTROL, "ff1_alpha",           0.0f,    0.0f,    1.0f,  0,                     nullptr},
    {PARAM_FF2_ALPHA,               GROUP_CONTROL, "ff2_alpha",           0.0f,    0.0f,    1.0f,  0,                     nullptr},
    // AK45-10 motor torque constant [N·m/A] — hardware characteristic, not a
    // live tuning knob. READONLY, edit + reflash to change.
    {PARAM_FF1_KT_HIP,              GROUP_CONTROL, "ff1_kt_hip",          1.2732f, 0.0f,    5.0f,  PARAM_FLAG_READONLY, nullptr},
    // Jump controller (Phase 7)
    {PARAM_JUMP_ENABLE,             GROUP_CONTROL, "jump_enable",         0.0f,    0.0f,    1.0f,  0,                     nullptr},
    {PARAM_JUMP_TORQUE_MAX,         GROUP_CONTROL, "jump_torque_max",     0.0f,    0.0f,   18.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_JUMP_CROUCH_TIME_S,      GROUP_CONTROL, "jump_crouch_time",    0.30f,   0.05f,   1.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_JUMP_RAMP_UP_S,          GROUP_CONTROL, "jump_ramp_up",        0.05f,   0.005f,  0.5f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_JUMP_RAMP_DOWN_RAD,      GROUP_CONTROL, "jump_ramp_down",      0.08f,   0.01f,   0.5f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_JUMP_OMEGA_MAX,          GROUP_CONTROL, "jump_omega_max",     40.0f,    5.0f,  200.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_JUMP_HARDSTOP_MARGIN,    GROUP_CONTROL, "jump_hs_margin",      0.06f,   0.01f,   0.5f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_JUMP_KP,                 GROUP_CONTROL, "jump_kp",            80.0f,    0.0f,  500.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_JUMP_KD,                 GROUP_CONTROL, "jump_kd",             1.0f,    0.0f,    5.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_JUMP_EXTEND_KD,          GROUP_CONTROL, "jump_ext_kd",         0.1f,    0.0f,    5.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_JUMP_EXTEND_TIMEOUT_S,   GROUP_CONTROL, "jump_ext_timeout",    0.15f,   0.05f,   1.0f,  PARAM_FLAG_PERSISTENT, nullptr},

    // GROUP_WHEEL — wheel motor settings
    {PARAM_WM_ENC_TIMEOUT_MS, GROUP_WHEEL, "wm_enc_timeout_ms", 20.0f,  5.0f, 500.0f, PARAM_FLAG_PERSISTENT, nullptr},

    // GROUP_COMMAND — radio-derived setpoints (firmware-written, never persisted)
    {PARAM_RADIO_HIP_CMD,        GROUP_COMMAND, "radio_hip_cmd",        0.0f,  0.0f,    1.0f,  PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    // Firmware-copied from the active CH9 profile's vel_max/yaw_max (main.cpp radio_update());
    // not independently persisted — READONLY|COMMAND matches other profile/radio-derived params.
    {PARAM_RADIO_VEL_MAX,        GROUP_COMMAND, "radio_vel_max",        0.5f,  0.0f,    2.0f,  PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_RADIO_YAW_MAX,        GROUP_COMMAND, "radio_yaw_max",        1.0f,  0.0f,    4.0f,  PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_RADIO_PITCH_TRIM,     GROUP_COMMAND, "radio_pitch_trim",     0.0f, -0.0873f, 0.0873f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    // Speed profiles — each set of three is copied into the active params when CH9 switches
    {PARAM_PROFILE_1_VEL_MAX,    GROUP_COMMAND, "profile1_vel_max",     0.2f,  0.0f,    2.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_PROFILE_1_YAW_MAX,    GROUP_COMMAND, "profile1_yaw_max",     0.5f,  0.0f,    4.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_PROFILE_1_TORQUE_LIM, GROUP_COMMAND, "profile1_torque_lim",  0.01f,  0.0f,    7.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_PROFILE_2_VEL_MAX,    GROUP_COMMAND, "profile2_vel_max",     0.5f,  0.0f,    2.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_PROFILE_2_YAW_MAX,    GROUP_COMMAND, "profile2_yaw_max",     1.0f,  0.0f,    4.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_PROFILE_2_TORQUE_LIM, GROUP_COMMAND, "profile2_torque_lim",  0.05f,  0.0f,    7.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_PROFILE_3_VEL_MAX,    GROUP_COMMAND, "profile3_vel_max",     1.0f,  0.0f,    2.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_PROFILE_3_YAW_MAX,    GROUP_COMMAND, "profile3_yaw_max",     2.0f,  0.0f,    4.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_PROFILE_3_TORQUE_LIM, GROUP_COMMAND, "profile3_torque_lim",  0.1f,  0.0f,    7.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_ACTIVE_PROFILE,       GROUP_COMMAND, "active_profile",       0.0f,  0.0f,    2.0f,  PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},

    // GROUP_IBUS — RC receiver live channel readings (firmware-written via param_force_set)
    {PARAM_IBUS_CH0,   GROUP_IBUS, "ibus_ch0",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH1,   GROUP_IBUS, "ibus_ch1",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH2,   GROUP_IBUS, "ibus_ch2",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH3,   GROUP_IBUS, "ibus_ch3",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH4,   GROUP_IBUS, "ibus_ch4",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH5,   GROUP_IBUS, "ibus_ch5",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH6,   GROUP_IBUS, "ibus_ch6",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH7,   GROUP_IBUS, "ibus_ch7",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH8,   GROUP_IBUS, "ibus_ch8",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH9,   GROUP_IBUS, "ibus_ch9",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH10,  GROUP_IBUS, "ibus_ch10",  1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH11,  GROUP_IBUS, "ibus_ch11",  1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH12,  GROUP_IBUS, "ibus_ch12",  1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH13,  GROUP_IBUS, "ibus_ch13",  1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_ALIVE, GROUP_IBUS, "ibus_alive",    0.0f,    0.0f,    1.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
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
