#pragma once
// config.h — All firmware constants: pins, gains, geometry, timing.
// Single source of truth for the firmware build.

// ── Timing ──────────────────────────────────────────────────────────────────
#ifndef LOOP_RATE_HZ
#define LOOP_RATE_HZ        500       // [Hz] main control loop
#endif
#define LOOP_PERIOD_US      (1000000UL / LOOP_RATE_HZ)  // 2000 us

#define TELEMETRY_DIV       10        // send every 10 ticks → 50 Hz
#define COMMAND_DIV         50        // receive every 50 ticks → 10 Hz
#define LED_MATRIX_DIV      167       // update every 167 ticks → ~3 Hz

// ── WiFi credentials ────────────────────────────────────────────────────────
#define WIFI_SSID           "Minnexico"
#define WIFI_PASS           "Fenway724"
#define OTA_HOSTNAME        "BalanceBot"
#define OTA_PASSWORD        "arduino"

// ── Pin assignments ─────────────────────────────────────────────────────────
// BNO086 IMU (SPI)
#define PIN_IMU_CS          10
#define PIN_IMU_INT         2
#define PIN_IMU_RST         3
// SPI bus: SCK=13, MOSI=11, MISO=12 (hardware defaults)

// Built-in LED for timing verification
#define PIN_STATUS_LED      LED_BUILTIN

// ── CAN bus ─────────────────────────────────────────────────────────────────
#ifndef CAN_BAUD
#define CAN_BAUD            1000000   // 1 Mbps
#endif

// CAN node IDs
#define CAN_ID_HIP_L        1         // AK45-10 left hip (MIT CAN)
#define CAN_ID_HIP_R        2         // AK45-10 right hip (MIT CAN)
#define CAN_ID_ODESC         3         // ODESC dual-axis (ODrive CAN)
#define ODESC_AXIS_L        0         // left wheel
#define ODESC_AXIS_R        1         // right wheel

// ── IMU ─────────────────────────────────────────────────────────────────────
#ifndef IMU_RATE_HZ
#define IMU_RATE_HZ         500
#endif

// ── Safety thresholds ───────────────────────────────────────────────────────
#define FALL_ANGLE_RAD      0.785f    // 45° — robot considered fallen
#define IMU_TIMEOUT_MS      10        // IMU watchdog
#define CAN_TIMEOUT_MS      20        // CAN watchdog

// ── Robot geometry (baseline-1, run_id 51167) ───────────────────────────────
#define WHEEL_RADIUS        0.075f    // [m]
#define L_FEMUR             0.17378f  // [m]
#define L_STUB              0.03513f  // [m]
#define L_TIBIA             0.12939f  // [m]
#define L_COUPLER           0.15081f  // [m]
#define Q_RET              (-0.35071f) // [rad] fully retracted
#define Q_EXT              (-1.43161f) // [rad] fully extended
#define Q_NOM              (Q_RET + 0.30f * (Q_EXT - Q_RET))  // 30% stroke

// ── Motor limits ────────────────────────────────────────────────────────────
#define HIP_TORQUE_MAX      7.0f      // [N·m] AK45-10 peak
#define HIP_IMP_TORQUE_MAX  5.0f      // [N·m] impedance mode cap
#define WHEEL_TORQUE_MAX    6.825f    // [N·m] ODESC 50A × Kt

// ── Control gains (from simulation params.py — optimized) ───────────────────
// LQR cost weights (used by tools/export_gains.py to pre-compute K)
#define LQR_Q_PITCH         5.40254f
#define LQR_Q_PITCH_RATE    0.033408f
#define LQR_Q_VEL           0.00165213f
#define LQR_R               39.647f

// VelocityPI
#define VEL_PI_KP           0.05f
#define VEL_PI_KI           0.1f
#define VEL_PI_KFF          0.108498f  // ≈ 1/g feed-forward
#define VEL_PI_THETA_MAX    0.5f       // [rad] max lean angle
#define VEL_PI_INT_MAX      1.0f       // [rad·s] anti-windup
#define VEL_PI_RATE_LIM     5.0f       // [rad/s] theta_ref rate limit

// YawPI
#define YAW_PI_KP           0.193246f
#define YAW_PI_KI           0.221219f
#define YAW_PI_TORQUE_MAX   0.5f       // [N·m]
#define YAW_PI_INT_MAX      0.5f       // [N·m·s]

// Hip position PD
#define HIP_POS_KP          50.0f      // [N·m/rad]
#define HIP_POS_KD          3.0f       // [N·m·s/rad]

// Suspension (impedance + roll leveling)
#define SUSP_K_S            12.9624f   // [N·m/rad]
#define SUSP_B_S            0.109151f  // [N·m·s/rad]
#define SUSP_K_ROLL         45.2559f   // [rad/rad]
#define SUSP_D_ROLL         0.491916f  // [rad·s/rad]
