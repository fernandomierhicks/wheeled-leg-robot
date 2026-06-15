#pragma once

// --- CAN buses ---
#define PIN_CAN2_TX      1  // AK45 hip motors (Serial1 unavailable on these pins)
#define PIN_CAN2_RX      0
#define PIN_CAN3_TX     31  // ODrive wheel motors
#define PIN_CAN3_RX     30

// --- IMU (BNO086, SPI0) ---
#define PIN_IMU_CS      10
#define PIN_IMU_INT      9
#define PIN_IMU_RST      6

// --- RGB LED ---
#define PIN_LED_R        3
#define PIN_LED_G        2
#define PIN_LED_B        4

// --- Passive buzzer (PWM tone) ---
#define PIN_BUZZER       5

// --- AK45 UART (Serial2 / Serial3, standard 2-wire TX/RX) ---
#define PIN_AK45_1_RX    7
#define PIN_AK45_1_TX    8
#define PIN_AK45_2_TX   14
#define PIN_AK45_2_RX   15
#define AK45_UART_BAUD   921600UL

// --- RC receiver (FlySky iBUS, Serial4 RX only) ---
#define PIN_IBUS_RX     16

// --- ESP32 link (Serial5) ---
#define PIN_ESP32_TX    20
#define PIN_ESP32_RX    21

// --- ESP32 UART baud ---
#define ESP32_BAUD      1200000UL

// --- Control loop ---
#define CONTROL_HZ      500

// --- AK45-10 hip motor CAN IDs (MIT Cheetah protocol, CAN1) ---
#define AK45_ID_L           11          // left hip motor
#define AK45_ID_R           12          // right hip motor

// --- ODrive wheel motor CAN ---
#define ODESC_NODE_L        0           // ODrive axis 0 — left wheel
#define ODESC_NODE_R        1           // ODrive axis 1 — right wheel
#define CAN_BAUD            1000000UL   // 1 Mbps
#define CAN_TIMEOUT_MS      20          // encoder feedback watchdog
#define HIP_CAN_TIMEOUT_MS  50          // AK45 MIT feedback watchdog (looser than wheel encoders)
#define CAN_INTER_FRAME_US  500         // gap between back-to-back TX frames

// --- Hip hardstop calibration (STATE_CALIBRATION) ---
#define CALIB_SEEK_SPEED_RADS     (10.0f * PI / 180.0f)  // ramp speed while seeking a hardstop (~10 deg/s)
#define CALIB_KP                  2.0f    // gentle position gain while seeking
#define CALIB_KD                  0.05f   // damping while seeking
#define CALIB_HOLD_KP             1.0f    // position gain once homed, holding at midpoint
#define CALIB_HOLD_KD             0.05f   // damping once homed
#define CALIB_STALL_CUR_A         0.5f    // current threshold [A] indicating a hardstop
#define CALIB_STALL_DEADBAND_RAD  0.02f   // max position movement per tick to count as stalled
#define CALIB_STALL_CONFIRM_TICKS 15      // consecutive ticks before declaring a hardstop (30 ms @ 500 Hz)
#define CALIB_MARGIN_RAD          (10.0f * PI / 180.0f)  // software limit margin from each hardstop (~10 deg)
#define CALIB_SAFETY_BOUND_RAD    (90.0f * PI / 180.0f)  // abort if a seek ramps more than this far from the start position (~90 deg)

// Seek-direction sign toward each hip's "bottom" hardstop (the one zeroed to
// 0 rad). L and R may be mirrored — verify on first run and flip if a leg
// seeks toward its top hardstop first.
#define HIP_L_SEEK_DIR      (+1.0f)
#define HIP_R_SEEK_DIR      (-1.0f)
