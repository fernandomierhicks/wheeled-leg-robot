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

// --- Hip/leg NC limit switches to GND (LOW=released, HIGH=pressed) ---
#define PIN_LIMIT_LEFT  23  // leg 1 / motor 1 / channel 1
#define PIN_LIMIT_RIGHT 22

// --- ESP32 UART baud ---
#define ESP32_BAUD      4000000UL

// --- Control loop ---
#define CONTROL_HZ      500

// --- AK45-10 hip motor CAN IDs (MIT Cheetah protocol, CAN1) ---
#define AK45_ID_L           11          // left hip motor
#define AK45_ID_R           12          // right hip motor

// --- ODrive wheel motor CAN ---
#define ODESC_NODE_L        0           // ODrive axis 0 — left wheel
#define ODESC_NODE_R        1           // ODrive axis 1 — right wheel
#define CAN_BAUD            1000000UL   // 1 Mbps
#define HIP_CAN_TIMEOUT_MS  50          // AK45 MIT feedback watchdog (looser than wheel encoders)
#define CAN_INTER_FRAME_US  100         // gap between back-to-back TX frames — under
                                          // re-characterization, was 500; a standard 8-byte
                                          // CAN frame at 1 Mbps takes ~110-130us worst-case
                                          // (bit-stuffed), so 500 was ~4x that

// --- Hip retract-switch calibration ---
// Motor/wiring orientation is a hardware property, not a runtime parameter.
// Signs use the normalized firmware frame after the left motor's mirroring.
#define CALIB_L_SEEK_DIR  (+1.0f)
#define CALIB_R_SEEK_DIR  (+1.0f)
// Tunable motion, current, range, timeout, and ramp settings are in
// ParamRegistry (generated from protocol/schema.json).
