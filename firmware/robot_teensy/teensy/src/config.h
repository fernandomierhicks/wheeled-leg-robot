#pragma once

// --- CAN buses ---
#define PIN_CAN1_TX     22  // AK45 hip motors
#define PIN_CAN1_RX     23
#define PIN_CAN2_TX      1  // ODrive wheel motors (Serial1 unavailable on these pins)
#define PIN_CAN2_RX      0

// --- IMU (BNO086, SPI0) ---
#define PIN_IMU_CS      10
#define PIN_IMU_INT      2
#define PIN_IMU_RST      3

// --- RGB LED ---
#define PIN_LED_R        4
#define PIN_LED_G        5
#define PIN_LED_B        6

// --- RC receiver (FlySky iBUS, Serial4 RX only) ---
#define PIN_IBUS_RX     16

// --- ESP32 link (Serial5) ---
#define PIN_ESP32_TX    20
#define PIN_ESP32_RX    21

// --- ESP32 UART baud ---
#define ESP32_BAUD      1200000UL

// --- Control loop ---
#define CONTROL_HZ      500

// --- ODrive wheel motor CAN ---
#define ODESC_NODE_L        0           // ODrive axis 0 — left wheel
#define ODESC_NODE_R        1           // ODrive axis 1 — right wheel
#define CAN_BAUD            1000000UL   // 1 Mbps
#define CAN_TIMEOUT_MS      20          // encoder feedback watchdog
#define CAN_INTER_FRAME_US  500         // gap between back-to-back TX frames
