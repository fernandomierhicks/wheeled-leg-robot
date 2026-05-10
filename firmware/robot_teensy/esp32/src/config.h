#pragma once

// --- Teensy link (UART2: RX=GPIO16←Teensy pin20, TX=GPIO17→Teensy pin21) ---
#define TEENSY_UART_BAUD  1200000UL
#define TEENSY_UART_RX    16
#define TEENSY_UART_TX    17

// --- 2.0" TFT display (GMT020-02-9P, ST7789, SPI) ---
// VSPI bus: SCK=18, MOSI=23 (hardware defaults, shared)
#define TFT_CS   5
#define TFT_DC   2
#define TFT_RST  4
#define TFT_BLK  15   // backlight; tie to 3.3 V if GPIO control not needed

// --- WiFi telemetry ---
#define TELEM_UDP_PORT    5005
#define CMD_TCP_PORT      5006
