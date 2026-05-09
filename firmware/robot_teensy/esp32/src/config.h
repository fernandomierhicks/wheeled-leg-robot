#pragma once

// --- Teensy link (UART2: RX=GPIO16←Teensy pin20, TX=GPIO17→Teensy pin21) ---
#define TEENSY_UART_BAUD  1200000UL
#define TEENSY_UART_RX    16
#define TEENSY_UART_TX    17

// --- WiFi telemetry ---
#define TELEM_UDP_PORT    5005
#define CMD_TCP_PORT      5006
