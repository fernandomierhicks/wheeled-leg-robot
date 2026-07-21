#pragma once

// --- Teensy link (UART2: RX=GPIO16←Teensy pin20, TX=GPIO17→Teensy pin21) ---
#define TEENSY_UART_BAUD  4000000UL
#define TEENSY_UART_RX    16
#define TEENSY_UART_TX    17
#define UPLINK_QUEUE_LEN        24      // uplink_task queue depth — ~200 ms absorb at 100 frames/s
#define CONTROL_UPLINK_QUEUE_LEN 8      // command results/params/log messages, drained first
#define LOG_UPLINK_QUEUE_LEN     4      // ACK flow control keeps at most one data chunk in flight
#define UART_PARSE_BUDGET_BYTES  2048   // maximum Serial2 bytes parsed per loop pass
#define HOST_PARSE_BUDGET_BYTES   512   // maximum USB/TCP bytes parsed per loop pass
#define TEENSY_LINK_TIMEOUT_MS  1500    // single truth for "Teensy link up": TFT active-state,
                                         // Neopixel linked-state, and face-mode connected-state

// --- 2.0" TFT display (GMT020-02-9P, ST7789, SPI) ---
// VSPI bus: SCK=18, MOSI=23 (hardware defaults, shared)
#define TFT_CS   5
#define TFT_DC   2
#define TFT_RST  4
#define TFT_BLK  15   // backlight; tie to 3.3 V if GPIO control not needed

// --- WiFi telemetry ---
#ifndef WIFI_ENABLED
#define WIFI_ENABLED      1     // set to 0 to disable UDP broadcast + TCP server. Unlike the other
                                 // WiFi campaign toggles below, this wasn't previously overridable via
                                 // PLATFORMIO_BUILD_FLAGS — added the guard so a WiFi-free diagnostic
                                 // build (e.g. -D WIFI_ENABLED=0) is possible without editing this file.
#endif
#define TELEM_UDP_PORT    5005
#define CMD_TCP_PORT      5006
#define DISCOVERY_UDP_PORT 5007
#define WIFI_SESSION_LEASE_MS 3500

// --- ToF laser distance sensors (VL53L1X x4) ---
#define LASERS_ENABLED    0     // set to 1 once VL53L1X sensors are wired up

// --- Display personality ---
// 0 = PERS_ENGINEERING (telemetry panels), 1 = PERS_FACE (animated eyes)
// Change this to switch the power-on default; later the Teensy can override live.
#define DEFAULT_DISPLAY_PERSONALITY  0

// --- WiFi telemetry campaign build-time toggles ---
// All overridable via PLATFORMIO_BUILD_FLAGS (-D...) without editing this file,
// so test variants can be flashed without dirtying the tree.
#ifndef WIFI_TELEM_MODE
#define WIFI_TELEM_MODE      1   // 0=broadcast, 1=unicast to WIFI_UNICAST_IP (default —
                                 //   campaign result 2026-07-18: unicast eliminates the
                                 //   DTIM-clumping that made broadcast telemetry choppy)
#endif
#ifndef WIFI_UNICAST_IP
#define WIFI_UNICAST_IP      ""  // baked in by flash_monitor.py at flash time (auto-detected
                                 //   PC LAN IP); falls back to broadcast-like failure if unset —
                                 //   set explicitly via PLATFORMIO_BUILD_FLAGS if flashing outside the GUI
#endif
#ifndef WIFI_TELEM_COMBINED
#define WIFI_TELEM_COMBINED  1   // 1=single atomic TelemetryPayload datagram (production default)
#endif
#ifndef WIFI_TX_POWER_MAX
#define WIFI_TX_POWER_MAX    0   // 0=default, 1=WiFi.setTxPower(max) in setup()
#endif
#ifndef NEO_ENABLED
#define NEO_ENABLED          1   // 0=skip starting neo_task (FastLED isolation test)
#endif
#ifndef DISPLAY_ENABLED
#define DISPLAY_ENABLED      1   // 0=skip starting display_task (core-0 contention isolation —
                                 //   the display is the heaviest core-0 load)
#endif
#ifndef WIFI_DIAG_HZ
#define WIFI_DIAG_HZ         5   // bumped 2->5 (Phase 3, UARTplat.md): snappier GUI ESP32-alive detection
#endif
#ifndef ESP32_STATUS_HZ
#define ESP32_STATUS_HZ      5   // ESP32->Teensy link heartbeat rate (Phase 3, UARTplat.md)
#endif
#ifndef WIFI_TRANSPORT_GATING
#define WIFI_TRANSPORT_GATING 0  // 0=always send WiFi telemetry when connected (current), 1=honor
                                 //   CMD_ID_SET_TELEM_TRANSPORT and suppress WiFi telemetry sends
                                 //   when the GUI has announced it's reading USB instead
#endif
