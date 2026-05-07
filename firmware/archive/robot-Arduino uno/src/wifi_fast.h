#pragma once
// wifi_fast.h — Slack-time WiFi: send/receive only in dead time between ticks.
//
// Enabled by #define USE_WIFI 1 in config.h.  When USE_WIFI is 0, all
// functions are no-ops and telemetry/commands go via USB-UART instead.
//
// Architecture (Option A — slack-time WiFi):
//   tick_flag fires (every 2000 µs)
//   ├── Control work: IMU + LQR + split    (~400-800 µs)
//   ├── wifi_fast_fill_telemetry()          (fills back buffer, no UDP)
//   ├── LED matrix (every ~167 ms)          (~50 µs)
//   └── return to loop()
//
//   loop() idle time (1200-1600 µs of slack):
//   ├── wifi_try_send()     — UDP telemetry if due + slack available
//   ├── wifi_try_receive()  — UDP commands  if due + slack available
//   └── spin until next tick_flag
//
// Key rules:
//   • WiFi I/O only in idle spin (tick_flag == false)
//   • Double-buffered telemetry: control writes buf A, WiFi sends buf B
//   • Slack guard: never start UDP if < WIFI_SEND_MIN_SLACK_US remain
//   • If a WiFi call overruns into next tick, tick just runs late once

#include "config.h"
#include "robot_state.h"

#if USE_WIFI

// Blocking WiFi connect + UDP socket init.  Call once in setup().
// Returns true if WiFi connected successfully.
bool wifi_fast_init();

// Fill telemetry back-buffer from current state.  No UDP I/O.
// Call from tick path at 50 Hz after all state is computed.
void wifi_fast_fill_telemetry(const RobotState& state);

// Attempt UDP telemetry send if pending and enough slack remains.
// tick_start_us = micros() recorded at start of current tick.
// Safe to call repeatedly in idle spin — sends at most once per fill.
// Returns true if a UDP send was performed.
bool wifi_try_send(uint32_t tick_start_us);

// Attempt UDP command receive if due (10 Hz) and enough slack remains.
// Safe to call repeatedly — polls at most once per eligible tick.
// Returns true if a command was received and applied to state.
bool wifi_try_receive(RobotState& state, uint32_t tick, uint32_t tick_start_us);

// Poll ArduinoOTA in slack time.  Fast (~50 µs) when no update pending.
void wifi_fast_ota_poll();

// Returns true if WiFi is connected.
bool wifi_fast_connected();

// ── Profiling accessors ──
uint32_t wifi_last_send_us();   // duration of last UDP send [µs]
uint32_t wifi_last_recv_us();   // duration of last command receive [µs]
uint32_t wifi_send_skips();     // telemetry sends skipped (overwritten before sent)

#else  // !USE_WIFI — stubs so main.cpp compiles unconditionally

static inline bool     wifi_fast_init()          { return false; }
static inline void     wifi_fast_fill_telemetry(const RobotState&) {}
static inline bool     wifi_try_send(uint32_t)   { return false; }
static inline bool     wifi_try_receive(RobotState&, uint32_t, uint32_t) { return false; }
static inline void     wifi_fast_ota_poll()       {}
static inline bool     wifi_fast_connected()      { return false; }
static inline uint32_t wifi_last_send_us()        { return 0; }
static inline uint32_t wifi_last_recv_us()        { return 0; }
static inline uint32_t wifi_send_skips()          { return 0; }

#endif // USE_WIFI
