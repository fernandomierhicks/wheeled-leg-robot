#pragma once
// wifi_mgr.h — WiFi connection + OTA update management.
//
// When USE_WIFI=1, wifi_fast.cpp manages WiFi; wifi_is_connected() delegates
// there.  When USE_WIFI=0, these are stubs (no WiFi compiled in).

#include "config.h"

#if USE_WIFI

#include "wifi_fast.h"
static inline bool wifi_init()         { return wifi_fast_init(); }
static inline void wifi_ota_poll()     { wifi_fast_ota_poll(); }
static inline bool wifi_is_connected() { return wifi_fast_connected(); }

#else

// Blocking WiFi connect + OTA init. Call once in setup().
bool wifi_init();

// Non-blocking OTA poll. Call every loop() iteration. ~50 µs.
void wifi_ota_poll();

// Returns true if WiFi is currently connected.
bool wifi_is_connected();

#endif // USE_WIFI
