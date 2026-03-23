#pragma once
// wifi_mgr.h — WiFi connection + OTA update management.

// Blocking WiFi connect + OTA init. Call once in setup().
// Returns true if WiFi connected successfully.
bool wifi_init();

// Non-blocking OTA poll. Call every loop() iteration. ~50 µs.
void wifi_ota_poll();

// Returns true if WiFi is currently connected.
bool wifi_is_connected();
