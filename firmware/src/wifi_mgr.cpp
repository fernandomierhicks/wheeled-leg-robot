// wifi_mgr.cpp — WiFi connection + ArduinoOTA for UNO R4 WiFi.

#include "wifi_mgr.h"
#include "config.h"
#include <Arduino.h>
#include <WiFiS3.h>
#include <ArduinoOTA.h>

static bool s_wifi_connected = false;

bool wifi_init() {
    Serial.print("[WiFi] Connecting to ");
    Serial.println(WIFI_SSID);

    // WiFi.begin() on UNO R4 can block for several seconds per attempt.
    // Try up to 3 attempts (~10s max) so the superloop starts promptly.
    for (int attempt = 0; attempt < 3; attempt++) {
        Serial.print("[WiFi] Attempt ");
        Serial.println(attempt + 1);
        if (WiFi.begin(WIFI_SSID, WIFI_PASS) == WL_CONNECTED) {
            Serial.print("[WiFi] Connected — IP: ");
            Serial.println(WiFi.localIP());

            // Start OTA server
            ArduinoOTA.begin(WiFi.localIP(), OTA_HOSTNAME, OTA_PASSWORD, InternalStorage);
            Serial.println("[OTA]  Ready");
            s_wifi_connected = true;
            return true;
        }
        delay(1000);
    }

    Serial.println("[WiFi] FAILED — continuing without WiFi/OTA");
    return false;
}

void wifi_ota_poll() {
    if (s_wifi_connected) {
        ArduinoOTA.poll();
    }
}

bool wifi_is_connected() {
    return s_wifi_connected;
}
