#include <Arduino.h>
#include <WiFiS3.h>
#include <ArduinoOTA.h>
#include "ArduinoGraphics.h"
#include "Arduino_LED_Matrix.h"

const char* ssid     = "Minnexico";
const char* password = "Fenway724";

ArduinoLEDMatrix matrix;

void scrollText(const char* msg) {
    matrix.beginDraw();
    matrix.stroke(0xFFFFFFFF);
    matrix.textScrollSpeed(50);
    matrix.textFont(Font_5x7);
    matrix.beginText(0, 1, 0xFFFFFF);
    matrix.println(msg);
    matrix.endText(SCROLL_LEFT);
    matrix.endDraw();
}

void setup() {
    Serial.begin(115200);
    matrix.begin();

    // Connect to WiFi
    Serial.print("Connecting to WiFi...");
    while (WiFi.begin(ssid, password) != WL_CONNECTED) {
        delay(1000);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("Connected! IP: ");
    Serial.println(WiFi.localIP());

    // Start OTA
    ArduinoOTA.begin(WiFi.localIP(), "Arduino_R4", "arduino", InternalStorage);
    Serial.println("OTA ready");
}

void loop() {
    ArduinoOTA.poll();

    char buf[64];
    snprintf(buf, sizeof(buf), "OTA Ready - %s", WiFi.localIP().toString().c_str());
    scrollText(buf);
}
