#include <Arduino.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <SPI.h>
#include "config.h"

Adafruit_ST7789 tft = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);

// HSV→RGB565: hue 0-360
static uint16_t hsvToRgb565(float h) {
    float s = 1.0f, v = 1.0f;
    int i = (int)(h / 60.0f) % 6;
    float f = h / 60.0f - (int)(h / 60.0f);
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    float r, g, b;
    switch (i) {
        case 0: r=v; g=t; b=p; break;
        case 1: r=q; g=v; b=p; break;
        case 2: r=p; g=v; b=t; break;
        case 3: r=p; g=q; b=v; break;
        case 4: r=t; g=p; b=v; break;
        default: r=v; g=p; b=q; break;
    }
    return tft.color565((uint8_t)(r*255), (uint8_t)(g*255), (uint8_t)(b*255));
}

void drawRainbowBars() {
    int barH = tft.height() / 6;
    for (int i = 0; i < 6; i++) {
        uint16_t c = hsvToRgb565(i * 60.0f);
        tft.fillRect(0, i * barH, tft.width(), barH, c);
    }
}

void scrollText(const char* msg, uint16_t color, int y, int delayMs) {
    tft.setTextColor(color, ST77XX_BLACK);
    tft.setTextSize(3);
    int msgW = strlen(msg) * 18;  // 6px * 3 scale = 18px per char
    for (int x = tft.width(); x > -msgW; x -= 3) {
        tft.fillRect(0, y, tft.width(), 24, ST77XX_BLACK);
        tft.setCursor(x, y);
        tft.print(msg);
        delay(delayMs);
    }
}

void setup() {
    Serial.begin(115200);

    pinMode(TFT_BLK, OUTPUT);
    digitalWrite(TFT_BLK, HIGH);  // backlight on

    tft.init(240, 320);
    tft.setRotation(1);  // landscape: 320 wide, 240 tall
    tft.fillScreen(ST77XX_BLACK);

    Serial.println("TFT init OK");

    // --- Splash: rainbow bars ---
    drawRainbowBars();
    delay(1500);
    tft.fillScreen(ST77XX_BLACK);

    // --- Hello Robot centred ---
    tft.setTextSize(4);
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(20, 100);
    tft.print("Hello Robot!");
    delay(1500);
    tft.fillScreen(ST77XX_BLACK);
}

void loop() {
    static float hue = 0.0f;

    // Scrolling rainbow text
    scrollText("Hello  Robot! ", hsvToRgb565(hue), 108, 8);
    hue = fmod(hue + 40.0f, 360.0f);

    // Brief rainbow-bar flash between scroll passes
    drawRainbowBars();
    delay(600);
    tft.fillScreen(ST77XX_BLACK);
}
