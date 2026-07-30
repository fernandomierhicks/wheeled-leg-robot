#include <Arduino.h>
#include <FastLED.h>

#define PIN_NEO    13
#define NUM_LEDS   100
#define BRIGHTNESS 200

static CRGB leds[NUM_LEDS];

static void animate() {
    static uint16_t hue16  = 0;  // smooth 16-bit hue counter
    static uint16_t wave16 = 0;  // brightness wave phase
    hue16  += 60;
    wave16 += 150;

    for (int i = 0; i < NUM_LEDS; i++) {
        uint8_t hue  = (hue16 >> 8) + (uint8_t)(i * 256 / NUM_LEDS);
        uint8_t wave = sin8((wave16 >> 8) + (uint8_t)(i * 256 / NUM_LEDS));
        uint8_t bri  = map(wave, 0, 255, 60, 255);
        leds[i] = CHSV(hue, 230, bri);
    }

    // random white sparkle
    if (random8() < 25) {
        leds[random8(NUM_LEDS)] = CRGB::White;
    }
}

// Runs on core 0 — core 1 is never blocked
static void neo_task(void *) {
    FastLED.addLeds<WS2812B, PIN_NEO, GRB>(leds, NUM_LEDS);
    FastLED.setBrightness(BRIGHTNESS);
    for (;;) {
        animate();
        FastLED.show();
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

void setup() {
    Serial.begin(115200);
    xTaskCreatePinnedToCore(neo_task, "neo", 2048, nullptr, 1, nullptr, 0);
    Serial.println("NeoPixel task pinned to core 0");
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}
