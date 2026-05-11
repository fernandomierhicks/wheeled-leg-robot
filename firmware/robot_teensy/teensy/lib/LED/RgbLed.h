#pragma once
#include <Arduino.h>

// Non-blocking RGB LED driver with animations.
// Supports common-cathode (active-high, default) and common-anode (active-low) wiring.
// Call update() every loop iteration; all timing is millis()-based.

class RgbLed {
public:
    enum class Mode { SOLID, BLINK, PULSE, FADE };

    RgbLed(uint8_t pin_r, uint8_t pin_g, uint8_t pin_b, bool active_low = false);

    void begin();
    void update();  // must be called every loop

    // --- Animations ---

    // Immediate solid color. r/g/b in [0, 255].
    void solid(uint8_t r, uint8_t g, uint8_t b);
    void off();

    // Blink: full color for on_ms, off for off_ms, forever.
    void blink(uint8_t r, uint8_t g, uint8_t b,
               uint32_t on_ms = 200, uint32_t off_ms = 200);

    // Smooth breathe: sinusoidal fade 0→peak→0 at given period.
    void pulse(uint8_t r, uint8_t g, uint8_t b, uint32_t period_ms = 1500);

    // One-shot fade from current output to target over duration_ms, then holds as SOLID.
    void fade_to(uint8_t r, uint8_t g, uint8_t b, uint32_t duration_ms = 500);

    // --- State queries ---
    bool is_done() const;  // true once a fade_to completes

private:
    uint8_t _pr, _pg, _pb;
    bool    _active_low;
    Mode    _mode = Mode::SOLID;
    bool    _done = true;

    // Current target color
    uint8_t _r = 0, _g = 0, _b = 0;

    // Blink state
    uint32_t _on_ms  = 200;
    uint32_t _off_ms = 200;

    // Pulse / fade timing
    uint32_t _period_ms   = 1500;
    uint32_t _duration_ms = 500;
    uint32_t _anim_start  = 0;

    // Fade: source color captured at fade_to() call
    uint8_t _from_r = 0, _from_g = 0, _from_b = 0;

    void _write(uint8_t r, uint8_t g, uint8_t b);
    uint8_t _lerp(uint8_t a, uint8_t b, float t) const;
};
