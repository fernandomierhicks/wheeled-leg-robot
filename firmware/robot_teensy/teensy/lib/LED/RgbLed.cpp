#include "RgbLed.h"
#include <math.h>

RgbLed::RgbLed(uint8_t pin_r, uint8_t pin_g, uint8_t pin_b, bool active_low)
    : _pr(pin_r), _pg(pin_g), _pb(pin_b), _active_low(active_low) {}

void RgbLed::begin() {
    pinMode(_pr, OUTPUT);
    pinMode(_pg, OUTPUT);
    pinMode(_pb, OUTPUT);
    _write(0, 0, 0);
}

// --------------------------------------------------------------------------

void RgbLed::solid(uint8_t r, uint8_t g, uint8_t b) {
    _r = r; _g = g; _b = b;
    _mode = Mode::SOLID;
    _done = true;
    _write(r, g, b);
}

void RgbLed::off() { solid(0, 0, 0); }

void RgbLed::blink(uint8_t r, uint8_t g, uint8_t b,
                   uint32_t on_ms, uint32_t off_ms) {
    _r = r; _g = g; _b = b;
    _on_ms  = on_ms;
    _off_ms = off_ms;
    _mode = Mode::BLINK;
    _done = false;
    _anim_start = millis();
}

void RgbLed::pulse(uint8_t r, uint8_t g, uint8_t b, uint32_t period_ms) {
    _r = r; _g = g; _b = b;
    _period_ms = period_ms;
    _mode = Mode::PULSE;
    _done = false;
    _anim_start = millis();
}

void RgbLed::fade_to(uint8_t r, uint8_t g, uint8_t b, uint32_t duration_ms) {
    _from_r = _r; _from_g = _g; _from_b = _b;
    _r = r; _g = g; _b = b;
    _duration_ms = duration_ms;
    _mode = Mode::FADE;
    _done = false;
    _anim_start = millis();
}

void RgbLed::flash_then_pulse(uint8_t r, uint8_t g, uint8_t b,
                               uint32_t flash_ms, uint32_t period_ms) {
    _r = r; _g = g; _b = b;
    _duration_ms = flash_ms;
    _period_ms   = period_ms;
    _mode = Mode::FLASH;
    _done = false;
    _anim_start = millis();
}

bool RgbLed::is_done() const { return _done; }

// --------------------------------------------------------------------------

void RgbLed::update() {
    uint32_t now = millis();
    uint32_t elapsed = now - _anim_start;

    switch (_mode) {
        case Mode::SOLID:
            break;

        case Mode::BLINK: {
            uint32_t cycle = _on_ms + _off_ms;
            uint32_t phase = elapsed % cycle;
            if (phase < _on_ms) {
                _write(_r, _g, _b);
            } else {
                _write(0, 0, 0);
            }
            break;
        }

        case Mode::PULSE: {
            float t = (float)(elapsed % _period_ms) / (float)_period_ms;
            float brightness = (1.0f - cosf(2.0f * M_PI * t)) * 0.5f;
            _write((uint8_t)(_r * brightness),
                   (uint8_t)(_g * brightness),
                   (uint8_t)(_b * brightness));
            break;
        }

        case Mode::FLASH: {
            if (elapsed >= _duration_ms) {
                _mode = Mode::PULSE;
                _anim_start = now;
                _done = true;
                _write(_r, _g, _b);
            } else {
                _write(_r, _g, _b);
            }
            break;
        }

        case Mode::FADE: {
            if (elapsed >= _duration_ms) {
                _write(_r, _g, _b);
                _mode = Mode::SOLID;
                _done = true;
            } else {
                float t = (float)elapsed / (float)_duration_ms;
                _write(_lerp(_from_r, _r, t),
                       _lerp(_from_g, _g, t),
                       _lerp(_from_b, _b, t));
            }
            break;
        }
    }
}

// --------------------------------------------------------------------------

void RgbLed::_write(uint8_t r, uint8_t g, uint8_t b) {
    if (_active_low) {
        analogWrite(_pr, 255 - r);
        analogWrite(_pg, 255 - g);
        analogWrite(_pb, 255 - b);
    } else {
        analogWrite(_pr, r);
        analogWrite(_pg, g);
        analogWrite(_pb, b);
    }
}

uint8_t RgbLed::_lerp(uint8_t a, uint8_t b, float t) const {
    return (uint8_t)(a + (b - a) * t);
}
