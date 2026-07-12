#include "Buzzer.h"
#include <math.h>

Buzzer::Buzzer(uint8_t pin) : _pin(pin) {}

void Buzzer::begin() {
    pinMode(_pin, OUTPUT);
    analogWriteResolution(8);
    _silence();
}

// ── Public API ────────────────────────────────────────────────────────────────

void Buzzer::set_enabled(bool en) {
    _enabled = en;
    if (!_enabled) {
        _silence();
        _notes = nullptr;
        _done  = true;
    }
}

void Buzzer::tone(uint16_t freq_hz, uint8_t volume, uint32_t duration_ms) {
    if (!_enabled) return;
    _notes          = nullptr;
    _tone_start     = millis();
    _tone_duration  = duration_ms;
    _done           = (duration_ms == 0) ? false : false;
    _in_gap         = false;
    _set_pwm(freq_hz, volume);
}

void Buzzer::midi(uint8_t note, uint8_t volume, uint32_t duration_ms) {
    tone((uint16_t)midi_to_hz(note), volume, duration_ms);
}

void Buzzer::play(const BuzzerNote* notes, uint8_t count, uint8_t volume, bool loop) {
    if (!_enabled) return;
    if (!notes || count == 0) return;
    _notes       = notes;
    _count       = count;
    _volume      = volume;
    _loop        = loop;
    _done        = false;
    _index       = 0;
    _start_note(0);
}

void Buzzer::off() {
    _silence();
    _notes = nullptr;
    _done  = true;
}

bool Buzzer::is_playing() const { return !_done; }
bool Buzzer::is_done()    const { return _done;  }

// ── update ────────────────────────────────────────────────────────────────────

void Buzzer::update() {
    if (_done) return;

    uint32_t now     = millis();
    uint32_t elapsed = now - _phase_start;

    if (_notes == nullptr) {
        // Single timed tone
        if (_tone_duration > 0 && (now - _tone_start) >= _tone_duration) {
            _silence();
            _done = true;
        }
        return;
    }

    // Melody
    if (_in_gap) {
        if (_notes[_index].gap_ms > 0 && elapsed >= _notes[_index].gap_ms)
            _advance();
    } else {
        if (elapsed >= _notes[_index].on_ms) {
            _silence();
            if (_notes[_index].gap_ms > 0) {
                _in_gap      = true;
                _phase_start = now;
            } else {
                _advance();
            }
        }
    }
}

// ── Static utility ────────────────────────────────────────────────────────────

float Buzzer::midi_to_hz(uint8_t note) {
    // A4 = MIDI 69 = 440 Hz; each semitone is a factor of 2^(1/12)
    return 440.0f * powf(2.0f, (note - 69) / 12.0f);
}

// ── Private helpers ───────────────────────────────────────────────────────────

void Buzzer::_set_pwm(uint16_t freq_hz, uint8_t volume) {
    if (freq_hz == 0 || volume == 0) {
        _silence();
        return;
    }
    // Map volume 0-255 → duty 0-127 (0-50%).
    // 50% duty maximises power to a passive buzzer; above 50% it falls again.
    uint8_t duty = (uint8_t)(((uint16_t)volume * 127u) / 255u);
    analogWriteFrequency(_pin, freq_hz);
    analogWrite(_pin, duty);
}

void Buzzer::_silence() {
    analogWrite(_pin, 0);
}

void Buzzer::_start_note(uint8_t idx) {
    _in_gap      = false;
    _phase_start = millis();
    if (_notes[idx].midi == 0) {
        _silence();
    } else {
        _set_pwm((uint16_t)midi_to_hz(_notes[idx].midi), _volume);
    }
}

void Buzzer::_advance() {
    _index++;
    if (_index >= _count) {
        if (_loop) {
            _index = 0;
        } else {
            _silence();
            _notes = nullptr;
            _done  = true;
            return;
        }
    }
    _start_note(_index);
}
