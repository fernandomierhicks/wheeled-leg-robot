#pragma once
#include <Arduino.h>

// A single step in a melody.  midi=0 is a rest.
struct BuzzerNote {
    uint8_t  midi;      // MIDI note number (21–108), 0 = rest
    uint16_t on_ms;     // note-on duration in ms
    uint16_t gap_ms;    // silent gap after note in ms (0 = no gap)
};

// Non-blocking passive buzzer driver.
// Volume is controlled by PWM duty cycle; frequency sets pitch.
// Call update() every loop iteration; all timing is millis()-based.
//
// Wiring: Teensy PWM pin → 1 kΩ → NPN base; collector → buzzer → V+; emitter → GND.

class Buzzer {
public:
    explicit Buzzer(uint8_t pin);

    void begin();
    void update();  // must be called every loop

    // Single tone: hold until off() or a new command.  duration_ms=0 → hold forever.
    void tone(uint16_t freq_hz, uint8_t volume = 255, uint32_t duration_ms = 0);

    // MIDI note number → frequency, then same as tone().
    void midi(uint8_t note, uint8_t volume = 255, uint32_t duration_ms = 0);

    // Non-blocking melody playback.  loop=true wraps around indefinitely.
    void play(const BuzzerNote* notes, uint8_t count, uint8_t volume = 255, bool loop = false);

    // Silence immediately; clears any active melody.
    void off();

    bool is_playing() const;  // true while a tone or melody is active
    bool is_done()    const;  // true once a melody or timed tone has finished

    // Pure utility: MIDI note number to frequency in Hz (A4=69=440 Hz).
    static float midi_to_hz(uint8_t note);

private:
    uint8_t _pin;

    // Single-tone state
    uint32_t _tone_start    = 0;
    uint32_t _tone_duration = 0;  // 0 = hold

    // Melody state
    const BuzzerNote* _notes  = nullptr;
    uint8_t           _count  = 0;
    uint8_t           _index  = 0;
    uint8_t           _volume = 255;
    bool              _loop   = false;
    bool              _done   = true;
    bool              _in_gap = false;
    uint32_t          _phase_start = 0;

    void _set_pwm(uint16_t freq_hz, uint8_t volume);
    void _silence();
    void _start_note(uint8_t idx);
    void _advance();
};
