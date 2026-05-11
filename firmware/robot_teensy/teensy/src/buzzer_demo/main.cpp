// Buzzer demo — Hello World melody + volume sweep.
//
// Phase 1 — "Twinkle Twinkle Little Star" at full volume (proves pitch & melody engine).
// Phase 2 — C major scale sweep at decreasing volume steps (proves volume control).
// Phase 3 — Short SOS pattern (proves rest notes & looping turned off).
//
// Serial output names each note as it plays so you can follow along.

#include <Arduino.h>
#include "config.h"
#include "Buzzer.h"

static Buzzer buzzer(PIN_BUZZER);

// ── MIDI convenience aliases ───────────────────────────────────────────────────
// Octave 4: C4=60 D4=62 E4=64 F4=65 G4=67 A4=69 B4=71
// Octave 5: C5=72 D5=74 E5=76 F5=77 G5=79 A5=81
static constexpr uint8_t nC4=60, nD4=62, nE4=64, nF4=65, nG4=67, nA4=69, nB4=71;
static constexpr uint8_t nC5=72, nG5=79, nA5=81;
static constexpr uint8_t nREST=0;

static constexpr uint16_t Q  = 350;   // quarter note ms
static constexpr uint16_t H  = 700;   // half note ms
static constexpr uint16_t G  = 50;    // inter-note gap ms

// ── Phase 1: Twinkle Twinkle Little Star ─────────────────────────────────────
static const BuzzerNote TWINKLE[] = {
    {nC4,Q,G},{nC4,Q,G},{nG4,Q,G},{nG4,Q,G},{nA4,Q,G},{nA4,Q,G},{nG4,H,G},
    {nF4,Q,G},{nF4,Q,G},{nE4,Q,G},{nE4,Q,G},{nD4,Q,G},{nD4,Q,G},{nC4,H,G},
    {nG4,Q,G},{nG4,Q,G},{nF4,Q,G},{nF4,Q,G},{nE4,Q,G},{nE4,Q,G},{nD4,H,G},
    {nG4,Q,G},{nG4,Q,G},{nF4,Q,G},{nF4,Q,G},{nE4,Q,G},{nE4,Q,G},{nD4,H,G},
    {nC4,Q,G},{nC4,Q,G},{nG4,Q,G},{nG4,Q,G},{nA4,Q,G},{nA4,Q,G},{nG4,H,G},
    {nF4,Q,G},{nF4,Q,G},{nE4,Q,G},{nE4,Q,G},{nD4,Q,G},{nD4,Q,G},{nC4,H,G},
};
static constexpr uint8_t TWINKLE_LEN = sizeof(TWINKLE) / sizeof(TWINKLE[0]);

// ── Phase 2: C major scale (volume sweep, one note per volume step) ───────────
static const BuzzerNote SCALE[] = {
    {nC4,400,50},{nD4,400,50},{nE4,400,50},{nF4,400,50},
    {nG4,400,50},{nA4,400,50},{nB4,400,50},{nC5,600,50},
};
static constexpr uint8_t SCALE_LEN = sizeof(SCALE) / sizeof(SCALE[0]);

// ── Phase 3: SOS in Morse (short=200 ms, long=600 ms, gap between letters) ───
static const BuzzerNote SOS[] = {
    // S: · · ·
    {nA5,200,80},{nA5,200,80},{nA5,200,200},
    // O: — — —
    {nA5,600,80},{nA5,600,80},{nA5,600,200},
    // S: · · ·
    {nA5,200,80},{nA5,200,80},{nA5,200,600},
};
static constexpr uint8_t SOS_LEN = sizeof(SOS) / sizeof(SOS[0]);

// ── Demo state machine ────────────────────────────────────────────────────────

enum class Phase : uint8_t { TWINKLE, SCALE_SWEEP, SOS, DONE };
static Phase   g_phase      = Phase::TWINKLE;
static uint8_t g_vol_step   = 0;

static const uint8_t VOL_STEPS[]  = {255, 200, 150, 100, 60, 30};
static const uint8_t N_VOL_STEPS  = sizeof(VOL_STEPS) / sizeof(VOL_STEPS[0]);

static void start_phase(Phase p) {
    g_phase = p;
    switch (p) {
        case Phase::TWINKLE:
            Serial.println("\n[Phase 1] Twinkle Twinkle — full volume");
            buzzer.play(TWINKLE, TWINKLE_LEN, 255);
            break;
        case Phase::SCALE_SWEEP:
            Serial.printf("\n[Phase 2] C-major scale — volume %d/255\n", VOL_STEPS[g_vol_step]);
            buzzer.play(SCALE, SCALE_LEN, VOL_STEPS[g_vol_step]);
            break;
        case Phase::SOS:
            Serial.println("\n[Phase 3] SOS — demo done, looping SOS pattern");
            buzzer.play(SOS, SOS_LEN, 200, /*loop=*/true);
            break;
        default:
            break;
    }
}

// ── Arduino entrypoints ───────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}
    Serial.println("\n=== Buzzer Hello World Demo ===");
    Serial.printf("PIN_BUZZER = %d\n", PIN_BUZZER);
    Serial.printf("A4 (MIDI 69) = %.1f Hz\n", Buzzer::midi_to_hz(69));
    Serial.println("Wire: Teensy PWM -> 1kΩ -> NPN base; collector -> buzzer -> 3.3V; emitter -> GND\n");

    buzzer.begin();
    start_phase(Phase::TWINKLE);
}

void loop() {
    buzzer.update();

    if (!buzzer.is_done()) return;

    switch (g_phase) {
        case Phase::TWINKLE:
            g_vol_step = 0;
            start_phase(Phase::SCALE_SWEEP);
            break;
        case Phase::SCALE_SWEEP:
            g_vol_step++;
            if (g_vol_step < N_VOL_STEPS) {
                start_phase(Phase::SCALE_SWEEP);
            } else {
                start_phase(Phase::SOS);
            }
            break;
        case Phase::SOS:
            // loops forever — nothing to do
            break;
        default:
            break;
    }
}
