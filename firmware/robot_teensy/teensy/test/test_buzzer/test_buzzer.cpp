#include <Arduino.h>
#include <unity.h>
#include <math.h>
#include "config.h"
#include "Buzzer.h"
#include "../test_led.h"

static Buzzer bz(PIN_BUZZER);

// ── MIDI frequency maths ──────────────────────────────────────────────────────

void test_midi_a4_is_440hz(void) {
    float hz = Buzzer::midi_to_hz(69);
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 440.0f, hz);
}

void test_midi_a5_is_880hz(void) {
    // One octave above A4 = exactly double
    float hz = Buzzer::midi_to_hz(81);
    TEST_ASSERT_FLOAT_WITHIN(0.5f, 880.0f, hz);
}

void test_midi_c4_approx(void) {
    float hz = Buzzer::midi_to_hz(60);
    TEST_ASSERT_FLOAT_WITHIN(1.0f, 261.63f, hz);
}

void test_midi_octave_doubles_freq(void) {
    float lo = Buzzer::midi_to_hz(60);
    float hi = Buzzer::midi_to_hz(72);
    TEST_ASSERT_FLOAT_WITHIN(0.5f, lo * 2.0f, hi);
}

// ── Driver state after construction ──────────────────────────────────────────

void test_initial_state_is_done(void) {
    Buzzer b(PIN_BUZZER);
    b.begin();
    TEST_ASSERT_TRUE(b.is_done());
    TEST_ASSERT_FALSE(b.is_playing());
}

// ── tone() ────────────────────────────────────────────────────────────────────

void test_tone_sets_playing(void) {
    bz.tone(440, 128);
    TEST_ASSERT_FALSE(bz.is_done());
    TEST_ASSERT_TRUE(bz.is_playing());
    bz.off();
}

void test_off_stops_tone(void) {
    bz.tone(880, 200);
    bz.off();
    TEST_ASSERT_TRUE(bz.is_done());
    TEST_ASSERT_FALSE(bz.is_playing());
}

void test_timed_tone_auto_stops(void) {
    bz.tone(1000, 200, 150);  // 150 ms duration
    TEST_ASSERT_FALSE(bz.is_done());
    delay(200);  // wait past the duration
    for (int i = 0; i < 10; i++) { bz.update(); delayMicroseconds(100); }
    TEST_ASSERT_TRUE_MESSAGE(bz.is_done(), "timed tone should auto-stop after 150 ms");
}

void test_indefinite_tone_does_not_auto_stop(void) {
    bz.tone(500, 100, 0);  // duration=0 → hold forever
    delay(100);
    for (int i = 0; i < 20; i++) { bz.update(); delayMicroseconds(100); }
    TEST_ASSERT_FALSE_MESSAGE(bz.is_done(), "indefinite tone must not auto-stop");
    bz.off();
}

// ── midi() ────────────────────────────────────────────────────────────────────

void test_midi_sets_playing(void) {
    bz.midi(69, 100, 0);  // A4, indefinite
    TEST_ASSERT_TRUE(bz.is_playing());
    bz.off();
}

// ── play() — melody ───────────────────────────────────────────────────────────

void test_melody_completes(void) {
    static const BuzzerNote notes[] = {
        {60, 80, 20},  // C4
        {64, 80, 20},  // E4
        {67, 80, 20},  // G4
    };
    bz.play(notes, 3, 200);
    TEST_ASSERT_FALSE(bz.is_done());

    // Total time = 3 * (80 + 20) = 300 ms; spin 400 ms to be safe
    uint32_t end = millis() + 400;
    while (millis() < end) { bz.update(); delayMicroseconds(500); }

    TEST_ASSERT_TRUE_MESSAGE(bz.is_done(), "3-note melody should finish within 400 ms");
}

void test_rest_note_completes(void) {
    static const BuzzerNote notes[] = {
        {60,  80, 10},  // C4
        { 0, 100, 10},  // rest
        {67,  80, 10},  // G4
    };
    bz.play(notes, 3, 200);
    uint32_t end = millis() + 500;
    while (millis() < end) { bz.update(); delayMicroseconds(500); }
    TEST_ASSERT_TRUE_MESSAGE(bz.is_done(), "melody with rest should complete");
}

void test_off_interrupts_melody(void) {
    static const BuzzerNote notes[] = {
        {60, 5000, 0},  // very long note
    };
    bz.play(notes, 1, 200);
    delay(20);
    bz.off();
    TEST_ASSERT_TRUE_MESSAGE(bz.is_done(), "off() should immediately terminate melody");
}

// ── Demo sequence (runs after tests) ─────────────────────────────────────────

static constexpr uint8_t nC4=60, nD4=62, nE4=64, nF4=65, nG4=67, nA4=69, nB4=71;
static constexpr uint8_t nC5=72, nA5=81;
static constexpr uint16_t Q=350, H=700, G=50;

static const BuzzerNote TWINKLE[] = {
    {nC4,Q,G},{nC4,Q,G},{nG4,Q,G},{nG4,Q,G},{nA4,Q,G},{nA4,Q,G},{nG4,H,G},
    {nF4,Q,G},{nF4,Q,G},{nE4,Q,G},{nE4,Q,G},{nD4,Q,G},{nD4,Q,G},{nC4,H,G},
    {nG4,Q,G},{nG4,Q,G},{nF4,Q,G},{nF4,Q,G},{nE4,Q,G},{nE4,Q,G},{nD4,H,G},
    {nG4,Q,G},{nG4,Q,G},{nF4,Q,G},{nF4,Q,G},{nE4,Q,G},{nE4,Q,G},{nD4,H,G},
    {nC4,Q,G},{nC4,Q,G},{nG4,Q,G},{nG4,Q,G},{nA4,Q,G},{nA4,Q,G},{nG4,H,G},
    {nF4,Q,G},{nF4,Q,G},{nE4,Q,G},{nE4,Q,G},{nD4,Q,G},{nD4,Q,G},{nC4,H,G},
};
static const BuzzerNote SCALE[] = {
    {nC4,400,50},{nD4,400,50},{nE4,400,50},{nF4,400,50},
    {nG4,400,50},{nA4,400,50},{nB4,400,50},{nC5,600,50},
};
static const BuzzerNote SOS[] = {
    {nA5,200,80},{nA5,200,80},{nA5,200,200},
    {nA5,600,80},{nA5,600,80},{nA5,600,200},
    {nA5,200,80},{nA5,200,80},{nA5,200,600},
};

static const uint8_t VOL_STEPS[] = {255, 200, 150, 100, 60, 30};
static const uint8_t N_VOL_STEPS = sizeof(VOL_STEPS) / sizeof(VOL_STEPS[0]);

enum class DemoPhase : uint8_t { TWINKLE, SCALE_SWEEP, SOS };
static DemoPhase g_demo_phase   = DemoPhase::TWINKLE;
static uint8_t   g_vol_step     = 0;
static bool      g_demo_running = false;

static void demo_start_phase(DemoPhase p) {
    g_demo_phase = p;
    switch (p) {
        case DemoPhase::TWINKLE:
            Serial.println("\n[Demo] Twinkle Twinkle — full volume");
            bz.play(TWINKLE, sizeof(TWINKLE)/sizeof(TWINKLE[0]), 255);
            break;
        case DemoPhase::SCALE_SWEEP:
            Serial.printf("\n[Demo] C-major scale — volume %d/255\n", VOL_STEPS[g_vol_step]);
            bz.play(SCALE, sizeof(SCALE)/sizeof(SCALE[0]), VOL_STEPS[g_vol_step]);
            break;
        case DemoPhase::SOS:
            Serial.println("\n[Demo] SOS loop — done");
            bz.play(SOS, sizeof(SOS)/sizeof(SOS[0]), 200, /*loop=*/true);
            break;
    }
}

// ── Entry ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    delay(1000);
    test_led_begin();
    bz.begin();

    UNITY_BEGIN();

    RUN_TEST(test_midi_a4_is_440hz);
    RUN_TEST(test_midi_a5_is_880hz);
    RUN_TEST(test_midi_c4_approx);
    RUN_TEST(test_midi_octave_doubles_freq);

    RUN_TEST(test_initial_state_is_done);

    RUN_TEST(test_tone_sets_playing);
    RUN_TEST(test_off_stops_tone);
    RUN_TEST(test_timed_tone_auto_stops);
    RUN_TEST(test_indefinite_tone_does_not_auto_stop);
    RUN_TEST(test_midi_sets_playing);

    RUN_TEST(test_melody_completes);
    RUN_TEST(test_rest_note_completes);
    RUN_TEST(test_off_interrupts_melody);

    test_led_done(UNITY_END());

    Serial.println("\n=== Buzzer Demo ===");
    g_vol_step     = 0;
    g_demo_running = true;
    demo_start_phase(DemoPhase::TWINKLE);
}

void loop() {
    bz.update();

    if (!g_demo_running || !bz.is_done()) return;

    switch (g_demo_phase) {
        case DemoPhase::TWINKLE:
            g_vol_step = 0;
            demo_start_phase(DemoPhase::SCALE_SWEEP);
            break;
        case DemoPhase::SCALE_SWEEP:
            g_vol_step++;
            if (g_vol_step < N_VOL_STEPS)
                demo_start_phase(DemoPhase::SCALE_SWEEP);
            else
                demo_start_phase(DemoPhase::SOS);
            break;
        case DemoPhase::SOS:
            break;  // loops forever
    }
}
