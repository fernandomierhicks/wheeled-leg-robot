// RGB LED demo — pin-validation sequence + animation showcase.
//
// Phase 1 — Pin validation (each held 3 s so you can confirm the correct
//            physical LED channel lights up):
//   RED only → GREEN only → BLUE only → WHITE (all)
//
// Phase 2 — Animation showcase:
//   Blink cyan  → Pulse magenta → Fade orange → Pulse white → Blink green/red alternating

#include <Arduino.h>
#include "config.h"
#include "RgbLed.h"

static RgbLed led(PIN_LED_R, PIN_LED_G, PIN_LED_B);

// ── Demo sequence ─────────────────────────────────────────────────────────────

struct Step {
    const char* label;
    uint32_t    hold_ms;
    void (*start)(RgbLed&);
};

static void do_red   (RgbLed& l) { l.solid(255,   0,   0); }
static void do_green (RgbLed& l) { l.solid(  0, 255,   0); }
static void do_blue  (RgbLed& l) { l.solid(  0,   0, 255); }
static void do_white (RgbLed& l) { l.solid(255, 255, 255); }
static void do_off   (RgbLed& l) { l.off(); }

// Animation steps — hold_ms is how long we let the animation run before moving on
static void do_blink_cyan   (RgbLed& l) { l.blink(  0, 255, 255, 150, 150); }
static void do_pulse_magenta(RgbLed& l) { l.pulse(255,   0, 255, 1200); }
static void do_fade_orange  (RgbLed& l) { l.fade_to(255, 80, 0, 800); }
static void do_pulse_white  (RgbLed& l) { l.pulse(255, 255, 255, 1800); }
static void do_blink_green  (RgbLed& l) { l.blink(  0, 255,   0, 100, 400); }

static const Step STEPS[] = {
    // ── Phase 1: pin validation ───────────────────────────────────────────
    { "[PIN TEST] RED   — only the RED channel should light up",   3000, do_red    },
    { "[PIN TEST] GREEN — only the GREEN channel should light up", 3000, do_green  },
    { "[PIN TEST] BLUE  — only the BLUE channel should light up",  3000, do_blue   },
    { "[PIN TEST] WHITE — all three channels on",                  2000, do_white  },
    { "[PIN TEST] OFF",                                             500, do_off    },

    // ── Phase 2: animation showcase ──────────────────────────────────────
    { "[ANIM] Blink cyan (150 ms on / 150 ms off)",    3000, do_blink_cyan    },
    { "[ANIM] Pulse magenta (1.2 s breathe)",           4000, do_pulse_magenta },
    { "[ANIM] Fade to orange over 800 ms",              1500, do_fade_orange   },
    { "[ANIM] Pulse white (1.8 s breathe)",             5000, do_pulse_white   },
    { "[ANIM] Blink green — fast on, slow off",         4000, do_blink_green   },
};

static const uint8_t N_STEPS = sizeof(STEPS) / sizeof(STEPS[0]);

static uint8_t  g_step       = 0;
static uint32_t g_step_start = 0;

// ── Arduino entrypoints ───────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}  // wait up to 3 s for USB serial
    Serial.println("\n=== RGB LED Demo ===");
    Serial.println("Phase 1: PIN VALIDATION — confirm each colour is wired correctly.");
    Serial.println("Phase 2: ANIMATIONS — non-blocking driver showcase.\n");

    led.begin();
    STEPS[0].start(led);
    Serial.println(STEPS[0].label);
    g_step_start = millis();
}

void loop() {
    led.update();

    // Advance to next step when hold_ms has elapsed
    if (millis() - g_step_start >= STEPS[g_step].hold_ms) {
        g_step = (g_step + 1) % N_STEPS;
        STEPS[g_step].start(led);
        Serial.println(STEPS[g_step].label);
        g_step_start = millis();
    }
}
