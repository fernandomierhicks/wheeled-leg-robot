#include <Arduino.h>
#include <TFT_eSPI.h>
#include <math.h>

// Pin config comes from build flags (USER_SETUP_LOADED)
#define BL_PIN 15
#define SW     320
#define SH     240
#define CX     (SW / 2)
#define CY     (SH / 2)

static TFT_eSPI    tft;
static TFT_eSprite spr(&tft);
static int         spr_h = 0;   // actual allocated sprite height (0 = failed)

struct BenchResult { const char* label; float fps; float ms; };
static BenchResult results[5];
static int         n_res = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

static void record(const char* lbl, uint32_t us, int n) {
    float fps = (float)n * 1e6f / (float)us;
    float ms  = (float)us / 1000.0f / (float)n;
    results[n_res++] = {lbl, fps, ms};
    Serial.printf("  %-30s %6.1f FPS   %5.2f ms/frame\n", lbl, fps, ms);
}

// Fast integer hue (0-359) → RGB565
static uint16_t hue565(uint16_t h) {
    h %= 360;
    uint8_t s = h / 60, f = (uint8_t)((h % 60) * 255 / 60);
    uint8_t r, g, b;
    switch (s) {
        case 0: r=255;   g=f;     b=0;     break;
        case 1: r=255-f; g=255;   b=0;     break;
        case 2: r=0;     g=255;   b=f;     break;
        case 3: r=0;     g=255-f; b=255;   break;
        case 4: r=f;     g=0;     b=255;   break;
        default:r=255;   g=0;     b=255-f; break;
    }
    return ((uint16_t)(r >> 3) << 11) | ((uint16_t)(g >> 2) << 5) | (b >> 3);
}

// ── Sprite allocation: probe downward until something fits ────────────────────

static void alloc_anim_sprite() {
    spr.setColorDepth(16);
    for (int h = SH; h >= 120; h -= 20) {
        if (spr.createSprite(SW, h)) {
            spr_h = h;
            Serial.printf("  Anim sprite: 320x%d (%u KB)  free heap: %u\n",
                          h, (SW * h * 2) / 1024, ESP.getFreeHeap());
            return;
        }
    }
    Serial.printf("  [ERR] sprite alloc failed at all sizes  heap: %u\n", ESP.getFreeHeap());
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

static void bench_fillscreen() {
    const int N = 200;
    uint32_t t = micros();
    for (int i = 0; i < N; i++)
        tft.fillScreen(i & 1 ? TFT_RED : TFT_NAVY);
    record("fillScreen (320x240)", micros() - t, N);
}

static void bench_fullsprite() {
    if (!spr_h) { Serial.println("  [SKIP] fullsprite"); return; }
    const int N = 200;
    uint32_t t = micros();
    for (int i = 0; i < N; i++) {
        spr.fillSprite(i & 1 ? TFT_DARKGREEN : TFT_DARKCYAN);
        spr.pushSprite(0, 0);
    }
    char lbl[32]; snprintf(lbl, sizeof(lbl), "320x%d sprite push", spr_h);
    record(lbl, micros() - t, N);
}

static void bench_quad_sprite() {
    TFT_eSprite q(&tft);
    const int N = 100, QW = SW/2, QH = SH/2;
    q.setColorDepth(16);
    if (!q.createSprite(QW, QH)) { Serial.println("  [SKIP] quad alloc"); return; }
    const uint16_t cols[4] = {TFT_NAVY, TFT_MAROON, TFT_DARKGREEN, TFT_DARKCYAN};
    uint32_t t = micros();
    for (int i = 0; i < N; i++)
        for (int qd = 0; qd < 4; qd++) {
            q.fillSprite(cols[qd]);
            q.pushSprite((qd & 1) ? QW : 0, (qd & 2) ? QH : 0);
        }
    record("4x 160x120 sprite push", micros() - t, N);
    q.deleteSprite();
}

static void bench_complex() {
    TFT_eSprite c(&tft);
    const int N = 100, CW = 170, CH = 156;
    c.setColorDepth(16);
    if (!c.createSprite(CW, CH)) { Serial.println("  [SKIP] complex alloc"); return; }
    uint32_t t = micros();
    for (int i = 0; i < N; i++) {
        c.fillSprite(TFT_BLACK);
        for (int j = 0; j < 30; j++) c.drawLine(j*5, 0, CW-j*5, CH-1, TFT_WHITE);
        for (int j = 0; j < 10; j++) c.fillCircle(CW/2, CH/2, 48-j*4, j&1?TFT_RED:TFT_BLUE);
        for (int j = 0; j < 5;  j++) c.fillRect(j*28, j*18, 40, 16, TFT_GREEN);
        c.setTextColor(TFT_YELLOW, TFT_BLACK); c.setTextSize(1);
        c.setCursor(4, 4);  c.printf("Pitch: %+.1f", i*0.3f);
        c.setCursor(4, 13); c.printf("Roll:  %+.1f", i*0.2f);
        c.pushSprite(0, 0);
    }
    record("170x156 complex sprite", micros() - t, N);
    c.deleteSprite();
}

// ── Results screen ────────────────────────────────────────────────────────────

static uint16_t fps_color(float fps) {
    return fps >= 100.f ? TFT_GREEN : fps >= 40.f ? TFT_YELLOW : TFT_RED;
}

static void show_results() {
    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_WHITE, TFT_BLACK); tft.setTextSize(2);
    tft.setCursor(8, 6); tft.print("TFT BENCHMARK");
    tft.setTextSize(1); tft.setTextColor(tft.color565(120,120,120), TFT_BLACK);
    tft.setCursor(8, 24);
    tft.printf("ST7789 320x240  SPI 80MHz  sprite:320x%d", spr_h);

    uint16_t dim = tft.color565(50,50,50);
    tft.drawFastHLine(0, 36, SW, dim);

    int y = 44;
    for (int i = 0; i < n_res; i++) {
        tft.setTextSize(1); tft.setTextColor(tft.color565(140,140,140), TFT_BLACK);
        tft.setCursor(8, y); tft.print(results[i].label);
        tft.setTextSize(2); tft.setTextColor(fps_color(results[i].fps), TFT_BLACK);
        tft.setCursor(8, y+10); tft.printf("%.0f FPS", results[i].fps);
        tft.setTextSize(1); tft.setTextColor(tft.color565(100,100,100), TFT_BLACK);
        tft.setCursor(115, y+14); tft.printf("%.2f ms", results[i].ms);
        y += 40;
    }
    tft.drawFastHLine(0, y+2, SW, dim);
    tft.setTextSize(1); tft.setTextColor(tft.color565(80,80,80), TFT_BLACK);
    tft.setCursor(8, y+9);
    tft.print(spr_h ? "Starting animation..." : "No sprite RAM — animation skipped");
    delay(6000);
}

// ── Animation: rainbow background + bouncing ball ─────────────────────────────

static void run_animation() {
    const int top_y  = (SH - spr_h) / 2;
    const int scx    = SW / 2;
    const int scy    = spr_h / 2;
    const int BALL_R = 18;

    // Clear margins (only if sprite doesn't fill full height)
    if (top_y > 0) {
        tft.fillRect(0, 0,           SW, top_y,              TFT_BLACK);
        tft.fillRect(0, top_y+spr_h, SW, SH-(top_y+spr_h),  TFT_BLACK);
    }

    uint32_t frame    = 0;
    uint32_t fps_t    = micros();
    uint32_t serial_t = millis();
    float    live_fps = 0.0f;

    while (true) {
        // Rainbow background: each row gets a hue that shifts with frame
        uint16_t hue_offset = (uint16_t)(frame * 2);
        for (int y = 0; y < spr_h; y++)
            spr.drawFastHLine(0, y, SW, hue565(hue_offset + (uint16_t)(y * 360 / spr_h)));

        // Ball: Lissajous path (x and y at incommensurate frequencies → fills the space)
        float t  = (float)frame * 0.025f;
        int bx = scx + (int)((scx - BALL_R - 4) * sinf(t * 1.3f));
        int by = scy + (int)((scy - BALL_R - 4) * cosf(t * 0.7f));

        // Shadow
        spr.fillCircle(bx + 3, by + 3, BALL_R, tft.color565(0, 0, 0));
        // Ball body (white)
        spr.fillCircle(bx, by, BALL_R, TFT_WHITE);
        // Glint
        spr.fillCircle(bx - 6, by - 6, 5, tft.color565(255, 255, 255));

        // FPS counter (top-right, dark backing so it's readable on any color)
        spr.fillRect(SW - 88, 3, 86, 13, tft.color565(0, 0, 0));
        spr.setTextColor(TFT_WHITE, tft.color565(0,0,0));
        spr.setTextSize(1);
        spr.setCursor(SW - 86, 5);
        spr.printf("%.0f FPS  320x%d", live_fps, spr_h);

        spr.pushSprite(0, top_y);
        frame++;

        if (frame % 30 == 0) {
            uint32_t now = micros();
            live_fps = 30.0f * 1e6f / (float)(now - fps_t);
            fps_t    = now;
        }
        if (millis() - serial_t >= 1000) {
            Serial.printf("  Anim: %.1f FPS\n", live_fps);
            serial_t = millis();
        }
    }
}

// ── Entry points ──────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(921600);
    delay(500);

    pinMode(BL_PIN, OUTPUT);
    digitalWrite(BL_PIN, HIGH);

    tft.init();
    tft.setRotation(1);
    tft.fillScreen(TFT_BLACK);

    Serial.println("\n========================================");
    Serial.println(" TFT BENCHMARK  ST7789 320x240 @ 80MHz");
    Serial.println("========================================");
    Serial.printf("  Free heap at start: %u bytes\n", ESP.getFreeHeap());

    // Allocate animation sprite first while heap is freshest
    alloc_anim_sprite();

    tft.setTextColor(TFT_WHITE, TFT_BLACK); tft.setTextSize(1);
    tft.setCursor(8, 8);
    tft.printf("Sprite 320x%d  Running benchmarks...", spr_h);

    bench_fillscreen();
    bench_fullsprite();   // reuses spr in-place
    bench_quad_sprite();  // local sprite, freed after
    bench_complex();      // local sprite, freed after

    Serial.println("----------------------------------------");
    Serial.printf("  %d stages complete.\n", n_res);

    show_results();

    if (spr_h)
        run_animation();  // never returns
    else
        Serial.println("[ERR] No animation sprite — done.");
}

void loop() {}
