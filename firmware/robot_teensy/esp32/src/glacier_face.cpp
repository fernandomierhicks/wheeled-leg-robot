// GLACIER "Visor Eyes" face — see glacier_face.h.
//
// Ported from the approved browser preview. The preview drew into a 320x240 array of
// 4-bit palette indices with the same primitives used here (scanline polygons, thick
// segments as quads, per-pixel octagon tests), so shapes and timings carry over
// number for number.
//
// Nothing has a loop point: every value is a continuous function of time, gestures and
// state changes ease in and out of the resting pose (state changes morph over MORPH_S),
// and pulses ride an accumulated distance so speed changes never make them jump.

#include "glacier_face.h"
#include <Arduino.h>
#include <TFT_eSPI.h>
#include <math.h>
#include <string.h>

#ifndef GLACIER_FACE_FPS
#define GLACIER_FACE_FPS 0   // 1 = draw frames/s top-left. Serial carries CommLink frames, so no prints.
#endif

namespace {

constexpr int   FW = 320, FH = 240, ROW_BYTES = FW / 2;
constexpr float PI_F    = 3.14159265f;
constexpr float MORPH_S = 0.35f;   // state changes morph over this long instead of snapping

TFT_eSprite* s_spr = nullptr;
uint8_t*     s_buf = nullptr;      // 4 bpp: byte (x + y*FW)/2, even x in bits 7..4

inline int   imin(int a, int b) { return a < b ? a : b; }
inline int   imax(int a, int b) { return a > b ? a : b; }
inline float clamp01(float x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
inline float sstep(float a, float b, float x) { float u = clamp01((x - a) / (b - a)); return u * u * (3 - 2 * u); }
inline float env(float u) { return sstep(0, 0.15f, u) * (1 - sstep(0.85f, 1, u)); }   // 0 at both ends of a gesture

// Eased keyframes: k = {u0,v0, u1,v1, ...}, n pairs.
float kf(float u, const float* k, int n) {
    if (u <= k[0]) return k[1];
    for (int i = 0; i + 1 < n; i++)
        if (u <= k[2 * i + 2]) return k[2 * i + 1] + (k[2 * i + 3] - k[2 * i + 1]) * sstep(k[2 * i], k[2 * i + 2], u);
    return k[2 * n - 1];
}

// Integer hash -> [0,1). Replaces the preview's sin()-hash, which loses precision in float.
float h01(int32_t k, uint32_t salt) {
    uint32_t x = ((uint32_t)k * 0x9E3779B1u) ^ (salt * 0x85EBCA77u);
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
    return (x >> 8) * (1.0f / 16777216.0f);
}

// ── Palette ─────────────────────────────────────────────────────────────────
// 0-5 fixed neutrals, 6-11 the glow ramp (rebuilt from one base colour per state),
// 12-15 fixed extras. GLACIER graphite and blue come from cad/solidworks_api/swstyle.py.
const uint8_t FIXED_RGB[16][3] = {
    {0, 0, 0}, {10, 12, 15}, {30, 33, 39}, {64, 69, 77}, {126, 135, 149}, {237, 240, 242},
    {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
    {22, 25, 30}, {46, 50, 57}, {226, 190, 46}, {255, 255, 255}};

// Glow base colour per GlacierState: GLACIER blue except reject, E-stop and not-linked.
const uint8_t BASE_RGB[GF_COUNT][3] = {
    {78, 92, 112},    // GF_OFFLINE  steel
    {33, 115, 242},   // GF_STARTUP
    {33, 115, 242},   // GF_CALIB
    {33, 115, 242},   // GF_STANDBY
    {33, 115, 242},   // GF_RUNNING
    {33, 115, 242},   // GF_MANUAL
    {33, 115, 242},   // GF_JUMP
    {255, 120, 0},    // GF_REJECT   orange
    {235, 30, 30},    // GF_ESTOP    red
    {33, 115, 242},   // GF_STANDUP
    {33, 115, 242}};  // GF_DISARM

uint16_t rgb565(float r, float g, float b) {
    int ri = (int)lroundf(r), gi = (int)lroundf(g), bi = (int)lroundf(b);
    ri = imin(255, imax(0, ri)); gi = imin(255, imax(0, gi)); bi = imin(255, imax(0, bi));
    return (uint16_t)(((ri & 0xF8) << 8) | ((gi & 0xFC) << 3) | (bi >> 3));
}

void set_ramp(float r, float g, float b) {
    static const float K[5] = {0.09f, 0.2f, 0.38f, 0.66f, 1.0f};
    for (int i = 0; i < 5; i++) s_spr->setPaletteColor(6 + i, rgb565(r * K[i], g * K[i], b * K[i]));
    s_spr->setPaletteColor(11, rgb565(r + (255 - r) * 0.6f, g + (255 - g) * 0.6f, b + (255 - b) * 0.6f));
}

// ── Raster primitives on the 4 bpp buffer ──────────────────────────────────
inline void put(int x, int y, uint8_t c) {
    uint8_t* p = s_buf + y * ROW_BYTES + (x >> 1);
    *p = (x & 1) ? (uint8_t)((*p & 0xF0) | c) : (uint8_t)((*p & 0x0F) | (c << 4));
}

void hspan(int x0, int x1, int y, uint8_t c) {   // inclusive, already clipped to the screen
    if (x0 > x1) return;
    if (x0 & 1) { put(x0, y, c); x0++; }
    if (x0 > x1) return;
    if (!(x1 & 1)) { put(x1, y, c); x1--; }
    if (x0 <= x1) memset(s_buf + y * ROW_BYTES + (x0 >> 1), (c << 4) | c, ((x1 - x0) >> 1) + 1);
}

void clear(uint8_t c) { memset(s_buf, (c << 4) | c, ROW_BYTES * FH); }

// Even-odd scanline fill sampled at pixel centres. With row_mod set, only rows where
// (y - row_ref) mod row_mod < row_keep are drawn (the slatted glow inside an open mouth).
void poly(const float* p, int n, uint8_t c, int row_mod = 0, int row_keep = 0, int row_ref = 0) {
    float y0 = 1e9f, y1 = -1e9f;
    for (int k = 0; k < n; k++) { y0 = fminf(y0, p[2 * k + 1]); y1 = fmaxf(y1, p[2 * k + 1]); }
    int js = imax(0, (int)ceilf(y0 - 0.5f)), je = imin(FH - 1, (int)floorf(y1 - 0.5f));
    float xs[32];
    for (int j = js; j <= je; j++) {
        if (row_mod && (((j - row_ref) % row_mod + row_mod) % row_mod) >= row_keep) continue;
        float sy = j + 0.5f;
        int m = 0;
        for (int k = 0; k < n; k++) {
            int b = (k + 1) % n;
            float ax = p[2 * k], ay = p[2 * k + 1], bx = p[2 * b], by = p[2 * b + 1];
            if (((ay <= sy) && (by > sy)) || ((by <= sy) && (ay > sy)))
                if (m < 32) xs[m++] = ax + (sy - ay) / (by - ay) * (bx - ax);
        }
        for (int a = 1; a < m; a++) {   // insertion sort; m is tiny
            float v = xs[a]; int q = a - 1;
            while (q >= 0 && xs[q] > v) { xs[q + 1] = xs[q]; q--; }
            xs[q + 1] = v;
        }
        for (int q = 0; q + 1 < m; q += 2)
            hspan(imax(0, (int)ceilf(xs[q] - 0.5f)), imin(FW - 1, (int)ceilf(xs[q + 1] - 0.5f) - 1), j, c);
    }
}

// Thick line with square caps: one quad (two triangles' worth).
void seg(float x0, float y0, float x1, float y1, float w, uint8_t c) {
    float dx = x1 - x0, dy = y1 - y0, L = sqrtf(dx * dx + dy * dy);
    if (L < 1e-6f) { dx = 1; dy = 0; L = 1; }
    float ux = dx / L, uy = dy / L, h = w * 0.5f, nx = -uy * h, ny = ux * h, ex = ux * h, ey = uy * h;
    const float q[8] = {x0 - ex + nx, y0 - ey + ny, x1 + ex + nx, y1 + ey + ny,
                        x1 + ex - nx, y1 + ey - ny, x0 - ex - nx, y0 - ey - ny};
    poly(q, 4, c);
}

void path(const float* P, int n, float w, uint8_t c) {
    for (int k = 0; k + 1 < n; k++) seg(P[2 * k], P[2 * k + 1], P[2 * k + 2], P[2 * k + 3], w, c);
}

float path_len(const float* P, int n) {
    float L = 0;
    for (int k = 0; k + 1 < n; k++) L += hypotf(P[2 * k + 2] - P[2 * k], P[2 * k + 3] - P[2 * k + 1]);
    return L;
}

// The stretch of a path between arc lengths s0..s1.
void part(const float* P, int n, float s0, float s1, float w, uint8_t c) {
    if (s1 <= s0) return;
    float acc = 0;
    for (int k = 0; k + 1 < n; k++) {
        float x0 = P[2 * k], y0 = P[2 * k + 1], x1 = P[2 * k + 2], y1 = P[2 * k + 3], l = hypotf(x1 - x0, y1 - y0);
        float a = fmaxf(s0, acc), b = fminf(s1, acc + l);
        if (b > a && l > 0) {
            float ta = (a - acc) / l, tb = (b - acc) / l;
            seg(x0 + (x1 - x0) * ta, y0 + (y1 - y0) * ta, x0 + (x1 - x0) * tb, y0 + (y1 - y0) * tb, w, c);
        }
        acc += l;
        if (acc >= s1) break;
    }
}

struct Trace { float p[12]; uint8_t n; float len; float phase; };

// A pulse: bright head, dimmer tail behind it.
void pulse(const Trace& t, float s, float w) {
    static const uint8_t TAIL[4][3] = {{40, 24, 8}, {24, 12, 9}, {12, 4, 10}, {4, 0, 11}};
    for (int i = 0; i < 4; i++) part(t.p, t.n, fmaxf(0, s - TAIL[i][0]), fminf(t.len, s - TAIL[i][1]), w, TAIL[i][2]);
}

// ── Scene: traces, combs and plates (left half, mirrored) ──────────────────
Trace s_tr[6];
float s_combs[16][4];
float s_plates[2][10];
const float PLATE_TOP[12] = {140, 6, 180, 6, 180, 15, 172, 23, 148, 23, 140, 15};
const float VISOR[12]     = {150, 11, 170, 11, 173, 14, 170, 17, 150, 17, 147, 14};

void build_scene() {
    static const float TR[3][12] = {{0, 34, 18, 34, 38, 54, 38, 98, 46, 106},
                                    {56, 0, 56, 10, 68, 22, 124, 22, 132, 30, 132, 46},
                                    {0, 146, 10, 146, 26, 162, 26, 206, 38, 218, 118, 218}};
    static const uint8_t TRN[3] = {5, 6, 6};
    for (int i = 0; i < 3; i++) {
        Trace& a = s_tr[2 * i];
        Trace& b = s_tr[2 * i + 1];
        a.n = b.n = TRN[i];
        for (int k = 0; k < 2 * TRN[i]; k++) { a.p[k] = TR[i][k]; b.p[k] = (k % 2 == 0) ? FW - TR[i][k] : TR[i][k]; }
        a.len = b.len = path_len(a.p, a.n);
        a.phase = i * 53.0f;
        b.phase = i * 53.0f + 29;
    }
    int n = 0;
    for (float y : {68.f, 75.f, 82.f})  { float c[4] = {32, y + 6, 44, y - 6}; memcpy(s_combs[n++], c, sizeof c); }
    for (float x : {86.f, 93.f, 100.f}) { float c[4] = {x - 6, 28, x + 6, 16}; memcpy(s_combs[n++], c, sizeof c); }
    for (float y : {176.f, 183.f})      { float c[4] = {20, y + 6, 32, y - 6}; memcpy(s_combs[n++], c, sizeof c); }
    for (int i = 0; i < 8; i++) {
        s_combs[8 + i][0] = FW - s_combs[i][0]; s_combs[8 + i][1] = s_combs[i][1];
        s_combs[8 + i][2] = FW - s_combs[i][2]; s_combs[8 + i][3] = s_combs[i][3];
    }
    static const float PL[10] = {6, 62, 28, 62, 28, 104, 14, 118, 6, 118};
    for (int k = 0; k < 10; k++) { s_plates[0][k] = PL[k]; s_plates[1][k] = (k % 2 == 0) ? FW - PL[k] : PL[k]; }
}

// ── Mouth shapes: upper + lower lip, 8 points each, every segment flat or 45 deg ──
enum Mood : uint8_t { M_SLEEP, M_SNORE, M_NEUTRAL, M_FLAT, M_SMILE, M_GRIN, M_ZIG, M_THINK, M_SMIRK,
                      M_O, M_OSTOP, M_YAWN, M_STRAIN, M_DRIVE, M_COUNT };
struct Shape { float u[16]; float l[16]; };
Shape SH[M_COUNT];

const uint8_t MOOD_OF[GF_COUNT] = {M_SLEEP, M_NEUTRAL, M_THINK, M_SMILE, M_DRIVE, M_DRIVE,
                                   M_GRIN, M_ZIG, M_OSTOP, M_STRAIN, M_NEUTRAL};

void same(Shape& s, const float* u) { memcpy(s.u, u, sizeof s.u); memcpy(s.l, u, sizeof s.l); }
void oct_shape(Shape& s, float ax, float ay) {   // an octagon written as an upper and a lower lip
    float c = roundf(fminf(ax, ay) * 0.42f);
    const float u[16] = {-ax, 0, -ax, -(ay - c), -(ax - c), -ay, -2, -ay, 2, -ay, ax - c, -ay, ax, -(ay - c), ax, 0};
    for (int i = 0; i < 16; i++) { s.u[i] = u[i]; s.l[i] = (i & 1) ? -u[i] : u[i]; }
}

void build_shapes() {
    static const float SLEEP[16]   = {-22, 0, -15, 0, -8, 0, -2, 0, 2, 0, 8, 0, 15, 0, 22, 0};
    static const float NEUTRAL[16] = {-46, -6, -40, 0, -24, 0, -8, 0, 8, 0, 24, 0, 40, 0, 46, -6};
    static const float FLAT[16]    = {-44, 0, -30, 0, -18, 0, -6, 0, 6, 0, 18, 0, 30, 0, 44, 0};
    static const float SMILE[16]   = {-54, -14, -44, -4, -34, 6, -12, 6, 12, 6, 34, 6, 44, -4, 54, -14};
    static const float GRIN_U[16]  = {-58, -16, -48, -6, -44, -2, -26, -2, 26, -2, 44, -2, 48, -6, 58, -16};
    static const float GRIN_L[16]  = {-58, -16, -46, -4, -34, 8, -26, 16, 26, 16, 34, 8, 46, -4, 58, -16};
    static const float ZIG[16]     = {-36, 0, -30, -6, -18, 6, -6, -6, 6, 6, 18, -6, 30, 6, 36, 0};
    static const float THINK[16]   = {-14, 4, -4, 4, 6, 4, 14, 4, 20, 4, 24, 4, 28, 0, 32, -4};
    static const float SMIRK[16]   = {-40, 2, -28, 2, -16, 2, -4, 2, 8, 2, 20, 2, 30, -8, 38, -16};
    static const float STRAIN[16]  = {-40, 6, -34, 0, -20, 0, -6, 0, 6, 0, 20, 0, 34, 0, 40, 6};   // gritted, corners down
    same(SH[M_SLEEP], SLEEP);     oct_shape(SH[M_SNORE], 9, 4);
    same(SH[M_NEUTRAL], NEUTRAL); same(SH[M_FLAT], FLAT);
    same(SH[M_SMILE], SMILE);     memcpy(SH[M_GRIN].u, GRIN_U, sizeof GRIN_U); memcpy(SH[M_GRIN].l, GRIN_L, sizeof GRIN_L);
    same(SH[M_ZIG], ZIG);         same(SH[M_THINK], THINK);    same(SH[M_SMIRK], SMIRK);
    oct_shape(SH[M_O], 12, 11);   oct_shape(SH[M_OSTOP], 15, 14); oct_shape(SH[M_YAWN], 17, 21);
    same(SH[M_STRAIN], STRAIN);   same(SH[M_DRIVE], FLAT);     // drive is computed live in mouth_shape()
}

void mix(const Shape& a, const Shape& b, float w, Shape& o) {
    for (int i = 0; i < 16; i++) { o.u[i] = a.u[i] + (b.u[i] - a.u[i]) * w; o.l[i] = a.l[i] + (b.l[i] - a.l[i]) * w; }
}
void mirror(const Shape& s, Shape& o) {
    for (int k = 0; k < 8; k++) {
        o.u[2 * k] = -s.u[2 * (7 - k)]; o.u[2 * k + 1] = s.u[2 * (7 - k) + 1];
        o.l[2 * k] = -s.l[2 * (7 - k)]; o.l[2 * k + 1] = s.l[2 * (7 - k) + 1];
    }
}

// ── Gestures: one-shots that start and end at the resting pose ─────────────
enum Gesture : uint8_t { G_GLANCE, G_DBLINK, G_WINK, G_SQUINT, G_SCAN, G_JOY, G_LOOKUP, G_SURGE, G_ROLL,
                         G_SIDEEYE, G_SMIRK, G_YAWN, G_PEEK, G_BREATH, G_PONDER, G_FLINCH, G_NERVOUS,
                         G_COUNT, G_NONE = 0xFF };
const float G_DUR[G_COUNT] = {1.9f, 0.55f, 0.7f, 1.4f, 0.9f, 1.2f, 1.4f, 0.9f, 1.5f,
                              1.9f, 1.6f, 2.4f, 2.2f, 2.2f, 2.0f, 0.8f, 1.2f};

const uint8_t GS_OFFLINE[] = {G_PEEK, G_BREATH};
const uint8_t GS_STARTUP[] = {G_DBLINK, G_GLANCE};
const uint8_t GS_CALIB[]   = {G_PONDER, G_DBLINK};
const uint8_t GS_STANDBY[] = {G_GLANCE, G_DBLINK, G_WINK, G_LOOKUP, G_ROLL, G_SIDEEYE, G_JOY, G_SMIRK, G_YAWN};
const uint8_t GS_RUNNING[] = {G_GLANCE, G_DBLINK, G_WINK, G_SQUINT, G_SCAN, G_JOY, G_SURGE, G_SIDEEYE, G_SMIRK};
const uint8_t GS_ESTOP[]   = {G_FLINCH, G_NERVOUS};

// Random gesture sets per state, with the gap between gestures. Jump, reject, stand-up
// and disarm are short transitions and get none.
struct GSet { const uint8_t* ids; uint8_t n; uint16_t min_ms, max_ms; };
const GSet GSETS[GF_COUNT] = {
    {GS_OFFLINE, sizeof GS_OFFLINE, 8000, 15000},   // GF_OFFLINE
    {GS_STARTUP, sizeof GS_STARTUP, 4000, 8000},    // GF_STARTUP (after the boot sequence)
    {GS_CALIB,   sizeof GS_CALIB,   5000, 9000},    // GF_CALIB
    {GS_STANDBY, sizeof GS_STANDBY, 6000, 12000},   // GF_STANDBY
    {GS_RUNNING, sizeof GS_RUNNING, 4000, 9000},    // GF_RUNNING
    {GS_RUNNING, sizeof GS_RUNNING, 4000, 9000},    // GF_MANUAL
    {nullptr, 0, 0, 0},                             // GF_JUMP
    {nullptr, 0, 0, 0},                             // GF_REJECT
    {GS_ESTOP,   sizeof GS_ESTOP,   4000, 8000},    // GF_ESTOP
    {nullptr, 0, 0, 0},                             // GF_STANDUP
    {nullptr, 0, 0, 0}};                            // GF_DISARM

// ── Expression parameters ───────────────────────────────────────────────────
struct FaceParams {
    float   hw, hh, ch, lookX, lookY, blink, lidTop, lidBot, rake, happyT;   // rake/happyT 200 = no cut
    float   eyeLid[2], eyeBlink[2];
    float   surge;                 // < 0: none
    float   scan;   bool hasScan;
    float   grow, fill; bool hasGrow;
    int8_t  strobe;                // -1 none, 0 dark phase, 1 lit phase
    bool    dim, glint, flash;
    float   bounce, shake, v, thr; // v = pulse speed px/s, thr = running "throttle" 0..1
    uint8_t mFrom, mTo; float mW; int8_t mSideFrom, mSide;
};

float blink_at(float x) { return (x >= 0 && x <= 0.17f) ? 1 - 0.93f * sinf(PI_F * x / 0.17f) : 1; }

// Each 1 s slot blinks with probability q, at a random offset; a quarter are doubles.
float blink_rand(float t, uint32_t seed, float q) {
    int32_t k = (int32_t)floorf(t);
    float f = t - k, b = 1;
    if (h01(k, seed) < q) {
        float off = h01(k, seed + 11) * 0.55f;
        b = blink_at(f - off);
        if (h01(k, seed + 5) < 0.25f) b = fminf(b, blink_at(f - off - 0.26f));
    }
    return b;
}

// Gaze target that hops every `period` s, eased over 0.22 s.
void look_hop(float t, float period, uint32_t seed, float& x, float& y) {
    int32_t k = (int32_t)floorf(t / period);
    float f = t - k * period;
    float a0 = h01(k, seed) * 2 - 1, a1 = h01(k, seed + 57) * 2 - 1;
    float b0 = h01(k - 1, seed) * 2 - 1, b1 = h01(k - 1, seed + 57) * 2 - 1;
    float e = fminf(1, f / 0.22f), s = e * e * (3 - 2 * e);
    x = b0 + (a0 - b0) * s;
    y = b1 + (a1 - b1) * s;
}

// Quick look away, hold, come back, all inside one slot.
void glances(float t, uint32_t seed, float slot, float q, float ax, float ay, float& x, float& y) {
    int32_t k = (int32_t)floorf(t / slot);
    float f = t - k * slot;
    x = y = 0;
    if (h01(k, seed) >= q) return;
    float hold = 0.5f + 0.6f * h01(k, seed + 21), s0 = 0.1f * slot * h01(k, seed + 9);
    float w = sstep(s0, s0 + 0.12f, f) * (1 - sstep(s0 + 0.12f + hold, s0 + 0.3f + hold, f));
    x = (h01(k, seed + 3) * 2 - 1) * ax * w;
    y = (h01(k, seed + 7) * 2 - 1) * ay * w;
}

FaceParams face_params(float t, GlacierState st, float ts) {
    FaceParams p;
    memset(&p, 0, sizeof p);
    p.hw = 44; p.hh = 48; p.ch = 20; p.blink = 1; p.rake = 200; p.happyT = 200;
    p.eyeBlink[0] = p.eyeBlink[1] = 1; p.surge = -1; p.fill = 1; p.strobe = -1; p.glint = true;
    p.v = 60; p.mSideFrom = p.mSide = 1;
    float x, y, c;
    switch (st) {
    case GF_OFFLINE:
        p.lidTop = 54; p.dim = true; p.v = 22; p.glint = false;
        break;
    case GF_STARTUP:   // boot sequence once: traces power on, slats fill bottom-up, one blink
        if (ts < 1.9f) {
            p.hasGrow = true; p.grow = clamp01(ts / 0.8f); p.fill = clamp01((ts - 0.7f) / 0.9f); p.v = 0;
            if (ts > 1.7f) p.blink = blink_at(ts - 1.7f);
        } else {
            float k = sstep(1.9f, 2.4f, ts);
            look_hop(t, 2.6f, 17, x, y); p.lookX = x * 6 * k; p.lookY = y * 3 * k;
            p.blink = blink_rand(t, 13, 0.25f);
        }
        break;
    case GF_CALIB:
        p.hh = 26; p.ch = 12; p.hasScan = true; p.scan = sinf(t * 2.4f); p.v = 90;
        break;
    case GF_STANDBY:
        look_hop(t, 2.6f, 3, x, y); p.lookX = x * 9; p.lookY = y * 4;
        p.blink = blink_rand(t, 3, 0.27f);
        break;
    case GF_RUNNING:
    case GF_MANUAL:    // c = synthetic throttle, drifting slowly 0..1
        c = 0.5f + 0.5f * (0.6f * sinf(t * 0.21f) + 0.4f * sinf(t * 0.53f + 2));
        glances(t, 31, 1.8f, 0.45f, 9, 5, x, y);
        p.lookX = 2.5f * sinf(t * 0.37f) + 1.5f * sinf(t * 0.93f + 1) + 3 * c * sinf(t * 0.8f) + x;
        p.lookY = 2 + 1.2f * sinf(t * 0.61f) + y;
        p.lidTop = 10 + 10 * c; p.lidBot = 6 + 6 * c;
        p.blink = blink_rand(t, 7, 0.22f); p.v = 110 + 130 * c; p.thr = c;
        break;
    case GF_JUMP:
        p.happyT = 20; p.ch = 36; p.lidBot = 20; p.bounce = -fabsf(sinf(t * 5.2f)) * 9; p.v = 260;
        break;
    case GF_REJECT:    // plays once on entry: head-shake and flashing traces
        p.rake = -8; p.shake = sinf(ts * 38) * 7 * expf(-ts * 2.6f); p.v = 0;
        p.flash = ts < 0.9f && ((int)floorf(ts * 10) % 2 == 0);
        break;
    case GF_ESTOP:
        p.hw = 34; p.hh = 34; p.ch = 14; p.strobe = ((int)floorf(t * 4) % 2 == 0) ? 1 : 0;
        p.v = 0; p.flash = p.strobe == 1; p.glint = false;
        break;
    case GF_STANDUP:   // effort: squeezed lids, a slight tremble, fast pulses
        p.lidTop = 16; p.lidBot = 12; p.lookY = -3; p.shake = sinf(t * 31) * 1.2f; p.v = 200;
        break;
    case GF_DISARM:    // settling: lids sink as power winds down
        c = clamp01(ts / 1.2f); p.lidTop = 26 * c; p.v = 60 * (1 - c) + 20;
        break;
    default:
        break;
    }
    return p;
}

void apply_gesture(FaceParams& p, uint8_t id, float u, int8_t side) {
    u = clamp01(u);
    float e = env(u);
    int i = side < 0 ? 0 : 1;
    switch (id) {
    case G_GLANCE: { static const float K[] = {0, 0, 0.14f, -14, 0.42f, -14, 0.58f, 14, 0.86f, 14, 1, 0};
        p.lookX += kf(u, K, 6); break; }
    case G_DBLINK:
        p.blink = fminf(p.blink, fminf(blink_at(u * 0.55f), blink_at(u * 0.55f - 0.27f))); break;
    case G_WINK: { static const float K1[] = {0, 1, 0.2f, 0.07f, 0.6f, 0.07f, 0.85f, 1};
                   static const float K2[] = {0, 0, 0.2f, 1, 0.6f, 1, 0.85f, 0};
        p.eyeBlink[i] = kf(u, K1, 4); p.mTo = M_SMIRK; p.mSide = side; p.mW = kf(u, K2, 4); break; }
    case G_SQUINT:
        p.lidTop += 16 * e; p.lidBot += 12 * e; p.hw -= 3 * e; break;
    case G_SCAN:
        if (u > 0.05f && u < 0.95f) { p.hasScan = true; p.scan = -1 + 2 * sstep(0.05f, 0.95f, u); }
        break;
    case G_JOY:
        p.happyT += (20 - p.happyT) * e; p.ch += (36 - p.ch) * e; p.lidTop *= 1 - e; p.lidBot += (20 - p.lidBot) * e;
        p.bounce -= fabsf(sinf(u * PI_F * 3)) * 7 * e; p.mTo = M_GRIN; p.mW = e; break;
    case G_LOOKUP:
        p.lookY -= 9 * e; p.lidTop *= 1 - e; p.lidBot *= 1 - e; p.hh += 3 * e; p.mTo = M_O; p.mW = e; break;
    case G_SURGE:
        p.surge = u; break;
    case G_ROLL:
        p.lookX += 11 * sinf(2 * PI_F * u) * e; p.lookY -= 6 * cosf(2 * PI_F * u) * e; break;
    case G_SIDEEYE:
        p.lookX += 12 * side * e; p.lidTop += 18 * e; p.eyeLid[1 - i] += 8 * e;
        p.mTo = M_THINK; p.mSide = side; p.mW = e; break;
    case G_SMIRK:
        p.lookX += 6 * side * e; p.eyeLid[i] += 10 * e; p.mTo = M_SMIRK; p.mSide = side; p.mW = e; break;
    case G_YAWN: { static const float K[] = {0, 0, 0.3f, 1, 0.7f, 1, 1, 0}; float e2 = kf(u, K, 4);
        p.lidTop += 22 * e2; p.lidBot += 6 * e2; p.lookY -= 3 * e2; p.mTo = M_YAWN; p.mW = e2; break; }
    case G_PEEK: { static const float K[]  = {0, 0, 0.2f, 1, 0.8f, 1, 1, 0};                    // one eye opens and looks around
                   static const float KX[] = {0, 0, 0.3f, 0, 0.42f, -8, 0.62f, 8, 0.78f, 0, 1, 0};
        p.eyeLid[i] -= 40 * kf(u, K, 4); p.lookX += kf(u, KX, 6); break; }
    case G_BREATH:                                                                                // lids lift, mouth opens wide
        p.lidTop -= 14 * e; p.mTo = M_O; p.mW = 0.6f * e; break;
    case G_PONDER:                                                                                // look up to one side, "hmm" there
        p.lookX += 8 * side * e; p.lookY -= 6 * e; p.mTo = M_THINK; p.mSide = side; p.mW = e; break;
    case G_FLINCH: { static const float K[] = {0, 1, 0.15f, 0.25f, 0.5f, 0.25f, 0.8f, 1};
        p.blink = fminf(p.blink, kf(u, K, 4)); p.shake += sinf(u * 60) * 4 * e; break; }
    case G_NERVOUS: { static const float K[] = {0, 0, 0.15f, -10, 0.35f, -10, 0.5f, 10, 0.7f, 10, 0.85f, -6, 1, 0};
        p.lookX += kf(u, K, 7); break; }
    default:
        break;
    }
}

// ── Face state ──────────────────────────────────────────────────────────────
uint8_t  s_state = GF_COUNT, s_prev = GF_COUNT;   // GF_COUNT = none yet
uint32_t s_state_ms = 0, s_last_ms = 0;
double   s_D = 0;          // accumulated pulse distance (double: stays exact over hours of uptime)
float    s_v = 60;
uint8_t  s_gest = G_NONE, s_glast = G_NONE;
uint32_t s_gest_ms = 0, s_gnext_ms = 0;
int8_t   s_gside = 1;

void tick_gesture(uint32_t now, float ts) {
    const GSet& gs = GSETS[s_state];
    if (s_gest != G_NONE && now - s_gest_ms >= (uint32_t)(G_DUR[s_gest] * 1000)) {
        s_gest = G_NONE;
        if (gs.n) s_gnext_ms = now + random(gs.min_ms, gs.max_ms + 1);
    }
    if (s_gest != G_NONE || !gs.n || (int32_t)(now - s_gnext_ms) < 0) return;
    if (s_state == GF_STARTUP && ts < 2.5f) return;   // let the boot sequence finish first
    uint8_t pick;
    do { pick = gs.ids[random(gs.n)]; } while (gs.n > 1 && pick == s_glast);   // never the same twice in a row
    s_gest = pick; s_glast = pick; s_gest_ms = now; s_gside = random(2) ? 1 : -1;
}

FaceParams state_params(float t, float ts, uint32_t now) {
    FaceParams p = face_params(t, (GlacierState)s_state, ts);
    p.mFrom = p.mTo = MOOD_OF[s_state];
    if (s_prev < GF_COUNT && ts < MORPH_S) {          // morph out of the previous state instead of snapping
        FaceParams q = face_params(t, (GlacierState)s_prev, ts + 30);
        float w = sstep(0, MORPH_S, ts);
#define BLEND(f) p.f = q.f + (p.f - q.f) * w
        BLEND(hw); BLEND(hh); BLEND(ch); BLEND(lidTop); BLEND(lidBot); BLEND(rake); BLEND(happyT);
        BLEND(bounce); BLEND(shake); BLEND(v); BLEND(lookX); BLEND(lookY); BLEND(thr);
#undef BLEND
        p.mFrom = MOOD_OF[s_prev]; p.mW = w;
    }
    if (s_gest != G_NONE) apply_gesture(p, s_gest, (now - s_gest_ms) * 0.001f / G_DUR[s_gest], s_gside);
    return p;
}

// ── Drawing ─────────────────────────────────────────────────────────────────
struct Eye { float cx, cy, hw, hh, ch, lidTop, lidBot, rake, T; int inner; };

// Chamfered plate grown by g px, minus the expression cuts (lids, 45 deg rake, chevron).
inline bool eye_in(float x, float y, const Eye& e, float g) {
    float dx = fabsf(x - e.cx), dy = fabsf(y - e.cy);
    if (dx > e.hw + g || dy > e.hh + g || dx + dy > e.hw + e.hh - e.ch + g * 1.4142f) return false;
    float top = e.cy - e.hh;
    if (e.lidTop > 0 && y < top + e.lidTop - g) return false;
    if (e.lidBot > 0 && y > e.cy + e.hh - e.lidBot + g) return false;
    if ((y - top) < e.inner * (x - e.cx) - e.rake - g * 1.4142f) return false;
    if ((y - top) - dx > e.T + g * 1.4142f) return false;
    return true;
}

void draw_eye(const Eye& e, const FaceParams& p, float t) {
    float top = e.cy - e.hh, hgt = fmaxf(1, 2 * e.hh);
    int   cy0 = (int)lroundf(e.cy);
    float gx = e.cx + p.lookX * 0.4f, gy = e.cy - e.hh * 0.5f, gw = e.hw * 0.5f;
    float sx = p.hasScan ? e.cx + p.scan * (e.hw - 6) : -999;
    float by = top - 6 + sstep(0.6f, 1, p.surge) * (hgt + 12);   // surge band washing down
    bool  breathe = sinf(t * 1.3f) > 0.2f;
    int x0 = imax(0, (int)floorf(e.cx - e.hw - 6)), x1 = imin(FW, (int)ceilf(e.cx + e.hw + 6));
    int y0 = imax(0, (int)floorf(top - 6)),         y1 = imin(FH, (int)ceilf(e.cy + e.hh + 6));
    for (int j = y0; j < y1; j++) {
        float y = j + 0.5f, vf = (y - top) / hgt, ay = fabsf(y - gy);
        bool slat = ((j - cy0) % 8 + 8) % 8 >= 3;
        for (int i = x0; i < x1; i++) {
            float x = i + 0.5f;
            int c = -1;
            if (eye_in(x, y, e, 0)) {
                if (p.dim)                               c = slat ? (breathe ? 9 : 8) : 6;
                else if (p.fill < 1 && vf < 1 - p.fill)  c = slat ? 6 : 0;
                else if (fabsf(x - sx) < 4)              c = slat ? 11 : 9;
                else if (p.strobe >= 0)                  c = slat ? (p.strobe ? 10 : 8) : 7;
                else if (!slat)                          c = 7;
                else if (p.surge >= 0 && fabsf(y - by) < 6) c = 11;
                else if (p.glint && ay < 5 && fabsf(x - gx) < gw + (5 - ay)) c = 11;
                else                                     c = vf < 0.6f ? 10 : 9;
            } else if (eye_in(x, y, e, 4) && !eye_in(x, y, e, 2)) {
                c = p.dim ? 6 : 7;
            }
            if (c >= 0) put(i, j, (uint8_t)c);
        }
    }
}

// Some moods keep moving on their own.
void mouth_shape(uint8_t id, const FaceParams& p, float t, int8_t side, Shape& o) {
    switch (id) {
    case M_SMILE: mix(SH[M_NEUTRAL], SH[M_SMILE], 0.75f + 0.25f * sinf(t * 1.1f), o); return;          // breathing smile
    case M_DRIVE: mix(SH[M_FLAT], SH[M_SMILE], 0.15f + 0.5f * p.thr, o); return;                        // grins as throttle rises
    case M_GRIN:  mix(SH[M_SMILE], SH[M_GRIN], 0.7f + 0.3f * fabsf(sinf(t * 5.2f)), o); return;        // open, bouncing
    case M_SLEEP: mix(SH[M_SLEEP], SH[M_SNORE], 0.5f + 0.5f * sinf(t * 1.3f), o); return;              // slow snore
    case M_ZIG: {
        float k = 0.7f + 0.3f * sinf(t * 30);
        for (int i = 0; i < 16; i++) o.u[i] = o.l[i] = (i & 1) ? SH[M_ZIG].u[i] * k : SH[M_ZIG].u[i];
        return;
    }
    case M_THINK:
    case M_SMIRK:
        if (side < 0) mirror(SH[id], o); else o = SH[id];
        return;
    default:
        o = SH[id];
        return;
    }
}

// Raked 45 deg comb bars off each mouth corner, fading outward.
void mouth_combs(const FaceParams& p, float lx, float ly, float rx, float ry) {
    for (int k = 0; k < 3; k++) {
        float d = 9 + k * 7;
        uint8_t c = p.dim ? (k > 1 ? 6 : 7) : (p.strobe >= 0 ? (p.strobe ? 10 : 8) : (p.flash ? 11 : 10 - k));
        seg(rx + d - 5, ry + 5, rx + d + 5, ry - 5, 2, c);
        seg(lx - d + 5, ly + 5, lx - d - 5, ly - 5, 2, c);
    }
}

void draw_mouth(const FaceParams& p, float t, double D) {
    float mx = 160 + p.lookX * 0.6f + p.shake, my = 186 + p.lookY * 0.5f + p.bounce * 0.6f;
    Shape a, b, S;
    mouth_shape(p.mFrom, p, t, p.mSideFrom, a);
    mouth_shape(p.mTo, p, t, p.mSide, b);
    mix(a, b, p.mW, S);
    float tilt = 0.006f * p.lookX;                     // corners tilt toward the gaze
    float U[16], Lo[16];
    for (int i = 0; i < 16; i += 2) {
        U[i]  = mx + S.u[i]; U[i + 1]  = my + S.u[i + 1] - tilt * S.u[i];
        Lo[i] = mx + S.l[i]; Lo[i + 1] = my + S.l[i + 1] - tilt * S.l[i];
    }
    float open = 0;
    for (int i = 1; i < 16; i += 2) open = fmaxf(open, Lo[i] - U[i]);
    uint8_t core = p.dim ? 8 : (p.strobe >= 0 ? (p.strobe ? 10 : 8) : (p.flash ? 11 : 10));
    uint8_t halo = p.dim ? 6 : 7;
    float   L = path_len(U, 8);
    if (open > 1.5f) {                                 // open mouth: dim slatted glow between the lips
        float f[32];
        memcpy(f, U, sizeof U);
        for (int k = 0; k < 8; k++) { f[16 + 2 * k] = Lo[2 * (7 - k)]; f[17 + 2 * k] = Lo[2 * (7 - k) + 1]; }
        poly(f, 16, 6);
        if (!p.dim) poly(f, 16, 8, 4, 2, (int)lroundf(my));
    }
    path(U, 8, 9, halo);
    if (open > 1.5f) path(Lo, 8, 9, halo);
    if (p.fill < 1) {
        path(U, 8, 5, 6);
        part(U, 8, 0, L * p.fill, 5, core);
    } else {
        path(U, 8, 5, core);
        if (open > 1.5f) path(Lo, 8, 5, core);
        if (!p.dim && p.strobe < 0 && p.v > 0.5f) {
            float s = (float)fmod(D, 200.0);
            part(U, 8, fmaxf(0, s - 12), fminf(L, s), 5, 11);
        }
    }
    mouth_combs(p, U[0], U[1], U[14], U[15]);
}

void render(const FaceParams& p, float t, double D) {
    clear(0);
    poly(s_plates[0], 5, 2);
    poly(s_plates[1], 5, 2);
    poly(PLATE_TOP, 6, 2);
    poly(VISOR, 6, p.dim ? 6 : 8);
    uint8_t tc = p.flash ? 10 : (p.dim ? 6 : 7);
    for (const Trace& tr : s_tr) {
        if (p.hasGrow) {                               // boot: traces power on from the screen edge
            path(tr.p, tr.n, 3, 6);
            part(tr.p, tr.n, 0, tr.len * p.grow, 3, 7);
            if (p.grow < 1) pulse(tr, tr.len * p.grow, 3);
        } else {
            path(tr.p, tr.n, 3, tc);
        }
    }
    for (const auto& cb : s_combs) seg(cb[0], cb[1], cb[2], cb[3], 2, p.dim ? 7 : 9);
    if (p.v > 0.5f)
        for (const Trace& tr : s_tr) pulse(tr, (float)fmod(D + tr.phase, (double)tr.len + 140.0), 3);
    if (p.surge >= 0)
        for (const Trace& tr : s_tr) pulse(tr, p.surge * (tr.len + 40), 3);
    for (int idx = 0; idx < 2; idx++) {
        int   side = idx ? 1 : -1;
        float bk = p.blink * p.eyeBlink[idx], hh = p.hh * bk;
        Eye e;
        e.cx = (side < 0 ? 100 : 220) + p.lookX + p.shake;
        e.cy = 106 + p.lookY + p.bounce;
        e.hw = p.hw; e.hh = hh; e.ch = fminf(p.ch, fminf(hh, p.hw)); e.inner = -side;
        e.lidTop = (p.lidTop + p.eyeLid[idx]) * bk; e.lidBot = p.lidBot * bk;
        e.rake = p.rake; e.T = p.happyT;
        draw_eye(e, p, t);
    }
    draw_mouth(p, t, D);
}

}  // namespace

bool glacier_face_begin(TFT_eSPI* tft) {
    if (!s_spr) s_spr = new TFT_eSprite(tft);
    s_spr->setColorDepth(4);
    if (!s_spr->createSprite(FW, FH)) return false;
    s_buf = (uint8_t*)s_spr->getPointer();
    uint16_t pal[16];
    for (int i = 0; i < 16; i++) pal[i] = rgb565(FIXED_RGB[i][0], FIXED_RGB[i][1], FIXED_RGB[i][2]);
    s_spr->createPalette(pal);
    build_scene();
    build_shapes();
    s_state = s_prev = GF_COUNT;
    s_last_ms = 0;
    s_gest = s_glast = G_NONE;
    return true;
}

void glacier_face_end() {
    if (s_spr) s_spr->deleteSprite();
    s_buf = nullptr;
}

void glacier_face_update(GlacierState st, uint32_t now_ms) {
    if (!s_buf) return;
    if (st >= GF_COUNT) st = GF_STANDBY;
    if (st != s_state) {
        s_prev = s_state; s_state = st; s_state_ms = now_ms;
        s_gest = G_NONE;
        s_gnext_ms = now_ms + GSETS[st].min_ms / 2;
    }
    float t  = now_ms * 0.001f;
    float ts = (now_ms - s_state_ms) * 0.001f;
    float dt = s_last_ms ? fminf((now_ms - s_last_ms) * 0.001f, 0.1f) : 0;
    s_last_ms = now_ms;
    s_D += (double)s_v * dt;

    tick_gesture(now_ms, ts);
    FaceParams p = state_params(t, ts, now_ms);
    s_v = p.v;

    // Glow ramp: the state's base colour, cross-faded during the morph.
    const uint8_t* c1 = BASE_RGB[s_state];
    float r = c1[0], g = c1[1], b = c1[2];
    if (s_prev < GF_COUNT && ts < MORPH_S) {
        const uint8_t* c0 = BASE_RGB[s_prev];
        float w = sstep(0, MORPH_S, ts);
        r = c0[0] + (r - c0[0]) * w; g = c0[1] + (g - c0[1]) * w; b = c0[2] + (b - c0[2]) * w;
    }
    set_ramp(r, g, b);

    render(p, t, s_D);

#if GLACIER_FACE_FPS
    static uint32_t win_ms = 0, frames = 0, fps = 0;
    frames++;
    if (now_ms - win_ms >= 1000) { fps = frames; frames = 0; win_ms = now_ms; }
    s_spr->setTextColor(15);
    s_spr->setTextSize(2);
    s_spr->setCursor(4, 4);
    s_spr->print(fps);
#endif

    s_spr->pushSprite(0, 0);
}
