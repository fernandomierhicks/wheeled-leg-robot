// CubeMars AK45-10 — MIT mode springiness exploration, dual motor
// Hardware: Arduino UNO R4 WiFi
// CAN wiring: CANTX → D4, CANRX → D5, common GND with motor driver
//
// Both motors hold position 0. Every PHASE_MS the firmware steps to the
// next KP/KD preset so you can feel the difference by hand-disturbing the shaft.
// LED matrix always shows the preset number. Serial prints state at 10 Hz.

#include <Arduino.h>
#include <Arduino_CAN.h>
#include <Arduino_LED_Matrix.h>

static ArduinoLEDMatrix matrix;

// =====================================================================
//  ** KNOBS **
//
//  PHASE_MS   — how long each preset is held before stepping to the next (ms)
//  TARGET_POS — shaft position both motors try to hold (rad, 0 = power-on zero)
// =====================================================================
static const uint32_t PHASE_MS   = 4000;
static const float    TARGET_POS = 0.0f;
// =====================================================================

// AK45-10 MIT mode limits
static const float P_MIN  = -12.5f, P_MAX  =  12.5f;
static const float V_MIN  = -65.0f, V_MAX  =  65.0f;
static const float KP_MIN =   0.0f, KP_MAX = 500.0f;
static const float KD_MIN =   0.0f, KD_MAX =   5.0f;
static const float T_MIN  = -18.0f, T_MAX  =  18.0f;

static const uint8_t M1_ID    = 64;
static const uint8_t M2_ID    = 1;
static const uint8_t ENTER_CMD[8] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFC};

static const uint32_t INTER_FRAME_US = 200;

// --- Presets ---
// display_num is the number shown on the LED matrix (original preset numbering)
struct Preset { const char* name; float kp; float kd; uint8_t display_num; };
static const Preset PRESETS[] = {
    { "Floppy     ", 0.5f,  0.05f, 1 },
    { "Soft       ", 5.0f,  0.3f,  2 },
    { "Underdamped", 30.0f, 0.1f,  3 },
    { "Medium     ", 30.0f, 1.0f,  4 },
    { "Floppy-lite", 0.1f,  0.01f, 5 },  // softer than Floppy — barely any force
    { "Very Soft  ", 2.0f,  0.15f, 6 },  // softer than Soft
    { "Soft Underd", 10.0f, 0.05f, 7 },  // softer than Underdamped — gentler oscillation
};
static const uint8_t NUM_PRESETS = sizeof(PRESETS) / sizeof(PRESETS[0]);

static uint8_t  presetIdx   = 0;

// Timing
static uint32_t lastPollUs  = 0;
static bool     pollM1      = true;
static uint32_t lastPhaseMs = 0;
static uint32_t lastPrintMs = 0;

// Motor state
static float m1Pos = 0, m1Vel = 0, m1Cur = 0;
static float m2Pos = 0, m2Vel = 0, m2Cur = 0;

// --- LED matrix digit renderer ---
// 5-wide × 7-tall pixel font. Each byte is one row; bit4 = leftmost pixel.
// Digits available: 1 2 3 4 7
static const uint8_t DIGIT_KEYS[NUM_PRESETS]       = {1,  2,  3,  4,  5,  6,  7};
static const uint8_t DIGIT_FONT[NUM_PRESETS][7]    = {
    { 0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 },  // 1
    { 0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111 },  // 2
    { 0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110 },  // 3
    { 0b10001, 0b10001, 0b11111, 0b00001, 0b00001, 0b00001, 0b00001 },  // 4
    { 0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110 },  // 5
    { 0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110 },  // 6
    { 0b11111, 0b00001, 0b00010, 0b00100, 0b00100, 0b00100, 0b00100 },  // 7
};

// Renders digit d centred on the 12×8 matrix and leaves it displayed.
static void show_digit(uint8_t d) {
    uint8_t fi = 0;
    for (uint8_t i = 0; i < NUM_PRESETS; i++) if (DIGIT_KEYS[i] == d) { fi = i; break; }

    byte frame[8][12];
    memset(frame, 0, sizeof(frame));
    // 5-wide glyph centred in 12 cols → left edge at col 3; top-aligned row 0
    for (int row = 0; row < 7; row++) {
        uint8_t pat = DIGIT_FONT[fi][row];
        for (int col = 0; col < 5; col++)
            if (pat & (1 << (4 - col))) frame[row][col + 3] = 1;
    }
    matrix.renderBitmap(frame, 8, 12);
}

// --- CAN helpers ---
static uint16_t float_to_uint(float x, float xmin, float xmax, int bits) {
    uint32_t maxv = (1u << bits) - 1;
    float c = x < xmin ? xmin : (x > xmax ? xmax : x);
    return (uint16_t)((c - xmin) / (xmax - xmin) * maxv);
}

static void pack_mit(uint8_t buf[8], float pos, float vel, float kp, float kd, float torq) {
    uint16_t p   = float_to_uint(pos,  P_MIN,  P_MAX,  16);
    uint16_t v   = float_to_uint(vel,  V_MIN,  V_MAX,  12);
    uint16_t kp_ = float_to_uint(kp,  KP_MIN, KP_MAX,  12);
    uint16_t kd_ = float_to_uint(kd,  KD_MIN, KD_MAX,  12);
    uint16_t t   = float_to_uint(torq, T_MIN,  T_MAX,  12);
    buf[0] = p >> 8;
    buf[1] = p & 0xFF;
    buf[2] = v >> 4;
    buf[3] = ((v & 0xF) << 4) | (kp_ >> 8);
    buf[4] = kp_ & 0xFF;
    buf[5] = kd_ >> 4;
    buf[6] = ((kd_ & 0xF) << 4) | (t >> 8);
    buf[7] = t & 0xFF;
}

static void enter_mit(uint8_t id) {
    CanMsg msg(CanStandardId(id), 8, (uint8_t*)ENTER_CMD);
    CAN.write(msg);
}

static void send_spring(uint8_t id, float kp, float kd) {
    uint8_t buf[8];
    pack_mit(buf, TARGET_POS, 0.0f, kp, kd, 0.0f);
    CanMsg cmd(CanStandardId(id), 8, buf);
    CAN.write(cmd);
}

static void decode_reply(const CanMsg& msg) {
    if (msg.data_length < 6) return;
    uint8_t  id      = msg.data[0];
    uint16_t raw_pos = ((uint16_t)msg.data[1] << 8) | msg.data[2];
    uint16_t raw_vel = ((uint16_t)msg.data[3] << 4) | (msg.data[4] >> 4);
    uint16_t raw_cur = ((uint16_t)(msg.data[4] & 0xF) << 8) | msg.data[5];
    float pos = raw_pos / 65535.0f * (P_MAX - P_MIN) + P_MIN;
    float vel = raw_vel /  4095.0f * (V_MAX - V_MIN) + V_MIN;
    float cur = raw_cur /  4095.0f * 40.0f - 20.0f;
    if (id == M1_ID) { m1Pos = pos; m1Vel = vel; m1Cur = cur; }
    else              { m2Pos = pos; m2Vel = vel; m2Cur = cur; }
}

static void print_preset_banner(uint8_t idx) {
    Serial.print("\n>>> Preset ");
    Serial.print(PRESETS[idx].display_num);
    Serial.print("  ");
    Serial.print(PRESETS[idx].name);
    Serial.print("  KP=");
    Serial.print(PRESETS[idx].kp, 1);
    Serial.print("  KD=");
    Serial.println(PRESETS[idx].kd, 2);
}

void setup() {
    Serial.begin(115200);
    matrix.begin();
    delay(100);
    CAN.begin(CanBitRate::BR_1000k);
    delay(50);

    enter_mit(M1_ID); delay(10);
    delayMicroseconds(INTER_FRAME_US);
    enter_mit(M2_ID); delay(10);
    send_spring(M1_ID, PRESETS[0].kp, PRESETS[0].kd); delay(5);
    send_spring(M2_ID, PRESETS[0].kp, PRESETS[0].kd); delay(5);

    show_digit(PRESETS[0].display_num);

    Serial.println("=== AK45 Springiness Explorer ===");
    Serial.print("Hold time per preset: "); Serial.print(PHASE_MS); Serial.println(" ms");
    Serial.println("Both motors hold 0 rad. Disturb shaft to feel each preset.");
    Serial.println("Columns: preset  KP  KD  m1pos  m1vel  m1cur  m2pos  m2vel  m2cur");
    print_preset_banner(0);

    lastPollUs  = micros();
    lastPhaseMs = millis();
    lastPrintMs = millis();
}

void loop() {
    uint32_t nowUs = micros();
    uint32_t nowMs = millis();

    // Drain RX
    while (CAN.available()) decode_reply(CAN.read());

    // Step to next preset
    if (nowMs - lastPhaseMs >= PHASE_MS) {
        lastPhaseMs = nowMs;
        presetIdx = (presetIdx + 1) % NUM_PRESETS;
        enter_mit(M1_ID);
        delayMicroseconds(INTER_FRAME_US);
        enter_mit(M2_ID);
        delay(5);
        send_spring(M1_ID, PRESETS[presetIdx].kp, PRESETS[presetIdx].kd);
        delayMicroseconds(INTER_FRAME_US);
        send_spring(M2_ID, PRESETS[presetIdx].kp, PRESETS[presetIdx].kd);
        show_digit(PRESETS[presetIdx].display_num);
        print_preset_banner(presetIdx);
    }

    // Alternate M1/M2 at 500 Hz each (every 1 ms)
    if (nowUs - lastPollUs >= 1000) {
        lastPollUs = nowUs;
        if (pollM1) send_spring(M1_ID, PRESETS[presetIdx].kp, PRESETS[presetIdx].kd);
        else        send_spring(M2_ID, PRESETS[presetIdx].kp, PRESETS[presetIdx].kd);
        pollM1 = !pollM1;
    }

    // Print at 10 Hz
    if (nowMs - lastPrintMs >= 100) {
        lastPrintMs = nowMs;
        Serial.print(PRESETS[presetIdx].name);
        Serial.print("  kp="); Serial.print(PRESETS[presetIdx].kp, 1);
        Serial.print(" kd=");  Serial.print(PRESETS[presetIdx].kd, 2);
        Serial.print("  m1: pos="); Serial.print(m1Pos, 3);
        Serial.print(" vel=");      Serial.print(m1Vel, 2);
        Serial.print(" cur=");      Serial.print(m1Cur, 2);
        Serial.print("  m2: pos="); Serial.print(m2Pos, 3);
        Serial.print(" vel=");      Serial.print(m2Vel, 2);
        Serial.print(" cur=");      Serial.println(m2Cur, 2);
    }
}
