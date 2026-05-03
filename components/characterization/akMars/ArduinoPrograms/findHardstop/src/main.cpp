// Hardstop finder — dual motor AK45-10
//
// WAIT state: repeatedly sends enter_mit + hold-zero so the motor is ready
//             the instant it finishes booting (watch motor LED).
// SEEK state: ramps position target at SEEK_SPEED rad/s toward FINAL_TARGET.
//
// Send 'S' over serial to transition WAIT → SEEK.
// LED matrix: blank dot while waiting, 'S' glyph while seeking.

#include <Arduino.h>
#include <Arduino_CAN.h>
#include <Arduino_LED_Matrix.h>

static ArduinoLEDMatrix matrix;

// ** KNOBS **
static const float    SEEK_SPEED   = 1.0f;    // rad/s — lower = slower
static const float    FINAL_TARGET = 12.5f;   // rad
static const float    KP           = 0.5f;
static const float    KD           = 0.05f;
static const float    HOLD_KP      = 1.0f;
static const float    HOLD_KD      = 0.05f;
static const float    CUR_THRESHOLD = 0.5f;   // A — stall current
static const float    POS_DEADBAND  = 0.02f;  // rad — max movement per tick to count as stalled
static const uint8_t  CONFIRM_COUNT = 15;     // consecutive ticks before declaring hardstop

static const float P_MIN  = -12.5f, P_MAX  =  12.5f;
static const float V_MIN  = -65.0f, V_MAX  =  65.0f;
static const float KP_MIN =   0.0f, KP_MAX = 500.0f;
static const float KD_MIN =   0.0f, KD_MAX =   5.0f;
static const float T_MIN  = -18.0f, T_MAX  =  18.0f;

static const uint8_t M1_ID        = 64;
static const uint8_t M2_ID        = 1;
static const uint8_t ENTER_CMD[8] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFC};
static const uint8_t EXIT_CMD[8]  = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFD};
static const uint8_t ZERO_CMD[8]  = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE};
static const uint32_t INTER_FRAME_US = 200;

// --- LED glyphs ---
static const uint8_t GLYPH_S[7] = {
    0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110,
};
static const uint8_t GLYPH_F[7] = {
    0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
};

static void show_glyph(const uint8_t g[7]) {
    byte frame[8][12];
    memset(frame, 0, sizeof(frame));
    for (int r = 0; r < 7; r++)
        for (int c = 0; c < 5; c++)
            if (g[r] & (1 << (4 - c))) frame[r][c + 3] = 1;
    matrix.renderBitmap(frame, 8, 12);
}

static void show_dot() {
    byte frame[8][12];
    memset(frame, 0, sizeof(frame));
    frame[3][5] = 1;
    matrix.renderBitmap(frame, 8, 12);
}

// --- State ---
static bool  seeking    = false;
static float rampTarget = 0.0f;

static float   m1PrevPos = 0, m2PrevPos = 0;
static uint8_t m1Stall   = 0, m2Stall   = 0;
static bool    m1Done    = false, m2Done = false, bothDone = false;

static uint32_t lastPollUs  = 0;
static bool     pollM1      = true;
static uint32_t lastPrintMs = 0;
static uint32_t lastEnterMs = 0;

static float m1Pos = 0, m1Vel = 0, m1Cur = 0;
static float m2Pos = 0, m2Vel = 0, m2Cur = 0;

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
    buf[0] = p >> 8;  buf[1] = p & 0xFF;
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

static void exit_mit(uint8_t id) {
    CanMsg msg(CanStandardId(id), 8, (uint8_t*)EXIT_CMD);
    CAN.write(msg);
}

static void set_zero(uint8_t id) {
    CanMsg msg(CanStandardId(id), 8, (uint8_t*)ZERO_CMD);
    CAN.write(msg);
}

static void send_cmd(uint8_t id, float pos) {
    uint8_t buf[8];
    pack_mit(buf, pos, 0.0f, KP, KD, 0.0f);
    CanMsg cmd(CanStandardId(id), 8, buf);
    CAN.write(cmd);
}

static void send_hold(uint8_t id) {
    uint8_t buf[8];
    pack_mit(buf, 0.0f, 0.0f, HOLD_KP, HOLD_KD, 0.0f);
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
    if      (id == M1_ID) { m1Pos = pos; m1Vel = vel; m1Cur = cur; }
    else if (id == M2_ID) { m2Pos = pos; m2Vel = vel; m2Cur = cur; }
}

static void do_enter_and_cmd(float pos) {
    enter_mit(M1_ID); delayMicroseconds(INTER_FRAME_US);
    enter_mit(M2_ID); delay(5);
    send_cmd(M1_ID, pos); delayMicroseconds(INTER_FRAME_US);
    send_cmd(M2_ID, pos);
}

void setup() {
    Serial.begin(115200);
    matrix.begin();
    delay(100);
    CAN.begin(CanBitRate::BR_1000k);
    delay(50);

    do_enter_and_cmd(0.0f);
    show_dot();

    Serial.println("=== Hardstop Finder ===");
    Serial.print("seek_speed="); Serial.print(SEEK_SPEED, 2);
    Serial.print(" rad/s  kp="); Serial.print(KP, 2);
    Serial.print(" kd="); Serial.println(KD, 3);
    Serial.println("Send 'S' to start seek.");
    Serial.println("Columns: state  ramp  m1pos  m1vel  m1cur  m2pos  m2vel  m2cur");

    lastPollUs  = micros();
    lastPrintMs = millis();
    lastEnterMs = millis();
}

void loop() {
    uint32_t nowUs = micros();
    uint32_t nowMs = millis();

    while (CAN.available()) decode_reply(CAN.read());

    // Serial command
    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'S' || c == 's') {
            seeking    = true;
            rampTarget = 0.0f;
            m1PrevPos  = m1Pos;  m2PrevPos = m2Pos;
            m1Stall    = 0;      m2Stall   = 0;
            m1Done     = false;  m2Done    = false;  bothDone = false;
            show_glyph(GLYPH_S);
            Serial.println("--- SEEK started ---");
        }
    }

    // Periodic enter_mit + command every 4 s
    if (!seeking && nowMs - lastEnterMs >= 1000) {
        lastEnterMs = nowMs;
        do_enter_and_cmd(0.0f);
    }

    // 1 kHz poll — alternate motors
    if (nowUs - lastPollUs >= 1000) {
        lastPollUs = nowUs;

        if (seeking && rampTarget < FINAL_TARGET)
            rampTarget = min(rampTarget + SEEK_SPEED * 0.001f, FINAL_TARGET);

        // Hardstop detection (only while seeking, per motor)
        if (seeking) {
            if (!m1Done) {
                bool stalled = fabsf(m1Pos - m1PrevPos) < POS_DEADBAND &&
                               fabsf(m1Cur) > CUR_THRESHOLD;
                m1Stall = stalled ? m1Stall + 1 : 0;
                if (m1Stall >= CONFIRM_COUNT) {
                    m1Done = true;
                    exit_mit(M1_ID);  delay(50);
                    set_zero(M1_ID);  delay(50);
                    enter_mit(M1_ID); delayMicroseconds(INTER_FRAME_US);
                    send_hold(M1_ID);
                    Serial.print("*** M1 hardstop — zero set at "); Serial.print(m1Pos, 3); Serial.println(" rad ***");
                }
            }
            m1PrevPos = m1Pos;

            if (!m2Done) {
                bool stalled = fabsf(m2Pos - m2PrevPos) < POS_DEADBAND &&
                               fabsf(m2Cur) > CUR_THRESHOLD;
                m2Stall = stalled ? m2Stall + 1 : 0;
                if (m2Stall >= CONFIRM_COUNT) {
                    m2Done = true;
                    exit_mit(M2_ID);  delay(50);
                    set_zero(M2_ID);  delay(50);
                    enter_mit(M2_ID); delayMicroseconds(INTER_FRAME_US);
                    send_hold(M2_ID);
                    Serial.print("*** M2 hardstop — zero set at "); Serial.print(m2Pos, 3); Serial.println(" rad ***");
                }
            }
            m2PrevPos = m2Pos;

            if (m1Done && m2Done && !bothDone) {
                bothDone = true;
                show_glyph(GLYPH_F);
                Serial.println("=== Both motors zeroed ===");
            }
        }

        float cmdPos = seeking ? rampTarget : 0.0f;
        if (pollM1) { if (m1Done) send_hold(M1_ID); else send_cmd(M1_ID, cmdPos); }
        else        { if (m2Done) send_hold(M2_ID); else send_cmd(M2_ID, cmdPos); }
        pollM1 = !pollM1;
    }

    if (nowMs - lastPrintMs >= 100) {
        lastPrintMs = nowMs;
        Serial.print(seeking ? "SEEK" : "WAIT");
        Serial.print("  ramp="); Serial.print(rampTarget, 2);
        Serial.print("  m1: pos="); Serial.print(m1Pos, 3);
        Serial.print(" vel=");      Serial.print(m1Vel, 2);
        Serial.print(" cur=");      Serial.print(m1Cur, 2);
        Serial.print("  m2: pos="); Serial.print(m2Pos, 3);
        Serial.print(" vel=");      Serial.print(m2Vel, 2);
        Serial.print(" cur=");      Serial.println(m2Cur, 2);
    }
}
