// CubeMars AK45-10 — MIT mode control, dual motor
// Hardware: Arduino UNO R4 WiFi
// CAN wiring: CANTX → D4, CANRX → D5, common GND with motor driver
// Motor 1: CAN ID 1  — alternates ±TARGET_AMP every 2 s
// Motor 2: CAN ID 64 — alternates ∓TARGET_AMP (opposite phase to M1)

#include <Arduino.h>
#include <Arduino_CAN.h>

// AK45-10 MIT mode parameter ranges (from CubeMars datasheet)
static const float P_MIN  = -12.5f, P_MAX  =  12.5f;  // rad
static const float V_MIN  = -65.0f, V_MAX  =  65.0f;  // rad/s
static const float KP_MIN =   0.0f, KP_MAX = 500.0f;  // N·m/rad
static const float KD_MIN =   0.0f, KD_MAX =   5.0f;  // N·m·s/rad
static const float T_MIN  = -18.0f, T_MAX  =  18.0f;  // N·m

static uint16_t float_to_uint(float x, float x_min, float x_max, int bits) {
    uint32_t max_val = (1u << bits) - 1;
    float clamped = x < x_min ? x_min : (x > x_max ? x_max : x);
    return (uint16_t)((clamped - x_min) / (x_max - x_min) * max_val);
}

void pack_mit_frame(uint8_t buf[8], float pos, float vel, float kp, float kd, float torque) {
    uint16_t p   = float_to_uint(pos,    P_MIN,  P_MAX,  16);
    uint16_t v   = float_to_uint(vel,    V_MIN,  V_MAX,  12);
    uint16_t kp_ = float_to_uint(kp,    KP_MIN, KP_MAX, 12);
    uint16_t kd_ = float_to_uint(kd,    KD_MIN, KD_MAX, 12);
    uint16_t t   = float_to_uint(torque, T_MIN,  T_MAX,  12);

    buf[0] = p >> 8;
    buf[1] = p & 0xFF;
    buf[2] = v >> 4;
    buf[3] = ((v & 0xF) << 4) | (kp_ >> 8);
    buf[4] = kp_ & 0xFF;
    buf[5] = kd_ >> 4;
    buf[6] = ((kd_ & 0xF) << 4) | (t >> 8);
    buf[7] = t & 0xFF;
}

static const uint8_t ENTER_CMD[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC};

// --- Tuning ---
static const float KP         =  0.5f;  // N·m/rad   spring stiffness (both motors)
static const float KD         =  0.1f;  // N·m·s/rad damping (both motors)
static const float TARGET_AMP =  2.0f;  // rad       both motors alternate ±this
// --------------

static const uint8_t M1_ID = 64;
static const uint8_t M2_ID = 1;

// Each motor has its own target; only one flips per 2 s turn
static float    m1Tgt       =  TARGET_AMP;
static float    m2Tgt       = -TARGET_AMP;
static bool     m1Turn      = true;   // whose turn to move next

static uint32_t lastPollUs  = 0;    // alternates M1/M2 every 1 ms → 500 Hz each
static bool     pollM1      = true;
static uint32_t lastPhaseMs = 0;
static uint32_t lastPrintMs = 0;

static float m1Pos = 0, m1Vel = 0, m1Cur = 0;
static float m2Pos = 0, m2Vel = 0, m2Cur = 0;

void enter_mit(uint8_t id) {
    CanMsg msg(CanStandardId(id), 8, (uint8_t*)ENTER_CMD);
    CAN.write(msg);
}

void send_spring(uint8_t id, float target) {
    uint8_t buf[8];
    pack_mit_frame(buf, target, 0.0f, KP, KD, 0.0f);
    CanMsg cmd(CanStandardId(id), 8, buf);
    CAN.write(cmd);
}

void decode_reply(const CanMsg& msg) {
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

void setup() {
    Serial.begin(115200);
    delay(100);
    CAN.begin(CanBitRate::BR_1000k);
    delay(50);

    enter_mit(M1_ID); delay(10);
    enter_mit(M2_ID); delay(10);
    send_spring(M1_ID, m1Tgt); delay(5);
    send_spring(M2_ID, m2Tgt); delay(5);
    Serial.println("Dual MIT spring — motors move one at a time");

    lastPollUs  = micros();
    lastPhaseMs = millis();
    lastPrintMs = millis();
}

void loop() {
    uint32_t nowUs = micros();
    uint32_t nowMs = millis();

    // Every 2 s, flip only the motor whose turn it is; the other keeps holding
    if (nowMs - lastPhaseMs >= 2000) {
        lastPhaseMs = nowMs;
        if (m1Turn) {
            m1Tgt = -m1Tgt;
            enter_mit(M1_ID); delay(5);
            send_spring(M1_ID, m1Tgt);
            Serial.print("M1 moves -> "); Serial.print(m1Tgt, 1);
            Serial.print("  M2 holds "); Serial.println(m2Tgt, 1);
        } else {
            m2Tgt = -m2Tgt;
            enter_mit(M2_ID); delay(5);
            send_spring(M2_ID, m2Tgt);
            Serial.print("M1 holds "); Serial.print(m1Tgt, 1);
            Serial.print("  M2 moves -> "); Serial.println(m2Tgt, 1);
        }
        m1Turn = !m1Turn;
    }

    // Drain RX every tick — replies arrive asynchronously
    while (CAN.available()) decode_reply(CAN.read());

    // Alternate M1/M2 every 1 ms — each motor gets 500 Hz, never simultaneous
    if (nowUs - lastPollUs >= 1000) {
        lastPollUs = nowUs;
        if (pollM1) send_spring(M1_ID, m1Tgt);
        else        send_spring(M2_ID, m2Tgt);
        pollM1 = !pollM1;
    }

    // Print at 10 Hz
    if (nowMs - lastPrintMs >= 100) {
        lastPrintMs = nowMs;
        Serial.print("M1 tgt="); Serial.print(m1Tgt, 1);
        Serial.print(" pos=");   Serial.print(m1Pos, 3);
        Serial.print(" cur=");   Serial.print(m1Cur, 2);
        Serial.print("  |  M2 tgt="); Serial.print(m2Tgt, 1);
        Serial.print(" pos=");        Serial.print(m2Pos, 3);
        Serial.print(" cur=");        Serial.println(m2Cur, 2);
    }
}
