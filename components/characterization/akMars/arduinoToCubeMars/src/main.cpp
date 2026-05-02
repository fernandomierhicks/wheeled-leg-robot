// CubeMars AK45-10 — MIT mode control
// Hardware: Arduino UNO R4 WiFi
// CAN wiring: CANTX → D4, CANRX → D5, common GND with motor driver

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

// --- Spring demo tuning ---
// Reduce KP for a softer spring (more compliant / easier to push against).
// Reduce KD for less damping (bouncier). Increase for critically damped.
// TARGET_AMP sets how far apart the two alternating set-points are (rad).
static const float KP         =  0.5f;   // N·m/rad  — spring stiffness
static const float KD         =  0.1f;   // N·m·s/rad — damping
static const float TARGET_AMP =  2.0f;   // rad      — ±set-point
// -------------------------

static bool     posPhase    = true;   // true → +TARGET_AMP, false → −TARGET_AMP
static uint32_t lastPollUs  = 0;
static uint32_t lastPhaseMs = 0;      // 2 s set-point alternation
static uint32_t lastPrintMs = 0;
static float    lastPos = 0, lastVel = 0, lastCur = 0;

void send_spring() {
    float target = posPhase ? TARGET_AMP : -TARGET_AMP;
    uint8_t buf[8];
    // vel=0: pure spring/damper, no velocity feedforward
    pack_mit_frame(buf, target, 0.0f, KP, KD, 0.0f);
    CanMsg cmd(CanStandardId(0x01), 8, buf);
    CAN.write(cmd);
}

void setup() {
    Serial.begin(115200);
    delay(100);
    CAN.begin(CanBitRate::BR_1000k);
    delay(50);

    CanMsg enter_msg(CanStandardId(0x01), 8, (uint8_t*)ENTER_CMD);
    CAN.write(enter_msg);
    delay(10);
    send_spring();
    Serial.println("MIT spring mode — kp=20 kd=0.5 target=+2 rad");

    lastPollUs  = micros();
    lastPhaseMs = millis();
    lastPrintMs = millis();
}

void loop() {
    uint32_t nowUs = micros();
    uint32_t nowMs = millis();

    // Alternate set-point every 2 seconds
    if (nowMs - lastPhaseMs >= 2000) {
        posPhase    = !posPhase;
        lastPhaseMs = nowMs;
        CanMsg enter_msg(CanStandardId(0x01), 8, (uint8_t*)ENTER_CMD);
        CAN.write(enter_msg);
        delay(10);
        send_spring();
        Serial.print("Set-point -> "); Serial.println(posPhase ? TARGET_AMP : -TARGET_AMP, 2);
    }

    // Poll telemetry at 500 Hz
    if (nowUs - lastPollUs >= 2000) {
        lastPollUs = nowUs;
        send_spring();
        if (CAN.available()) {
            CanMsg reply = CAN.read();
            if (reply.data_length >= 6) {
                uint16_t raw_pos = ((uint16_t)reply.data[1] << 8) | reply.data[2];
                uint16_t raw_vel = ((uint16_t)reply.data[3] << 4) | (reply.data[4] >> 4);
                uint16_t raw_cur = ((uint16_t)(reply.data[4] & 0xF) << 8) | reply.data[5];
                lastPos = raw_pos / 65535.0f * (P_MAX - P_MIN) + P_MIN;
                lastVel = raw_vel /  4095.0f * (V_MAX - V_MIN) + V_MIN;
                lastCur = raw_cur /  4095.0f * 40.0f - 20.0f;
            }
        }
    }

    // Print at 10 Hz
    if (nowMs - lastPrintMs >= 100) {
        lastPrintMs = nowMs;
        Serial.print("tgt="); Serial.print(posPhase ? TARGET_AMP : -TARGET_AMP, 2);
        Serial.print("  pos="); Serial.print(lastPos, 3);
        Serial.print("  vel="); Serial.print(lastVel, 3);
        Serial.print("  cur="); Serial.println(lastCur, 3);
    }
}
