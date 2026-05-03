// CubeMars AK45-10 — CAN timing stress test, dual motor
// Hardware: Arduino UNO R4 WiFi
// CAN wiring: CANTX → D4, CANRX → D5, common GND with motor driver
//
// Default behaviour mirrors arduinoToCubeMars_HelloWorld exactly.
// Loosen the knobs one at a time to find limits.

#include <Arduino.h>
#include <Arduino_CAN.h>
#include <math.h>

// =====================================================================
//  ** KNOBS — start here, matches HelloWorld defaults **
//
//  TARGET_HZ          — per-motor command rate (Hz)
//                       HelloWorld = 500.  Each tick only commands one motor
//                       (alternating), so total CAN writes = 2 × TARGET_HZ.
//
//  BOTH_MOTORS_PER_TICK — false = alternate M1/M2 (HelloWorld behaviour)
//                          true  = send both motors every tick (stress mode)
//
//  INTER_FRAME_DELAY_US — pause between M1 and M2 writes when both sent per tick.
//                         Only used when BOTH_MOTORS_PER_TICK = true.
//
//  REENTER_MIT_MS     — resend the enter-MIT command this often (ms).
//                       HelloWorld does this every 2000 ms before each flip.
//                       Set to 0 to disable (skip re-entry entirely).
//
//  REENTER_DELAY_MS   — blocking delay after enter-MIT, before the motion
//                       command.  HelloWorld uses 5 ms.
// =====================================================================
static const uint32_t TARGET_HZ             = 1500;
static const bool     BOTH_MOTORS_PER_TICK  = true;
static const uint32_t INTER_FRAME_DELAY_US  = 150;
static const uint32_t REENTER_MIT_MS        = 1000;
static const uint32_t REENTER_DELAY_MS      = 5;
// =====================================================================

// Sine sweep
static const float SWEEP_AMP_RAD = 1.0f;   // peak position (rad)
static const float SWEEP_FREQ_HZ = 0.5f;   // sweep frequency (Hz)

// Spring gains
static const float KP = 0.5f;
static const float KD = 0.1f;

// AK45-10 MIT mode limits
static const float P_MIN  = -12.5f, P_MAX  =  12.5f;
static const float V_MIN  = -65.0f, V_MAX  =  65.0f;
static const float KP_MIN =   0.0f, KP_MAX = 500.0f;
static const float KD_MIN =   0.0f, KD_MAX =   5.0f;
static const float T_MIN  = -18.0f, T_MAX  =  18.0f;

static const uint8_t M1_ID    = 64;
static const uint8_t M2_ID    = 1;
static const uint8_t ENTER_CMD[8] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFC};

// Derived timing
static const uint32_t LOOP_PERIOD_US = 1000000UL / TARGET_HZ;

// State
static uint32_t startUs     = 0;  // used for float-safe elapsed time
static uint32_t lastLoopUs  = 0;
static uint32_t prevLoopUs  = 0;
static uint32_t lastReenterMs = 0;
static uint32_t lastStatsMs = 0;
static bool     pollM1      = true;  // alternating flag

static float m1Pos = 0, m1Vel = 0, m1Cur = 0;
static float m2Pos = 0, m2Vel = 0, m2Cur = 0;

// --- Timing stats ---
struct TimingStat {
    uint32_t minUs, maxUs;
    uint64_t sumUs;
    uint32_t count;
    void reset() { minUs = UINT32_MAX; maxUs = 0; sumUs = 0; count = 0; }
    void record(uint32_t us) {
        if (us < minUs) minUs = us;
        if (us > maxUs) maxUs = us;
        sumUs += us; count++;
    }
    uint32_t avg() { return count ? (uint32_t)(sumUs / count) : 0; }
};
static TimingStat statSend;   // CAN write(s) per tick
static TimingStat statRx;     // RX drain per tick
static TimingStat statPeriod; // actual tick-to-tick interval

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

static uint32_t send_mit(uint8_t id, float pos) {
    uint8_t buf[8];
    pack_mit(buf, pos, 0.0f, KP, KD, 0.0f);
    CanMsg cmd(CanStandardId(id), 8, buf);
    uint32_t t0 = micros();
    CAN.write(cmd);
    return micros() - t0;
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

void setup() {
    Serial.begin(115200);
    delay(100);
    CAN.begin(CanBitRate::BR_1000k);
    delay(50);

    // Mirror HelloWorld setup exactly: enter MIT, delay, send zero
    enter_mit(M1_ID); delay(10);
    enter_mit(M2_ID); delay(10);
    send_mit(M1_ID, 0.0f); delay(5);
    send_mit(M2_ID, 0.0f); delay(5);

    Serial.println("=== AK45 CAN timing stress test ===");
    Serial.print("TARGET_HZ="); Serial.print(TARGET_HZ);
    Serial.print(" LOOP_PERIOD_US="); Serial.print(LOOP_PERIOD_US);
    Serial.print(" BOTH_PER_TICK="); Serial.print(BOTH_MOTORS_PER_TICK);
    Serial.print(" INTER_FRAME_US="); Serial.print(INTER_FRAME_DELAY_US);
    Serial.print(" REENTER_MS="); Serial.print(REENTER_MIT_MS);
    Serial.print(" REENTER_DELAY_MS="); Serial.println(REENTER_DELAY_MS);
    Serial.println("send[min/avg/max us]  rx[min/avg/max us]  period[min/avg/max us]  ticks/s  m1pos  m2pos");

    statSend.reset(); statRx.reset(); statPeriod.reset();
    startUs      = micros();
    lastLoopUs   = startUs;
    prevLoopUs   = startUs;
    lastReenterMs = millis();
    lastStatsMs  = millis();
}

void loop() {
    uint32_t nowUs = micros();
    uint32_t nowMs = millis();

    // Periodic re-enter MIT (mirrors HelloWorld's enter+delay before each flip)
    if (REENTER_MIT_MS > 0 && nowMs - lastReenterMs >= REENTER_MIT_MS) {
        lastReenterMs = nowMs;
        enter_mit(M1_ID);
        delayMicroseconds(INTER_FRAME_DELAY_US);
        enter_mit(M2_ID);
        if (REENTER_DELAY_MS > 0) delay(REENTER_DELAY_MS);
    }

    // Drain RX
    {
        uint32_t r0 = micros();
        while (CAN.available()) decode_reply(CAN.read());
        statRx.record(micros() - r0);
    }

    // Command tick
    if (nowUs - lastLoopUs >= LOOP_PERIOD_US) {
        statPeriod.record(nowUs - prevLoopUs);
        prevLoopUs = nowUs;
        lastLoopUs = nowUs;

        // Float-safe elapsed seconds (relative to startUs, resets every ~71 min)
        float elapsedS = (nowUs - startUs) * 1e-6f;
        float pos = SWEEP_AMP_RAD * sinf(2.0f * (float)M_PI * SWEEP_FREQ_HZ * elapsedS);

        uint32_t t0 = micros();
        if (BOTH_MOTORS_PER_TICK) {
            send_mit(M1_ID,  pos);
            if (INTER_FRAME_DELAY_US > 0) delayMicroseconds(INTER_FRAME_DELAY_US);
            send_mit(M2_ID, -pos);
        } else {
            // Alternate M1/M2 — exactly like HelloWorld
            if (pollM1) send_mit(M1_ID,  pos);
            else        send_mit(M2_ID, -pos);
            pollM1 = !pollM1;
        }
        statSend.record(micros() - t0);
    }

    // Print stats at 1 Hz
    if (nowMs - lastStatsMs >= 1000) {
        lastStatsMs = nowMs;
        Serial.print("send[");
        Serial.print(statSend.minUs); Serial.print("/");
        Serial.print(statSend.avg()); Serial.print("/");
        Serial.print(statSend.maxUs);
        Serial.print(" us]  rx[");
        Serial.print(statRx.minUs == UINT32_MAX ? 0 : statRx.minUs); Serial.print("/");
        Serial.print(statRx.avg()); Serial.print("/");
        Serial.print(statRx.maxUs);
        Serial.print(" us]  period[");
        Serial.print(statPeriod.minUs); Serial.print("/");
        Serial.print(statPeriod.avg()); Serial.print("/");
        Serial.print(statPeriod.maxUs);
        Serial.print(" us]  ticks="); Serial.print(statSend.count);
        Serial.print("  m1="); Serial.print(m1Pos, 2);
        Serial.print("  m2="); Serial.println(m2Pos, 2);
        statSend.reset(); statRx.reset(); statPeriod.reset();
    }
}
