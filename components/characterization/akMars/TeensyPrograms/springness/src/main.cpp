// AK45-10 springiness explorer — Teensy 4.1 port
// Original: components/characterization/akMars/ArduinoPrograms/springness
//
// CAN1: TX=pin22, RX=pin23, 1 Mbps via SN65HVD230 transceiver
// Motor IDs: L=0x11, R=0x12 (MIT Cheetah protocol)
//
// Cycles through two soft presets every PHASE_MS.
// Serial prints state at 10 Hz. Send any key to advance preset immediately.

#include <Arduino.h>
#include <FlexCAN_T4.h>

// ── Knobs ─────────────────────────────────────────────────────────────────────
static constexpr uint32_t PHASE_MS   = 4000;   // ms per preset
static constexpr float    TARGET_POS = 0.0f;   // hold position [rad]
// ─────────────────────────────────────────────────────────────────────────────

// AK45-10 MIT mode limits
static constexpr float P_MIN  = -12.5f, P_MAX  =  12.5f;
static constexpr float V_MIN  = -65.0f, V_MAX  =  65.0f;
static constexpr float KP_MIN =   0.0f, KP_MAX = 500.0f;
static constexpr float KD_MIN =   0.0f, KD_MAX =   5.0f;
static constexpr float T_MIN  = -18.0f, T_MAX  =  18.0f;
static constexpr float I_MIN  = -20.0f, I_MAX  =  20.0f;

static constexpr uint8_t  ID_L           = 11;
static constexpr uint8_t  ID_R           = 12;
static constexpr uint32_t INTER_FRAME_US = 500;

static const uint8_t ENTER_MIT[8] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFC};

// Soft presets only (scenarios 1 and 2 from original)
struct Preset { const char* name; float kp; float kd; };
static constexpr Preset PRESETS[] = {
    { "Floppy", 0.5f, 0.05f },
    { "Soft",   5.0f, 0.30f },
};
static constexpr uint8_t NUM_PRESETS = 2;

static FlexCAN_T4<CAN1, RX_SIZE_64, TX_SIZE_16> can1;

static uint8_t  preset_idx  = 0;
static uint32_t last_phase  = 0;
static uint32_t last_send   = 0;
static uint32_t last_print  = 0;

// Motor state (written from ISR — read only in loop after brief settle)
static volatile float l_pos = 0, l_vel = 0, l_cur = 0;
static volatile float r_pos = 0, r_vel = 0, r_cur = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

static uint16_t to_uint(float x, float xmin, float xmax, int bits) {
    uint32_t maxv = (1u << bits) - 1;
    if (x < xmin) x = xmin;
    if (x > xmax) x = xmax;
    return (uint16_t)((x - xmin) / (xmax - xmin) * maxv);
}

static float from_uint(uint16_t x, float xmin, float xmax, int bits) {
    return (float)x / (float)((1u << bits) - 1) * (xmax - xmin) + xmin;
}

static void send_raw(uint8_t id, const uint8_t data[8]) {
    CAN_message_t msg = {};
    msg.id  = id;
    msg.len = 8;
    memcpy(msg.buf, data, 8);
    can1.write(msg);
}

static void send_spring(uint8_t id, float kp, float kd) {
    uint16_t p   = to_uint(TARGET_POS, P_MIN, P_MAX, 16);
    uint16_t v   = to_uint(0.0f,       V_MIN, V_MAX, 12);
    uint16_t kp_ = to_uint(kp,         KP_MIN, KP_MAX, 12);
    uint16_t kd_ = to_uint(kd,         KD_MIN, KD_MAX, 12);
    uint16_t t   = to_uint(0.0f,       T_MIN,  T_MAX,  12);

    uint8_t buf[8];
    buf[0] = p >> 8;
    buf[1] = p & 0xFF;
    buf[2] = v >> 4;
    buf[3] = ((v & 0xF) << 4) | (kp_ >> 8);
    buf[4] = kp_ & 0xFF;
    buf[5] = kd_ >> 4;
    buf[6] = ((kd_ & 0xF) << 4) | (t >> 8);
    buf[7] = t & 0xFF;
    send_raw(id, buf);
}

static void enter_mit_both() {
    send_raw(ID_L, ENTER_MIT);
    delayMicroseconds(INTER_FRAME_US);
    send_raw(ID_R, ENTER_MIT);
}

static void rx_callback(const CAN_message_t& msg) {
    if (msg.len < 6) return;
    uint16_t raw_pos = ((uint16_t)msg.buf[1] << 8) | msg.buf[2];
    uint16_t raw_vel = ((uint16_t)msg.buf[3] << 4) | (msg.buf[4] >> 4);
    uint16_t raw_cur = ((uint16_t)(msg.buf[4] & 0xF) << 8) | msg.buf[5];
    float pos = from_uint(raw_pos, P_MIN, P_MAX, 16);
    float vel = from_uint(raw_vel, V_MIN, V_MAX, 12);
    float cur = from_uint(raw_cur, I_MIN, I_MAX, 12);
    if      (msg.id == ID_L) { l_pos = pos; l_vel = vel; l_cur = cur; }
    else if (msg.id == ID_R) { r_pos = pos; r_vel = vel; r_cur = cur; }
}

static void print_banner(uint8_t idx) {
    Serial.printf("\n>>> Preset %u/%u  %s  kp=%.1f  kd=%.2f\n",
                  idx + 1, NUM_PRESETS,
                  PRESETS[idx].name, PRESETS[idx].kp, PRESETS[idx].kd);
}

// ── Entry ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    delay(200);

    can1.begin();
    can1.setBaudRate(1000000);
    can1.setMaxMB(16);
    can1.enableFIFO();
    can1.enableFIFOInterrupt();
    can1.onReceive(rx_callback);

    // Proven startup: enter MIT, settle 10 ms per motor, then first command
    enter_mit_both();
    delay(20);
    send_spring(ID_L, PRESETS[0].kp, PRESETS[0].kd);
    delayMicroseconds(INTER_FRAME_US);
    send_spring(ID_R, PRESETS[0].kp, PRESETS[0].kd);
    delay(5);

    Serial.println("=== AK45 Springiness Explorer — Teensy 4.1 ===");
    Serial.printf("Presets: %u  |  Hold time: %lu ms  |  Send any key to advance\n",
                  NUM_PRESETS, PHASE_MS);
    Serial.println("Both motors hold 0 rad. Disturb shaft to feel each preset.");
    Serial.println();
    print_banner(0);

    last_phase = last_send = last_print = millis();
}

void loop() {
    uint32_t now = millis();

    // Advance preset on keypress
    if (Serial.available()) {
        while (Serial.available()) Serial.read();
        preset_idx = (preset_idx + 1) % NUM_PRESETS;
        enter_mit_both();
        delay(20);
        last_phase = now;
        print_banner(preset_idx);
    }

    // Auto-advance preset on timer
    if (now - last_phase >= PHASE_MS) {
        last_phase = now;
        preset_idx = (preset_idx + 1) % NUM_PRESETS;
        enter_mit_both();
        delay(20);
        print_banner(preset_idx);
    }

    // Send spring command at 100 Hz
    if (now - last_send >= 10) {
        last_send = now;
        send_spring(ID_L, PRESETS[preset_idx].kp, PRESETS[preset_idx].kd);
        delayMicroseconds(INTER_FRAME_US);
        send_spring(ID_R, PRESETS[preset_idx].kp, PRESETS[preset_idx].kd);
    }

    // Print at 10 Hz
    if (now - last_print >= 100) {
        last_print = now;
        Serial.printf("%-6s kp=%4.1f kd=%4.2f  |  L: pos=%+7.3f vel=%+6.2f cur=%+5.2f  |  R: pos=%+7.3f vel=%+6.2f cur=%+5.2f\n",
                      PRESETS[preset_idx].name,
                      PRESETS[preset_idx].kp, PRESETS[preset_idx].kd,
                      (float)l_pos, (float)l_vel, (float)l_cur,
                      (float)r_pos, (float)r_vel, (float)r_cur);
    }
}
