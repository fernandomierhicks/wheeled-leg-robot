// CubeMars AK45-10 reference test sketch — UNO R4 WiFi (built-in CAN)
// Ported from CubeMarsTestArduinoCode.ino (MCP2515) to Arduino_CAN.
// Wiring: CANTX → D4, CANRX → D5, common GND.
// Motor CAN ID set below (CAN_ID = 104). Servo feedback reply ID: 0x2968.
// To select a control mode, change the literal in down()'s switch() statement:
//   0=duty  1=current  2=current_brake  3=rpm  4=pos  5=origin  6=pos+spd

#include <Arduino.h>
#include <Arduino_CAN.h>

// ── CAN transmit ──────────────────────────────────────────────────────────────
void comm_can_transmit_sid(uint32_t id, const uint8_t *data, uint8_t len) {
    CanMsg msg(CanStandardId(id), len, data);
    CAN.write(msg);
}

void comm_can_transmit_eid(uint32_t id, const uint8_t *data, uint8_t len) {
    CanMsg msg(CanExtendedId(id), len, data);
    CAN.write(msg);
}

// ── Buffer helpers (big-endian) ───────────────────────────────────────────────
void buffer_append_int16(uint8_t *buf, int16_t n, int32_t *i)  { buf[(*i)++] = n >> 8; buf[(*i)++] = n; }
void buffer_append_uint16(uint8_t *buf, uint16_t n, int32_t *i) { buf[(*i)++] = n >> 8; buf[(*i)++] = n; }
void buffer_append_int32(uint8_t *buf, int32_t n, int32_t *i) {
    buf[(*i)++] = n >> 24; buf[(*i)++] = n >> 16; buf[(*i)++] = n >> 8; buf[(*i)++] = n;
}
void buffer_append_uint32(uint8_t *buf, uint32_t n, int32_t *i) {
    buf[(*i)++] = n >> 24; buf[(*i)++] = n >> 16; buf[(*i)++] = n >> 8; buf[(*i)++] = n;
}

// ── Packet ID enum ────────────────────────────────────────────────────────────
enum {
    CAN_PACKET_SET_DUTY = 0,
    CAN_PACKET_SET_CURRENT,
    CAN_PACKET_SET_CURRENT_BRAKE,
    CAN_PACKET_SET_RPM,
    CAN_PACKET_SET_POS,
    CAN_PACKET_SET_ORIGIN_HERE,
    CAN_PACKET_SET_POS_SPD
} CAN_PACKET_ID;

static const uint8_t CAN_ID = 104;

// ── Servo-mode commands ───────────────────────────────────────────────────────
void comm_can_set_duty(uint8_t id, float duty) {
    int32_t idx = 0; uint8_t buf[4];
    buffer_append_int32(buf, (int32_t)(duty * 100000.0f), &idx);
    comm_can_transmit_eid(id | ((uint32_t)CAN_PACKET_SET_DUTY << 8), buf, idx);
}
void comm_can_set_current(uint8_t id, float current) {
    int32_t idx = 0; uint8_t buf[4];
    buffer_append_int32(buf, (int32_t)(current * 1000.0f), &idx);
    comm_can_transmit_eid(id | ((uint32_t)CAN_PACKET_SET_CURRENT << 8), buf, idx);
}
void comm_can_set_cb(uint8_t id, float current) {
    int32_t idx = 0; uint8_t buf[4];
    buffer_append_int32(buf, (int32_t)(current * 1000.0f), &idx);
    comm_can_transmit_eid(id | ((uint32_t)CAN_PACKET_SET_CURRENT_BRAKE << 8), buf, idx);
}
void comm_can_set_rpm(uint8_t id, float rpm) {
    int32_t idx = 0; uint8_t buf[4];
    buffer_append_int32(buf, (int32_t)rpm, &idx);
    comm_can_transmit_eid(id | ((uint32_t)CAN_PACKET_SET_RPM << 8), buf, idx);
}
void comm_can_set_pos(uint8_t id, float pos) {
    int32_t idx = 0; uint8_t buf[4];
    buffer_append_int32(buf, (int32_t)(pos * 10000.0f), &idx);
    comm_can_transmit_eid(id | ((uint32_t)CAN_PACKET_SET_POS << 8), buf, idx);
}
void comm_can_set_origin(uint8_t id, uint8_t mode) {
    comm_can_transmit_eid(id | ((uint32_t)CAN_PACKET_SET_ORIGIN_HERE << 8), &mode, 0);
}
void comm_can_set_pos_spd(uint8_t id, float pos, int16_t spd, int16_t rpa) {
    int32_t idx = 0, idx1 = 4; uint8_t buf[8];
    buffer_append_int32(buf, (int32_t)(pos * 10000.0f), &idx);
    buffer_append_int16(buf, (int16_t)(spd / 10.0f), &idx1);
    buffer_append_int16(buf, (int16_t)(rpa / 10.0f), &idx1);
    comm_can_transmit_eid(id | ((uint32_t)CAN_PACKET_SET_POS_SPD << 8), buf, idx1);
}

// ── MIT-mode commands ─────────────────────────────────────────────────────────
void set_mit_mode_run()  { uint8_t d[8]={0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFC}; comm_can_transmit_sid(0x00,d,8); }
void set_mit_mode_idle() { uint8_t d[8]={0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFD}; comm_can_transmit_sid(0x00,d,8); }
void set_mit_current_position_zero_positong() { uint8_t d[8]={0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE}; comm_can_transmit_sid(0x00,d,8); }

// ── MIT float ↔ uint ──────────────────────────────────────────────────────────
#define P_MIN -12.5f
#define P_MAX  12.5f
#define V_MIN -45.0f
#define V_MAX  45.0f
#define T_MIN -18.0f
#define T_MAX  18.0f
#define KP_MIN  0.0f
#define KP_MAX 500.0f
#define KD_MIN  0.0f
#define KD_MAX   5.0f

unsigned int float_to_uint(float x, float x_min, float x_max, unsigned int bits) {
    if (x < x_min) x = x_min; else if (x > x_max) x = x_max;
    return (unsigned int)((x - x_min) * ((float)(((long)1 << bits) - 1) / (x_max - x_min)));
}
float uint_to_float(uint16_t x_int, float x_min, float x_max, int bits) {
    return ((float)x_int) * (x_max - x_min) / ((float)(((long)1 << bits) - 1)) + x_min;
}

void pack_cmd(uint8_t id, float p_des, float v_des, float kp, float kd, float t_ff) {
    int p_int  = float_to_uint(p_des, P_MIN,  P_MAX,  16);
    int v_int  = float_to_uint(v_des, V_MIN,  V_MAX,  12);
    int kp_int = float_to_uint(kp,    KP_MIN, KP_MAX, 12);
    int kd_int = float_to_uint(kd,    KD_MIN, KD_MAX, 12);
    int t_int  = float_to_uint(t_ff,  T_MIN,  T_MAX,  12);
    uint8_t buf[8] = {
        (uint8_t)(p_int >> 8),
        (uint8_t)(p_int & 0xFF),
        (uint8_t)(v_int >> 4),
        (uint8_t)((v_int & 0xF) << 4 | (kp_int >> 8)),
        (uint8_t)(kp_int & 0xFF),
        (uint8_t)(kd_int >> 4),
        (uint8_t)((kd_int & 0xF) << 4 | (t_int >> 8)),
        (uint8_t)(t_int & 0xFF)
    };
    comm_can_transmit_sid(id, buf, 8);
}

// ── Feedback parsers ──────────────────────────────────────────────────────────
void motor_receive_servo(const CanMsg *m) {
    int16_t pos_int = (int16_t)((m->data[0] << 8) | m->data[1]);
    int16_t spd_int = (int16_t)((m->data[2] << 8) | m->data[3]);
    int16_t cur_int = (int16_t)((m->data[4] << 8) | m->data[5]);
    float pos = pos_int * 0.1f;
    float spd = spd_int * 10.0f;
    float cur = cur_int * 0.01f;
    int8_t temp  = (int8_t)m->data[6];
    int8_t error = (int8_t)m->data[7];
    Serial.print(pos);  Serial.print(" ");
    Serial.print(spd);  Serial.print(" ");
    Serial.print(cur);  Serial.print(" ");
    Serial.print(temp); Serial.print(" ");
    Serial.println(error);
}

void motor_receive_mit(const CanMsg *m) {
    uint16_t pos_int = (uint16_t)((m->data[1] << 8) | m->data[2]);
    uint16_t spd_int = (uint16_t)((m->data[3] << 4) | (m->data[4] >> 4));
    uint16_t t_int   = (uint16_t)(((m->data[4] & 0xF) << 8) | m->data[5]);
    float pos  = uint_to_float(pos_int, P_MIN, P_MAX, 16);
    float vel  = uint_to_float(spd_int, V_MIN, V_MAX, 12);
    float torq = uint_to_float(t_int,   T_MIN, T_MAX, 12);
    int8_t temp  = (int8_t)m->data[6];
    int8_t error = (int8_t)m->data[7];
    Serial.print(pos);   Serial.print(" ");
    Serial.print(vel);   Serial.print(" ");
    Serial.print(torq);  Serial.print(" ");
    Serial.print(temp);  Serial.print(" ");
    Serial.println(error);
}

// ── Command dispatch — change switch literal to select mode ───────────────────
void down() {
    switch (6) {
        case CAN_PACKET_SET_DUTY:          comm_can_set_duty(CAN_ID, 0.1f);               break;
        case CAN_PACKET_SET_CURRENT:       comm_can_set_current(CAN_ID, 1.0f);            break;
        case CAN_PACKET_SET_CURRENT_BRAKE: comm_can_set_cb(CAN_ID, 1.0f);                break;
        case CAN_PACKET_SET_RPM:           comm_can_set_rpm(CAN_ID, 5000.0f);             break;
        case CAN_PACKET_SET_POS:           comm_can_set_pos(CAN_ID, 180.0f);              break;
        case CAN_PACKET_SET_ORIGIN_HERE:   comm_can_set_origin(CAN_ID, 0);               break;
        case CAN_PACKET_SET_POS_SPD:       comm_can_set_pos_spd(CAN_ID, 180, 2000, 2000); break;
    }
}

// ── Setup / loop ──────────────────────────────────────────────────────────────
static uint8_t cntr = 0;

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.begin(115200);
    while (!Serial);

    if (CAN.begin(CanBitRate::BR_1000k)) {
        Serial.println("CAN init ok!");
    } else {
        Serial.println("CAN init fault!");
        while (1);
    }
}

void loop() {
    down();

    if (CAN.available()) {
        CanMsg msg = CAN.read();
        // bit 31 flags extended frame (same convention as MCP2515 CAN_EFF_FLAG)
        uint32_t raw_id = msg.id & 0x1FFFFFFFUL;

        if (msg.id & 0x80000000UL) Serial.print("eid ");
        else                        Serial.print("sid ");

        Serial.print(raw_id, HEX);
        Serial.print(" len ");
        Serial.print(msg.data_length, HEX);
        Serial.print(" data ");
        for (int i = 0; i < msg.data_length; i++) {
            Serial.print(msg.data[i], HEX);
            Serial.print(" ");
        }
        Serial.println();

        if (raw_id == 0x2968) motor_receive_servo(&msg);
        // motor_receive_mit(&msg);

        cntr++;
    }
}
