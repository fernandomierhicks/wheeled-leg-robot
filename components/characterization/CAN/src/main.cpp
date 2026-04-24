// ODrive CAN characterization sketch
//
// Measures how fast we can command an ODrive via CAN and how fast it sends
// encoder feedback back.  Both axes (node 0 = left, node 1 = right) are tested.
//
// Bus topology this sketch is designed for:
//   Node 0  — ODrive axis 0 (wheel L)   arb_ids 0x000–0x01F
//   Node 1  — ODrive axis 1 (wheel R)   arb_ids 0x020–0x03F
//   0x41    — AK45-10 hip L  (future)   MIT CAN, no collision with ODrive range
//   0x42    — AK45-10 hip R  (future)   MIT CAN, no collision with ODrive range
//   Unknown IDs are silently ignored so the AK45s can be on the bus while testing.
//
// Pre-requisites (configure ODrive via odrivetool before running):
//   odrv0.can.config.baud_rate               = 1000000
//   odrv0.axis0.config.can.node_id           = 0
//   odrv0.axis1.config.can.node_id           = 1
//   odrv0.axis0.config.can.encoder_rate_ms   = 2   <- try 1, 2, 5, 10
//   odrv0.axis1.config.can.encoder_rate_ms   = 2
//   odrv0.save_configuration()
//
// Serial commands (115200 baud, send any key then Enter or use monitor):
//   e   — enable velocity mode (CLOSED_LOOP, both axes)
//   d   — disable (IDLE, both axes)
//   +   — increase velocity setpoint by 0.5 turns/s
//   -   — decrease velocity setpoint by 0.5 turns/s
//   0   — zero velocity setpoint
//   s   — print stats snapshot now
//   c   — clear ODrive errors
//   h   — print this help

#include <Arduino.h>
#include <Arduino_CAN.h>

// ── ODrive CAN constants ──────────────────────────────────────────────────────
static constexpr uint8_t  CMD_HEARTBEAT        = 0x01;
static constexpr uint8_t  CMD_ESTOP            = 0x02;
static constexpr uint8_t  CMD_SET_AXIS_STATE   = 0x07;
static constexpr uint8_t  CMD_ENCODER_ESTIMATE = 0x09;
static constexpr uint8_t  CMD_SET_CTRL_MODES   = 0x0B;
static constexpr uint8_t  CMD_SET_INPUT_VEL    = 0x0D;
static constexpr uint8_t  CMD_CLEAR_ERRORS     = 0x12;

static constexpr uint32_t AXIS_STATE_IDLE        = 1;
static constexpr uint32_t AXIS_STATE_CLOSED_LOOP = 8;
static constexpr uint32_t CTRL_MODE_VELOCITY     = 2;
static constexpr uint32_t INPUT_MODE_PASSTHROUGH = 1;

static constexpr uint8_t  NODE_L = 0;
static constexpr uint8_t  NODE_R = 1;

// ── Config ────────────────────────────────────────────────────────────────────
// Command rate: how often we send velocity setpoints.
// 1000 Hz = every 1 ms.  Lower if you see TX drops in stats.
static constexpr uint32_t CMD_RATE_HZ    = 1000;
static constexpr uint32_t CMD_PERIOD_US  = 1000000UL / CMD_RATE_HZ;

// Stats print interval.
static constexpr uint32_t STATS_PERIOD_MS = 1000;

// ── State ─────────────────────────────────────────────────────────────────────
static float    g_vel_sp      = 0.0f;   // velocity setpoint (turns/s), both axes
static bool     g_enabled     = false;

// TX stats
static uint32_t g_tx_sent     = 0;
static uint32_t g_tx_failed   = 0;

// RX stats (per axis)
struct AxisStats {
    float    pos_turns = 0.0f;
    float    vel_turns = 0.0f;
    uint32_t rx_frames = 0;          // frames received this stats window
    uint32_t last_rx_us = 0;         // micros() at last frame
    uint32_t min_interval_us = UINT32_MAX;
    uint32_t max_interval_us = 0;
    uint64_t sum_interval_us = 0;
    uint32_t interval_count  = 0;
    uint32_t heartbeats      = 0;
    uint32_t axis_error      = 0;
    uint8_t  axis_state      = 0;
    // TX→RX pipeline latency: gap between last TX command and next encoder frame
    uint32_t last_tx_us      = 0;    // micros() when last TX was issued
    uint32_t min_pipe_us     = UINT32_MAX;
    uint32_t max_pipe_us     = 0;
    uint64_t sum_pipe_us     = 0;
    uint32_t pipe_count      = 0;
};
static AxisStats g_ax[2];   // index 0 = L, 1 = R

// Timing
static uint32_t g_last_cmd_us   = 0;
static uint32_t g_last_stats_ms = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────
static inline uint32_t arb_id(uint8_t node, uint8_t cmd) {
    return ((uint32_t)node << 5) | cmd;
}
static inline uint8_t node_from_id(uint32_t id) { return (uint8_t)((id >> 5) & 0x3F); }
static inline uint8_t cmd_from_id(uint32_t id)  { return (uint8_t)(id & 0x1F); }

static inline void pack_f32(uint8_t* buf, uint8_t off, float v) { memcpy(buf + off, &v, 4); }
static inline float unpack_f32(const uint8_t* buf, uint8_t off) {
    float v; memcpy(&v, buf + off, 4); return v;
}
static inline uint32_t unpack_u32(const uint8_t* buf, uint8_t off) {
    uint32_t v; memcpy(&v, buf + off, 4); return v;
}

static bool can_send(uint32_t id, const uint8_t* data, uint8_t len) {
    CanMsg msg(CanStandardId(id), len, data);
    return CAN.write(msg) > 0;
}

// ── ODrive commands ───────────────────────────────────────────────────────────
static void odrive_set_axis_state(uint8_t node, uint32_t state) {
    uint8_t d[4]; memcpy(d, &state, 4);
    can_send(arb_id(node, CMD_SET_AXIS_STATE), d, 4);
}

static void odrive_set_ctrl_mode(uint8_t node, uint32_t ctrl, uint32_t inp) {
    uint8_t d[8]; memcpy(d, &ctrl, 4); memcpy(d+4, &inp, 4);
    can_send(arb_id(node, CMD_SET_CTRL_MODES), d, 8);
}

static void odrive_clear_errors(uint8_t node) {
    uint8_t d[1] = {0};
    can_send(arb_id(node, CMD_CLEAR_ERRORS), d, 0);
}

static void odrive_send_vel(uint8_t node, float vel_turns_s) {
    uint8_t d[8];
    float ff = 0.0f;
    pack_f32(d, 0, vel_turns_s);
    pack_f32(d, 4, ff);
    uint32_t t = micros();
    bool ok = can_send(arb_id(node, CMD_SET_INPUT_VEL), d, 8);
    if (ok) {
        g_ax[node].last_tx_us = t;
        g_tx_sent++;
    } else {
        g_tx_failed++;
    }
}

// ── Enable / disable ──────────────────────────────────────────────────────────
static void enable_velocity_mode() {
    Serial.println("[CAN] Clearing errors...");
    odrive_clear_errors(NODE_L);
    odrive_clear_errors(NODE_R);
    delay(30);

    Serial.println("[CAN] Setting velocity mode...");
    odrive_set_ctrl_mode(NODE_L, CTRL_MODE_VELOCITY, INPUT_MODE_PASSTHROUGH);
    odrive_set_ctrl_mode(NODE_R, CTRL_MODE_VELOCITY, INPUT_MODE_PASSTHROUGH);
    delay(10);

    Serial.println("[CAN] Requesting CLOSED_LOOP...");
    odrive_set_axis_state(NODE_L, AXIS_STATE_CLOSED_LOOP);
    odrive_set_axis_state(NODE_R, AXIS_STATE_CLOSED_LOOP);
    g_enabled = true;
    Serial.println("[CAN] Enabled.");
}

static void disable() {
    odrive_set_axis_state(NODE_L, AXIS_STATE_IDLE);
    odrive_set_axis_state(NODE_R, AXIS_STATE_IDLE);
    g_enabled = false;
    Serial.println("[CAN] Disabled — IDLE.");
}

// ── Stats ─────────────────────────────────────────────────────────────────────
static void reset_stats() {
    for (int i = 0; i < 2; i++) {
        g_ax[i].rx_frames       = 0;
        g_ax[i].min_interval_us = UINT32_MAX;
        g_ax[i].max_interval_us = 0;
        g_ax[i].sum_interval_us = 0;
        g_ax[i].interval_count  = 0;
        g_ax[i].heartbeats      = 0;
        g_ax[i].min_pipe_us     = UINT32_MAX;
        g_ax[i].max_pipe_us     = 0;
        g_ax[i].sum_pipe_us     = 0;
        g_ax[i].pipe_count      = 0;
    }
    g_tx_sent   = 0;
    g_tx_failed = 0;
}

static void print_axis_stats(uint8_t node, const char* label) {
    AxisStats& a = g_ax[node];

    float avg_interval_us = (a.interval_count > 0)
        ? (float)(a.sum_interval_us / a.interval_count) : 0.0f;
    float rx_hz = (avg_interval_us > 0.0f) ? (1e6f / avg_interval_us) : 0.0f;

    float min_pipe = (a.min_pipe_us == UINT32_MAX) ? 0.0f : (float)a.min_pipe_us;
    float max_pipe = (float)a.max_pipe_us;
    float avg_pipe = (a.pipe_count > 0) ? (float)(a.sum_pipe_us / a.pipe_count) : 0.0f;

    Serial.print("  Axis "); Serial.print(label);
    Serial.print("  frames="); Serial.print(a.rx_frames);
    Serial.print("  rate=");   Serial.print(rx_hz, 1); Serial.print(" Hz");
    Serial.print("  jitter min/avg/max=");
    Serial.print(a.min_interval_us == UINT32_MAX ? 0 : (int)a.min_interval_us);
    Serial.print("/");
    Serial.print((int)avg_interval_us);
    Serial.print("/");
    Serial.print((int)a.max_interval_us);
    Serial.print(" us");
    Serial.print("  pipe(TX->RX) min/avg/max=");
    Serial.print((int)min_pipe); Serial.print("/");
    Serial.print((int)avg_pipe); Serial.print("/");
    Serial.print((int)max_pipe); Serial.print(" us");
    Serial.print("  hb="); Serial.print(a.heartbeats);
    if (a.axis_error) {
        Serial.print("  ERR=0x"); Serial.print(a.axis_error, HEX);
    }
    Serial.print("  state="); Serial.print(a.axis_state);
    Serial.print("  pos="); Serial.print(a.pos_turns, 3);
    Serial.print(" t  vel="); Serial.print(a.vel_turns, 3); Serial.print(" t/s");
    Serial.println();
}

static void print_stats() {
    Serial.println("──────────────────────────────────────────────────");
    Serial.print("[CAN stats]  TX sent="); Serial.print(g_tx_sent);
    Serial.print("  failed=");             Serial.print(g_tx_failed);
    Serial.print("  vel_sp=");             Serial.print(g_vel_sp, 2);
    Serial.print(" t/s  enabled=");        Serial.println(g_enabled ? "YES" : "NO");
    print_axis_stats(NODE_L, "L");
    print_axis_stats(NODE_R, "R");
    reset_stats();
}

// ── RX handler ────────────────────────────────────────────────────────────────
static void process_rx() {
    while (CAN.available()) {
        CanMsg msg = CAN.read();
        uint32_t id  = msg.getStandardId();
        uint8_t  node = node_from_id(id);
        uint8_t  cmd  = cmd_from_id(id);

        // Ignore AK45-10 hip motor IDs (0x41, 0x42) and anything > node 1 range
        if (node > 1) continue;

        uint32_t now_us = micros();
        AxisStats& a = g_ax[node];

        if (cmd == CMD_ENCODER_ESTIMATE && msg.data_length >= 8) {
            a.pos_turns = unpack_f32(msg.data, 0);
            a.vel_turns = unpack_f32(msg.data, 4);

            // Inter-frame interval
            if (a.last_rx_us != 0) {
                uint32_t dt = now_us - a.last_rx_us;
                if (dt < a.min_interval_us) a.min_interval_us = dt;
                if (dt > a.max_interval_us) a.max_interval_us = dt;
                a.sum_interval_us += dt;
                a.interval_count++;
            }
            a.last_rx_us = now_us;
            a.rx_frames++;

            // TX → RX pipeline gap
            if (a.last_tx_us != 0 && now_us > a.last_tx_us) {
                uint32_t pipe = now_us - a.last_tx_us;
                // Only count if plausible (< 50 ms; avoids wrap-around noise)
                if (pipe < 50000UL) {
                    if (pipe < a.min_pipe_us) a.min_pipe_us = pipe;
                    if (pipe > a.max_pipe_us) a.max_pipe_us = pipe;
                    a.sum_pipe_us += pipe;
                    a.pipe_count++;
                }
                a.last_tx_us = 0;  // consume — wait for next TX
            }

        } else if (cmd == CMD_HEARTBEAT && msg.data_length >= 5) {
            a.axis_error = unpack_u32(msg.data, 0);
            a.axis_state = msg.data[4];
            a.heartbeats++;
        }
    }
}

// ── Serial command handler ────────────────────────────────────────────────────
static void print_help() {
    Serial.println("Commands:");
    Serial.println("  e  — enable velocity mode");
    Serial.println("  d  — disable (IDLE)");
    Serial.println("  +  — vel +0.5 turns/s");
    Serial.println("  -  — vel -0.5 turns/s");
    Serial.println("  0  — vel = 0");
    Serial.println("  s  — print stats now");
    Serial.println("  c  — clear ODrive errors");
    Serial.println("  h  — help");
}

static void handle_serial() {
    while (Serial.available()) {
        char ch = (char)Serial.read();
        switch (ch) {
            case 'e': enable_velocity_mode(); break;
            case 'd': disable(); break;
            case '+': g_vel_sp += 0.5f;
                      Serial.print("[CAN] vel_sp="); Serial.println(g_vel_sp);
                      break;
            case '-': g_vel_sp -= 0.5f;
                      Serial.print("[CAN] vel_sp="); Serial.println(g_vel_sp);
                      break;
            case '0': g_vel_sp = 0.0f;
                      Serial.println("[CAN] vel_sp=0");
                      break;
            case 's': print_stats(); break;
            case 'c':
                odrive_clear_errors(NODE_L);
                odrive_clear_errors(NODE_R);
                Serial.println("[CAN] clear_errors sent");
                break;
            case 'h': print_help(); break;
            default:  break;
        }
    }
}

// ── Arduino entry points ──────────────────────────────────────────────────────
void setup() {
    Serial.begin(1000000);
    delay(500);
    Serial.println("\n=== ODrive CAN Characterization ===");
    Serial.print("CMD_RATE_HZ="); Serial.println(CMD_RATE_HZ);

    if (!CAN.begin(CanBitRate::BR_1000k)) {
        Serial.println("[CAN] INIT FAILED — check wiring");
        while (true) {}
    }
    Serial.println("[CAN] bus 1 Mbps OK");
    print_help();

    reset_stats();
    g_last_cmd_us   = micros();
    g_last_stats_ms = millis();
}

void loop() {
    uint32_t now_us = micros();
    uint32_t now_ms = millis();

    // ── TX: send velocity commands at CMD_RATE_HZ ─────────────────────────────
    if (g_enabled && (now_us - g_last_cmd_us) >= CMD_PERIOD_US) {
        g_last_cmd_us += CMD_PERIOD_US;
        odrive_send_vel(NODE_L,  g_vel_sp);
        odrive_send_vel(NODE_R, -g_vel_sp);  // R motor is mirrored
    }

    // ── RX: drain CAN buffer every loop iteration ──────────────────────────────
    process_rx();

    // ── Stats: print every STATS_PERIOD_MS ───────────────────────────────────
    if ((now_ms - g_last_stats_ms) >= STATS_PERIOD_MS) {
        g_last_stats_ms += STATS_PERIOD_MS;
        print_stats();
    }

    // ── Serial: check for user commands ──────────────────────────────────────
    handle_serial();
}
