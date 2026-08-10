#include "wheel_motors.h"
#include "config.h"
#include "param_registry.h"
#include "wheel_safety.h"
#include "comm_protocol.h"  // comm_log — driver messages must be visible over WiFi too (audit W7)
#include <Arduino.h>
#include <FlexCAN_T4.h>

// Max consecutive implausible encoder samples the filter may suppress before
// it gives up and accepts reality. 3 ticks bounds the blind window while
// still swallowing the isolated single-sample spikes seen on the bench.
static constexpr uint8_t WM_VEL_MAX_CONSEC_REJECT = 3;

// ODrive CAN command IDs (5-bit, ORed into bits [4:0] of the 11-bit frame ID)
#define CMD_HEARTBEAT       0x001
#define CMD_ENCODER_EST     0x009
#define CMD_SET_AXIS_STATE  0x007
#define CMD_SET_CTRL_MODE   0x00B
#define CMD_SET_INPUT_POS   0x00C
#define CMD_SET_INPUT_VEL   0x00D
#define CMD_SET_INPUT_TRQ   0x00E
#define CMD_CLEAR_ERRORS    0x018
#define CMD_GET_VBUS        0x017

// ODrive axis state values
#define AXIS_IDLE           1u
#define AXIS_CLOSED_LOOP    8u

// ODrive control mode values
#define CTRL_TORQUE         1u
#define CTRL_VELOCITY       2u
#define CTRL_POSITION       3u

// ODrive input mode
#define INPUT_PASSTHROUGH   1u

WheelAxisState wm_L    = {};
WheelAxisState wm_R    = {};
WheelMode      wm_mode = WheelMode::IDLE;

// CAN3 on Teensy 4.1 uses pins 30 (RX) and 31 (TX) — matches config.h PIN_CAN3_*.
static FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16> can3;
static bool s_can_initialized = false;

// ── CAN TX health ────────────────────────────────────────────────────────────
// FlexCAN_T4::write() returns 1 when the frame went straight into a hardware
// mailbox and -1 when every mailbox was busy and it was pushed onto the bounded
// software TX queue instead (TX_SIZE_16 here). A single -1 is therefore NOT a
// lost frame and must not fault on its own — the queue drains normally. What is
// dangerous is a sustained run of them: the queue is then the only thing moving
// frames, and struct2queueTx() drops silently once it fills, so torque setpoints
// and mode changes disappear with nothing anywhere to say so. Until this change
// the return value was discarded entirely and that failure was invisible.
//
// Threshold is a consecutive-deferral run, not a rate: ~200 deferrals is ~100
// control ticks (wheel_motors_send() emits 2 frames/tick in TORQUE), i.e. about
// 200 ms of a bus that has stopped draining.
static constexpr uint32_t WM_TX_DEFER_FAULT_RUN = 200;
static uint32_t s_tx_defer_total = 0;  // lifetime count, diagnostic
static uint32_t s_tx_defer_run   = 0;  // current consecutive run
static bool     s_tx_stalled     = false;

// ── helpers ──────────────────────────────────────────────────────────────────

static void note_tx(int rc) {
    if (rc == 1) { s_tx_defer_run = 0; return; }
    s_tx_defer_total++;
    if (s_tx_defer_run < UINT32_MAX) s_tx_defer_run++;
}

static void send_frame(uint8_t node_id, uint8_t cmd_id, const void* data, uint8_t len) {
    CAN_message_t msg = {};
    msg.id  = ((uint32_t)node_id << 5) | cmd_id;
    msg.len = len;
    memcpy(msg.buf, data, len);
    note_tx(can3.write(msg));
}

uint32_t wheel_motors_tx_defer_count() { return s_tx_defer_total; }

// ── CAN RX callback (called from FlexCAN_T4 ISR) ─────────────────────────────

static void rx_callback(const CAN_message_t& msg) {
    uint8_t node_id = (msg.id >> 5) & 0x3F;
    uint8_t cmd_id  = msg.id & 0x1F;

    WheelAxisState* ax;
    if      (node_id == ODESC_NODE_L) ax = &wm_L;
    else if (node_id == ODESC_NODE_R) ax = &wm_R;
    else return;

    if (cmd_id == CMD_ENCODER_EST && msg.len >= 8) {
        float pos, vel;
        memcpy(&pos, msg.buf + 0, 4);
        memcpy(&vel, msg.buf + 4, 4);
        // Right motor is physically mirrored — negate so positive = robot forward.
        if (node_id == ODESC_NODE_R) { pos = -pos; vel = -vel; }
        ax->pos_turns       = pos;
        // Raw only — wheel_motors_poll() runs the plausibility filter and
        // publishes vel_turns_s. Doing it here would mean calling param_get()
        // from an ISR and would leave the filter running at CAN frame rate
        // rather than at a known control-tick cadence.
        ax->vel_raw_turns_s = vel;
        ax->fb_seq++;
        ax->last_fb_ms      = millis();
        ax->ever_heard      = true;
    } else if (cmd_id == CMD_HEARTBEAT && msg.len >= 5) {
        uint32_t err;
        memcpy(&err, msg.buf + 0, 4);
        ax->error         = err;
        ax->axis_state    = msg.buf[4];
        ax->last_hb_ms    = millis();
        ax->hb_ever_heard = true;
    } else if (cmd_id == CMD_GET_VBUS && msg.len >= 4) {
        memcpy(&ax->vbus, msg.buf + 0, 4);
    }
}

// ── public API ────────────────────────────────────────────────────────────────

bool wheel_motors_init() {
    can3.begin();
    can3.setBaudRate(CAN_BAUD);
    can3.setMaxMB(16);
    can3.enableFIFO();
    can3.enableFIFOInterrupt();
    can3.onReceive(rx_callback);
    s_can_initialized = true;
    comm_log(LOG_LEVEL_INFO, "WheelMotors: CAN3 init OK %lu kbps node_L=%u node_R=%u",
             (unsigned long)(CAN_BAUD / 1000), (unsigned)ODESC_NODE_L, (unsigned)ODESC_NODE_R);
    return true;
}

// ── ODrive axis-state confirmation ───────────────────────────────────────────
// Encoder freshness alone says the ODrive is powered and talking; it says
// nothing about whether the axis is still in the mode we commanded. An axis
// that has dropped out of CLOSED_LOOP_CONTROL keeps streaming perfectly good
// encoder estimates while ignoring every torque command, which used to read as
// a healthy wheel right up until the robot fell over.
//
// Two deliberate escape hatches, both to avoid turning a diagnostic into a
// brick:
//   * Only enforced once a heartbeat has actually been seen (hb_ever_heard).
//     Heartbeat transmission is an ODrive-side config item; if it is off, this
//     degrades to exactly the old behaviour plus a one-time warning, rather
//     than failing every axis permanently.
//   * Suspended for WM_MODE_SETTLE_MS after a mode change. The ODrive needs a
//     few heartbeat periods to actually reach CLOSED_LOOP, and faulting during
//     that window would bounce us straight back to IDLE and oscillate.
static constexpr uint32_t WM_HB_TIMEOUT_MS    = 500;  // ODrive default heartbeat is 100 ms
static constexpr uint32_t WM_MODE_SETTLE_MS   = 500;
static uint32_t s_mode_change_ms = 0;

static bool axis_confirmed(const WheelAxisState& ax, uint32_t now) {
    if (!ax.hb_ever_heard) return true;                       // heartbeats not configured
    if ((now - ax.last_hb_ms) >= WM_HB_TIMEOUT_MS) return false;
    if (wm_mode == WheelMode::IDLE) return true;              // nothing to confirm
    if ((now - s_mode_change_ms) < WM_MODE_SETTLE_MS) return true;
    return ax.axis_state == AXIS_CLOSED_LOOP;
}

void wheel_motors_poll() {
    // FlexCAN_T4 interrupt-driven — no polling call needed.
    uint32_t now = millis();

    // One-time notice if the ODrives never sent a heartbeat: the axis-state and
    // ODrive-error checks are both inert in that case, and that is worth
    // knowing before trusting either of them on the bench.
    static bool s_hb_warned = false;
    if (!s_hb_warned && now > 5000 && !wm_L.hb_ever_heard && !wm_R.hb_ever_heard) {
        s_hb_warned = true;
        comm_log(LOG_LEVEL_WARN,
                 "WheelMotors: no ODrive heartbeat seen — axis-state and error "
                 "checks are inactive (enable heartbeat in ODrive CAN config)");
    }

    // ── Encoder-velocity plausibility filter ─────────────────────────────────
    // Only evaluated when a new encoder frame actually arrived, and the
    // allowance scales with the time since the last accepted sample, so a gap
    // in CAN frames widens the window instead of rejecting the sample that
    // ends the gap. See wheel_safety.h for why this exists.
    {
        float max_accel = param_get(PARAM_WM_VEL_SLEW_MAX);
        WheelAxisState* axes[2] = { &wm_L, &wm_R };
        for (WheelAxisState* ax : axes) {
            if (ax->fb_seq == ax->fb_seq_seen) continue;  // nothing new this tick
            ax->fb_seq_seen = ax->fb_seq;
            float dt_s = (ax->vel_accept_ms == 0) ? 0.0f
                                                  : (now - ax->vel_accept_ms) / 1000.0f;
            uint8_t run_before = ax->vel_reject_run;
            ax->vel_turns_s = wheel_vel_glitch_filter(
                ax->vel_raw_turns_s, ax->vel_turns_s, max_accel, dt_s,
                WM_VEL_MAX_CONSEC_REJECT, &ax->vel_reject_run);
            if (ax->vel_reject_run > run_before) {
                ax->vel_glitch_count++;   // held last-good: this sample was rejected
            } else {
                ax->vel_accept_ms = now;  // accepted (or fail-open) — restart the clock
            }
        }
    }

    // ── CAN TX stall ─────────────────────────────────────────────────────────
    // Latches once the deferral run crosses the threshold; cleared only by
    // wheel_motors_clear_errors(), like a latched ODrive fault. Reported once
    // per latch rather than per tick — a stalled bus would otherwise emit a
    // log line at 500 Hz and starve the loop it is trying to warn about.
    if (!s_tx_stalled && s_tx_defer_run >= WM_TX_DEFER_FAULT_RUN) {
        s_tx_stalled = true;
        comm_log(LOG_LEVEL_ERROR,
                 "WheelMotors: CAN3 TX stalled — %lu consecutive deferrals (%lu total); "
                 "torque commands are no longer reaching the bus",
                 (unsigned long)s_tx_defer_run, (unsigned long)s_tx_defer_total);
    }

    uint32_t enc_timeout = (uint32_t)param_get(PARAM_WM_ENC_TIMEOUT_MS);
    // ever_heard guard: without it, ok is spuriously true for the first
    // enc_timeout ms after boot (last_fb_ms == 0) even with no ODrive present.
    // A latched TX stall clears ok on both axes: feedback can still be arriving
    // perfectly while nothing we send gets out, and an axis we cannot command
    // is not an axis we can balance on.
    wm_L.ok = wm_L.ever_heard && !s_tx_stalled && (now - wm_L.last_fb_ms) < enc_timeout
              && axis_confirmed(wm_L, now);
    wm_R.ok = wm_R.ever_heard && !s_tx_stalled && (now - wm_R.last_fb_ms) < enc_timeout
              && axis_confirmed(wm_R, now);

    bool l_bad = (param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f) && (!wm_L.ok || wm_L.error);
    bool r_bad = (param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f) && (!wm_R.ok || wm_R.error);
    bool fault = (l_bad || r_bad);
    if (fault && wm_mode != WheelMode::IDLE) {
        comm_log(LOG_LEVEL_ERROR, "WheelMotors FAULT -> IDLE  L(ok=%d err=0x%lX)  R(ok=%d err=0x%lX)",
                 (int)wm_L.ok, (unsigned long)wm_L.error, (int)wm_R.ok, (unsigned long)wm_R.error);
        wheel_motors_set_mode(WheelMode::IDLE);
    }
}

void wheel_motors_forgive_feedback_stall() {
    // Companion to hip_motors_forgive_feedback_stall(): after a deliberate,
    // known-blocking main-loop operation (SD-log open/finalize) froze the tick,
    // reset the encoder-freshness clock so the self-inflicted stall isn't read
    // as an encoder dropout on the next poll(). Only the freshness timeout is
    // forgiven — a real ODrive error flag (wm_*.error) still faults.
    uint32_t now = millis();
    wm_L.last_fb_ms = now;
    wm_R.last_fb_ms = now;
    // Heartbeat freshness feeds ok too (axis_confirmed), and a frozen tick
    // ages it exactly the same way, so forgive both clocks or the stall just
    // reappears as a heartbeat timeout instead of an encoder one.
    wm_L.last_hb_ms = now;
    wm_R.last_hb_ms = now;
}

void wheel_motors_set_mode(WheelMode mode) {
    bool l_en = param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f;
    bool r_en = param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f;
    if (mode == WheelMode::IDLE) {
        uint32_t s = AXIS_IDLE;
        if (l_en) send_frame(ODESC_NODE_L, CMD_SET_AXIS_STATE, &s, 4);
        delayMicroseconds(CAN_INTER_FRAME_US);
        if (r_en) send_frame(ODESC_NODE_R, CMD_SET_AXIS_STATE, &s, 4);
    } else {
        uint32_t ctrl;
        switch (mode) {
            case WheelMode::VELOCITY: ctrl = CTRL_VELOCITY; break;
            case WheelMode::POSITION: ctrl = CTRL_POSITION; break;
            case WheelMode::TORQUE:
            default:                  ctrl = CTRL_TORQUE;   break;
        }
        uint32_t ctrl_data[2] = { ctrl, INPUT_PASSTHROUGH };
        if (l_en) send_frame(ODESC_NODE_L, CMD_SET_CTRL_MODE, ctrl_data, 8);
        delayMicroseconds(CAN_INTER_FRAME_US);
        if (r_en) send_frame(ODESC_NODE_R, CMD_SET_CTRL_MODE, ctrl_data, 8);
        delayMicroseconds(CAN_INTER_FRAME_US);
        uint32_t s = AXIS_CLOSED_LOOP;
        if (l_en) send_frame(ODESC_NODE_L, CMD_SET_AXIS_STATE, &s, 4);
        delayMicroseconds(CAN_INTER_FRAME_US);
        if (r_en) send_frame(ODESC_NODE_R, CMD_SET_AXIS_STATE, &s, 4);
    }
    wm_mode = mode;
    s_mode_change_ms = millis();  // opens the axis-state settle window
}

void wheel_motor_disable_L() {
    if (s_can_initialized) {
        uint32_t idle = AXIS_IDLE;
        send_frame(ODESC_NODE_L, CMD_SET_AXIS_STATE, &idle, 4);
    }
    if (param_get(PARAM_WHEEL_R_ENABLE) < 0.5f) wm_mode = WheelMode::IDLE;
}

void wheel_motor_disable_R() {
    if (s_can_initialized) {
        uint32_t idle = AXIS_IDLE;
        send_frame(ODESC_NODE_R, CMD_SET_AXIS_STATE, &idle, 4);
    }
    if (param_get(PARAM_WHEEL_L_ENABLE) < 0.5f) wm_mode = WheelMode::IDLE;
}

void wheel_motors_send(float L, float R) {
    // Right motor is physically mirrored — apply sign convention once here.
    float L_hw =  L;
    float R_hw = -R;
    bool l_en = param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f;
    bool r_en = param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f;

    switch (wm_mode) {
        case WheelMode::IDLE:
            break;

        case WheelMode::VELOCITY: {
            // rad/s → turns/s
            float vel_L[2] = { L_hw / TWO_PI, 0.0f };
            float vel_R[2] = { R_hw / TWO_PI, 0.0f };
            if (l_en) send_frame(ODESC_NODE_L, CMD_SET_INPUT_VEL, vel_L, 8);
            delayMicroseconds(CAN_INTER_FRAME_US);
            if (r_en) send_frame(ODESC_NODE_R, CMD_SET_INPUT_VEL, vel_R, 8);
            break;
        }

        case WheelMode::POSITION: {
            // rad → turns; vel_ff and torque_ff both zero
            float pos_L = L_hw / TWO_PI;
            float pos_R = R_hw / TWO_PI;
            uint8_t buf[8] = {};
            if (l_en) {
                memcpy(buf + 0, &pos_L, 4);
                send_frame(ODESC_NODE_L, CMD_SET_INPUT_POS, buf, 8);
            }
            delayMicroseconds(CAN_INTER_FRAME_US);
            if (r_en) {
                memcpy(buf + 0, &pos_R, 4);
                send_frame(ODESC_NODE_R, CMD_SET_INPUT_POS, buf, 8);
            }
            break;
        }

        case WheelMode::TORQUE: {
            if (l_en) send_frame(ODESC_NODE_L, CMD_SET_INPUT_TRQ, &L_hw, 4);
            delayMicroseconds(CAN_INTER_FRAME_US);
            if (r_en) send_frame(ODESC_NODE_R, CMD_SET_INPUT_TRQ, &R_hw, 4);
            break;
        }
    }
}

void wheel_motors_pet_watchdog() {
    // 50 Hz divider (audit D4): petting at the full 500 Hz call rate put a
    // standing ~1000 frames/s (~13% of CAN3) on the bus for no benefit — the
    // ODrive watchdog only needs a frame well inside its timeout window.
    // Coverage in the other modes: TORQUE is petted by the control loop's
    // unconditional wheel_motors_send(); VELOCITY/POSITION are petted by the
    // 50 Hz wheel_motors_request_vbus() poll (any CAN frame addressed to the
    // axis feeds the ODrive watchdog) — a zero-vel keepalive there would
    // stomp the GUI's live setpoint, so IDLE keeps the only explicit pet.
    static uint8_t div = 0;
    if (++div < 10) return;
    div = 0;
    if (wm_mode == WheelMode::IDLE) {
        float zero[2] = { 0.0f, 0.0f };
        if (param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f) send_frame(ODESC_NODE_L, CMD_SET_INPUT_VEL, zero, 8);
        delayMicroseconds(CAN_INTER_FRAME_US);
        if (param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f) send_frame(ODESC_NODE_R, CMD_SET_INPUT_VEL, zero, 8);
    }
}

void wheel_motors_request_vbus() {
    CAN_message_t msg = {};
    msg.flags.remote = 1;
    msg.len = 8;
    if (param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f) {
        msg.id = ((uint32_t)ODESC_NODE_L << 5) | CMD_GET_VBUS;
        note_tx(can3.write(msg));
    }
    delayMicroseconds(CAN_INTER_FRAME_US);
    if (param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f) {
        msg.id = ((uint32_t)ODESC_NODE_R << 5) | CMD_GET_VBUS;
        note_tx(can3.write(msg));
    }
}

void wheel_motors_clear_errors() {
    uint32_t ident = 0;
    if (param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f) send_frame(ODESC_NODE_L, CMD_CLEAR_ERRORS, &ident, 4);
    delayMicroseconds(CAN_INTER_FRAME_US);
    if (param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f) send_frame(ODESC_NODE_R, CMD_CLEAR_ERRORS, &ident, 4);
    wm_L.error = 0;
    wm_R.error = 0;
    s_tx_stalled   = false;   // latched TX stall clears with the ODrive faults
    s_tx_defer_run = 0;
    comm_log(LOG_LEVEL_INFO, "WheelMotors: clear_errors sent");
}

bool wheel_motors_ok() {
    bool l_ok = (param_get(PARAM_WHEEL_L_ENABLE) < 0.5f) || (wm_L.ok && !wm_L.error);
    bool r_ok = (param_get(PARAM_WHEEL_R_ENABLE) < 0.5f) || (wm_R.ok && !wm_R.error);
    return l_ok && r_ok;
}
