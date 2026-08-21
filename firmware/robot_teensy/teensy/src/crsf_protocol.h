#pragma once
// CRSF (Crossfire) wire protocol — pure, no Arduino, no globals.
//
// Split out from the driver on purpose: everything here compiles natively, so
// the failsafe semantics can be unit-tested on a desktop with no radio, no
// receiver and no Teensy. That matters more here than anywhere else in this
// firmware — see the FAILSAFE CONTRACT below.
//
// Frame layout (simple frames, type < 0x28):
//
//   [addr][len][type][payload ...][crc8]
//              |<--------- len ------->|
//
//   addr    device address (0xC8 = flight controller) or sync byte
//   len     number of bytes AFTER len, i.e. 1 (type) + N (payload) + 1 (crc)
//   crc8    poly 0xD5, init 0, computed over [type][payload]
//
// Reference: EdgeTX radio/src/telemetry/crossfire.{h,cpp} — the decoder on the
// other end of this link. Scaling of every telemetry field below was read out
// of that file rather than assumed.

#include <stdint.h>
#include <string.h>

// ── Addresses ────────────────────────────────────────────────────────────────
static constexpr uint8_t CRSF_ADDR_BROADCAST        = 0x00;
static constexpr uint8_t CRSF_ADDR_FLIGHT_CONTROLLER= 0xC8;
static constexpr uint8_t CRSF_ADDR_RADIO            = 0xEA;
static constexpr uint8_t CRSF_ADDR_RECEIVER         = 0xEC;
static constexpr uint8_t CRSF_ADDR_TRANSMITTER      = 0xEE;

// ── Frame types we care about ────────────────────────────────────────────────
static constexpr uint8_t CRSF_FT_BATTERY   = 0x08;  // we send
static constexpr uint8_t CRSF_FT_LINK      = 0x14;  // we receive
static constexpr uint8_t CRSF_FT_CHANNELS  = 0x16;  // we receive
static constexpr uint8_t CRSF_FT_ATTITUDE  = 0x1E;  // we send
static constexpr uint8_t CRSF_FT_FLIGHTMODE= 0x21;  // we send

// Private frame carrying the robot-specific numeric fields. EdgeTX pushes any
// frame type it does not natively decode to the Lua telemetry queue
// (crossfire.cpp, `default:` -> pushTelemetryDataToQueues), which is how the
// WLRHUD widget gets at these. Chosen below 0x28 so it stays a simple frame
// rather than an extended-header one, and outside every id in EdgeTX's
// crossfire.h.
//
// UNVERIFIED ON HARDWARE: ExpressLRS is known to relay ATTITUDE, BATTERY and
// FLIGHT_MODE, but whether it relays an arbitrary private type has not been
// confirmed. Everything the HUD genuinely needs is carried by the three
// standard frames; this one is additive, and the HUD renders its fields as
// MISSING if it never arrives.
static constexpr uint8_t CRSF_FT_WLR_STATE = 0x24;

static constexpr uint8_t CRSF_MAX_FRAME    = 64;   // addr+len+62
static constexpr uint8_t CRSF_MAX_PAYLOAD  = 60;
static constexpr uint8_t CRSF_NUM_CHANNELS = 16;

static constexpr uint32_t CRSF_BAUD = 420000;

// ── CRC8, poly 0xD5 ──────────────────────────────────────────────────────────
inline uint8_t crsf_crc8(const uint8_t* data, uint8_t len) {
    uint8_t crc = 0;
    for (uint8_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (uint8_t b = 0; b < 8; b++)
            crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0xD5) : (uint8_t)(crc << 1);
    }
    return crc;
}

// ── Channel scaling ──────────────────────────────────────────────────────────
// CRSF ticks 172..1811 map to 988..2012 us. This is the mapping every flight
// controller uses, so the numbers in telemetry read like ordinary RC
// microseconds and the GUI's channel mirrors stay comparable with the iBUS
// captures that predate the swap.
//
// NORMALISE AT THE BOUNDARY. Every threshold in main.cpp (> 1990, < 1010) is
// an absolute microsecond value and must stay untouched — there are a dozen of
// them across arm, rescue and calibration paths, and converting at each site
// is a dozen chances to get it wrong.
static constexpr uint16_t CRSF_TICK_MIN = 172;
static constexpr uint16_t CRSF_TICK_MID = 992;
static constexpr uint16_t CRSF_TICK_MAX = 1811;

inline uint16_t crsf_ticks_to_us(uint16_t ticks) {
    // us = ticks * 1024/1639 + 881  (integer form of Betaflight's 0.62477 * t + 881)
    return (uint16_t)(((uint32_t)ticks * 1024U) / 1639U + 881U);
}

inline uint16_t crsf_us_to_ticks(uint16_t us) {
    if (us < 881) return 0;
    return (uint16_t)((((uint32_t)us - 881U) * 1639U) / 1024U);
}

// Unpack 16 channels of 11 bits each, LSB-first, from a 22-byte payload.
inline void crsf_unpack_channels(const uint8_t* p, uint16_t* out_ticks) {
    uint32_t bits = 0;
    uint8_t  nbits = 0;
    uint8_t  src = 0;
    for (uint8_t ch = 0; ch < CRSF_NUM_CHANNELS; ch++) {
        while (nbits < 11) {
            bits |= (uint32_t)p[src++] << nbits;
            nbits += 8;
        }
        out_ticks[ch] = (uint16_t)(bits & 0x7FF);
        bits >>= 11;
        nbits -= 11;
    }
}

inline void crsf_pack_channels(const uint16_t* ticks, uint8_t* out22) {
    uint32_t bits = 0;
    uint8_t  nbits = 0;
    uint8_t  dst = 0;
    for (uint8_t ch = 0; ch < CRSF_NUM_CHANNELS; ch++) {
        bits |= (uint32_t)(ticks[ch] & 0x7FF) << nbits;
        nbits += 11;
        while (nbits >= 8) {
            out22[dst++] = (uint8_t)(bits & 0xFF);
            bits >>= 8;
            nbits -= 8;
        }
    }
    if (nbits > 0) out22[dst++] = (uint8_t)(bits & 0xFF);
}

// ── LINK_STATISTICS (0x14) payload, 10 bytes ─────────────────────────────────
struct CrsfLinkStats {
    uint8_t up_rssi_1;      // dBm * -1
    uint8_t up_rssi_2;
    uint8_t up_lq;          // uplink link quality, percent
    int8_t  up_snr;
    uint8_t active_antenna;
    uint8_t rf_mode;
    uint8_t up_tx_power;    // index into a power table
    uint8_t down_rssi;
    uint8_t down_lq;
    int8_t  down_snr;
};

inline void crsf_parse_link_stats(const uint8_t* p, CrsfLinkStats* s) {
    s->up_rssi_1      = p[0];
    s->up_rssi_2      = p[1];
    s->up_lq          = p[2];
    s->up_snr         = (int8_t)p[3];
    s->active_antenna = p[4];
    s->rf_mode        = p[5];
    s->up_tx_power    = p[6];
    s->down_rssi      = p[7];
    s->down_lq        = p[8];
    s->down_snr       = (int8_t)p[9];
}

// ── Frame builders ───────────────────────────────────────────────────────────
// Each writes a complete frame into `out` and returns its total length.
// `out` must have room for CRSF_MAX_FRAME bytes.

inline uint8_t crsf_build_frame(uint8_t* out, uint8_t addr, uint8_t type,
                                const uint8_t* payload, uint8_t payload_len) {
    out[0] = addr;
    out[1] = (uint8_t)(payload_len + 2);   // type + payload + crc
    out[2] = type;
    if (payload_len) memcpy(out + 3, payload, payload_len);
    out[3 + payload_len] = crsf_crc8(out + 2, (uint8_t)(payload_len + 1));
    return (uint8_t)(payload_len + 4);
}

// Round, do not truncate. Float scaling is exact almost nowhere: 4.2f * 10.0f
// is 41.99999, which a cast turns into 41 and puts a 0.1 A error on the
// telemetry. The same applies to every scaled field below.
static inline int16_t crsf_round16(float v) {
    return (int16_t)(v >= 0.0f ? (v + 0.5f) : (v - 0.5f));
}

static inline void crsf_put_be16(uint8_t* p, int16_t v) {
    p[0] = (uint8_t)((uint16_t)v >> 8);
    p[1] = (uint8_t)((uint16_t)v & 0xFF);
}

static inline void crsf_put_be24(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)((v >> 16) & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)(v & 0xFF);
}

// ATTITUDE 0x1E — pitch, roll, yaw as int16 BE in radians * 10000.
// EdgeTX decodes with value/10 and precision 3, i.e. it displays raw/10000
// radians. Pitch and roll are *the* state variables for a balancing robot, so
// these are the highest-value three bytes on the whole link.
//
// NOTE the sensor unit EdgeTX auto-creates is RADIANS, not degrees. The HUD
// converts for display; leave the sensor's unit alone in Telemetry setup or
// EdgeTX will convert too and the HUD will double-count.
inline uint8_t crsf_build_attitude(uint8_t* out, float pitch_rad,
                                   float roll_rad, float yaw_rad) {
    uint8_t p[6];
    crsf_put_be16(p + 0, crsf_round16(pitch_rad * 10000.0f));
    crsf_put_be16(p + 2, crsf_round16(roll_rad  * 10000.0f));
    crsf_put_be16(p + 4, crsf_round16(yaw_rad   * 10000.0f));
    return crsf_build_frame(out, CRSF_ADDR_FLIGHT_CONTROLLER, CRSF_FT_ATTITUDE, p, 6);
}

// BATTERY_SENSOR 0x08 — voltage 0.1 V, current 0.1 A, used capacity mAh,
// remaining percent.
//
// Fields the robot does not measure are sent as all-0xFF, which is CRSF's
// "no data". EdgeTX's getCrossfireTelemetryValue() returns false when every
// byte of a field is 0xFF and then skips the sensor entirely
// (crossfire.cpp), so the sensor is never created rather than being created
// and reading a confident, wrong zero.
//
// That distinction matters: nothing on this robot measures bus current, and a
// Curr sensor sitting at a flat 0.0 A is exactly the kind of reading someone
// eventually builds an alarm on. Better for it to be visibly absent.
static constexpr float CRSF_BATT_NO_DATA = -1.0f;   // pass for an unmeasured field

inline uint8_t crsf_build_battery(uint8_t* out, float volts, float amps,
                                  int32_t used_mah, uint8_t remaining_pct) {
    uint8_t p[8];
    if (volts < 0.0f) { p[0] = 0xFF; p[1] = 0xFF; }
    else              crsf_put_be16(p + 0, crsf_round16(volts * 10.0f));
    if (amps < 0.0f)  { p[2] = 0xFF; p[3] = 0xFF; }
    else              crsf_put_be16(p + 2, crsf_round16(amps * 10.0f));
    if (used_mah < 0) { p[4] = 0xFF; p[5] = 0xFF; p[6] = 0xFF; }
    else              crsf_put_be24(p + 4, (uint32_t)used_mah & 0xFFFFFFu);
    p[7] = remaining_pct;
    return crsf_build_frame(out, CRSF_ADDR_FLIGHT_CONTROLLER, CRSF_FT_BATTERY, p, 8);
}

// FLIGHT_MODE 0x21 — null-terminated text. EdgeTX truncates at
// min(16, len) (crossfire.cpp), so the usable string is 13 characters plus the
// terminator. Carries the state name, swapped for the fault name in ESTOP,
// which is what makes state and fault visible with zero Lua.
static constexpr uint8_t CRSF_FLIGHTMODE_MAX = 13;

inline uint8_t crsf_build_flight_mode(uint8_t* out, const char* mode) {
    uint8_t p[CRSF_FLIGHTMODE_MAX + 1];
    uint8_t n = 0;
    while (n < CRSF_FLIGHTMODE_MAX && mode[n]) { p[n] = (uint8_t)mode[n]; n++; }
    p[n++] = 0;
    return crsf_build_frame(out, CRSF_ADDR_FLIGHT_CONTROLLER, CRSF_FT_FLIGHTMODE, p, n);
}

// ── WLR_STATE 0x24 — the robot-specific numerics ─────────────────────────────
// 16-byte payload, big-endian throughout to match every other CRSF frame.
// Kept small deliberately: CRSF telemetry is a thin slice of the link, and the
// .wlog on the robot remains the authoritative record. Do not grow this into a
// mirror of the 247-byte TelemetryPayload.
//
//  [0]  state          uint8   RobotStateEnum
//  [1]  fault          uint8   FAULT_*
//  [2]  jump_state     uint8   0..4
//  [3]  standup_state  uint8   0..3
//  [4]  alpha          uint8   gain_sched_alpha * 200 (0..200)
//  [5]  profile        uint8   active_profile
//  [6..7]   health     uint16  health_flags
//  [8..9]   hip_l      int16   N.m * 100
//  [10..11] hip_r      int16   N.m * 100
//  [12..13] wheel      int16   m/s * 100
//  [14]     esp32_ok   uint8
//  [15..16] glitch     uint16  wm_L + wm_R vel_glitch_count, saturating
//
// The glitch count is a free-running total and the annunciator warns on it
// RISING, so a byte-wide field would saturate early in a bad session and then
// look like it had stopped climbing -- silencing the warning exactly when it
// was becoming true. 16 bits costs one byte and removes the problem.
static constexpr uint8_t CRSF_WLR_STATE_LEN = 17;

struct CrsfWlrState {
    uint8_t  state;
    uint8_t  fault;
    uint8_t  jump_state;
    uint8_t  standup_state;
    float    alpha;
    uint8_t  profile;
    uint16_t health_flags;
    float    hip_l_nm;
    float    hip_r_nm;
    float    wheel_ms;
    uint8_t  esp32_ok;
    uint32_t glitch_count;
};

static inline uint8_t crsf_sat_u8(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 254.5f) return 255;
    return (uint8_t)(v + 0.5f);
}

inline uint8_t crsf_build_wlr_state(uint8_t* out, const CrsfWlrState& s) {
    uint8_t p[CRSF_WLR_STATE_LEN];
    p[0] = s.state;
    p[1] = s.fault;
    p[2] = s.jump_state;
    p[3] = s.standup_state;
    p[4] = crsf_sat_u8(s.alpha * 200.0f);
    p[5] = s.profile;
    p[6] = (uint8_t)(s.health_flags >> 8);
    p[7] = (uint8_t)(s.health_flags & 0xFF);
    crsf_put_be16(p + 8,  crsf_round16(s.hip_l_nm * 100.0f));
    crsf_put_be16(p + 10, crsf_round16(s.hip_r_nm * 100.0f));
    crsf_put_be16(p + 12, crsf_round16(s.wheel_ms * 100.0f));
    p[14] = s.esp32_ok;
    uint32_t g = (s.glitch_count > 65535u) ? 65535u : s.glitch_count;
    p[15] = (uint8_t)(g >> 8);
    p[16] = (uint8_t)(g & 0xFF);
    return crsf_build_frame(out, CRSF_ADDR_FLIGHT_CONTROLLER, CRSF_FT_WLR_STATE,
                            p, CRSF_WLR_STATE_LEN);
}

// ═════════════════════════════════════════════════════════════════════════════
//  FAILSAFE CONTRACT — read before touching anything below
// ═════════════════════════════════════════════════════════════════════════════
//
// The iBUS driver this replaces had two properties that the whole safety design
// leans on, and they are NOT incidental:
//
//   1. alive() goes false when the link stops.
//   2. channel() returns 0 — NOT the last received value — whenever the link
//      is not alive.
//
// Property 2 is the subtle one. The rescue combo tests
//      channel(3) > 1990 && channel(2) > 1990 && channel(1) < 1010 && channel(4) < 1010
// and the calibration combo is its mirror. If a dead link returned the last
// value, or held a value, a radio that died with the sticks anywhere near a
// corner could satisfy the two "< 1010" halves for free. Returning 0 makes the
// low tests trivially true, which is precisely why every combo in main.cpp is
// ALSO guarded by an explicit `alive &&`. Both halves of that belt-and-braces
// have to survive this port.
//
// radio_update()'s disarm interlock computes `armed = alive && (ch10 > 1990)`
// for the same reason.
//
// Three ways this link can look alive when it is not, all handled:
//   a. Frames stop entirely     -> the timeout below.
//   b. Frames keep arriving but the receiver has lost the transmitter and is
//      sending its configured failsafe values -> uplink LQ collapses to 0, so
//      LQ is treated as a liveness signal once we have ever seen link stats.
//      This is strictly better than the FlySky sentinel/noise heuristics it
//      replaces, which existed only because iBUS gave nothing better.
//   c. Garbage on the wire that happens to frame up -> CRC8 rejection plus a
//      warm-up count, so one lucky frame cannot make a dead link look live.
//
// Set the receiver's own failsafe to "no pulses", never "hold". The model file
// ships failsafeMode NOT_SET for this reason.
//
static constexpr uint32_t CRSF_LINK_TIMEOUT_MS = 100;
static constexpr uint8_t  CRSF_WARMUP_FRAMES   = 5;

// Pure decoder + channel state. Time is injected so the liveness rules can be
// tested exhaustively without a clock.
class CrsfCore {
public:
    void reset() {
        _idx = 0; _expect = 0;
        _frames = 0; _crc_err = 0; _last_frame_ms = 0;
        _have_link_stats = false;
        _link = CrsfLinkStats{};
        for (uint8_t i = 0; i < CRSF_NUM_CHANNELS; i++) _ch_us[i] = CRSF_CH_MID;
    }

    // Feed one received byte. Returns true when a complete, CRC-valid frame was
    // consumed (of any type).
    bool feed(uint8_t b, uint32_t now_ms) {
        if (_idx == 0) {
            // Resync on either a flight-controller-addressed frame or a bare
            // sync byte; receivers in the wild use both.
            if (b != CRSF_ADDR_FLIGHT_CONTROLLER && b != CRSF_ADDR_BROADCAST) return false;
            _buf[_idx++] = b;
            return false;
        }
        if (_idx == 1) {
            if (b < 2 || b > CRSF_MAX_PAYLOAD + 2) { _idx = 0; return false; }
            _expect = b;
            _buf[_idx++] = b;
            return false;
        }
        _buf[_idx++] = b;
        if (_idx < (uint16_t)(_expect + 2)) return false;

        const uint8_t total = (uint8_t)(_expect + 2);
        const uint8_t type  = _buf[2];
        const uint8_t plen  = (uint8_t)(_expect - 2);
        const uint8_t crc   = _buf[total - 1];
        _idx = 0;

        if (crsf_crc8(_buf + 2, (uint8_t)(plen + 1)) != crc) { _crc_err++; return false; }

        _last_type = type;
        _last_plen = plen;
        memcpy(_last_payload, _buf + 3, plen);

        if (type == CRSF_FT_CHANNELS && plen >= 22) {
            uint16_t ticks[CRSF_NUM_CHANNELS];
            crsf_unpack_channels(_buf + 3, ticks);
            for (uint8_t i = 0; i < CRSF_NUM_CHANNELS; i++)
                _ch_us[i] = crsf_ticks_to_us(ticks[i]);
            _last_frame_ms = now_ms;
            if (_frames < CRSF_WARMUP_FRAMES) _frames++;
        } else if (type == CRSF_FT_LINK && plen >= 10) {
            crsf_parse_link_stats(_buf + 3, &_link);
            _have_link_stats = true;
        }
        return true;
    }

    bool alive(uint32_t now_ms, uint32_t timeout_ms = CRSF_LINK_TIMEOUT_MS) const {
        if (_frames < CRSF_WARMUP_FRAMES) return false;
        if ((uint32_t)(now_ms - _last_frame_ms) >= timeout_ms) return false;
        // Case (b): frames still flowing, transmitter gone.
        if (_have_link_stats && _link.up_lq == 0) return false;
        return true;
    }

    // 1-indexed to match the transmitter, the GUI and radio_channels.md.
    // Returns 0 when the link is not alive — see the FAILSAFE CONTRACT above.
    uint16_t channel_us(uint8_t n, uint32_t now_ms,
                        uint32_t timeout_ms = CRSF_LINK_TIMEOUT_MS) const {
        if (!alive(now_ms, timeout_ms)) return 0;
        if (n < 1 || n > CRSF_NUM_CHANNELS) return CRSF_CH_MID;
        return _ch_us[n - 1];
    }

    // Raw last-known value, ignoring liveness. Diagnostics only — never feed
    // this to an arm, rescue or calibration test.
    uint16_t channel_us_raw(uint8_t n) const {
        if (n < 1 || n > CRSF_NUM_CHANNELS) return CRSF_CH_MID;
        return _ch_us[n - 1];
    }

    const CrsfLinkStats& link() const   { return _link; }
    bool     have_link_stats() const    { return _have_link_stats; }
    uint32_t crc_errors() const         { return _crc_err; }
    uint8_t  last_frame_type() const    { return _last_type; }
    uint8_t  last_payload_len() const   { return _last_plen; }
    const uint8_t* last_payload() const { return _last_payload; }

    static constexpr uint16_t CRSF_CH_MID = 1500;

private:
    uint8_t  _buf[CRSF_MAX_FRAME]  = {};
    uint16_t _idx                  = 0;
    uint8_t  _expect               = 0;
    uint16_t _ch_us[CRSF_NUM_CHANNELS] = {
        1500,1500,1500,1500,1500,1500,1500,1500,
        1500,1500,1500,1500,1500,1500,1500,1500};
    uint32_t _last_frame_ms        = 0;
    uint8_t  _frames               = 0;
    uint32_t _crc_err              = 0;
    CrsfLinkStats _link            = {};
    bool     _have_link_stats      = false;
    uint8_t  _last_type            = 0;
    uint8_t  _last_plen            = 0;
    uint8_t  _last_payload[CRSF_MAX_PAYLOAD] = {};
};
