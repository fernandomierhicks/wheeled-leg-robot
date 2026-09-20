#pragma once
// CRSF telemetry scheduler — the return half of the link.
//
// Budget, measured rather than assumed. This file used to claim "~350 bytes/s"
// while actually emitting ~590, and to describe ATTITUDE/WLR_STATE as 10 Hz
// when three of eight slots at 40 frames/s is 15 Hz. Both figures were wrong,
// and the second mistake helped hide the first.
//
// What matters is the other end. An ExpressLRS downlink at a 1:128 telemetry
// ratio carries a couple of packets per second — order tens of bytes/s. At
// ~590 B/s the receiver discarded most of what was sent and *which* frames
// survived was arbitrary, so FLIGHT_MODE — the one frame carrying state and
// fault as guaranteed-to-relay text — was among the least likely to arrive.
//
// So the schedule now sends only what is operationally critical:
//
//   WLR_STATE       2 Hz   state, fault, profile            7 B/frame
//   FLIGHT_MODE     2 Hz   state name, or !FAULT text   12-18 B/frame
//   BATTERY_SENSOR  1 Hz   pack volts and percent          12 B/frame
//
// ~52 B/s total, a ~11x cut. Pack voltage sits at 1 Hz because it moves on a
// timescale of minutes; there is nothing to be gained from sampling a slowly
// draining battery twice a second, and it is the one reading where a stale
// value is still a useful value.
//
// ATTITUDE (0x1E) is no longer emitted at all. Its builder remains in
// crsf_protocol.h and stays unit-tested, so restoring it is a one-line
// addition to the switch below: this is a scheduling decision, not a loss of
// capability. Note the HUD never read `yaw` anyway.
//
// The HUD keeps every readout it had. The ones with no data behind them render
// MISSING, which is the honest display for "not being sent" and is exactly
// what telem.lua's status handling exists to do. The .wlog on the robot
// remains the authoritative record for everything trimmed here.
//
// State and fault deliberately go out TWICE — as numbers in WLR_STATE and as
// text in FLIGHT_MODE. FLIGHT_MODE is a standard frame ExpressLRS is known to
// relay; the private 0x24 type is not confirmed to relay at all. Sending both
// costs 7 B/frame and means the two critical readings survive either way. It
// also turns the HUD's profile readout into a live test of whether 0x24
// relays: a number means it does, MISSING means it does not.

#include <Arduino.h>
#include "crsf_protocol.h"
#include "Crsf.h"
#include "generated_names.h"

// Everything the emitter needs, gathered by the caller. Passing a plain struct
// keeps this file free of any dependency on robot_state.h or the param
// registry, so it stays easy to reason about and to test.
struct CrsfTelemSources {
    uint8_t robot_state;
    uint8_t fault_code;
    uint8_t active_profile;
    float   pack_volts;     // CRSF_BATT_NO_DATA when unmeasured
    float   pack_amps;      // CRSF_BATT_NO_DATA when unmeasured
    uint8_t pack_pct;
};

class CrsfTelemetry {
public:
    // Call every control tick; it rate-limits itself.
    void tick(Crsf& link, uint32_t now_ms, const CrsfTelemSources& s) {
        if ((uint32_t)(now_ms - _last_ms) < SLOT_MS) return;
        _last_ms = now_ms;

        uint8_t f[CRSF_MAX_FRAME];
        uint8_t n = 0;

        switch (_slot) {
            case 0:
            case 2:
                n = build_state(f, s);
                break;
            case 1:
            case 3:
                n = crsf_build_flight_mode(f, mode_text(s));
                break;
            case 4:
                n = crsf_build_battery(f, s.pack_volts, s.pack_amps, -1, s.pack_pct);
                break;
            default:
                break;
        }
        _slot = (uint8_t)((_slot + 1) % SLOTS);
        if (n) link.send(f, n);
    }

    // The string EdgeTX shows as its FM sensor, and the one piece of robot
    // state visible with no Lua at all. A fault wins over the state name,
    // prefixed with '!' so it is unmistakable at a glance.
    static const char* mode_text(const CrsfTelemSources& s) {
        static char buf[CRSF_FLIGHTMODE_MAX + 1];
        if (s.fault_code != 0) {
            const char* nm = (s.fault_code < 16) ? FAULT_SHORT_NAMES[s.fault_code] : "FAULT";
            buf[0] = '!';
            uint8_t i = 0;
            while (i < CRSF_FLIGHTMODE_MAX - 1 && nm[i]) { buf[1 + i] = nm[i]; i++; }
            buf[1 + i] = 0;
            return buf;
        }
        if (s.robot_state < 10) return STATE_NAMES[s.robot_state];
        return "?";
    }

private:
    static uint8_t build_state(uint8_t* f, const CrsfTelemSources& s) {
        CrsfWlrState w;
        w.state   = s.robot_state;
        w.fault   = s.fault_code;
        w.profile = s.active_profile;
        return crsf_build_wlr_state(f, w);
    }

    // 5 slots at 200 ms is a 1 s cycle: state and flight mode twice per cycle
    // (2 Hz), battery once (1 Hz). State changes are events, so the worst-case
    // 500 ms of display latency buys an 11x cut in link load.
    static constexpr uint32_t SLOT_MS = 200;
    static constexpr uint8_t  SLOTS   = 5;

    uint32_t _last_ms = 0;
    uint8_t  _slot    = 0;
};
