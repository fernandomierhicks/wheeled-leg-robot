#pragma once
// CRSF telemetry scheduler — the return half of the link.
//
// Budget, deliberately: about 10 fields at 5-10 Hz, never a mirror of the
// 247-byte TelemetryPayload at 50 Hz. CRSF telemetry is a thin slice of the
// radio link, and the .wlog on the robot stays the authoritative record. If
// you find yourself wanting another field here, ask first whether the answer
// is really "look at the log afterwards".
//
// One frame per tick, round-robin, so telemetry leaves as a trickle rather
// than a burst that collides with the receiver's own uplink slot.
//
//   ATTITUDE    10 Hz   pitch/roll/yaw -- the state variables that matter
//   WLR_STATE   10 Hz   the robot-specific numerics, for the Lua HUD
//   BATTERY      2 Hz   pack volts/amps
//   FLIGHT_MODE  2 Hz   state name, or !FAULT while in ESTOP
//
// ~350 bytes/s total.

#include <Arduino.h>
#include "crsf_protocol.h"
#include "Crsf.h"
#include "generated_names.h"

// Everything the emitter needs, gathered by the caller. Passing a plain struct
// keeps this file free of any dependency on robot_state.h or the param
// registry, so it stays easy to reason about and to test.
struct CrsfTelemSources {
    float    pitch_rad;
    float    roll_rad;
    float    yaw_rad;
    float    pack_volts;
    float    pack_amps;
    uint8_t  pack_pct;
    uint8_t  robot_state;
    uint8_t  fault_code;
    uint8_t  jump_state;
    uint8_t  standup_state;
    float    gain_sched_alpha;
    uint8_t  active_profile;
    uint16_t health_flags;
    float    hip_l_torque_nm;
    float    hip_r_torque_nm;
    float    wheel_vel_avg_ms;
    uint8_t  esp32_link_ok;
    uint16_t vel_glitch_count;
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
                n = crsf_build_attitude(f, s.pitch_rad, s.roll_rad, s.yaw_rad);
                break;
            case 1:
                n = build_state(f, s);
                break;
            case 2:
                n = crsf_build_attitude(f, s.pitch_rad, s.roll_rad, s.yaw_rad);
                break;
            case 3:
                n = build_state(f, s);
                break;
            case 4:
                n = crsf_build_battery(f, s.pack_volts, s.pack_amps, 0, s.pack_pct);
                break;
            case 5:
                n = crsf_build_attitude(f, s.pitch_rad, s.roll_rad, s.yaw_rad);
                break;
            case 6:
                n = build_state(f, s);
                break;
            case 7:
                n = crsf_build_flight_mode(f, mode_text(s));
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
        w.state         = s.robot_state;
        w.fault         = s.fault_code;
        w.jump_state    = s.jump_state;
        w.standup_state = s.standup_state;
        w.alpha         = s.gain_sched_alpha;
        w.profile       = s.active_profile;
        w.health_flags  = s.health_flags;
        w.hip_l_nm      = s.hip_l_torque_nm;
        w.hip_r_nm      = s.hip_r_torque_nm;
        w.wheel_ms      = s.wheel_vel_avg_ms;
        w.esp32_ok      = s.esp32_link_ok;
        w.glitch_count  = s.vel_glitch_count;
        return crsf_build_wlr_state(f, w);
    }

    // 8 slots at 25 ms gives attitude and state 10 Hz each, battery and
    // flight mode 5 Hz.
    static constexpr uint32_t SLOT_MS = 25;
    static constexpr uint8_t  SLOTS   = 8;

    uint32_t _last_ms = 0;
    uint8_t  _slot    = 0;
};
