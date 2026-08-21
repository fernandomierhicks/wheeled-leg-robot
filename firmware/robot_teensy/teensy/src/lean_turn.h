#pragma once
// Coordinated-turn lean: the roll setpoint a balancing two-wheeler needs to
// hold while turning, so the resultant of gravity and centripetal acceleration
// stays along its own vertical instead of trying to tip it over.
//
// Pure and Arduino-free on purpose. The sign convention below is the one thing
// in this feature that will hurt if it is wrong -- lean the wrong way and the
// robot falls harder and faster than with no feature at all -- so it is
// unit-tested rather than reasoned about once and trusted.
//
// ── The physics ──────────────────────────────────────────────────────────────
//
//   lateral acceleration in a turn   a = v * omega
//   coordinated lean                 phi = atan(a / g)
//
// ── The sign chain, derived from this repo's conventions ─────────────────────
//
//   Frame (CLAUDE.md):        +X forward, +Y left, +Z up
//   Yaw command (main.cpp):   stick left -> yaws left -> omega_cmd POSITIVE
//                             (right-hand rule about +Z: left/CCW is positive)
//   Roll command (main.cpp):  stick right -> roll_cmd POSITIVE
//   Roll physically:          right-hand rule about +X rotates +Y toward +Z,
//                             so positive roll lifts the left side, i.e. the
//                             robot leans RIGHT
//
//   A LEFT turn (omega > 0) throws the mass to the right, so the robot must
//   lean LEFT, which is NEGATIVE roll. Hence the leading minus:
//
//       roll_cmd = -gain * atan(v * omega / g)
//
// VERIFY THIS ON THE BENCH, SUSPENDED, before any floor test. Command a slow
// left turn and confirm the body leans left. This repo has paid for a
// convention mistake before (AngleRetractedExt.md); the cost of checking is
// two minutes.
//
// ── Why the gain is not 1.0 ──────────────────────────────────────────────────
//
// This is not a motorcycle lean. The track is fixed and the wheels stay put, so
// tilting the body via differential hip extension SHIFTS THE CoM sideways
// rather than rotating the whole vehicle about its contact line. How much CoM
// shift you get per radian of body tilt depends on CoM height above the hip
// pivot, which is not something to derive on paper for this machine. So
// lean_gain is an empirical scale, defaulting to 0 (inert) and walked up a
// ladder the way jump_effort is.

#include <math.h>

static constexpr float LEAN_G = 9.81f;

// Returns the roll setpoint in radians for a coordinated turn.
//
//   v_ms        forward velocity, m/s   (measured: accurate and slow-changing)
//   omega_rads  yaw rate, rad/s         (commanded: leads the turn, no lag)
//   gain        empirical scale; 0 disables
//   max_rad     hard clamp, must stay well inside roll_watchdog_limit
//   min_ms      speed deadband; below this there is no meaningful centripetal
//               term and spinning in place should not command lean
//
// Returns 0 when disabled or inside the deadband, so the caller can add this
// to a stick command unconditionally.
inline float lean_turn_setpoint(float v_ms, float omega_rads, float gain,
                                float max_rad, float min_ms) {
    if (gain <= 0.0f) return 0.0f;
    if (fabsf(v_ms) < min_ms) return 0.0f;

    // atan, not the small-angle approximation: the argument reaches ~0.8 at
    // the robot's own limits (2 m/s, 4 rad/s), where the approximation is
    // already 12% high and biased the wrong way -- it would over-lean exactly
    // where over-leaning is least recoverable.
    const float lean = -gain * atanf((v_ms * omega_rads) / LEAN_G);

    if (lean >  max_rad) return  max_rad;
    if (lean < -max_rad) return -max_rad;
    return lean;
}

// Fraction of the requested lean the leg geometry can actually deliver, given
// the normalised ride height t.
//
// The roll controller's own headroom clamp is min(t, 1-t) * span: a
// differential offset spends retract headroom on one leg and extend headroom
// on the other, so travel collapses to zero at both ends of the stroke. That
// clamp keeps things SAFE, but it does so silently -- at full crouch or full
// extension the robot simply will not lean, however hard it is asked to.
//
// This exists so the HUD can say so out loud rather than leaving the operator
// wondering why the machine is not doing what the stick says.
inline float lean_turn_authority(float ride_height_t) {
    if (ride_height_t < 0.0f) ride_height_t = 0.0f;
    if (ride_height_t > 1.0f) ride_height_t = 1.0f;
    const float h = (ride_height_t < 1.0f - ride_height_t)
                        ? ride_height_t : (1.0f - ride_height_t);
    return h * 2.0f;   // 0 at either end of the stroke, 1.0 at mid-height
}
