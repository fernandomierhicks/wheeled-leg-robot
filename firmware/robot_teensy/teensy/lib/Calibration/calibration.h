#pragma once

// Hip hardstop calibration (STATE_CALIBRATION).
//
// Runs both hip axes in parallel: seeks toward each axis's "bottom" hardstop
// (HIP_{L,R}_SEEK_DIR in config.h) at CALIB_SEEK_SPEED_RADS, zeros the encoder
// there (persists in AK45 flash), then seeks the opposite direction to find
// the "top" hardstop. From the measured range it computes software position
// limits (hm_limits_L/R in hip_motors.h) with a CALIB_MARGIN_RAD margin, then
// returns each axis to the midpoint and holds.
//
// Limits are RAM-only — `calibration_start()` invalidates them, so a fault or
// reboot before completion leaves setpoints unclamped again.

// Reset both axes and begin seeking. Call once on entering STATE_CALIBRATION.
void calibration_start();

// Advance both axes by one control tick. Call every tick while in
// STATE_CALIBRATION.
void calibration_update();

// True once both axes have found their hardstops, computed limits, and
// returned to their home (midpoint) position.
bool calibration_done();

// True if either axis ramped CALIB_SAFETY_BOUND_RAD without finding a
// hardstop.
bool calibration_failed();

// Abort an in-progress calibration (e.g. operator bailed out via GUI).
// Resets internal axis state so a subsequent calibration_start() is clean.
// Does NOT invalidate already-computed limits.
void calibration_abort();
