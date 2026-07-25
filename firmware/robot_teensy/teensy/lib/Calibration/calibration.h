#pragma once

// Retract-switch hip calibration (STATE_CALIBRATION).
//
// Each enabled axis releases an already-pressed switch, seeks the switch from
// the released side, zeros its encoder, confirms a fresh near-zero CAN feedback
// frame, and backs off to the configured safe retracted position. The extended
// software limit is a configured angle from switch zero; no extended hardstop
// is contacted.

void calibration_start();
void calibration_update();

// True once both axes have homed, computed limits, backed off, and ramped
// calibration torque to zero.
bool calibration_done();

// True if either axis fails a switch, travel, timeout, feedback, or current
// safety check.
bool calibration_failed();

// Begin/update a graceful operator cancellation. The begin call snapshots each
// active calibration setpoint; update tapers its gains to zero over
// calib_rampdown_s and returns true while the ramp is still active.
void calibration_begin_disarm();
bool calibration_run_disarm();

// Cancel an unfinished calibration while returning to STANDBY.
void calibration_abort();
