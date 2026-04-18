"""
run_anticogging.py — Anticogging calibration for ODrive 0.5.6 axis0.

Uses whatever gains are already configured (same as normal operation).
Extra fixes:
  - Disable anticogging_enabled before calibrating (was fighting itself on stale map)
  - Allow reasonable regen current (dc_max_negative_current was -0.01 A)
  - Relax calib thresholds for reasonable sweep speed
  - 10 s timeout at every step; goes IDLE immediately if motor doesn't move
"""
import time
import sys
import odrive
from odrive.enums import (
    AXIS_STATE_IDLE,
    AXIS_STATE_FULL_CALIBRATION_SEQUENCE,
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    CONTROL_MODE_VELOCITY_CONTROL,
    CONTROL_MODE_POSITION_CONTROL,
    INPUT_MODE_PASSTHROUGH,
)

AXIS = 0
MOVE_TIMEOUT = 10.0

def dump(ax, label=""):
    if label:
        print(f"\n=== {label} ===")
    print(f"  state={ax.current_state}  ax_err=0x{ax.error:08X}  mot_err=0x{ax.motor.error:08X}  enc_err=0x{ax.encoder.error:08X}")
    print(f"  calibrated={ax.motor.is_calibrated}  enc_ready={ax.encoder.is_ready}")
    cc = ax.controller.config
    print(f"  vel_gain={cc.vel_gain:.4f}  vel_integrator_gain={cc.vel_integrator_gain:.4f}  vel_limit={cc.vel_limit}")
    print(f"  vel={ax.encoder.vel_estimate:+.4f} t/s  Iq={ax.motor.current_control.Iq_measured:+.3f} A")
    acog = cc.anticogging
    print(f"  acog.index={acog.index}  pre_calibrated={acog.pre_calibrated}  enabled={acog.anticogging_enabled}")
    print(f"  acog.calib_vel_threshold={acog.calib_vel_threshold}  calib_pos_threshold={acog.calib_pos_threshold}")

def safe_idle(ax):
    try:
        ax.requested_state = AXIS_STATE_IDLE
    except Exception:
        pass

def abort(ax, msg):
    print(f"\nABORT: {msg}")
    safe_idle(ax)       # always go idle before exiting — no humming/hanging
    time.sleep(0.3)
    dump(ax, "STATE AT ABORT")
    sys.exit(1)

def wait_idle(ax, timeout=60):
    t0 = time.monotonic()
    while ax.current_state != AXIS_STATE_IDLE:
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(0.2)
    return True

def enter_closed_loop(ax, ctrl_mode, inp_mode=INPUT_MODE_PASSTHROUGH):
    ax.controller.config.control_mode = ctrl_mode
    ax.controller.config.input_mode   = inp_mode
    ax.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
    time.sleep(0.4)
    return ax.current_state == AXIS_STATE_CLOSED_LOOP_CONTROL

# ── Connect ────────────────────────────────────────────────────────────────────
print("Connecting...")
odrv = odrive.find_any(timeout=10)
if odrv is None:
    print("ABORT: no ODrive found"); sys.exit(1)
ax = getattr(odrv, f"axis{AXIS}")
print(f"Connected: {hex(odrv.serial_number)}  fw={odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}  Vbus={odrv.vbus_voltage:.1f}V")
odrv.clear_errors()
dump(ax, "INITIAL")

# ── Calibrate if needed ────────────────────────────────────────────────────────
if not ax.motor.is_calibrated or not ax.encoder.is_ready:
    print("\n[CAL] Running FULL_CALIBRATION_SEQUENCE...")
    ax.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
    if not wait_idle(ax, 60):
        abort(ax, "calibration timeout")
    if ax.error or ax.motor.error or ax.encoder.error:
        abort(ax, "calibration errors")
    print("  Calibration OK.")
else:
    print("\n[CAL] Already calibrated.")

# ── Pre-calibration setup ──────────────────────────────────────────────────────
print("\n[SETUP]")

# Disable anticogging — was running on stale/empty map, fights itself during calibration
ax.controller.config.anticogging.anticogging_enabled = False
print("  anticogging_enabled = False")

# Allow reasonable regen (was -0.01 A — nearly zero — clamped any braking current)
saved_neg_cur = odrv.config.dc_max_negative_current
odrv.config.dc_max_negative_current = -5.0
print(f"  dc_max_negative_current: {saved_neg_cur} → -5.0 A")

# Relax calibration thresholds (default 1.0 t/s causes 1 index per minute)
ax.controller.config.anticogging.calib_vel_threshold = 5.0
ax.controller.config.anticogging.calib_pos_threshold = 5.0
print("  calib_vel_threshold = 5.0  calib_pos_threshold = 5.0")

# Report gains — not changing them, using whatever normal operation uses
cc = ax.controller.config
print(f"  Using existing gains: vel_gain={cc.vel_gain:.4f}  vel_integrator={cc.vel_integrator_gain:.4f}  vel_limit={cc.vel_limit}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP A: Velocity smoke test — same as normal operation, no gain changes
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[A] Velocity smoke test (2 t/s for up to {MOVE_TIMEOUT:.0f}s)...")
odrv.clear_errors()
ax.controller.input_vel = 0.0

if not enter_closed_loop(ax, CONTROL_MODE_VELOCITY_CONTROL):
    abort(ax, f"velocity closed loop entry failed (state={ax.current_state})")

ax.controller.input_vel = 2.0
t0 = time.monotonic()
moved = False
while time.monotonic() - t0 < MOVE_TIMEOUT:
    vel = ax.encoder.vel_estimate
    iq  = ax.motor.current_control.Iq_measured
    print(f"  t={time.monotonic()-t0:.1f}s  vel={vel:+.4f} t/s  Iq={iq:+.3f} A", end="\r")
    if abs(vel) > 0.3:
        moved = True
        break
    if ax.error or ax.motor.error or ax.encoder.error:
        ax.controller.input_vel = 0.0
        abort(ax, "error during velocity test")
    time.sleep(0.05)

ax.controller.input_vel = 0.0
safe_idle(ax)           # go idle immediately — no hanging
time.sleep(0.5)
print()

if not moved:
    abort(ax, f"motor did not move in {MOVE_TIMEOUT:.0f}s — check motor state before anticogging")
print("  Velocity test OK!")

# ══════════════════════════════════════════════════════════════════════════════
# STEP B: Anticogging calibration
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[B] Anticogging calibration (position control, existing gains)...")
print("    Motor sweeps all positions — takes several minutes. Ctrl+C to abort.")
odrv.clear_errors()

if not enter_closed_loop(ax, CONTROL_MODE_POSITION_CONTROL):
    abort(ax, f"position closed loop entry failed (state={ax.current_state})")

print("  Calling start_anticogging_calibration()...")
ax.controller.start_anticogging_calibration()
time.sleep(1.0)

# Wait up to 10 s to confirm motor started moving
t0 = time.monotonic()
started = False
while time.monotonic() - t0 < MOVE_TIMEOUT:
    vel = ax.encoder.vel_estimate
    iq  = abs(ax.motor.current_control.Iq_measured)
    idx = ax.controller.config.anticogging.index
    print(f"  t={time.monotonic()-t0:.1f}s  vel={vel:+.4f} t/s  Iq={iq:.3f} A  idx={idx}", end="\r")
    if abs(vel) > 0.05 or iq > 0.5:
        started = True
        break
    if ax.error or ax.motor.error or ax.encoder.error:
        abort(ax, "error immediately after start_anticogging_calibration()")
    time.sleep(0.1)

print()
if not started:
    abort(ax, f"motor did not start moving within {MOVE_TIMEOUT:.0f}s after calibration call")

print("  Sweep running!")
print(f"  {'Time':>8}  {'idx':>6}  {'vel':>10}  {'Iq':>8}  state")

t0 = time.monotonic()
prev_idx = -1
try:
    while True:
        elapsed = time.monotonic() - t0
        if elapsed > 600:
            print(f"\n  TIMEOUT — 10 min"); break

        acog  = ax.controller.config.anticogging
        idx   = acog.index
        vel   = ax.encoder.vel_estimate
        iq    = ax.motor.current_control.Iq_measured
        state = ax.current_state

        if ax.error or ax.motor.error or ax.encoder.error:
            print()
            abort(ax, f"error during sweep: ax=0x{ax.error:08X} mot=0x{ax.motor.error:08X}")

        if idx != prev_idx:
            print(f"  {elapsed:8.1f}s  {idx:6d}  {vel:+10.4f}  {iq:+8.3f}  {state}")
            prev_idx = idx

        if state == AXIS_STATE_IDLE and elapsed > 3.0:
            print(f"\n  Axis returned to IDLE — calibration complete! (t={elapsed:.1f}s)")
            break

        time.sleep(0.3)

except KeyboardInterrupt:
    print("\n  Interrupted.")
    safe_idle(ax)

# ── Finish ──────────────────────────────────────────────────────────────────
dump(ax, "FINAL STATE")
safe_idle(ax)
time.sleep(0.3)

try:
    ax.controller.remove_anticogging_bias()
    print("  remove_anticogging_bias() OK")
except Exception as e:
    print(f"  remove_anticogging_bias: {e}")

# Restore original regen limit
odrv.config.dc_max_negative_current = saved_neg_cur
print(f"  dc_max_negative_current restored to {saved_neg_cur}")

ans = input("\nSave anticogging map and reboot? [y/N] ").strip().lower()
if ans == "y":
    ax.controller.config.anticogging.anticogging_enabled = True
    ax.controller.config.anticogging.pre_calibrated      = True
    odrv.save_configuration()
    print("  Saved. Rebooting...")
    try:
        odrv.reboot()
    except Exception:
        pass
    print("  Done.")
else:
    print("  Not saved.")
