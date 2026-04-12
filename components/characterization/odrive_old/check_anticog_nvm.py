"""
check_anticog_nvm.py — Check anticogging map status and quality on ODrive 0.5.6.

Quality test: hold a fixed position, measure Iq variance with anticogging ON vs OFF.
A real map reduces current ripple; a flat/empty map shows no difference.
"""
import time
import statistics
import odrive
from odrive.enums import (
    AXIS_STATE_IDLE,
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    CONTROL_MODE_POSITION_CONTROL,
    INPUT_MODE_PASSTHROUGH,
)

AXIS      = 0
HOLD_SECS = 4      # seconds to sample Iq at each setting
SAMPLE_HZ = 20     # samples per second

print("Connecting...")
odrv = odrive.find_any(timeout=10)
if odrv is None:
    print("ABORT: no ODrive found"); raise SystemExit(1)

ax   = odrv.axis0
acog = ax.controller.config.anticogging

print(f"fw  {odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}")
print(f"\nanticogging map status:")
print(f"  pre_calibrated      = {acog.pre_calibrated}")
print(f"  anticogging_enabled = {acog.anticogging_enabled}")
print(f"  index               = {acog.index}  (resets to 0 after reboot — normal)")
print(f"  calib_anticogging   = {getattr(acog, 'calib_anticogging', '(n/a)')}")
print(f"  cogging_ratio       = {getattr(acog, 'cogging_ratio', '(n/a)')}")

if not acog.pre_calibrated:
    print("\nRESULT: pre_calibrated=False — map not flagged, will not be applied on boot.")
    raise SystemExit(0)

if not ax.motor.is_calibrated or not ax.encoder.is_ready:
    print("\nSkipping quality test — motor not calibrated / encoder not ready.")
    raise SystemExit(0)

print("\n── Quality test: Iq variance with anticogging ON vs OFF ──")
print("  Entering position hold...")

ax.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
ax.controller.config.input_mode   = INPUT_MODE_PASSTHROUGH
ax.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
time.sleep(0.4)

if ax.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
    print(f"  Could not enter closed loop (state={ax.current_state}) — skipping quality test.")
    raise SystemExit(0)

hold_pos = ax.encoder.pos_estimate
ax.controller.input_pos = hold_pos
time.sleep(0.3)

def sample_iq(label, secs=HOLD_SECS):
    samples = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        samples.append(abs(ax.motor.current_control.Iq_measured))
        time.sleep(1.0 / SAMPLE_HZ)
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples)
    print(f"  {label:20s}  mean Iq={mean:.4f} A  stdev={stdev:.4f} A  (n={len(samples)})")
    return mean, stdev

acog.anticogging_enabled = True
time.sleep(0.2)
mean_on,  std_on  = sample_iq("anticogging ON")

acog.anticogging_enabled = False
time.sleep(0.2)
mean_off, std_off = sample_iq("anticogging OFF")

# Restore
acog.anticogging_enabled = True

ax.controller.input_pos = hold_pos   # keep holding
ax.requested_state = AXIS_STATE_IDLE
time.sleep(0.3)

print()
if std_off > 0 and std_on < std_off * 0.8:
    reduction = (1 - std_on / std_off) * 100
    print(f"RESULT: map looks VALID — Iq stdev reduced by {reduction:.0f}% with anticogging ON.")
elif std_on <= std_off:
    print("RESULT: map has marginal effect — Iq stdev similar ON vs OFF.")
    print("        Map may be low quality (high thresholds) or motor has low cogging to begin with.")
else:
    print("RESULT: anticogging ON increased Iq variance — map may be corrupt or misaligned.")
