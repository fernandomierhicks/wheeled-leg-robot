# AK45-10 Characterization — CubeMars Tool V1.32

## Tool
`CubeMarstool_V1.32.exe` — VESC-derived upper computer software.

## Max Speed

**Datasheet:** 180 rpm output (no-load, 24 V) = **25,200 ERPM** (10:1 gearbox, 14 pole pairs).

**Verified:** Visual inspection confirmed the motor reaches ~3 RPS (180 rpm) at full command — matches datasheet.

## Known Issue: Speed Display in deg/s is Wrong

The GUI reports output speed in deg/s with a large scaling error (~4.7× too low). At full speed the display shows ~230 deg/s when the true output is ~1,080 deg/s (180 rpm × 6).

**Use ERPM as the ground truth** (25,200 ERPM = 180 rpm = 1,080 deg/s at no load, 24 V).

The bug is likely a wrong gear ratio or pole-pair count in the GUI's unit conversion — do not trust the deg/s readout.
