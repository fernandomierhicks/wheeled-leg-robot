# IMU

> **AI maintenance note:** If you find anything here that is stale while
> working in this tree, update this README in the same change.

Robust BNO086 SPI driver for the 500 Hz robot control loop.

## Wiring (SPI0)

| Signal | Teensy 4.1 pin |
|---|---|
| CS | 10 |
| INT | 9 |
| RST | 6 |
| SCK | 13 |
| MOSI | 11 |
| MISO | 12 |

PS0 and PS1 must both be high when the BNO086 samples its boot mode to select
SPI. After that first H_INTN assertion, PS0 changes function and is the
active-low WAKE input. The current module bridges both pins high, so Teensy
cannot assert WAKE. The driver compensates with explicit, bounded
reset-as-wake handshakes. A future board revision should route PS0 to a Teensy
GPIO that holds it high for SPI boot selection and can drive it low afterward.

## API and safety contract

```cpp
imu_init();
while (imu_state() == ImuState::INITIALIZING) imu_update();

// Once per 500 Hz control tick:
imu_update();
if (imu_state() == ImuState::NOMINAL) {
    float pitch = imu_pitch();
    float rate  = imu_pitch_rate();
}
```

Only `NOMINAL` permits motion. `DEGRADED` is used immediately when the required
paired report is stale or while automatic recovery is in progress, so the
robot's state machine can remove torque before recovery does any blocking reset
work. `ERROR` means a recovery attempt failed; the driver remains safe and
retries a bounded hardware-reset recovery every second. A full, potentially
multi-second `begin_SPI()` is boot-only.

`imu_last_update_ms()` returns the arrival time of the required paired report
(or the older stream time in a legacy A/B build). `imu_get_diagnostics()`
exposes report counts, per-stream maximum gaps,
poll gaps, invalid values, resets, recovery attempts, queue overflows, decode
errors, and transport errors.

## Report-rate budget

| Report | Requested | Measured on robot | Role |
|---|---:|---:|---|
| Gyro Integrated Rotation Vector | 400 Hz | 399.7 Hz | paired quaternion + angular velocity to LQR; **required** |
| Linear acceleration | disabled | 0 Hz | retired; telemetry fields remain zero for protocol compatibility |

The interval sent to SH2 is a request, not an exact output schedule. The
dedicated integrated report is the only continuous production stream and
measures 399.7 Hz at the real 500 Hz host polling cadence. The older
Game-Rotation-Vector + calibrated-gyro + linear-acceleration architecture
measured 191.0/191.0/12.3 Hz and remains available as `test_imu_legacy`.

At 399.7 Hz a normal 500 Hz tick still sees less than one report on average. The
bounded drain limit is 16 so a delayed tick can catch up without turning into
unbounded loop work. Every rate or report-set change must be checked with the
physical `test_imu` test using the complete production configuration.

### Gyro-Integrated Rotation Vector selection

The BNO086 also has a dedicated `SH2_GYRO_INTEGRATED_RV` SHTP channel. One
14-byte report supplies a quaternion and calibrated XYZ angular velocity from
the same gyro sample. It is genuinely faster and lower latency. It cannot be
combined with useful linear acceleration on this unit, so production now uses
gyro-only landing and disables acceleration.

Physical A/B results from 2026-08-13, all with the host polling at 500 Hz:

| Configuration | Attitude + rate | Linear accel | Delivery age | Result |
|---|---:|---:|---:|---|
| Legacy GRV + gyro, 160 Hz requests | 191.0 Hz | 12.3 Hz | 5.42/3.42 ms | all tests pass |
| Integrated, 500 Hz request | 490.0 Hz | 0 Hz | 0.258 ms | reject: no jump acceleration |
| Integrated, 400 Hz request | 399.3 Hz | 0.7 Hz | 0.179 ms | reject: acceleration starved |
| Integrated, 250 Hz request | 400.0 Hz | 0.7 Hz | 0.178 ms | reject: request quantizes to 400 Hz |
| **Production integrated-only, 400 Hz request** | **399.7 Hz** | **disabled, 0 Hz** | **0.179 ms** | **all robustness tests pass** |

The 400 Hz integrated run survived a 150 ms host pause, a forced BNO086 reset,
and a 10-second soak with no spontaneous recovery, queue overflow, decode
error, transport error, or invalid sample. However, a repeated 500 Hz run had
three spontaneous stall/reset/recovery cycles (four total including the
deliberately forced reset) and still produced no linear acceleration. Its report
also has no per-sample sequence field, so the current
standard-stream loss metric cannot prove losslessness. The driver now closes
that gap by checking the dedicated channel's SPI SHTP sequence directly.

The integrated-only configuration cannot be issued as a single feature command
on the present tied-high-WAKE board: a direct post-reset integrated request is
acknowledged but produces no reports. The robust sequence briefly enables
linear acceleration during the reset wake window, enables Gyro RV, then waits
for a sensor-originated interrupt and disables acceleration while the hub is
provably awake. The completed initialization exposes zero acceleration reports.

The final integrated-only run produced 399.7 Hz attitude and angular velocity,
then passed a 150 ms no-poll pause, a forced hardware reset, and a 60-second
500 Hz-polling soak. It consumed 25,188 paired reports with zero spontaneous
stall/reset/recovery, SHTP sequence gap, queue overflow, decode error, transport
error, or invalid value. This removes the earlier robustness objection under
the explicit assumption that landing uses gyro evidence only.

Across two runs at rest, production pitch noise was 0.000038-0.000104 rad RMS
versus 0.000343-0.001062 rad integrated. Production gyro-axis noise was
0.00169-0.00215 rad/s RMS versus 0.00401-0.00486 rad/s integrated. The much
lower integrated delivery-age number is useful but is not a full physical
sensor-to-controller latency measurement: CEVA's SH2 library stamps every
Gyro-Integrated report with its SHTP transfer timestamp, whereas normal reports
include the SH2 report-delay timestamp. A motion fixture or closed-loop robot
test would be required to measure phase improvement directly. First-order gyro
filter probes showed that 40/60/80 Hz cutoffs reduced stationary RMS by only
about 2-6% while adding approximately 2.86/1.60/0.99 ms low-frequency delay.
Even 10/20 Hz filtering reduced RMS only about 8-22% while adding 14.70/6.77 ms.
The variation is therefore predominantly low-frequency drift, not noise that a
light control-safe filter removes. Start dynamic tests with the raw integrated
rate; do not spend the latency advantage on smoothing without measured need.

Production defaults are `IMU_USE_GYRO_INTEGRATED_RV=1`,
`IMU_REQUIRED_RATE_HZ=400`, and `IMU_ENABLE_LINEAR_ACCEL=0`. The reproducible
comparison environments are `test_imu_integrated_only` (the same report set
with a 60-second soak), `test_imu_legacy` (191 Hz three-stream baseline),
`test_imu_integrated` (400 Hz with acceleration requested),
`test_imu_integrated_500`, and `test_imu_integrated_250`.

## Why the old driver dropped out

The failures found in host logs were deterministic, not random sensor noise:

1. Starting an SD log preallocates a file and blocked the 500 Hz loop for
   roughly 90-190 ms.
2. The BNO086 data sheet says an unserviced H_INTN transaction times out after
   about 10 ms, deasserts, and retries. Repeatedly delaying service can starve
   its processing and cause output errors.
3. The old Adafruit callback stored one decoded report. If one SHTP packet held
   several reports, each callback overwrote the previous one; whichever stream
   decoded last survived. This explains cases where GRV remained live while
   gyro became stale.
4. PS0 is tied high on this hardware, although it must act as active-low WAKE
   after SPI mode is selected. A host command could therefore find the hub
   asleep. The old HAL responded to a 500 ms write timeout by silently resetting
   the BNO086. During multi-report setup, that reset erased reports which had
   already been enabled, while later commands appeared to succeed.
5. After 100 ms of either-stream silence, the old wrapper entered terminal
   `ERROR`. A benign scheduler stall therefore required a robot reboot.
6. The 400 Hz result from the one-report characterization was incorrectly
   applied to four simultaneous streams, overloading the BNO086's internal
   scheduler and making auxiliary output especially sparse.

## Robustness design

- The vendored Adafruit callback uses a 48-report FIFO, large enough for all
  decoded reports from the maximum 384-byte incoming SHTP payload. Overflow
  drops the oldest entry, retains the newest evidence, and increments a counter.
- A no-data SPI poll returns immediately. Payload and write handshakes have
  explicit 5 ms and 100 ms bounds. The longer write bound covers measured boot
  traffic; it remains finite. The low-level HAL reports failure and never
  performs a surprise reset.
- Initialization explicitly resets to open a known wake window. Required
  reports are sent first; auxiliary configurations are staged on later
  sensor-originated interrupts, when the hub is known to be awake. A failure
  closes the SH2 session cleanly so the next attempt is real.
- Events are drained before freshness is evaluated. A delayed scheduler tick
  can therefore consume retained reports instead of producing a false dropout.
- The production integrated report has dedicated SHTP-channel sequence-gap and
  silence tracking. Legacy GRV and gyro builds retain independent tracking.
  Validity checks reject non-finite vectors and malformed quaternions; accepted
  quaternions are normalized before Euler conversion.
- Silence first enters `DEGRADED`, with a 20 ms grace for the BNO086's own retry.
  If both streams do not return, the driver issues a fixed 30 ms reset pulse,
  waits up to 500 ms for the reset notification, re-enables all reports, and
  requires fresh paired attitude/rate data before returning to `NOMINAL`.
- An unexpected sensor reset follows the same reconfigure-and-prove-data path.
  Failed runtime recovery remains non-nominal and retries once per second. Boot
  initialization failures also retry without requiring a power cycle.
- Startup does not claim `NOMINAL` merely because configuration succeeded. It
  requires real samples from both control streams.

## Validation

`test/test_imu/test_imu.cpp` runs at the production 500 Hz polling cadence and:

- measures production attitude/rate and confirms acceleration stays disabled;
- requires less than 2% required-stream sequence loss;
- records sensor-timestamp gaps, arrival gaps, delivery age, and stationary
  pitch/gyro noise;
- checks SPI-aware sequence continuity on the dedicated Gyro RV channel;
- pauses polling for 150 ms to reproduce the measured SD-start stall;
- pulses the physical reset pin and verifies automatic report reconfiguration;
- soaks for 10 seconds and requires no spontaneous recovery plus zero queue,
  decode, transport, and invalid-report errors.

The final pre-promotion 2026-08-13 integrated-only run passed every assertion.
It measured 399.7 Hz paired attitude/rate and zero acceleration. The 150 ms
no-poll test recovered automatically, a forced reset reconfigured and proved
fresh paired data, and the 60-second soak consumed 25,188 reports with zero
packet loss, spontaneous recovery, SHTP sequence gap, queue overflow, decode
error, transport error, or invalid value. The post-promotion production GUI
capture recorded 260 telemetry frames over 5.18 seconds: IMU-NOMINAL was set in
all 260, IMU packet loss and link drops stayed at zero, all acceleration fields
stayed zero, and all three gyro axes continued updating.

Build it with:

```text
pio test -e test_teensy -f test_imu --without-uploading --without-testing
```

See `firmware/robot_teensy/README.md` for flashing and monitoring through the
GUI while it owns the Teensy serial port.

## Euler convention

`quat_to_euler()` uses intrinsic Z-Y-X (yaw, pitch, roll). Pitch is the
nose-down lean of body +X independent of heading, which is the quantity the
balance controller regulates. Its singularity is at pitch +/-90 degrees,
outside the permitted operating envelope.
