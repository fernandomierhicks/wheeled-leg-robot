# Wheeled-Leg Robot Reliability and Improvement Plan

**Date:** July 30, 2026  
**Baseline:** `main` at commit `e906227b81ae77779c286fdf929363551aac8854`  
**Status:** Proposed implementation plan; no fixes in this document have been implemented merely by creating it.

## 1. Purpose

This plan consolidates the reliability, safety, controller, state-machine, ESP32,
GUI, testing, and AI-assisted tuning improvements identified during a review of
the three software deployments:

- Teensy 4.1 real-time firmware
- ESP32 communications, display, ToF, and status firmware
- Python/PyQt GUI, logging, analysis, remote-control, and tuning tools

The immediate goal is to make commanded motion expire reliably, prevent unsafe
state transitions, preserve balance authority during actuator saturation, and
make controller-tuning experiments repeatable and recoverable. The longer-term
goal is to support AI-assisted tuning without allowing an AI process, GUI hang,
network client, or stale parameter to bypass deterministic firmware safety.

## 2. Scope and priorities

### P0 — Complete before more unattended or AI-assisted tuning

1. Separate the motion-command deadman from the general GUI heartbeat.
2. Add a universal ARMING/preflight state before entering RUNNING.
3. Validate all sensor and actuator values for finiteness and physical
   plausibility.
4. Fix STARTUP recovery and state-entry-relative timeouts.
5. Allow preallocated SD logging to continue while RUNNING.
6. Replace independent wheel clipping with a balance-priority torque allocator.
7. Restrict safety-critical parameter writes and simulation overrides.
8. Make explicit motion release, lease expiration, and source changes perform
   the same deterministic safe cleanup.

### P1 — Reliability and authority hardening

1. Refactor the top-level state machine and remove command rejection as a mode.
2. Confirm ODrive state/mode using fresh heartbeat data before arming.
3. Add CAN transmit-error handling and coherent ISR-to-control snapshots.
4. Bound communication and storage work so it cannot starve the control loop.
5. Harden ESP32 TCP sessions, partial writes, queues, and log ownership.
6. Move blocking GUI networking, logging, and log assembly off the Qt thread.
7. Make parameter updates transactional and correlated end to end.
8. Correct fault severity, protocol IDs, documentation, and generated metadata.

### P2 — Operator experience and tuning quality

1. Reorganize the GUI around Overview, Drive, Tune, Diagnostics, and Logs.
2. Redesign the ESP32 screen around arm readiness, faults, and torque headroom.
3. Correct trim-aware tuning metrics and expand safety rejection criteria.
4. Add repeatable excitation profiles, repeated trials, and uncertainty
   estimates.
5. Add full trial manifests and reproducible analysis bundles.

### P3 — AI-augmented capabilities

1. Offline tuning and fault-analysis copilot.
2. System identification and digital-twin comparison.
3. Deterministic guarded experiment runner.
4. Constrained gain optimizer with rollback.
5. Predictive maintenance and anomaly detection.

## 3. Safety architecture principles

All implementation work should follow these rules:

1. **The Teensy owns motion safety.** GUI, ESP32, radio, and AI checks improve
   usability but are not substitutes for firmware-enforced limits.
2. **Motion authority is leased and expires.** A connection heartbeat proves
   that software exists; only a fresh motion command proves that its setpoint is
   current.
3. **No invalid number reaches an actuator.** NaN, infinity, invalid enum
   values, stale data, and incoherent snapshots must be rejected before control
   output.
4. **Balance has highest actuator priority.** Optional yaw, roll interaction,
   and feedforward must not unexpectedly consume the torque required to balance.
5. **Every energetic transition is deliberate.** Entry into motor torque mode
   requires fresh hardware state, a safe pose, neutral commands, and explicit
   authority.
6. **Every energetic state has one immediate escape path.** All aborts should
   enter DISARMING or a fault state without waiting for normal phase completion.
7. **Bench overrides are visibly different from operating modes.** Simulation,
   missing-motor bypasses, forced gain-schedule values, and disabled watchdogs
   belong in a dedicated BENCH_TEST mode.
8. **Experiments are transactions.** Candidate parameters, excitation, results,
   abort reason, and rollback are correlated and reproducible.

## 4. Workstream A — Motion command and authority model

### A1. Problem

The Teensy currently refreshes the GUI watchdog for any valid command. The GUI
sends PING approximately every 100 ms, so a stale nonzero velocity/yaw command
can remain active while the GUI is open even if the process responsible for
motion commands has stopped.

Relevant areas:

- `firmware/robot_teensy/teensy/src/main.cpp`
- `firmware/robot_teensy/teensy/src/state_machine.cpp`
- `software/gui/main.py`
- `software/gui/tabs/remote_control.py`
- `software/gui/tabs/comm_commands.py`

### A2. Protocol changes

Add semantic motion commands rather than using three independent parameter
writes:

```text
MOTION_ACQUIRE(session_id, lease_id, requested_ttl_ms)
MOTION_SET(session_id, lease_id, sequence, v_m_s, omega_rad_s, ttl_ms)
MOTION_RELEASE(session_id, lease_id, sequence)
```

Requirements:

- `MOTION_SET` atomically updates velocity, yaw, lease sequence, and deadline.
- Only a valid `MOTION_SET` refreshes the motion deadline.
- PING refreshes only a connection/manual-supervision heartbeat.
- Duplicate sequence numbers are idempotent.
- Old sequence numbers are rejected.
- A different session cannot reuse a previous session's request/result cache.
- TTL has a conservative firmware maximum.
- Expiration atomically zeros velocity/yaw and releases GUI motion ownership.
- Motion state and remaining TTL are present in telemetry.

### A3. Teensy changes

- Store motion owner/session, latest sequence, expiration time, and last applied
  setpoint as one coherent structure.
- Check expiration at control-loop priority, not only inside slower radio or
  communications processing.
- Use the same cleanup routine for:
  - explicit release;
  - lease expiration;
  - GUI source loss;
  - Wi-Fi/USB source change;
  - state exit;
  - ESTOP;
  - MCU command-session reset.
- Zero both setpoints before clearing the owner.
- Ensure a stale motion command cannot be re-applied after leaving and
  re-entering RUNNING.

### A4. GUI and AI changes

- Make motion sending asynchronous and owned by a dedicated transport worker.
- Display lease owner, source, last sequence, and TTL prominently.
- Require an explicit human action to grant AI motion authority.
- Do not treat `authorize_motion=true` supplied by a remote caller as human
  consent.
- Make takeover require confirmation unless the previous lease has expired.
- Prevent generic `param_set` from writing motion-source parameters.

### A5. Acceptance criteria

- With nonzero motion commanded, stopping only the motion sender zeros motion
  within the configured TTL even while normal GUI PING continues.
- Closing the GUI, unplugging USB, dropping Wi-Fi, changing sources, releasing
  the lease, and expiring the lease all produce the same zero/release result.
- Duplicated and reordered motion packets never extend the lease incorrectly.
- A GUI restart cannot inherit a prior motion lease.
- Firmware tests cover sequence wrap, duplicates, expiry, source change, and
  release during every energetic state.

## 5. Workstream B — State machine and arming

### B1. Proposed top-level state model

```text
BOOT
  -> SELF_TEST
  -> STANDBY
  -> ARMING
  -> RUNNING

STANDBY <-> CALIBRATION
STANDBY <-> MANUAL
STANDBY <-> BENCH_TEST

RUNNING -> JUMPING -> RUNNING
RUNNING -> STANDING_UP/RECOVERY -> RUNNING

Any energetic state -> DISARMING -> STANDBY
Any state -> FAULT_LATCHED
FAULT_LATCHED -> SELF_TEST or STANDBY, according to generated recovery policy
```

`CMD_REJECT` should become a correlated command-result event and temporary UI
notification. It should not replace the robot's actual operating state.

### B2. ARMING preflight

Entering RUNNING should require all of the following, regardless of whether
stand-up is enabled:

- IMU nominal and fresh.
- Pitch and roll within configurable capture envelopes.
- Pitch, roll, and yaw rates below limits.
- Wheel speeds below an arm limit.
- Hip and wheel feedback fresh.
- ODrive heartbeat fresh.
- ODrive axes confirmed in CLOSED_LOOP_CONTROL.
- Expected ODrive control/input modes confirmed.
- Hip calibration valid or BENCH_TEST explicitly active.
- Both velocity and yaw commands neutral.
- No simulation injection active.
- No unapproved hardware/watchdog bypass active.
- Battery above the arm threshold.
- Radio/GUI authority consistent with the requested arm source.
- No active logger/storage operation capable of blocking entry.

ARMING should have:

- an entry-relative deadline;
- one enumerated blocker/failure reason;
- telemetry of every preflight item;
- no wheel torque until all checks pass;
- optional controlled hip authority only if explicitly required.

### B3. STARTUP and reset correction

Current STARTUP checks rely on absolute MCU uptime and `ever_heard`. Replace this
with:

- `state_entry_ms`;
- a per-device feedback generation counter or post-entry timestamp;
- a fresh heartbeat/encoder requirement for each enabled actuator;
- a bounded timeout for every prerequisite;
- explicit differentiation between never detected, stale, wrong mode, hardware
  fault, and configuration-disabled.

Do not send a soft hardware-recovery reset back into an incompletely
reinitialized STARTUP. Either:

- perform a complete subsystem reinitialization with fresh-generation checks;
  or
- use a controlled MCU reboot and report the reboot reason.

### B4. Transition behavior

- Repeating `SET_MODE(current_state)` should succeed without side effects.
- Centralize mode requests in a transition table rather than independent
  condition blocks that can overwrite acceptance.
- All energetic states must prioritize safety checks before abort or normal
  completion.
- Calibration initiated by radio should cancel or ramp down on radio loss.
- Command results should include requested state, resulting state, status,
  rejection reason, and active blocker.

### B5. Jump corrections

- Remove the fixed outer three-second completion deadline.
- Give CROUCH, EXTEND, FLIGHT/LAND if used, and RETRACT separate deadlines.
- Track left and right leg completion independently.
- Require both legs to reach a terminal condition.
- Treat phase timeout as a fault/abort, not success.
- Detect and fault excessive left/right position, velocity, or current
  asymmetry.
- Log invalid limits only once per condition rather than at 500 Hz.
- Consider IMU/contact/ToF evidence for liftoff and landing. If there is no
  flight detection, name the behavior a launch sequence rather than implying a
  complete jump state estimator.

### B6. Stand-up corrections

- Refuse stand-up if the configured gains, torque limit, capture bounds, or
  divergence bounds form an incoherent/zero-authority preset.
- Apply pitch, roll, wheel-speed, motor-state, and sensor-freshness watchdogs in
  every stand-up phase.
- Require pitch, rate, wheel speed, roll, saturation, and neutral commands
  before handing off to RUNNING.
- Give retries reduced authority or require operator reauthorization rather
  than repeating an identical failed attempt.
- Publish phase, retry, convergence metrics, saturation, and failure cause.

### B7. Acceptance criteria

- It is impossible to enter RUNNING while outside the arm envelope.
- Soft reset at long MCU uptime cannot immediately timeout or remain in STARTUP
  forever.
- Every state has deterministic timeout and fault behavior.
- Repeated mode requests are idempotent.
- Jump and stand-up tests cover asymmetric legs, stale sensors, timeout,
  saturation, abort, and operator disarm.

## 6. Workstream C — Sensor, CAN, and actuator integrity

### C1. Finite and plausibility validation

For every decoded sensor value:

- reject NaN and infinity;
- validate enum values;
- validate quaternion norm before conversion;
- validate gyro, acceleration, position, velocity, current, voltage, and
  temperature against physical ranges;
- update freshness only after validation succeeds;
- retain a bounded-age last-known-good value;
- count consecutive and total invalid samples;
- fault rather than silently fail open after a configured limit.

Before sending any actuator command:

- validate every field for finiteness and bounds;
- clamp only values that are valid and intentionally clampable;
- reject invalid modes and invalid node IDs;
- force torque to zero and fault if a nonfinite command is produced.

### C2. Coherent snapshots

CAN callbacks currently update structures that the main loop reads
asynchronously. Implement one of:

- interrupt-disabled copy of a complete snapshot;
- sequence-lock pattern;
- double-buffered immutable snapshots.

The control loop must never combine position from one CAN frame with timestamp
or status from another partially updated frame.

### C3. Motor health

Wheel readiness must include:

- fresh encoder estimate;
- fresh ODrive heartbeat;
- no ODrive error;
- expected axis state;
- expected controller/input mode;
- successful recent command transmission;
- plausible velocity and current.

Do not mark a requested mode as active until the heartbeat confirms it.

### C4. CAN transmit handling

- Check every `can.write()` return value.
- Maintain per-bus and per-device transmit-failure counters.
- Retry only where duplication is safe.
- Provide queue/backpressure telemetry.
- Fault or disarm after a bounded continuous failure.
- Avoid logging every failed tick; aggregate and rate-limit reports.

### C5. Acceptance criteria

- Injected NaN/Inf sensor and controller values always produce zero actuator
  output and an observable fault.
- Stale heartbeat with fresh encoder data cannot pass ARMING.
- Wrong ODrive axis state cannot pass ARMING.
- CAN TX saturation cannot silently lose torque-mode or zero-torque commands.
- Snapshot stress tests find no mixed-generation reads.

## 7. Workstream D — Controller interaction and timing

### D1. Balance-priority torque allocation

The current controller combines:

- symmetric LQR balance torque;
- velocity PI through lean reference;
- differential yaw PI torque;
- FF1 hip-reaction compensation;
- FF2 gravity compensation;
- wheel velocity governor and hard wheel limits.

The allocator should explicitly define priority:

1. Hard safety zero/governor.
2. Symmetric balance torque.
3. Differential yaw torque within remaining wheel headroom.
4. Feedforward within remaining authority according to configured policy.

Suggested calculation:

```text
tau_balance = clamp(requested_balance, symmetric_safe_limit)
remaining_left  = wheel_limit - abs(tau_balance)
remaining_right = wheel_limit - abs(tau_balance)
tau_yaw_applied = clamp(requested_yaw, differential_headroom)
tau_ff_applied  = allocate_ff(requested_ff, remaining_headroom)
```

Account for signs and asymmetric remaining headroom directly rather than using
only a scalar approximation.

### D2. Anti-windup

- Feed final applied yaw torque back to the yaw integrator.
- Feed final applied lean/symmetric authority back to the velocity PI.
- Freeze or back-calculate integrators during wheel governor action.
- Reset or transfer integrator state explicitly on state and profile changes.
- Publish integrator state and anti-windup reason.

### D3. Feedforward definition

FF1 currently derives hip torque from measured hip current. Decide between:

- true feedforward based on commanded hip torque/acceleration; or
- filtered disturbance compensation based on measured current.

If measured-current compensation remains:

- rename it;
- low-pass filter it;
- measure latency;
- bound its derivative and contribution;
- include validity and age;
- prevent double-counting with FF2 or the LQR.

### D4. Roll and leg-height interaction

- Ensure differential roll offsets preserve average commanded extension.
- Recompute or validate gain-schedule alpha from the actual left/right geometry.
- Bound leg asymmetry separately from total extension.
- Confirm the roll integrator cannot walk the center of gravity outside the
  lateral safety envelope.
- Apply roll watchdog protection during RUNNING, JUMPING, and STANDING_UP as
  appropriate.

### D5. Loop timing

- Replace the unconditional fixed `dt=0.002` assumption with measured, bounded
  `dt` for integrators and rate limiters.
- Keep the nominal 500 Hz schedule, but count missed deadlines.
- Put control and safety ahead of unbounded communications, display, logging,
  and filesystem work.
- Give communications a byte/frame budget per tick.
- On a large timing discontinuity:
  - freeze or reset integrators;
  - output safe torque;
  - record the overrun;
  - disarm if the interruption exceeds a safe threshold.
- Enable the hardware watchdog in deployed configurations and record reset
  cause in telemetry/log headers.

### D6. Telemetry additions

Publish:

- requested/applied balance torque;
- requested/applied yaw torque;
- requested/applied FF1 and FF2;
- left/right final torque;
- allocator clipping reason;
- per-controller saturation fraction;
- torque headroom;
- integrator values and anti-windup state;
- actual measured loop period;
- deadline misses and maximum overrun;
- active profile and profile authority source.

### D7. Acceptance criteria

- A saturated yaw request cannot reduce required average balance torque without
  an explicit balance-saturation indication.
- Integrators unwind after saturation and do not jump on re-entry.
- Large loop overruns produce deterministic safe behavior.
- Controller replay tests reproduce allocator output exactly.

## 8. Workstream E — Parameters and persistent storage

### E1. Generated write policy

Extend parameter schema metadata with:

```text
READONLY
COMMAND
LIVE_TUNABLE
INERT_ONLY
BOOT_ONLY
BENCH_ONLY
PERSISTENT
BOOT_CLEAR
```

Examples:

- Gains intended for controlled tuning: `LIVE_TUNABLE`.
- Motor-enable flags and hardware configuration: `BOOT_ONLY`.
- Watchdog thresholds: normally `INERT_ONLY`; disabling is `BENCH_ONLY`.
- Simulation values and enable flags: `BENCH_ONLY | BOOT_CLEAR`.
- Motion setpoints: commands, not ordinary parameters.
- Forced alpha/calibration bypass: `BENCH_ONLY | BOOT_CLEAR`.

### E2. Transactional updates

Add:

```text
PARAM_TX_BEGIN(transaction_id)
PARAM_TX_SET(transaction_id, parameter_id, value)
PARAM_TX_VALIDATE(transaction_id)
PARAM_TX_COMMIT(transaction_id)
PARAM_TX_ABORT(transaction_id)
```

Validation must cover coupled invariants, including:

- watchdog threshold versus commanded lean plus trim and safety margin;
- forward/backward sign and range;
- stand-up capture versus divergence limits;
- jump cutoff versus calibrated travel;
- roll offset and integral contribution versus calibrated hip limits;
- controller torque requests versus hard motor torque limits;
- valid speed profile ranges;
- enabled controller requiring meaningful gains and watchdogs.

### E3. Persistent storage

- Do not automatically format LittleFS following an ordinary mount failure.
- Report mount failure and continue with safe compiled defaults.
- Provide a deliberate factory-reset command available only in an inert state
  with human confirmation.
- Restore only parameters marked persistent.
- Ensure legacy migration cannot restore command, simulation, or read-only
  values accidentally.
- Store schema/version/hash with each parameter generation.
- Preserve two CRC-protected generations and report which one was selected.

### E4. GUI behavior

- Buffer edits locally until Apply.
- Show old value, requested value, validated/applied value, and persistence.
- Correlate SET results by request and transaction ID.
- Do not treat any report for the same parameter ID as confirmation.
- Support Dry Run, Apply, Undo, Export Diff, and Rollback.
- Lock BOOT_ONLY and BENCH_ONLY controls according to state.

### E5. Acceptance criteria

- Unsafe coupled parameter sets fail validation atomically.
- Power interruption during persistence recovers one valid generation.
- A stale bulk report cannot mark a parameter write successful.
- Import either applies completely or leaves the previous configuration intact.
- Bench overrides are cleared on reboot and visible while active.

## 9. Workstream F — Logging, playback, and analysis

### F1. SD logger

- Permit record appends to a pre-opened, preallocated log during energetic
  states.
- Keep directory operations, file creation, deletion, and finalization out of
  energetic states.
- Use bounded ring buffers.
- Publish logger active, buffer use, dropped records, write latency, and
  overflow.
- Finalize safely after DISARMING.

### F2. GUI host logger

- Move file writes to a bounded worker queue.
- Flush periodically or at safe checkpoints, not every record.
- Do not put cloud-synchronized filesystem latency on the Qt event loop.
- Include host monotonic time, firmware time, source, session, protocol hash,
  firmware hash, and parameter transaction events.

### F3. Log transfer

- Stream chunks directly into a temporary random-access file.
- Track received chunks with a bitmap rather than retaining all chunk payloads.
- Verify suffix and complete CRC incrementally.
- Rename the temporary file atomically only after successful verification.
- Keep one transfer owner/session from start through completion.
- Prevent USB/TCP commands from redirecting another client’s active log stream.

### F4. Playback and decoder

- Strictly validate log magic, format version, telemetry version, record size,
  sample rate, and complete record count before playback.
- Drop and count schema/decode failures rather than emitting partially decoded
  dictionaries.
- Reject nonfinite telemetry fields from metrics.
- Coalesce playback updates through the same UI-rate bus used for live data.
- Use recorded timestamps and skip UI frames when necessary instead of
  delivering hundreds of repaints per second.

### F5. Correct tuning metrics

Use the actual control target:

```text
pitch_error = pitch - theta_ref - pitch_trim
```

Safety evaluation should reject:

- NaN/Inf values;
- insufficient RUNNING duration;
- missing state coverage;
- excessive balance, yaw, or wheel saturation;
- wheel governor activity;
- loop overruns;
- sensor dropouts;
- fault flags;
- excessive current;
- excessive wheel velocity;
- excessive roll/pitch;
- incomplete or mixed parameter snapshots.

Thresholds should come from the trial’s captured parameter snapshot rather than
hardcoded analysis constants.

### F6. Acceptance criteria

- A 64 MB log transfers without holding multiple full copies in GUI memory.
- RUNNING logs contain complete telemetry and parameter provenance.
- Playback rejects incompatible formats before decoding fields.
- Metrics use trim-aware pitch error and reject nonfinite/incomplete trials.

## 10. Workstream G — ESP32 communications and peripherals

### G1. TCP session security and ownership

- Bind command TCP sessions to the active discovery/telemetry lease.
- Add a random session token and challenge response or pre-shared-key HMAC.
- Treat USB as a separate trusted local transport.
- Add connection read-idle timeout and TCP keepalive.
- Do not allow a stale client to hold the sole TCP slot indefinitely.
- Associate command results and log transfers with connection generation.

### G2. Correct partial-write handling

The current nonblocking TCP wrapper can send only part of a frame. Replace it
with:

- one outbound frame queue per connection;
- persistent frame offset;
- retry on later writable opportunities;
- no interleaving of a second frame before the first completes;
- disconnect-generation checks;
- counters for short writes, stalls, queue full, and dropped frames.

### G3. Queue policy

- Reserve capacity for command results and safety/status events.
- Never evict or silently drop a command result to make room for telemetry.
- Telemetry may remain lossy.
- Bulk logs should remain ACK-paced and isolated.
- Publish queue high-water marks and drop counts by lane.

### G4. Cross-core snapshots

Publish one immutable display/telemetry snapshot from the communications core
instead of reading many independent volatile globals. This avoids mixing values
from different telemetry frames and prevents command-log text tearing.

### G5. ToF robustness

- If sensor initialization fails, pull its XSHUT pin low before starting the
  next sensor.
- Track validity and last-update age for each sensor.
- Mark stale distance invalid rather than retaining it indefinitely.
- Add bounded runtime recovery/readdressing.
- Report sensor-specific initialization and runtime faults.

### G6. Configuration cleanup

- Remove obsolete `WIFI_UNICAST_IP` compile/flash behavior if runtime claim
  leasing is the actual source of the destination.
- Update the flash GUI and documentation accordingly.
- Pin production PlatformIO platforms and library versions.

### G7. Acceptance criteria

- An unleased LAN client cannot issue robot commands.
- Forced short TCP writes reconstruct exactly one valid frame.
- Command results survive telemetry and log congestion.
- A failed first ToF sensor does not block later sensors.
- Display data is frame-coherent across cores.

## 11. Workstream H — GUI authority and responsiveness

### H1. Remote-control server

- Fix lease takeover so a new actor cannot inherit motion authorization.
- Run safe motion cleanup on explicit release, not only expiration.
- Cap request-buffer size and add idle timeout for clients that never send a
  newline.
- Require a lease for service stop, flashing, and other disruptive operations.
- Require inert state, preflight, and human confirmation for firmware flashing.
- Risk-classify parameter IDs; do not rely on caller-supplied
  `acknowledge_risk`.

### H2. Operator bridge

- Give operator-relevant widgets explicit stable IDs/object names.
- Do not generate durable control IDs from dynamic traversal order or labels.
- Keep generic widget automation read-only for safety-critical actions.
- Route state, motion, parameter, reset, reboot, and flash actions through the
  same semantic authority API.

### H3. Transport threading

- Own TCP connect/send/receive in a worker thread.
- Do not mutate `SourceManager` state directly from the transport thread; use
  queued Qt signals.
- Validate discovery acknowledgement and complete CommLink frames before
  marking Wi-Fi connected or changing the ESP32 IP.
- Pin command routing during a control lease.
- On source change, atomically release motion or complete a deliberate
  authenticated handover.

### H4. Responsiveness

Move off the Qt thread:

- TCP connection and `sendall`;
- host-log writes and flush;
- large log assembly and CRC;
- expensive analysis;
- firmware build/flash subprocess monitoring where blocking is possible.

Add UI-event-loop latency telemetry so a frozen or overloaded GUI is visible.

### H5. Acceptance criteria

- The GUI remains responsive during a 64 MB download and host logging.
- Junk UDP cannot claim the active ESP32 address.
- Lease takeover cannot inherit motion authorization.
- Explicit release zeroes setpoints and disables motion ownership.
- A source switch cannot redirect commands silently.

## 12. Workstream I — GUI and ESP32 display redesign

### I1. Persistent GUI safety header

Always show:

- robot mode;
- READY TO ARM or the highest-priority arm blocker;
- active fault and recommended recovery;
- radio link and physical CH10 state;
- command source and lease owner;
- motion TTL;
- Teensy, ESP32, USB, and Wi-Fi health;
- hardware and pitch/roll watchdog state;
- battery voltage and arm threshold;
- ESTOP control.

### I2. GUI workspace organization

#### Overview

- system health;
- arm checklist;
- current mode and state history;
- battery and communications;
- active faults and recovery actions.

#### Drive

- velocity/yaw command;
- speed profile and limits;
- deadman/lease status;
- actual versus commanded motion;
- large release/stop controls.

#### Tune

- candidate parameter set and current baseline;
- synchronized plots for pitch, true pitch target, rate, wheel velocity, and
  reference;
- requested/applied torque and headroom;
- saturation and governor events;
- excitation timeline;
- Apply Candidate, Run Trial, Abort, Revert, and Accept buttons;
- complete trial manifest and notes.

#### Diagnostics

- raw sensor ages and validity;
- ODrive axis and controller modes;
- CAN errors and queue occupancy;
- loop timing;
- watchdogs and bypasses;
- ToF status.

#### Logs

- host and SD recording status;
- buffer/drop counters;
- transfers with owner/session;
- playback compatibility and analysis results.

### I3. ESP32 screen

Default operational screen:

- large mode;
- READY or one explicit blocker;
- fault code plus recovery action;
- RC/GUI/AI authority;
- Teensy and ESP32 heartbeat;
- battery;
- watchdog state.

RUNNING screen:

- pitch/roll horizon;
- actual and commanded velocity;
- requested/applied balance torque;
- torque headroom/saturation;
- wheel governor indication;
- logger state;
- motion TTL.

Other changes:

- Use telemetry enable/active bits instead of inferring controller state from a
  nonzero command or output.
- Add distinct STANDING_UP and DISARMING face/mood behavior.
- Show ToF value, validity, and age.
- Validate battery cell count and use configured undervoltage thresholds.
- Never rely on color as the only state indication.

## 13. Workstream J — Tuning methodology

### J1. Required tuning order

1. Validate sensor signs, motor signs, and torque allocation.
2. Tune retracted LQR pitch/rate with velocity, yaw, roll, and FF disabled.
3. Validate LQR at nominal and extended leg heights.
4. Fit or tune gain scheduling across leg-height alpha.
5. Tune velocity PI with deterministic forward/reverse profiles.
6. Tune yaw PI using remaining torque headroom.
7. Tune roll controller and verify lateral safety margins.
8. Add FF1/FF2 last, measuring whether they improve error without increasing
   saturation or noise.

### J2. Repeatable excitation

Replace manual hold instructions with versioned profiles:

- zero hold;
- small positive/negative velocity steps;
- ramps;
- chirps or PRBS where safe;
- yaw steps at zero and low forward velocity;
- roll commands where mechanically safe;
- explicit settle intervals.

Every excitation command must carry a short motion TTL and be aborted by the
firmware safety monitor independently of the GUI.

### J3. Candidate search

The current adaptive search scales gain pairs with one shared random delta,
which explores only a one-dimensional ray. Replace it with:

- independent/covariance-aware perturbations;
- bounded ratios and sign constraints;
- baseline repeats;
- interleaved candidate/baseline trials;
- confidence intervals;
- constrained Bayesian optimization or another safe optimizer;
- explicit maximum authority increase per trial;
- automatic rollback after any failed or aborted candidate.

### J4. Trial bundle

Every trial should contain:

- firmware, GUI, ESP32, schema, and analysis commit hashes;
- protocol/telemetry versions;
- complete starting parameter snapshot;
- staged parameter transaction and applied result;
- active profile and gain-schedule alpha;
- excitation profile and exact command timestamps;
- host and firmware logs;
- operator/AI identity and lease;
- abort/fault reason;
- computed metrics and plots;
- acceptance decision and confidence;
- resulting rollback or accepted baseline.

## 14. Workstream K — AI-augmented capabilities

### K1. Phase 1: Read-only tuning copilot

Capabilities:

- load and validate trial bundles;
- explain faults, saturation, and tracking errors;
- compare candidate versus baseline;
- detect insufficient or invalid experiments;
- propose the next bounded candidate;
- generate plots and a human-readable rationale.

No direct motion or parameter authority is required.

### K2. Phase 2: System identification

Estimate:

- balance dynamics versus leg extension;
- motor/drive latency;
- friction and dead zones;
- effective torque constants;
- battery-voltage dependence;
- yaw coupling;
- roll-to-pitch and hip-to-wheel coupling;
- sensor delay/noise;
- saturation and governor behavior.

Compare identified dynamics with the existing simulation/digital twin and flag
model drift.

### K3. Phase 3: Guarded experiment runner

Expose semantic operations:

```text
preflight()
create_trial(candidate, excitation_profile)
validate_trial()
request_human_authorization()
apply_candidate_transaction()
start_capture()
run_excitation()
monitor_trial()
abort(reason)
rollback()
analyze_trial()
accept_as_baseline()
```

The deterministic supervisor, not the language model, must:

- enforce parameter bounds and coupled invariants;
- enforce motion TTL;
- verify state and arm conditions;
- monitor hard safety limits;
- abort immediately;
- guarantee rollback;
- keep an immutable audit log.

### K4. Phase 4: Constrained optimizer

- Optimize only within firmware-approved envelopes.
- Model uncertainty and trial-to-trial variation.
- Prefer candidates with expected improvement and adequate safety margin.
- Stop on insufficient confidence, repeated aborts, model mismatch, or changing
  hardware conditions.
- Require human approval before expanding torque, speed, angle, or watchdog
  envelopes.

### K5. Phase 5: Anomaly and maintenance assistant

Detect trends such as:

- increasing CAN loss or latency;
- IMU packet-loss growth or bias shift;
- left/right motor current asymmetry;
- linkage friction and calibration drift;
- wheel velocity/torque asymmetry;
- battery sag and resistance growth;
- changed center of gravity;
- ToF degradation;
- increasing loop overruns or logger stalls.

## 15. Workstream L — Protocol, documentation, and build hygiene

- Give `COMM_TYPE_COMMAND_RESULT` and `COMM_TYPE_ESP32_STATUS` unique type IDs.
- Key command-result caching by source, session, request ID, and command ID,
  with expiry.
- Generate fault severity, recovery instruction, state names, parameter policy,
  and protocol IDs from one schema.
- Update stale telemetry-version references.
- Correct tuning documentation concerning:
  - motion deadman behavior;
  - SD logging during RUNNING;
  - `alpha_force_ret_en` persistence;
  - current telemetry version;
  - current protocol and transport behavior.
- Remove dead `WIFI_UNICAST_IP` flash-time configuration if runtime claims own
  the unicast destination.
- Correct stale CommLink comments.
- Pin Teensy platform and all production library versions.
- Repair the ESP32 desktop/native test target so it does not require
  `Arduino.h`, or give it an explicit native compatibility shim.
- Add CI jobs for:
  - protocol generation `--check`;
  - Teensy and ESP32 production builds;
  - native CommLink, parameter, command, controller, and allocator tests;
  - GUI tests with PyQt6, pyqtgraph, pyserial, and numpy installed;
  - documentation/schema consistency;
  - secret and oversized-file checks.

## 16. Delivery sequence

### Milestone 0 — Baseline preservation

- Preserve commit `e906227` as the reviewed baseline.
- Create a versioned hardware configuration and bench-test checklist.
- Capture one inert host log and one current safe hardware baseline.

### Milestone 1 — Motion lease and arm gate

- Implement Workstream A.
- Add ARMING and corrected STARTUP generation checks.
- Add explicit arm blockers to telemetry and GUI.
- Complete unit and bench tests before changing controller behavior.

### Milestone 2 — Numeric and actuator safety

- Add finite/range validation.
- Add coherent snapshots.
- Confirm ODrive heartbeat/mode state.
- Add CAN TX handling.
- Enable and report hardware watchdog behavior.

### Milestone 3 — Torque allocator and timing

- Implement balance-priority allocation.
- Add final-saturation anti-windup.
- Add requested/applied telemetry.
- Bound communication work and use measured/bounded timing.
- Replay existing logs through the controller to compare old and new behavior.

### Milestone 4 — Logging and parameter transactions

- Enable safe RUNNING recording.
- Add streaming GUI logger/download.
- Add transactional parameter API and generated policies.
- Remove automatic filesystem format.

### Milestone 5 — State behaviors

- Refactor command rejection and transition table.
- Correct jump and stand-up terminal conditions.
- Add BENCH_TEST.
- Generate fault recovery policy.

### Milestone 6 — ESP32 and GUI hardening

- Fix TCP sessions and partial writes.
- Fix queue and log ownership.
- Fix ToF recovery and cross-core snapshots.
- Move blocking GUI operations to workers.
- Harden remote leases and operator actions.

### Milestone 7 — Operator redesign

- Implement the safety header and task-based workspaces.
- Implement operational ESP32 screens.
- Add source, lease, TTL, blocker, and torque-headroom displays.

### Milestone 8 — Tuning and AI foundation

- Correct metrics.
- Add versioned excitation profiles and trial bundles.
- Add repeated trials and uncertainty-aware candidate selection.
- Deploy read-only AI analysis before granting guarded experiment authority.

## 17. Verification matrix

### Unit and native tests

- Motion TTL, sequence, duplicate, wrap, and release.
- State transition table and priority.
- STARTUP post-entry freshness.
- ARMING blockers.
- NaN/Inf and range rejection.
- Torque allocator and anti-windup.
- Parameter transactions and coupled validation.
- Command-result session cache.
- TCP partial writes and queue saturation.
- Log header and metrics validation.

### Software-in-the-loop

- Replay recorded telemetry through safety, state, and controller logic.
- Inject dropouts, latency, out-of-order packets, invalid values, and timing
  overruns.
- Sweep actuator saturation and verify balance priority.
- Exercise source handover and lease expiration.

### Bench tests with wheels raised or torque constrained

- Confirm motor signs and zero behavior.
- Confirm ODrive state acknowledgement.
- Confirm CAN loss and nonfinite-command response.
- Confirm motion TTL with GUI PING still running.
- Confirm hardware watchdog reset and recorded reset cause.
- Confirm logger operation without loop starvation.

### Tethered floor tests

- ARMING envelope and blocker display.
- Disarm from every energetic state.
- Balance-only saturation tests.
- Velocity/yaw tests at conservative torque.
- Roll watchdog and lateral envelope.
- Stand-up/jump only after their dedicated preflight and bench validation.

## 18. Definition of done

The reliability program is complete when:

- stale motion always expires at the Teensy;
- RUNNING cannot be entered without a complete fresh preflight;
- nonfinite data can never reach an actuator;
- balance torque priority is explicit and observable;
- every state has deterministic success, timeout, abort, and fault behavior;
- logging during RUNNING is reliable and bounded;
- parameter changes are policy-controlled and transactional;
- ESP32 and GUI transport cannot silently change command ownership;
- GUI and ESP32 clearly show authority, TTL, blockers, faults, and torque
  headroom;
- tuning metrics match the actual controller equations;
- every tuning trial is reproducible and automatically rollback-capable;
- AI interaction uses semantic guarded operations rather than unrestricted
  parameters or widget clicks;
- production builds, protocol generation, native tests, GUI tests, and
  documentation consistency run in CI.

