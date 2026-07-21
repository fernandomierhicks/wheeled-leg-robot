# Robot software reliability plan

Status: execution in progress on `codex/robot-reliability-phases`; progress and hardware gates are recorded below.

Reviewed: 2026-07-20

## Execution progress

Branch: `codex/robot-reliability-phases`

| Phase | Status | Evidence / notes |
| --- | --- | --- |
| 0 — software-operator replacement | **Complete; Wi-Fi transport retest deferred to Phase 4** | 29 semantic operations; deterministic inspection/control of 2,008 GUI widgets (461 actionable), 100% parity report, screenshots, GUI service lifecycle, USB discovery/source control, parameter/log/firmware operations, control lease and dead-man. GUI unit suite: 11/11 passed. Teensy COM12 and ESP32 COM14 both flashed successfully; post-flash telemetry advanced and parameter reads passed through both USB sources with zero GUI CRC/pair drops. Arm command was accepted under an authorized lease and the robot returned safely to STANDBY with all four motor-enable parameters confirmed `0`. Wi-Fi did not appear after the ESP32 flash: serial diagnostics show repeated UDP `endPacket` errors and `TASK_WDT` resets, so validation is explicitly carried into the transport phases. Implementation commit `90f4b78` pushed to `origin/codex/robot-reliability-phases`. |
| 1 — behavior baselines and metrics | **Complete** | Frozen cross-language frame vectors and ordered FSM contract; Windows-native CommLink suite now covers noise, truncation, bad length/CRC/end, short writes, max payload, overlapping magic, sequence wrap, and recovery (10/10 passed). GUI suite: 18/18 passed. Added a repeatable operator-API link baseline; post-flash all three GUI sources delivered fresh STANDBY/NONE telemetry and parameter transactions, with 116 parameters read back and all four motor enables `0`. Baseline exposed the expected shared-sequencer defect (ESP32/Wi-Fi GUI sequence gaps grow while CRC/pair errors stay zero) and Wi-Fi jitter (up to 200 ms sample age and one 388 ms command), now assigned to Phases 3–4. Teensy COM12 upload succeeded on retry after one loader write failure; ESP32 COM14 upload succeeded. GUI screenshot visually confirmed live data and connection, with the existing degraded-link banner matching these measured deficiencies. |
| 2 — command safety and state transitions | Pending | |
| 3 — UART ownership and ESP32 scheduling | Pending | |
| 4 — Wi-Fi session and traffic policy | Pending | |
| 5 — generated protocol and durable configuration | Pending | |
| 6 — staged peripheral startup and recovery | Pending | |
| Final mock tuning workflow | Pending | |

For every completed phase, record the tests run, Teensy and ESP32 flash results, GUI validation over Teensy USB, ESP32 USB, and Wi-Fi, commit hash, and pushed branch here.

## Scope

This plan covers the Teensy control firmware, the ESP32 communications/display firmware, and the Python GUI, with emphasis on:

- Teensy–ESP32 UART integrity and recovery
- ESP32–GUI Wi-Fi latency, loss, reconnects, and backpressure
- a single source of truth for wire protocols and shared definitions
- state-machine transition safety and diagnosability
- deterministic startup and degraded operation when peripherals fail
- a stable, safe interface for humans and agents to control the robot and retrieve telemetry and logs

The review was static. Findings marked **confirmed** follow directly from the current code. Findings marked **validate** are high-risk behaviors that should be reproduced or measured before choosing the final implementation.

## Executive summary

The architecture has a strong base: framed and checksummed UART traffic, separate high-rate control and communications processors, telemetry optimized into two bounded packets, extensive fault telemetry, UDP/TCP separation, runtime parameter discovery, log download support, a localhost GUI-control server, and existing fault-injection hooks.

The largest risks are at ownership and transition boundaries:

1. The ESP32 has two tasks writing through the same UART and network `CommLink` objects. Their bytes and sequence counters are not serialized as one stream.
2. The ESP32 task responsible for draining the Teensy UART can wait indefinitely for a TCP mutex, so a slow client can indirectly overflow the 4 Mbaud UART.
3. Commands have neither an end-to-end transaction ID nor an ACK/NACK from the Teensy. A GUI cannot reliably distinguish accepted, rejected, duplicated, or lost commands.
4. Several active-state abort paths are missing. In particular, arm loss or a software disarm request does not abort `JUMPING` or `STANDING_UP` immediately.
5. The Teensy accepts non-finite floating-point command and parameter values, and validates several variable command payloads too loosely.
6. Shared definitions are manually copied between C++, Python, tools, and documentation. Version, state, fault, parameter, and severity drift is already present.
7. Startup is largely blocking and some readiness checks reflect “heard once” rather than a fresh, confirmed operational handshake.
8. The existing localhost control server is useful, but response races, weak success semantics, and GUI lifecycle coupling keep it from being a dependable agent API.

The recommended sequence is: first make unsafe commands rejectable and observable; then establish single ownership and bounded work on communication paths; then improve Wi-Fi discovery/performance; then harden startup/peripheral recovery; finally promote the localhost control surface into a headless, transaction-oriented robot session API.

## Deficiencies and proposed changes

### P0 — safety or data-integrity issues

| ID | Finding | Evidence | Proposed change |
| --- | --- | --- | --- |
| P0.1 | **Confirmed:** the ESP32 uplink path has two writers. `uplink_task` and `log_uplink_task` both call `send_uplink_frame()`, which writes through shared USB/UART and network transports. `CommLink::_seq_tx` and serial frame writes are not protected, so task interleaving can corrupt a frame or race a sequence number. | `firmware/robot_esp32/src/main.cpp`, `uplink_task`, `log_uplink_task`, and `send_uplink_frame`; `firmware/shared/CommLink/CommLink.cpp`, `send` | Give every physical byte stream exactly one writer. Route normal responses, diagnostics, and log chunks into prioritized queues consumed by one uplink owner. Do not solve this with a lock around potentially blocking writes. |
| P0.2 | **Confirmed:** the core-1 loop that calls `g_teensy.update()` can block indefinitely on `g_tcp_mutex` while updating or accepting a TCP client. A slow/non-reading client can therefore delay Teensy UART parsing until the hardware buffer overflows. | `firmware/robot_esp32/src/main.cpp`, TCP update/status sections using `portMAX_DELAY` | Make UART draining independent of network locks. Use a dedicated UART ingress task or a strictly bounded core loop; hand complete frames to queues. Use bounded/nonblocking socket operations and disconnect clients that exceed a send deadline. |
| P0.3 | **Confirmed:** `CommLink::send()` returns `void` and ignores the result of `Stream::write()`. Short or failed writes cannot be retried, counted, or reported. | `firmware/shared/CommLink/CommLink.h`; `firmware/shared/CommLink/CommLink.cpp` | Return a structured transmit result. Count full sends, short writes, write failures, queue drops, and maximum queue age per output. Unit-test with a fake stream that deliberately performs short writes. |
| P0.4 | **Confirmed:** arm loss/software disarm is handled only from `RUNNING`. `JUMPING` and `STANDING_UP` do not have an immediate disarm/abort transition. | `firmware/robot_teensy/src/main.cpp`, arm-level request logic; `firmware/robot_teensy/src/state_machine.cpp`, transition table | Define a global active-state abort event for `RUNNING`, `JUMPING`, and `STANDING_UP`. Give it priority immediately below hard faults/ESTOP, cancel sequencers, command safe motor outputs, and report the abort reason. Test every energetic phase. |
| P0.5 | **Confirmed:** the arming guard checks that the IMU is enabled, but not that it is currently nominal and fresh. The system can leave a degraded `STANDBY` for `RUNNING` and only fault on a following tick. | `firmware/robot_teensy/src/state_machine.cpp`, `req_running_checks_common` and `running_imu_fault` | Require a nominal, fresh IMU sample in the same guarded transition that arms the robot. Apply equivalent freshness checks to every required motor/controller. Return a specific rejection reason. |
| P0.6 | **Confirmed:** command parsing mostly checks minimum length, not an exact schema or valid enum/range. Invalid wheel modes fall through to torque behavior; invalid hip IDs can affect both motors. | `firmware/robot_teensy/src/main.cpp`, `on_command`; wheel and hip motor setters | Validate command version, exact payload length, legal enum values, target IDs, and current-state permission before causing any effect. Unknown versions or commands must produce a NACK, never a best-effort interpretation. |
| P0.7 | **Confirmed:** non-finite floats are not rejected. A NaN bypasses normal parameter range comparisons and can be persisted; NaN/Inf motor commands can reach conversions or motor buses. | `firmware/robot_teensy/src/param_registry.cpp`, `param_set`; `firmware/robot_teensy/src/main.cpp`, motor command handlers | Apply `isfinite()` to every received, loaded, and derived floating-point value at the boundary. Reject the complete command atomically and record a validation fault/counter. Validate persisted values before applying them. |
| P0.8 | **Confirmed:** re-requesting ESTOP while already in ESTOP leaves a request flag that the ESTOP state does not consume. A later reset can enter `STARTUP` and immediately return to ESTOP because of the stale request. | `firmware/robot_teensy/src/state_machine.cpp`, `stateMachine_request_estop`, ESTOP entry, and transition ordering | Replace persistent request booleans with consumed events, or explicitly ignore/clear ESTOP requests in ESTOP. Add a regression test for repeated ESTOP followed by reset. |
| P0.9 | **Confirmed:** motor feedback structures are written in CAN receive callbacks and read by the main loop without a coherent snapshot. Fields and timestamps can be torn or observed from different frames. | `firmware/robot_teensy/src/wheel_motors.cpp`; `firmware/robot_teensy/src/hip_motors.cpp` | Copy ISR data into a small ring buffer or use a brief critical section/double-buffered generation counter. Consumers must read one coherent frame plus timestamp. Keep CAN callbacks minimal. |
| P0.10 | **Confirmed:** firmware enable/watchdog parameters are described as boot-time choices but are mutable live and take effect immediately. Enabling a previously uninitialized peripheral or disabling a health gate at runtime is not a defined transition. | `firmware/robot_teensy/src/param_registry.cpp`; uses in `firmware/robot_teensy/src/main.cpp` | Add a `REBOOT_REQUIRED` parameter flag and a pending-value store. Reject or defer live changes to boot-only parameters. Make the GUI display active versus pending values. Treat watchdog activation as a controlled boot policy. |

### P1 — communication reliability and observability

| ID | Finding | Evidence | Proposed change |
| --- | --- | --- | --- |
| P1.1 | **Confirmed:** independent `CommLink` instances write to the same logical GUI stream: USB diagnostics and normal USB traffic share `Serial`; Wi-Fi diagnostics and telemetry share UDP port 5005. Each instance starts its own sequence counter. The GUI uses one decoder/counter per stream, producing false sequence-gap statistics. | `firmware/robot_esp32/src/main.cpp`, `g_usb`, `g_usb_diag`, `g_telem_udp`, and `g_wifi_diag`; `software/gui/flash_monitor.py` | Use one sequencer per physical/logical stream, or add an explicit stream/channel ID and maintain sequence state per channel. Reset sequence accounting on a new boot/session ID. |
| P1.2 | **Confirmed:** after the uplink task is started, setup still prints plain text directly to `Serial`. This can be inserted between bytes of a binary USB frame. | `firmware/robot_esp32/src/main.cpp`, task creation followed by the final ready print | Once framed transport starts, forbid all direct writes to its stream. Route boot/status text through a framed diagnostic event or finish all text output before the owner task starts. Add a lint/search check for direct writes. |
| P1.3 | **Confirmed:** there is no end-to-end command identity or Teensy ACK/NACK. ESP32 restamping and telemetry-based inference cannot prove whether a command was accepted exactly once. | `firmware/shared/comm_protocol.h`; ESP32 forwarding; GUI confirmed-command helpers | Add a versioned command envelope containing boot/session ID, request ID, command ID, and payload length. Teensy returns ACK/NACK with the same request ID, result, reason, and resulting state/value. Maintain a small deduplication cache so retrying a timed-out request is safe. |
| P1.4 | **Confirmed:** `CommLink::update()` drains all currently available bytes with no byte/time budget. A continuously readable TCP client can monopolize the ESP32 loop. | `firmware/shared/CommLink/CommLink.cpp`, `update` | Add a configurable parse budget and return progress to the scheduler. UART should receive a larger/higher-priority budget than GUI ingress. Expose high-water marks and parser starvation counters. |
| P1.5 | **Confirmed:** the GUI stream decoder combines packet fragments as a byte stream even for UDP. A truncated datagram can borrow bytes from a later datagram before CRC recovery. | `software/gui/flash_monitor.py`, `PacketDecoder.feed`; `software/gui/wifi_transport.py` | Add a datagram decoder that accepts only complete, internally consistent frames in one datagram. Keep stream resynchronization only for USB/TCP. Count bad length, CRC, end marker, and unknown version separately. |
| P1.6 | **Confirmed:** an unmatched telemetry-B packet can be emitted as a partial dictionary, and the main window treats broadly received packets as fresh telemetry. This can mask a broken A/B stream. | `software/gui/flash_monitor.py`, A/B pairing; `software/gui/main.py`, packet freshness handling | Drop incomplete/mismatched telemetry halves and count them. Refresh robot-telemetry liveness only after a complete, decoded telemetry sample. Include sample age and source in the model. |
| P1.7 | **Validate:** the log uplink task subscribes to the task watchdog and then waits on its queue with `portMAX_DELAY`. Depending on the ESP-IDF watchdog configuration, an idle log queue can cause a watchdog reset. | `firmware/robot_esp32/src/main.cpp`, `log_uplink_task` and task watchdog registration | Reproduce with logs idle. Regardless of result, use a bounded queue wait and explicit watchdog policy, or do not subscribe an indefinitely blocked task. Record reset reason and task-watchdog diagnostics across reboot. |
| P1.8 | **Confirmed:** queue, mutex, task, and some display allocation results are not checked before use. | `firmware/robot_esp32/src/main.cpp`, setup and display initialization | Check every allocation/task creation and enter an explicit degraded mode on failure. Communications must start without TFT sprites, ToF, or NeoPixel. Publish a startup-health bitmap and reason strings/codes. |
| P1.9 | **Confirmed:** CAN transmit results are generally ignored, so a command that failed to enter the CAN queue looks successful to the control code. | Teensy wheel/hip motor transmit functions | Count and surface CAN enqueue failures. Apply a threshold appropriate to each control mode, then enter a safe fault rather than continuing with stale commands. |

### P1 — state-machine correctness and transition reporting

| ID | Finding | Evidence | Proposed change |
| --- | --- | --- | --- |
| P1.10 | **Confirmed:** the startup timeout uses absolute uptime (`millis() > 2000`) rather than elapsed time since entering `STARTUP` or starting a peripheral. Blocking setup already consumes much of that budget; a later software reset has no startup grace period at all. | `firmware/robot_teensy/src/state_machine.cpp`, STARTUP transitions; blocking setup in `firmware/robot_teensy/src/main.cpp` | Store entry time for every state and use wrap-safe elapsed calculations. Give each peripheral its own deadline/retry state. Test initial boot, software reset at long uptime, and `millis()` wrap. |
| P1.11 | **Confirmed:** `SET_MODE(STANDBY)` from `RUNNING` is effectively a no-op; tooling currently needs an ESTOP/reset workaround. | `firmware/robot_teensy/src/main.cpp`, `CMD_SET_MODE`; firmware README known behavior | Define explicit commands/events for `ARM`, `DISARM`, `ESTOP`, and `RESET_ESTOP`; do not overload “set arbitrary state.” Return legal-transition metadata and NACK unsupported targets. Remove the workaround after a compatibility window. |
| P1.12 | **Confirmed:** entering `STANDBY` does not necessarily mean torque is already zero; hip torque can continue ramping down. Other operations are allowed based on the visible state while disarming is still in progress. | `firmware/robot_teensy/src/state_machine.cpp`, `on_standby` ramp logic | Add an explicit `DISARMING` state or reported substate. Block re-arm, manual/calibration entry, and other incompatible operations until safe output is confirmed. Define the emergency path separately from the normal ramp. |
| P1.13 | **Confirmed:** requests are mostly booleans and guarded transitions silently consume, ignore, or defer intent without a reason visible to the caller. | `firmware/robot_teensy/src/state_machine.cpp`; GUI mode/parameter confirmation logic | Centralize events and guards. For each request, publish `from`, `requested action`, `accepted/rejected`, `reason`, and `actual state`, tied to the request ID. Maintain a short transition/event history in logs. |
| P1.14 | **Confirmed:** normal jump completion uses a fixed overall timer while its phase durations are separately tunable. These are two definitions of completion and can drift. | `firmware/robot_teensy/src/state_machine.cpp`, jump transitions and sequencer | Transition on sequencer completion. Retain a separately computed/configured hard watchdog as the abnormal path, with a distinct timeout fault. Apply the same pattern to stand-up. |
| P1.15 | **Confirmed:** state-machine documentation does not match active behavior in several places, including disarm behavior during energetic sequences; some linked design documents are missing or stale. | `firmware/robot_teensy/state_machine.md`; `firmware/robot_teensy/README.md` | Generate the state/event table and diagram from the canonical transition definition. Treat checked-in generated docs as a CI-verified artifact. Remove or repair dead links. |

### P1/P2 — Wi-Fi reliability and performance

| ID | Finding | Evidence | Proposed change |
| --- | --- | --- | --- |
| P1.16 | **Confirmed:** GUI liveness is based on whether neither UDP nor TCP is readable. Continuous TCP telemetry/readability can mask a dead UDP telemetry path indefinitely. | `software/gui/wifi_transport.py`, receive loop and timeout handling | Track UDP telemetry age, TCP command-channel age, and connection state independently. Display “commands connected, telemetry stale” as a first-class condition. |
| P1.17 | **Confirmed:** telemetry is sent over UDP and also copied onto TCP even though the GUI suppresses TCP telemetry. This consumes bandwidth and can delay command responses and log data behind telemetry. | `firmware/robot_esp32/src/main.cpp`, `send_uplink_frame`; `software/gui/wifi_transport.py` | Route high-rate telemetry only over UDP. Reserve TCP for command transactions, small events, and log transfer. Give command ACKs priority over bulk log traffic. |
| P1.18 | **Confirmed:** the unicast telemetry address is compiled/configured statically, so a host DHCP change or a different GUI computer requires reflashing/config changes. | `firmware/robot_esp32/config.h`; initialization in `firmware/robot_esp32/src/main.cpp` | Use low-rate broadcast or mDNS only for discovery, then learn a time-limited unicast telemetry endpoint from an explicit GUI session claim. Expire it on disconnect/timeout and permit controlled takeover. |
| P1.19 | **Confirmed:** a newly connecting TCP client silently replaces the current client. There is no session ownership, takeover policy, or rate limit. | `firmware/robot_esp32/src/main.cpp`, TCP accept logic | Add a session ID and owner lease. Reject a second client or require explicit takeover; close the displaced socket deliberately. Even on a trusted LAN, use a random per-boot/session token to prevent accidental control collisions. |
| P1.20 | **Validate:** socket writability before a `WiFiClient` write does not guarantee a bounded write, and the current path does not handle partial writes. | ESP32 TCP stream wrapper and shared `CommLink::send` | Use bounded nonblocking sends with per-client output buffers. Disconnect stalled clients and preserve UART ingress. Enable `TCP_NODELAY` for the accepted command socket and measure command RTT before/after. |
| P2.1 | **Confirmed:** split A/B telemetry doubles datagram opportunities for loss and pairing failure even though the combined payload is well below a normal Ethernet/Wi-Fi MTU. | `firmware/shared/comm_protocol.h`; ESP32 combined-telemetry option | Prefer one combined UDP telemetry datagram on Wi-Fi while retaining split packets for constrained/debug transports if needed. Version and test both during migration. |
| P2.2 | **Confirmed:** there is no explicit backpressure policy across telemetry, diagnostics, command replies, and log bulk data. Queue-full behavior is not fully visible to the GUI. | ESP32 uplink/log queue handling | Define traffic classes: safety/event and command ACK, diagnostics, telemetry latest-value, and bulk logs. Use bounded queues, replace-old policy for telemetry, never silently drop command ACKs, and report drops/high-water marks. |

### P1/P2 — startup and peripheral robustness

| ID | Finding | Evidence | Proposed change |
| --- | --- | --- | --- |
| P1.21 | **Confirmed:** wheel readiness is largely based on having heard encoder data once; state/error confirmation is separate. Local error fields can be cleared before a fresh controller heartbeat proves the clear, and requested closed-loop mode is not positively acknowledged before use. | `firmware/robot_teensy/src/wheel_motors.cpp`; startup checks | Implement a per-axis startup FSM: bus present, fresh heartbeat and encoder, error-clear observed, closed-loop request sent, closed-loop confirmed, then torque permission. Include retry limits, age, and failure reason in telemetry. |
| P1.22 | **Confirmed:** ESP32 ToF initialization leaves a failed sensor enabled at its default I2C address, which can collide with later sensors. Runtime readings have no age/validity and stale values can be forwarded indefinitely. | `firmware/robot_esp32/src/main.cpp`, ToF initialization and polling | Hold a failed sensor in XSHUT low before initializing the next sensor. Add per-sensor init/retry states, valid/age fields, stale invalidation, and bounded runtime recovery. Never use stale distance for future safety logic. |
| P1.23 | **Confirmed:** parameter persistence has no record CRC/generation and replaces the old file non-atomically; writes are not fully checked. Mount failure may format storage. Corrupt non-finite values can survive simple range checks. | `firmware/robot_teensy/src/param_registry.cpp`, load/save/mount paths | Use two slots or temp+atomic rename with magic, schema version, length, generation, and CRC. Validate every decoded value including finiteness. Fall back to defaults with a persistent recovery event; do not autoformat on a single unexplained mount failure. Add power-cut tests. |
| P1.24 | **Confirmed:** Teensy setup performs long blocking animation, IMU initialization, and waits before normal communication/state processing begins. | `firmware/robot_teensy/src/main.cpp`, setup | Convert startup to staged, nonblocking work while outputs remain safe. Service CommLink and the watchdog throughout. Emit boot stage/progress and per-peripheral status so a GUI can diagnose a stuck startup. |
| P1.25 | **Validate/policy:** the hardware watchdog defaults disabled while library/peripheral calls and startup code can block. Reset cause and reboot-loop behavior are not elevated to the control protocol. | watchdog parameter/default and setup code | Define development and production boot profiles. In production, arm the watchdog early enough to cover startup after safe parameter recovery. Persist reset cause and implement a boot-loop safe mode that keeps motors disabled but communications available. |
| P2.3 | **Confirmed:** fixed inter-frame busy delays are used around CAN sends, consuming part of the 500 Hz control budget. | Teensy CAN transmit paths and `CAN_INTER_FRAME_US` | Replace busy waits with per-bus scheduled/queued transmissions. Measure bus utilization, command age, loop maximum, and missed deadlines under worst-case telemetry and logging. |
| P2.4 | **Confirmed:** optional ESP32 peripherals can consume enough resources to threaten the communications bridge, but degraded-mode priority is implicit. | ESP32 TFT sprite/display, ToF, and task setup | Start and protect Teensy UART plus network control first. Initialize TFT/NeoPixel/ToF afterward. If memory or initialization fails, disable that subsystem while keeping the bridge operational and reporting health. |

### P1 — single source of truth

Current definitions are duplicated across at least:

- `firmware/shared/comm_protocol.h`
- `firmware/robot_teensy/src/robot_state.h`
- ESP32 local state/fault/display definitions
- `software/gui/telem_format.py`
- `software/gui/comm_commands.py`
- GUI fault severity, descriptions, state colors, and parameter descriptions
- standalone tools and README tables

Drift is already visible: documentation names older telemetry versions/sizes; state IDs and labels are copied; GUI severity coverage does not match all canonical fault actions; command IDs and CRC code are repeated; and parameter metadata is partly re-described in GUI code. `CommandPayload` also suggests a fixed command size even though real commands are variable length.

Proposed canonical protocol layout:

```text
protocol/
  robot_protocol.yaml       packet IDs, envelope, states, faults, actions, command schemas
  parameters.yaml           IDs, type, range, flags, units, descriptions, default policy
  generate.py
  generated/
    comm_protocol_generated.h
    robot_protocol_generated.py
    protocol.md
    test_vectors.json
```

Rules for the generator and migration:

1. Human-edited schema files are the only place IDs, types, lengths, severities, units, and enum values are assigned.
2. Generated C++ and Python files are checked in so embedded and GUI builds do not require the generator at build time.
3. CI runs the generator in `--check` mode and fails on drift, duplicate IDs, invalid lengths, or changed IDs without a protocol-version bump.
4. Golden encoded frames are decoded by both C++ and Python tests; Python-generated frames are decoded by C++ tests and vice versa.
5. Runtime parameter discovery remains available, but its metadata is generated from the same schema. The GUI does not maintain a competing registry.
6. State-machine transition definitions should likewise generate a documentation table/diagram, even if transition guards remain hand-written C++.
7. Protocol v2 should add session/request IDs, ACK/NACK, exact command schemas, stream/channel identity, and boot/reset identity. Run v1/v2 compatibility only for a bounded migration period.
8. A stronger CRC (CRC-16 or CRC-32) can be evaluated for v2, but it is secondary to single-writer ownership, exact validation, acknowledged commands, and measured recovery.

### P1/P2 — dependable GUI and agent control

The existing localhost server in `software/gui/tabs/remote_control.py` and `tools/robot_ctl.py` is a valuable foundation. It already exposes parameter, mode, logging, telemetry, and motion operations. It should be hardened rather than replaced immediately.

| ID | Finding | Proposed change |
| --- | --- | --- |
| P1.26 | Parameter get/set waiters are attached after sending, so a fast reply can arrive before the listener exists. Set confirmation matches an ID but not necessarily the expected value. | Register a request-ID waiter before send. Match the transaction, type, parameter ID, status, and returned value. Remove nested event-loop races by using one serialized asynchronous command dispatcher. |
| P1.27 | Several API operations return “success” when bytes were queued, not when Teensy accepted the operation. Transport send exceptions can be swallowed. | Return structured transaction results from the Teensy ACK/NACK. Distinguish `not_sent`, `transport_lost`, `timed_out`, `rejected`, and `applied`. |
| P1.28 | The client-facing state list includes states that are not valid direct command targets; firmware silently ignores some requests. | Publish capabilities and currently legal actions from the session/state model. Expose verbs such as arm/disarm/estop/reset rather than arbitrary state assignment. |
| P1.29 | A telemetry snapshot has weak freshness/session semantics. | Include sample timestamp and age, active source, ESP32 and Teensy boot IDs, link health, pending request/transition, state/substate, fault/action, queue drops, parser counters, and log status. Add `health` and `wait_for` endpoints. |
| P1.30 | Normal GUI close does not consistently stop/join the Wi-Fi worker, while automation has special shutdown handling. | Give transports explicit lifecycle ownership and always stop/join them on close, source replacement, and test teardown. Add a repeated open/close regression test. |
| P1.31 | A forced source can be selected while disconnected, after which commands may be silently dropped. | Make source availability part of command admission. Reject commands with a clear reason if the selected source is unavailable; surface automatic versus forced selection and source age. |
| P2.5 | The control server is coupled to a visible Qt process and accepts overlapping local clients/reentrant operations without a clear lease. | First serialize requests and enforce one control lease. Then extract transport/protocol/log ownership into a headless `RobotSession` service/library; make the GUI and agent CLI clients of the same stable localhost API. |
| P2.6 | Logs can be listed/downloaded, but live diagnosis is not a structured, queryable stream. | Add structured log tail/subscribe with monotonic timestamp, source, level, event ID, request ID, state transition, and bounded history. Retain binary file download for full-rate logs. |
| P1.32 | The proposed API does not yet guarantee access to every robot-related field and action exposed by the GUI. An agent could still encounter a GUI-only control or status value and require human transcription/clicking. | Maintain a machine-readable GUI parity manifest. Every robot-related field must map to a discoverable session property and every robot-related button/menu action must map to a typed API operation, or be explicitly classified as presentation-only. CI fails when a new relevant widget lacks a mapping. |
| P1.33 | Connection selection, process lifecycle, firmware/build identity, configuration, and recovery workflows are still partly operator-driven. | Add autonomous port/network discovery, connect/disconnect/reconnect, process start/restart, source selection, firmware/build information, configuration import/export, and supported recovery actions. Return progress and final results as structured operations. |
| P2.7 | Some visual-only behavior may not have a useful semantic representation, and future dialogs could temporarily precede API support. | Add a GUI automation fallback with stable accessibility names/test IDs, widget-tree inspection, screenshots, dialog discovery, and guarded action invocation. Use this only as a fallback; the semantic session API remains authoritative for robot state and commands. |
| P2.8 | Full software-operator replacement needs goals and workflows, not only individual primitive commands. | Add an automation runner for declarative, cancellable sequences with preconditions, timeouts, retries, assertions, cleanup, dry-run/torque-disabled modes, and a complete execution transcript. Provide standard workflows for connect, preflight, arm/disarm, parameter verification, logging, diagnostics, and safe shutdown. |

### Software-operator replacement requirement

Phase 0 must provide full software-side operational parity. After the human powers, connects, positions, and physically supervises the robot, an authorized agent must be able to perform every remaining routine operation without asking the human to copy/paste information, click a GUI control, select a tab, or run a command.

This requirement includes:

- Discover and connect to USB and Wi-Fi endpoints; select, switch, and recover transports.
- Start, stop, restart, and inspect the GUI/headless service and report process/connection failures.
- Read every robot-related value visible anywhere in the GUI, including hidden tabs, dialogs, plots, indicators, derived values, configuration, and status bars.
- Retrieve both the numeric data behind graphs and a rendered screenshot when visual layout or appearance matters.
- Enumerate every available field/action with name, type, units, valid range/options, current value, source, timestamp, age, validity, permissions, and safety classification.
- Invoke every robot-related button, menu action, keyboard operation, and workflow through a typed semantic operation with an acknowledged result.
- Change parameters and configuration, compare active/pending/persisted values, save/load profiles, and determine whether a reboot is required.
- Start, stop, list, tail, search, download, decode, and summarize logs without human file transfer.
- Inspect build, firmware, protocol, reset-cause, peripheral-health, and compatibility information.
- Run preflight checks, fault diagnostics, recovery procedures, calibration/manual-mode workflows, and safe shutdown sequences subject to the same interlocks as the GUI.
- Capture screenshots and operate the GUI through stable accessibility/test hooks when a truly visual or GUI-local task has no semantic equivalent.
- Produce an audit transcript containing observations, decisions, commands, acknowledgements, state transitions, timeouts, and cleanup actions.

The GUI must become a client of the same `RobotSession` model used by agents. Widget scraping must not be the primary way to obtain robot state: a displayed value and its API value must originate from the same model object. A generated parity report should list each relevant GUI control, its model property or action, API endpoint, permission/safety class, and automated coverage.

Operations that remain human responsibilities are deliberately limited to physical-world authority: supplying/removing power, plugging or repairing hardware when software cannot do so, positioning or restraining the robot, clearing the test area, supervising hazardous motion, granting control authorization, and using the physical ESTOP when communications are unavailable. These boundaries must not be used to hide a missing software capability.

Safety policy for the agent interface:

- Bind locally by default and require an explicit control lease.
- Default new clients to read-only; allow ESTOP without an arm lease.
- Require an operator-controlled hardware/software arm condition before torque-producing commands.
- Give motion commands a short TTL and automatic release/dead-man behavior.
- Audit actor/session, request ID, command, result, and transition in a durable log.
- Expose typed operations; do not expose unrestricted raw frames in the normal control API.
- Provide a simulator/fake transport so agents can exercise the full API without hardware.

Suggested minimal API surface:

```text
service.status / service.start / service.stop / service.restart
discovery.scan / connection.list / connection.connect / connection.disconnect
connection.select_source / connection.reconnect
health
capabilities.describe
ui.parity_report / ui.snapshot / ui.screenshot / ui.invoke_fallback
telemetry.snapshot
telemetry.subscribe
state.capabilities
state.arm / state.disarm / state.estop / state.reset_estop
parameter.list / parameter.get / parameter.set / parameter.compare
configuration.export / configuration.import / configuration.persist
firmware.info / protocol.info / peripheral.health
log.start / log.stop / log.list / log.download / log.tail / log.search
motion.acquire_lease / motion.set / motion.release
request.status / wait_for
workflow.list / workflow.validate / workflow.run / workflow.cancel / workflow.report
```

## Implementation phases

### Phase 0 — complete software-operator replacement

1. Inventory every robot-related field, plot, indicator, dialog, button, menu action, keyboard operation, connection control, configuration operation, and diagnostic workflow in the GUI. Classify presentation-only controls explicitly.
2. Create a checked GUI parity manifest that maps each relevant widget to its `RobotSession` property/action, API operation, permission/safety class, and automated test. Add a CI check that rejects unmapped additions.
3. Fix races and success semantics in the existing localhost control server.
4. Add typed health, capability discovery, transaction, transition, telemetry subscription, configuration, firmware identity, peripheral health, and structured log endpoints.
5. Extract a headless `RobotSession` layer shared by GUI and CLI/agents. Refactor GUI widgets to consume this model so GUI and API values cannot diverge.
6. Add autonomous USB/Wi-Fi discovery, connection/source management, reconnect/restart, compatibility checks, and structured recovery operations.
7. Add declarative workflows with preconditions, assertions, timeouts, cancellation, retries restricted to safe/idempotent operations, cleanup, and durable transcripts.
8. Add stable accessibility names/test IDs, widget-tree inspection, screenshots, and guarded GUI action invocation as a visual fallback for presentation-only or temporarily unmapped behavior.
9. Add leases, TTL/dead-man behavior, audit records, read-only defaults, explicit control authorization, and physical-safety gates.
10. Run parity and integration tests first against a simulator, then on a torque-disabled bench, then with normal hardware interlocks.

Exit criterion: after a human supplies power, positions the robot, clears the test area, and grants control authority, an agent can autonomously discover/connect, inspect every software-visible field, obtain graph data and screenshots, invoke every robot-related GUI operation, run and verify multi-step workflows, diagnose/recover supported failures, retrieve/tail/search logs, and shut down safely. The parity manifest has 100% coverage, all results are acknowledged and auditable, and no copy/paste, manual GUI clicking, or human-run software command is required.

Phase 0 must not wait for the later protocol rewrite. Build its first implementation on adapters around the current GUI, transports, and localhost server, with accessibility-driven GUI control as the fallback. Keep the `RobotSession` interface stable while later phases replace inferred results with authoritative request-ID ACK/NACK responses and improve transport internals.

### Phase 1 — freeze behavior and add tests/metrics

1. Capture current protocol golden frames from Teensy, ESP32, and GUI implementations.
2. Add host-native tests for `CommLink` noise, truncation, bad length/end/CRC, short writes, sequence wrap, and mid-frame reboot.
3. Add table-driven FSM tests before changing transitions.
4. Add counters for UART receive overflow, parser errors/timeouts, queue drops/high-water age, short writes, TCP stalls, UDP packet loss/pair loss, CAN send failures, control-loop maximum, and boot/reset cause.
5. Build a GUI/ESP32 traffic harness that can delay reads, drop/corrupt/reorder datagrams, reconnect sockets, and reboot either endpoint.

Exit criterion: current behavior is reproducible and failures can be attributed to a specific layer rather than inferred from missing telemetry.

### Phase 2 — command safety and state transitions

1. Reject non-finite values and invalid lengths/enums/targets immediately.
2. Add request IDs and ACK/NACK while preserving v1 telemetry during migration.
3. Add active-state abort/disarm, nominal/fresh arming guards, consumed events, and explicit transition rejection reasons.
4. Model normal torque ramp-down as `DISARMING`; separate it from immediate ESTOP output behavior.
5. Add exhaustive FSM regression tests and update generated state documentation.

Exit criterion: every command has one observable result, every unsafe payload is rejected without side effects, and every active state has a tested safe abort path.

### Phase 3 — single-owner UART and ESP32 scheduling

1. Establish one reader and one writer for each ESP32 serial/network stream.
2. Move complete frames between bounded, prioritized queues; remove direct post-start serial writes.
3. Prevent UART work from taking network mutexes or waiting for network I/O.
4. Add bounded parser work and transmit deadlines, including stalled-client disconnect.
5. Fix sequence ownership/channel identity and verify counters through reboots and wrap.

Exit criterion: UART remains loss-free under simultaneous telemetry, parameter dumps, display updates, Wi-Fi reconnects, and maximum-rate log transfer.

### Phase 4 — Wi-Fi session, discovery, and traffic policy

1. Separate UDP telemetry liveness from TCP command liveness.
2. Use combined telemetry datagrams and stop duplicating telemetry on TCP.
3. Add discovery followed by an explicit unicast session lease.
4. Implement traffic priorities and bounded output buffers; enable low-latency TCP settings.
5. Test AP outage, DHCP/IP change, GUI restart, second-client collision, packet loss/reorder, and a non-reading TCP client.

Exit criterion: Wi-Fi reconnect and latency/loss targets below are met without affecting Teensy UART or control-loop deadlines.

### Phase 5 — generated protocol and durable configuration

1. Introduce the schema/generator and make current IDs a compatibility baseline.
2. Generate C++, Python, docs, and cross-language test vectors.
3. Move parameter metadata and fault/state/action definitions to the schema.
4. Upgrade parameter persistence to an atomic CRC-protected format with recovery reporting.
5. Remove obsolete duplicate constants only after all consumers use generated artifacts.

Exit criterion: CI proves generated files and documentation match the schema, and a power interruption cannot leave parameters silently corrupted.

### Phase 6 — staged peripheral startup and recovery

1. Replace blocking setup waits with explicit subsystem startup FSMs.
2. Add confirmed motor-controller readiness, coherent CAN snapshots, send-failure handling, and runtime age checks.
3. Correct ToF XSHUT/address sequencing and add per-sensor retry/staleness.
4. Define production watchdog, reset-cause, and boot-loop safe-mode policy.
5. Prove that each optional peripheral can be absent or fail without disabling communications or producing torque.

Exit criterion: faults are deterministic and diagnosable for each missing, stuck, stale, or recovering peripheral.

## Acceptance tests and target metrics

Targets should be adjusted after a baseline run, but they must be explicit before release.

### Teensy–ESP32 UART

- Run at 4 Mbaud for at least 8 hours with telemetry, diagnostics, parameter dumps, Wi-Fi churn, display work, and log transfer active.
- Zero corrupted frames accepted as commands and zero unexplained command executions.
- Zero hardware UART receive overflows in the soak test.
- Recover from injected noise, truncated frames, invalid length, bad CRC/end byte, and either processor reboot by the next valid frame or a documented bounded resync interval.
- Reboot either endpoint at every possible byte offset of a representative maximum-length frame.
- Demonstrate exactly-once observable command results under dropped ACKs and safe retries.
- Keep ESP32 queue age and Teensy control-loop maximum below defined budgets; alert on any missed 500 Hz deadline.

### ESP32–GUI Wi-Fi

- Measure UDP telemetry loss, out-of-order count, sample age, command RTT p50/p95/p99, reconnect time, and queue high-water marks.
- Initial targets on a healthy local network: telemetry loss below 0.1%, command ACK p99 below 100 ms, telemetry age p99 below 50 ms, and recovery after AP/host reconnect below 3 seconds.
- Repeat with log download, a non-reading TCP client, a second connecting client, 5% injected UDP loss, burst reordering, DHCP/IP change, and GUI restart.
- Prove that loss or congestion on TCP cannot cause Teensy UART overflow.
- Prove that dead UDP is reported even while TCP remains connected/readable.

### State machine

- Table-test every legal and illegal `(state, event)` pair.
- Test simultaneous events using documented priority: ESTOP/fault, disarm/abort, completion, operator request.
- Test disarm in every phase of run, jump, and stand-up.
- Test repeated ESTOP, reset at long uptime, startup timeout from state entry, timer wrap, stale IMU/motor feedback, and a command arriving during `DISARMING`.
- Assert motor-output invariants at every transition, not only final state.

### Startup/peripherals

- Boot with each IMU, motor controller, SD card, ToF sensor, display allocation, and Wi-Fi connection absent/stuck independently.
- Test delayed appearance and recovery where supported.
- Interrupt every parameter-save step and verify either the previous or new complete record loads.
- Verify a boot-loop safe mode preserves communications and prevents torque.
- Verify reported health includes stage, retry count, age, and final reason.

### GUI/agent interface

- Run concurrent telemetry subscriptions, parameter transactions, mode transitions, and log transfer without response cross-matching.
- Close/reopen the GUI and headless session repeatedly with no orphan threads or sockets.
- Verify disconnected/stale sources reject commands explicitly.
- Verify lease expiry or client death releases motion within its TTL.
- Verify every accepted/rejected command is correlated in live and stored logs by request ID.
- Generate a parity report showing 100% mapping of robot-related GUI fields and actions to shared model properties and typed operations; fail CI for any unmapped relevant widget.
- For every displayed field, compare the GUI value and metadata with the headless API value from the same model update.
- Exercise every robot-related button/menu action through the semantic API and verify the same state change, acknowledgement, errors, and interlocks as direct GUI activation.
- Verify that plot source data, displayed time ranges, markers, and screenshots can be retrieved without opening or reading the GUI manually.
- Start from only “hardware powered and physically ready”; have an agent discover/connect, run preflight, collect status, start logging, execute an authorized torque-disabled workflow, diagnose an injected fault, recover, download logs, and disconnect with no human software interaction.
- Repeat the operator-replacement test with USB loss, Wi-Fi loss, GUI/service restart, stale telemetry, an incompatible protocol version, and a peripheral degraded at startup.
- Assert that no test step requires copied text, a human-run terminal command, a manual tab/dialog selection, or an unreported visual interpretation.
- Verify visual fallback by enumerating the widget tree, opening each supported dialog, capturing screenshots, and safely invoking presentation-only controls through stable accessibility/test identifiers.

## Recommended first change set

Bootstrap Phase 0 first so subsequent development and hardware validation can be driven directly by an agent:

1. Generate the initial GUI field/action inventory and parity manifest.
2. Give every relevant Qt widget a stable accessibility name/test ID and add widget-tree inspection, screenshots, and guarded invocation as an immediate full-GUI fallback.
3. Extend the localhost server with capability discovery, health, complete telemetry/model snapshots, connection/source control, structured results, and access to existing log and command workflows.
4. Provide one agent CLI/client that can launch or attach to the GUI, discover/connect to the robot, inspect fields, invoke actions, wait for results, retrieve logs, capture screenshots, and produce an audit transcript.
5. Add the control lease, dead-man/TTL, read-only default, ESTOP exception, and explicit torque authorization before enabling motion operations.
6. Prove the bootstrap on the simulator and torque-disabled bench with an end-to-end no-copy/paste and no-human-GUI-interaction test.

After this bootstrap is usable, continue Phase 0 by extracting the shared `RobotSession` model and eliminating GUI-only paths. Then begin the reliability work with protocol baselines/metrics followed by exact command validation, FSM safety tests, and request-ID ACK/NACK. Do not wait for the full protocol generator or storage migration before delivering operator access.

## Compatibility and rollout notes

- Treat on-wire changes as a new protocol version; reject incompatible versions explicitly.
- During a short transition, the ESP32 may translate selected v1 GUI requests into v2 transactions, but the Teensy should have one canonical internal command path.
- Add ESP32 and Teensy boot/session IDs before interpreting sequence counters as loss across reconnects.
- Keep a torque-disabled bench profile and simulation mode for automated integration tests.
- Update README and operator procedures in the same change that alters state semantics; do not let documentation lag behind firmware behavior.
- Gate each phase on its tests and metrics rather than landing all architectural changes at once.
