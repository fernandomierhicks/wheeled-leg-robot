# Firmware Framework Plan — Wheeled-Leg Balancing Robot

## Context

The robot's control system is fully designed and tuned in Python/MuJoCo simulation (500Hz cascaded LQR + VelocityPI + YawPI + Hip). No firmware exists yet — `firmware/src/` contains only `.gitkeep`. This plan ports the proven simulation controllers to C++ on the Arduino UNO R4 WiFi, with WiFi telemetry/commands and OTA support, in incremental phases that allow hardware validation at each step.

**MCU**: Arduino UNO R4 WiFi (RA4M1 @ 48MHz, 128KB usable flash with OTA, 32KB SRAM)
**Architecture**: Single-MCU superloop (no RTOS), timer-driven 500Hz tick, non-blocking WiFi

---

## File Structure

```
firmware/
  platformio.ini              (extend existing — add OTA env, WiFi + OTA libs)
  src/
    main.cpp                  (setup + loop — timing skeleton, calls modules)
    config.h                  (ALL constants: pins, gains, geometry, timing)
    robot_state.h             (central mutable state struct ~120 bytes)
    imu.h / imu.cpp           (BNO086 SPI init + poll + quat→pitch/roll)
    can_bus.h / can_bus.cpp   (CAN 1Mbps init, tx/rx helpers)
    odrive_can.h / odrive_can.cpp   (ODESC wheel torque cmd + encoder read)
    ak45_can.h / ak45_can.cpp      (MIT CAN hip torque cmd + feedback)
    lqr.h / lqr.cpp           (gain-scheduled 3-state balance controller)
    velocity_pi.h / velocity_pi.cpp (velocity → lean angle)
    yaw_pi.h / yaw_pi.cpp     (yaw rate → differential torque)
    hip_ctrl.h / hip_ctrl.cpp  (position PD + impedance + roll leveling)
    state_machine.h / state_machine.cpp (IDLE/BALANCE/DRIVE/JUMP/STAND_UP/FAULT)
    telemetry.h / telemetry.cpp (UDP binary send @ 50Hz)
    commands.h / commands.cpp   (UDP binary receive @ 10Hz)
    wifi_mgr.h / wifi_mgr.cpp  (WiFi connect + OTA init/poll)
    safety.h / safety.cpp      (fall detect, watchdogs, torque clamps)
```

Estimated: ~80KB flash (63%), ~10KB SRAM (31%) — comfortable margins.

---

## Main Loop Architecture

```
setup():
    wifi_init() + ota_init()        // blocking WiFi connect, ~3s one-time
    imu_init()                      // BNO086 SPI, enable GRV+Gyro @ 500Hz
    can_init()                      // 1Mbps
    timer_init(500Hz)               // hardware timer ISR sets tick_flag

loop():
    ArduinoOTA.poll()               // ~50us, non-blocking
    if (!tick_flag) return          // spin until 500Hz tick
    tick_flag = false

    // HARD REAL-TIME (~265us) ────────────────────────
    imu_poll(&state)                // ~100us SPI
    can_rx_poll(&state)             // ~50us  parse feedback
    if (mode >= BALANCE):
        velocity_pi_update(&state)  // ~5us
        lqr_update(&state)          // ~10us
        yaw_pi_update(&state)       // ~5us
        hip_ctrl_update(&state)     // ~10us
    can_tx_commands(&state)         // ~80us (3 motors)
    safety_check(&state)           // ~5us

    // SOFT REAL-TIME (remaining budget) ──────────────
    if (tick % 10 == 0) telemetry_send(&state)    // 50Hz, ~200us
    if (tick % 50 == 0) commands_receive(&state)   // 10Hz, ~100us
```

**Timing budget**: 2000us per tick, ~265us hard RT, ~1335us margin (67%).

---

## Controller Ports (from simulation → C++)

### LQR (`lqr.cpp`) — from `master_sim/controllers/lqr.py`
- **Pre-compute** K gains at Q_RET/Q_NOM/Q_EXT in Python, embed 9 floats in `config.h`
- **Pre-compute** equilibrium pitch at same 3 points, embed 3 floats
- Runtime: 3 linear interpolations by hip_q_avg + 3-element dot product
- `tau_sym = -(K[0]*(pitch - pitch_ff - theta_ref) + K[1]*pitch_rate + K[2]*(wheel_vel - v_ref))`
- Create `tools/export_gains.py` to automate extraction from simulation params

### VelocityPI (`velocity_pi.cpp`) — from `master_sim/controllers/velocity_pi.py`
- PI + feedforward: `theta_ref = Kp*v_err + Ki*integral + Kff*dv_cmd_dt`
- Anti-windup clamp, rate limiter, integrator reset on direction reversal
- Default gains: Kp=0.2, Ki=0.1, Kff=0.108, theta_max=0.5rad

### YawPI (`yaw_pi.cpp`) — from `master_sim/controllers/yaw_pi.py`
- `tau_yaw = Kp*yaw_err + Ki*integral`, clamped to +/-0.5Nm
- Split: `tau_L = tau_sym - tau_yaw`, `tau_R = tau_sym + tau_yaw`

### Hip Control (`hip_ctrl.cpp`) — from `master_sim/controllers/hip.py`
- **Position PD**: `tau = Kp*(q_target-q) + Kd*(0-dq)`, clamp 7Nm
- **Impedance**: `tau = -(K_s*(q-q_nom) + B_s*dq)`, clamp 5Nm
- **Roll leveling**: differential offsets `delta_q = K_roll*roll + D_roll*roll_rate`

---

## CAN Bus Layout

| Device | CAN ID | Protocol | Commands | Feedback |
|--------|--------|----------|----------|----------|
| AK45-10 Left Hip | 1 | MIT CAN | torque (pure: Kp=0, Kd=0) | pos, vel, current |
| AK45-10 Right Hip | 2 | MIT CAN | torque | pos, vel, current |
| ODESC Axis 0 (Left Wheel) | node 3, axis 0 | ODrive CAN | torque (0x0E) | encoder (0x09) |
| ODESC Axis 1 (Right Wheel) | node 3, axis 1 | ODrive CAN | torque (0x0E) | encoder (0x09) |

---

## State Machine

**Modes**: `IDLE → BALANCE → DRIVE → (JUMP, STAND_UP future) → FAULT`
**Gain presets within DRIVE**: Conservative (50%), Normal (100%), Aggressive (150%)

**Transitions**:
- `IDLE → BALANCE`: command + IMU healthy + |pitch| < 15deg
- `BALANCE → DRIVE`: v_cmd != 0
- `DRIVE → BALANCE`: v_cmd == 0 for 500ms
- `ANY → FAULT`: |pitch| > 45deg, IMU timeout 10ms, CAN timeout 20ms
- `FAULT → IDLE`: explicit reset command

---

## Telemetry (UDP, 50Hz, binary)

61-byte packed struct: timestamp, mode, pitch, pitch_rate, roll, wheel_vel, v_cmd, theta_ref, tau_sym, tau_yaw, tau_wheel_L/R, hip_q_avg, tau_hip_L/R, dt_us.

## Commands (UDP, 10Hz, binary)

Command types: drive (v, omega, hip), mode_change, gain_update (by ID), ping.

---

## platformio.ini Changes

Add to existing file:
- `lib_deps`: add `jandrassy/ArduinoOTA@^1.1.1`, `WiFiS3` (built-in)
- New `[env:ota]` section with custom upload protocol (from OTA characterization)
- Keep existing build flags (`LOOP_RATE_HZ=500`, `CAN_BAUD=1000000`, `IMU_RATE_HZ=500`)

---

## Phased Implementation (each phase = one conversation)

### Phase 0 — Skeleton + OTA
- `main.cpp`, `config.h`, `wifi_mgr.cpp`: timer ISR, tick_flag superloop, WiFi+OTA
- **Verify**: 500Hz LED toggle on scope, OTA reflash works

### Phase 1 — IMU
- `imu.cpp`, `robot_state.h`: BNO086 SPI init, GRV + gyro, quat→pitch/roll
- **Verify**: serial print pitch while tilting board

### Phase 2 — CAN Motors
- `can_bus.cpp`, `odrive_can.cpp`, `ak45_can.cpp`: init, zero-torque keepalive, parse feedback
- **Verify**: read encoders, command 0.1Nm, verify motion

### Phase 3 — Static Balance (critical milestone)
- `lqr.cpp`, `hip_ctrl.cpp`, `state_machine.cpp`, `safety.cpp`, `tools/export_gains.py`
- **Verify**: robot balances in place

### Phase 4 — Velocity + Yaw
- `velocity_pi.cpp`, `yaw_pi.cpp`
- **Verify**: drive forward/back, turn, via serial commands

### Phase 5 — WiFi Telemetry + Commands
- `telemetry.cpp`, `commands.cpp`
- **Verify**: drive via WiFi, plot telemetry from Python script

### Phase 6 — Impedance + Roll Leveling
- Update `hip_ctrl.cpp` with impedance mode
- **Verify**: legs adapt on tilted surface

### Phase 7 — Polish
- Gain presets, runtime gain tuning, watchdogs, LED status, self-test, error recovery

---

## Future Path: ESP32-S3 as Main MCU

> **Status**: Not planned — documenting for future reference if WiFiS3 latency
> becomes a blocker on the UNO R4 WiFi.

### Why Consider This

The UNO R4 WiFi's WiFiS3 library talks to the on-board ESP32-S3 coprocessor via
an AT-command-style UART bridge. Each `endPacket()` call **blocks 1–5 ms** waiting
for the ESP32 ACK, which can blow the 2000 µs tick budget. The RA4M1 is single-core
(Cortex-M4, 48 MHz) so there's no way to isolate WiFi from the control loop at the
hardware level.

### What Changes

Replace the UNO R4 WiFi with an **ESP32-S3 dev board** (e.g. ESP32-S3-DevKitC-1).

| Feature | UNO R4 WiFi (RA4M1) | ESP32-S3 |
|---|---|---|
| Cores | 1 (Cortex-M4, 48 MHz) | 2 (Xtensa LX7, 240 MHz) |
| WiFi | AT bridge to coprocessor (1–5 ms blocking) | Native, zero-copy, core-pinnable |
| RTOS | None (bare-metal superloop) | FreeRTOS built-in |
| CAN | Built-in CAN peripheral | TWAI peripheral (same CAN 2.0B protocol) |
| SPI | Works (BNO086 proven) | Works (faster clock available) |
| Flash | 256 KB (128 KB usable with OTA) | 8–16 MB external flash |
| RAM | 32 KB SRAM | 512 KB SRAM + 8 MB PSRAM |

### Dual-Core Architecture

```
Core 1 (dedicated real-time)          Core 0 (WiFi + housekeeping)
─────────────────────────             ─────────────────────────
500 Hz hw timer ISR → flag            FreeRTOS task: wifi_task()

control_task() pinned to core 1:      ├── AsyncUDP or ESP-NOW
├── IMU SPI read                      ├── Read telemetry from ring buffer
├── LQR + PI controllers              ├── Send 50 Hz to dashboard
├── CAN (TWAI) motor output           ├── Receive commands → queue
├── Write telemetry to ring buffer    ├── OTA handling
└── Read commands from queue          └── LED matrix (non-critical)

Budget: 2000 µs, zero WiFi jitter    No deadline — runs freely
```

Lock-free ring buffer between cores ensures control loop is never blocked by WiFi.

### WiFi Protocol Options

| | AsyncUDP | ESP-NOW |
|---|---|---|
| Needs router? | Yes | No — direct peer-to-peer |
| Latency | ~2–5 ms | ~1–2 ms |
| Max payload | ~1400 bytes (MTU) | 250 bytes (69B telemetry packet fits) |
| Dashboard side | Standard UDP socket — any PC | Needs ESP32 USB dongle as receiver |
| OTA | Works alongside ArduinoOTA | Requires switching to WiFi STA mode |
| Best for | Lab/workshop testing with router | Field testing, no infrastructure |

**AsyncUDP** is the pragmatic first choice (standard sockets on dashboard side, OTA
works natively). **ESP-NOW** is worth adding later for untethered field testing.

### Firmware Port Scope

The port is mostly mechanical — ESP32-S3 has Arduino framework support + ESP-IDF:

| Current (RA4M1) | ESP32-S3 Equivalent |
|---|---|
| `FspTimer` (hardware timer ISR) | `esp_timer_create()` or `hw_timer` |
| `WiFiS3.h` | Native `WiFi.h` (ESP-IDF backed) |
| Direct port register for IMU INT | `gpio_get_level()` (fast on ESP32) |
| Built-in CAN peripheral | TWAI driver (`driver/twai.h`) |
| `Serial.write()` telemetry | Still available, plus native UDP |
| Bare-metal superloop | FreeRTOS task pinned to core 1 |

### When to Pull This Trigger

- WiFiS3 `endPacket()` causes >3 ms spikes that disrupt balance control
- Need wireless telemetry at >50 Hz for real-time tuning
- Want untethered operation (no USB cable, no router)
- Need more flash/RAM for future features (path planning, SLAM, etc.)

---

## LED Matrix Display (12×8, UNO R4 WiFi built-in)

The LED matrix shows a single full-screen icon selected by priority, with small
corner overlays for connectivity status. Updated at ~6 Hz from the soft-RT
section of the main loop.

### Icon Priority (highest → lowest)

| Priority | Condition | Icon | Description |
|----------|-----------|------|-------------|
| 1 | `mode == FAULT` | **Blinking X** | Unmistakable fault indicator (75% on-time) |
| 2 | Loop overrun | **!** (exclamation) | Brief flash (~1 s), then returns to normal icon |
| 3 | `imu_ok == false` | **Blinking ?** | "I don't know which way is up" — IMU data stale |
| 4 | `mode == BALANCE` | **Pulsing blob** | Alternates small/large circle ~1.5 Hz — alive and balancing |
| 5 | `mode == DRIVE` | **→ arrow** | Right-pointing arrow |
| 6 | `mode == JUMP` | **↑ arrow** | Up-pointing arrow |
| 7 | `mode == IDLE` / other | **Smiley :)** | All good, waiting for commands |

### Corner Overlays

| Position | Indicator | Meaning |
|----------|-----------|---------|
| Bottom-right 3×3 | WiFi wave arcs (solid) | WiFi connected |
| Bottom-right 3×3 | WiFi wave arcs (blinking) | WiFi connecting |
| Bottom-right 3×3 | Off | WiFi disconnected |
| Bottom-left pixel | Solid dot | Host PC linked (`host_connected`) |

### Source Files

- `src/led_matrix.h` — public API (`led_matrix_init`, `led_matrix_update`)
- `src/led_matrix.cpp` — icon bitmaps (8×12 `const` arrays) + priority logic + overlays

---

## Key Source Files to Reference

| Firmware Module | Simulation Source |
|----------------|------------------|
| `config.h` | `master_sim/params.py` (all gains/geometry) |
| `lqr.cpp` | `master_sim/controllers/lqr.py` |
| `velocity_pi.cpp` | `master_sim/controllers/velocity_pi.py` |
| `yaw_pi.cpp` | `master_sim/controllers/yaw_pi.py` |
| `hip_ctrl.cpp` | `master_sim/controllers/hip.py` |
| `imu.cpp` | `characterization/IMU/include/bno086_common.h` |
| `wifi_mgr.cpp` | `characterization/Arduino/OTA/src/main.cpp` |
| main loop order | `master_sim/sim_loop.py` SimController.tick() |
