# Firmware

Arduino UNO R4 WiFi firmware for the wheeled-leg balancing robot.

---

## CAN Bus

### Hardware

- **Controller:** Renesas RA4M1 built-in CAN peripheral (hardware framing, CRC, ACK — no CPU involvement)
- **Baud rate:** 1 Mbps
- **RX buffering:** interrupt-driven ring buffer in `Arduino_CAN`; CPU only drains it in `loop()` — never blocks waiting for data
- **Frame time:** ~130 µs per standard frame at 1 Mbps

### Bus Topology

```
Node 0  — ODrive axis 0 (wheel L)    arb_id = (0 << 5) | cmd   → 0x000–0x01F
Node 1  — ODrive axis 1 (wheel R)    arb_id = (1 << 5) | cmd   → 0x020–0x03F
0x41    — AK45-10 hip L              MIT CAN protocol
0x42    — AK45-10 hip R              MIT CAN protocol
```

AK45-10 IDs (≥ 0x41) never overlap with ODrive node-0/1 IDs (≤ 0x3F) — all four motors coexist safely on one bus.

### Bus Load Budget (500 Hz control rate, all 4 motors)

| Traffic | Frames/s | µs/s |
|---------|----------|------|
| ODrive encoder broadcast × 2 axes | 1000 | 130 000 |
| AK45-10 status reply × 2 axes | 1000 | 130 000 |
| ODrive velocity commands × 2 axes | 1000 | 130 000 |
| AK45-10 commands × 2 axes | 1000 | 130 000 |
| ODrive heartbeats (~10 Hz × 2) | 20 | 2 600 |
| **Total** | **~4020** | **~523 000 µs/s** |

**≈ 52% bus load at 500 Hz.** Do not exceed 500 Hz on all four motors simultaneously; 1000 Hz on all four approaches 100% load and causes frame collisions under jitter.

---

## ODrive CAN Protocol (v0.5.6)

Reference: [docs.odriverobotics.com/v/0.5.6/can-protocol.html](https://docs.odriverobotics.com/v/0.5.6/can-protocol.html)

Arbitration ID encoding: `arb_id = (node_id << 5) | cmd_id`

All multi-byte values are **little-endian**.

Request/reply messages use the **RTR bit** on the request frame; ODrive replies with the same CMD ID and the payload.

### Broadcast messages (unsolicited, rate set on ODrive)

| CMD | Name | Payload | Rate config |
|-----|------|---------|-------------|
| 0x01 | Heartbeat | `[0:3]` axis_error (u32) · `[4]` axis_state (u8) | `heartbeat_rate_ms` (default ~100 ms) |
| 0x09 | Encoder Estimates | `[0:3]` pos (float, turns) · `[4:7]` vel (float, turns/s) | `encoder_rate_ms` |

Configure `encoder_rate_ms` on the ODrive:
```python
odrv0.axis0.config.can.encoder_rate_ms = 2   # 500 Hz
odrv0.axis1.config.can.encoder_rate_ms = 2
odrv0.save_configuration()
```

### Request/Reply messages (poll as needed)

| CMD | Name | Reply payload |
|-----|------|---------------|
| 0x03 | Get Motor Error | `[0:7]` motor_error (u64) |
| 0x04 | Get Encoder Error | `[0:3]` encoder_error (u32) |
| 0x05 | Get Sensorless Error | `[0:3]` sensorless_error (u32) |
| 0x0A | Get Encoder Count | `[0:3]` shadow_count (i32) · `[4:7]` count_in_CPR (i32) |
| 0x14 | **Get IQ** | `[0:3]` Iq_setpoint (float, A) · `[4:7]` Iq_measured (float, A) |
| 0x15 | Get Sensorless Estimates | `[0:3]` pos (float, turns) · `[4:7]` vel (float, turns/s) |
| 0x17 | **Get Bus Voltage & Current** | `[0:3]` voltage (float, V) · `[4:7]` current (float, A) |
| 0x1C | Get ADC Voltage | `[0:3]` voltage (float, V) |
| 0x1D | Get Controller Error | `[0:3]` controller_error (u32) |

Most useful for the robot:
- **0x14 Get IQ** — Iq_measured × motor Kt gives actual wheel torque
- **0x17 Get Bus Voltage & Current** — battery sag monitoring and power budget

Poll these at a slow divisor (e.g. every 10 control ticks = 50 Hz). Each poll costs 2 frames (request + reply) per axis; at 50 Hz × 2 axes = 200 extra frames/s — negligible.

### Command messages (host → ODrive, no reply)

| CMD | Name | Payload |
|-----|------|---------|
| 0x02 | Estop | — |
| 0x06 | Set Axis Node ID | `[0:3]` node_id (u32) |
| 0x07 | Set Axis Requested State | `[0:3]` state (u32) |
| 0x0B | Set Controller Modes | `[0:3]` ctrl_mode (i32) · `[4:7]` input_mode (i32) |
| 0x0C | Set Input Pos | `[0:3]` pos (float, turns) · `[4:5]` vel_ff (i16, turns/s × 1000) · `[6:7]` torque_ff (i16, N·m × 1000) |
| 0x0D | Set Input Vel | `[0:3]` vel (float, turns/s) · `[4:7]` torque_ff (float, N·m) |
| 0x0E | Set Input Torque | `[0:3]` torque (float, N·m) |
| 0x0F | Set Limits | `[0:3]` vel_limit (float, turns/s) · `[4:7]` current_limit (float, A) |
| 0x11 | Set Traj Vel Limit | `[0:3]` vel (float, turns/s) |
| 0x12 | Set Traj Accel Limits | `[0:3]` accel (float) · `[4:7]` decel (float) |
| 0x13 | Set Traj Inertia | `[0:3]` inertia (float) |
| 0x16 | Reboot ODrive | — |
| 0x18 | Clear Errors | — |
| 0x19 | Set Linear Count | `[0:3]` count (i32) |
| 0x1A | Set Position Gain | `[0:3]` pos_gain (float) |
| 0x1B | Set Vel Gains | `[0:3]` vel_gain (float) · `[4:7]` vel_integrator_gain (float) |

### Axis states (used with Set Axis Requested State)

| Value | State |
|-------|-------|
| 1 | IDLE |
| 8 | CLOSED_LOOP_CONTROL |

### Control modes (used with Set Controller Modes)

| ctrl_mode | Mode |
|-----------|------|
| 1 | TORQUE |
| 2 | VELOCITY |
| 3 | POSITION |

| input_mode | Mode |
|------------|------|
| 1 | PASSTHROUGH (use for direct commands) |
| 5 | TRAP_TRAJ (use with trajectory limits) |

---

## Characterization Sketch

`components/characterization/CAN/` — standalone PlatformIO project for measuring:

- TX command throughput (how fast we can push velocity/torque commands)
- RX encoder frame rate and inter-frame jitter
- TX → next-RX pipeline gap (proxy for command-to-feedback latency)

Flash with:
```
cd components/characterization/CAN
pio run -e usb_can --target upload
```

Serial monitor at 1 Mbaud. Commands: `e` enable · `d` disable · `+/-` speed · `s` stats · `c` clear errors · `h` help.
