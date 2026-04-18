# HANDOFF — AK45-10 Hip Motor CAN Control + Dashboard Tab

**Status:** Firmware + dashboard fully implemented. Pending flash & hardware verification.

---

## Goal

Add MIT CAN control of the AK45-10 hip motors to the Arduino firmware and a new "AK45 Motors" tab to `software/dashboard/dashboard.py`. One motor is wired and powered for initial testing.

---

## ⚠️ Pre-Flight (must do before firmware works)

The motor ships with MIT CAN ID = **1**, which collides with ODESC_NODE_L heartbeat (CAN ID 0x01) on the shared 1 MHz bus.

1. Flash `components/datasheets/Leg Motor/ak45-10-firmware-and-parameters/AK45-10/CMESC_MIT_APP_AK45_10.bin` to motor via CubeMars PC software (USB-C to motor board)
2. In CubeMars PC software: change motor CAN ID → **0x41** (65) for Hip L, **0x42** (66) for Hip R
3. In `firmware/src/config.h`: update `CAN_ID_HIP_L 0x41` and `CAN_ID_HIP_R 0x42`

---

## Protocol Reference (CubeMars Manual v1.0.18, §5.3)

**Frame type:** Standard CAN (11-bit ID), 1 MHz  
**TX ID** = motor MIT ID | **RX ID** = motor MIT ID (same)

### Special commands — 8 bytes sent to motor ID:
| Command  | Bytes                           |
|----------|---------------------------------|
| Enable   | `FF FF FF FF FF FF FF FC`       |
| Disable  | `FF FF FF FF FF FF FF FD`       |
| Set Zero | `FF FF FF FF FF FF FF FE`       |

### TX frame (pack_cmd) — AK45-10 specific ranges:
```
P:  [-12.5, 12.5] rad   16 bits
V:  [-20.0, 20.0] rad/s 12 bits
T:  [ -8.0,  8.0] N·m   12 bits
Kp: [    0,  500]        12 bits
Kd: [    0,    5]        12 bits

Data[0] = p_int >> 8
Data[1] = p_int & 0xFF
Data[2] = v_int >> 4
Data[3] = (v_int & 0xF) << 4 | (kp_int >> 8)
Data[4] = kp_int & 0xFF
Data[5] = kd_int >> 4
Data[6] = (kd_int & 0xF) << 4 | (t_int >> 8)
Data[7] = t_int & 0xFF
```

### RX frame (unpack_reply) — 8 bytes from motor:
```
Data[0]              = Drive ID
Data[1]<<8 | Data[2] = position int (16-bit) → uint_to_float(..., -12.5, 12.5, 16)
Data[3]<<4 | Data[4]>>4 = velocity int (12-bit) → uint_to_float(..., -20.0, 20.0, 12)
(Data[4]&0xF)<<8 | Data[5] = torque int (12-bit) → uint_to_float(..., -8.0, 8.0, 12)
Data[6]              = temperature (raw; temp_C = Data[6] - 40)
Data[7]              = error code
```

---

## Implementation Plan

### Step 1 — `firmware/src/config.h` ✅
Change:
```cpp
#define CAN_ID_HIP_L  0x41   // was 2 — changed to avoid ODESC collision
#define CAN_ID_HIP_R  0x42   // was 3
```

### Step 2 — CREATE `firmware/src/ak45_can.h` + `ak45_can.cpp` ✅
New MIT CAN driver. Mirrors `odesc_can.h/cpp` structure.

Public API:
```cpp
void ak45_init();
void ak45_enable(uint8_t id);
void ak45_disable(uint8_t id);
void ak45_set_zero(uint8_t id);
void ak45_send_cmd(uint8_t id, float p_des, float v_des,
                   float kp, float kd, float t_ff);
void ak45_parse_rx(uint32_t can_id, const uint8_t* data, RobotState& state);
bool ak45_hip_ok();
```

Implementation notes:
- `float_to_uint` / `uint_to_float` with AK45-10 ranges above
- `can_send_ak()` uses `CanStandardId(id)` + `CAN.write()` (same API as odesc)
- Watchdog: module-static `s_last_rx_L_ms`, `s_last_rx_R_ms`; `ak45_hip_ok()` checks both < `CAN_TIMEOUT_MS`
- `#ifdef NO_CAN` empty stubs for every public function (required for default build env)

### Step 3 — `firmware/src/robot_state.h` ✅
Add three fields:
```cpp
bool    hip_enabled;   // true = send MIT commands each tick
float   hip_temp;      // motor temperature °C
uint8_t hip_error;     // motor error code
```

### Step 4 — `firmware/src/odesc_can.cpp` ✅
Add `#include "ak45_can.h"`.  
In `odesc_can_poll()`'s CAN drain loop, after existing node-ID checks:
```cpp
if (std_id == CAN_ID_HIP_L || std_id == CAN_ID_HIP_R) {
    if (msg.data_length >= 6)
        ak45_parse_rx(std_id, msg.data, state);
}
```

### Step 5 — `firmware/src/main.cpp` ✅
- Add `#include "ak45_can.h"`
- In `setup()`: `ak45_init()` after `odesc_can_init()`
- In ISR control loop after `odesc_can_send_torque(...)`:
```cpp
if (state.hip_enabled) {
    ak45_send_cmd(CAN_ID_HIP_L, state.hip_q_target, 0.0f,
                  HIP_POS_KP, HIP_POS_KD, state.tau_hip_L);
    ak45_send_cmd(CAN_ID_HIP_R, state.hip_q_target, 0.0f,
                  HIP_POS_KP, HIP_POS_KD, state.tau_hip_R);
}
```

### Step 6 — Telemetry packet: 69 → 73 bytes ✅
**`firmware/src/telemetry.h`**: replace `float hip_q_avg` → `float hip_q_L`, add `float hip_q_R` after `tau_hip_R`. Update `static_assert` to 73.

**`firmware/src/telemetry.cpp`** and **`firmware/src/wifi_fast.cpp`**: fill `hip_q_L` and `hip_q_R` from `state.hip_q_L` / `state.hip_q_R`.

### Step 7 — `firmware/src/wifi_fast.cpp` — CMD_HIP ✅
Add `CMD_HIP = 5` to local enum.  
Enlarge `static uint8_t buf[16]` → `buf[24]`.  
New case — wire format: `[CMD_HIP u8][motor_id u8][cmd u8][p_des f32][v_des f32][kp f32][kd f32][t_ff f32]` = 23 bytes total:

| motor_id | meaning   | cmd | meaning   |
|----------|-----------|-----|-----------|
| 1        | Hip L     | 0   | Disable   |
| 2        | Hip R     | 1   | Enable    |
| 3        | Both      | 2   | Set Zero  |
|          |           | 3   | MIT cmd   |

### Step 8 — `software/dashboard/dashboard.py` ✅
1. Format: `'<IB16f'` → `'<IB17f'` (73 bytes); update assert
2. Unpack: rename `hip_q_avg` → `hip_q_L`, insert `hip_q_R` after `tau_hip_R`
3. Ring bufs: rename `hip_buf` → `hip_q_L_buf`, add `hip_q_R_buf`
4. Row-2 Hip Position plot: two lines (L=cyan, R=green)
5. Add `CMD_HIP = 5` constant
6. Add `CommandSender.send_hip(motor_id, hip_cmd, p_des, v_des, kp, kd, t_ff)`:
   ```python
   pkt = struct.pack('<BBBfffff', CMD_HIP, motor_id, hip_cmd,
                     p_des, v_des, kp, kd, t_ff)
   self.sock.sendto(pkt, (self.robot_ip, COMMAND_PORT))
   ```
7. New `QTabWidget` at bottom of main vbox — tab "AK45 Motors":
   - Left panel (motor_id=1) and Right panel (motor_id=2), each with:
     - Live labels: Position (deg), Velocity (rad/s), Torque (N·m), Temp (°C)
     - Small pyqtgraph plot: position + torque (last 15 s)
     - Enable / Disable / Zero buttons
     - MIT cmd: p_des (deg), kp, kd, t_ff spinboxes + Send button
   - Between panels: Enable Both / Disable Both / Zero Both

---

## Build & Flash

```bash
# Compile (wifi env — no -DNO_CAN flag)
pio run -e wifi

# Flash over USB
pio run -e wifi -t upload
```

---

## Verification Sequence

1. ✅ `pio run -e wifi` → zero errors; `static_assert(sizeof(TelemetryPacket) == 73)` passes
2. ✅ `pio run -e uno_r4_wifi` → NO_CAN stubs compile cleanly
3. ✅ `python -c "import struct; print(struct.calcsize('<IB17f'))"` → `73`
4. ⬜ Flash + open dashboard → all 10 existing plots still live; AK45 tab visible
5. ⬜ Motor configured to MIT ID=0x41: Enable L → motor stiffens; p_des=0.2 rad → motor moves; Disable → goes limp; position plot updates in real time
