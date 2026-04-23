# Firmware

## Projects

| Folder | Purpose |
|---|---|
| `robot/` | Main balancing robot firmware (LQR, IMU, CAN control) |
| `can_bridge/` | CAN ↔ USB-Serial bridge (SLCAN protocol) for bench testing from Python |

---

## CAN Bridge — Pin Assignments (UNO R4 WiFi)

> **Note:** The UNO R4 **WiFi** CAN pins differ from the Minima.

| Arduino Pin | CAN Signal | Connect to |
|---|---|---|
| **D10** | CANTX | Transceiver TX input |
| **D13** | CANRX | Transceiver RX output |
| 3.3V | VCC | Transceiver VCC |
| GND | GND | Transceiver GND |

Transceiver CANH/CANL → ODrive CAN H / CAN L.

Termination: 120 Ω at both ends of the bus (ODrive JP1 solder bridge + external resistor at transceiver).

### Confirmed working
- ODrive v3.6 hw, fw v0.5.6
- CAN baud: 1,000,000 bps (`odrv0.can.config.baud_rate = 1000000`)
- Axis node IDs: axis 0 → node 0, axis 1 → node 1 (`odrv0.axisX.config.can.node_id`)

### Python usage
```python
import can
bus = can.Bus(interface='slcan', channel='COM5', bitrate=1000000, sleep_after_open=0.1)
msg = can.Message(arbitration_id=0x007, data=[8,0,0,0,0,0,0,0], is_extended_id=False)
bus.send(msg)
```

See `can_bridge/probe_odrive.py` for a diagnostic script.

---

## Robot Firmware — Pin Assignments (UNO R4 WiFi)

| Arduino Pin | Signal |
|---|---|
| D10 | BNO086 CS (SPI) ← **CONFLICT** (see below) |
| D2 | BNO086 INT |
| D3 | BNO086 RST |
| D13 | BNO086 SCK / SPI CLK ← **CONFLICT** (see below) |
| D11 | BNO086 MOSI |
| D12 | BNO086 MISO |
| D10 | CAN TX (`PIN_CAN0_TX` on WiFi variant) |
| D13 | CAN RX (`PIN_CAN0_RX` on WiFi variant) |

### ⚠ Pin Conflict — IMU + CAN cannot coexist as currently wired

On the UNO R4 **WiFi** variant the CAN peripheral uses D10 (TX) and D13 (RX).
The BNO086 IMU also uses D10 (CS) and D13 (SCK), so both peripherals fight over the same pins.

**Symptoms:** IMU never initialises when CAN is active. Pitch/roll/yaw read zero.

**Fix required before full robot firmware works:**
1. Rewire IMU CS from D10 → free pin (e.g. **D9**) and update `PIN_IMU_CS` in `config.h`
2. Rewire IMU SCK from D13 → use a software SPI or the secondary SPI bus
   — or accept the conflict and drive the IMU from a separate microcontroller

Until fixed, use the `usb_can` env for CAN-only testing (wheel encoder data works, IMU data is zeroed).
