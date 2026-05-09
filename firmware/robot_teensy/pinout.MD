# Teensy 4.1 Pinout — Wheeled-Leg Robot

## Power

| Stage | Device | Notes |
|-------|--------|-------|
| 24V → 5V | Buck converter (e.g. Pololu D24V50F5, or LM2596 module) | Must supply ≥1A for Teensy + peripherals |
| 5V → 3.3V | Teensy 4.1 built-in regulator (250 mA) | Powers Teensy logic, IMU, CAN transceivers |
| 5V consumers | FlySky receiver, ESP32 DevKit V1 VIN pin, CAN transceiver VCC (if TJA1050) | |

Teensy 4.1 is **3.3V logic — not 5V tolerant**. Use 3.3V CAN transceivers (SN65HVD230) or add level shifters.

---

## Pin Assignments

| Pin | Signal | Peripheral | Interface | Voltage | Notes |
|-----|--------|-----------|-----------|---------|-------|
| 0 | CAN2_RX | ODrive wheel motors | CAN2 | 3.3V | Via SN65HVD230 transceiver |
| 1 | CAN2_TX | ODrive wheel motors | CAN2 | 3.3V | Pins 0/1 = CAN2 only; Serial1 unavailable |
| 2 | IMU_INT | BNO086 | SPI0 | 3.3V | Active-low interrupt |
| 3 | IMU_RST | BNO086 | SPI0 | 3.3V | Active-low reset |
| 4 | LED_R | RGB LED | PWM | 3.3V | 33 Ω series resistor to LED |
| 5 | LED_G | RGB LED | PWM | 3.3V | |
| 6 | LED_B | RGB LED | PWM | 3.3V | |
| 7 | AK45_1_RX | AK45-1 UART (optional) | Serial2 | 3.3V | Half-duplex; use 1 kΩ series + diode |
| 8 | AK45_1_TX | AK45-1 UART (optional) | Serial2 | 3.3V | |
| 9 | (spare) | — | — | — | Free GPIO / PWM |
| 10 | IMU_CS | BNO086 | SPI0 | 3.3V | Chip select, active low |
| 11 | SPI0_MOSI | BNO086 | SPI0 | 3.3V | Hardware SPI bus |
| 12 | SPI0_MISO | BNO086 | SPI0 | 3.3V | |
| 13 | SPI0_SCK | BNO086 | SPI0 | 3.3V | Also drives built-in orange LED |
| 14 | AK45_2_TX | AK45-2 UART (optional) | Serial3 | 3.3V | |
| 15 | AK45_2_RX | AK45-2 UART (optional) | Serial3 | 3.3V | |
| 16 | IBUS_RX | FlySky receiver iBUS | Serial4 RX | 3.3V* | See iBUS note below |
| 17 | (spare) | Serial4 TX unused | — | — | iBUS is RX-only |
| 18 | I2C_SDA | Spare / future sensor | Wire | 3.3V | |
| 19 | I2C_SCL | Spare / future sensor | Wire | 3.3V | |
| 20 | EXT_MCU_TX | Display MCU / ESP32 | Serial5 | 3.3V | Teensy TX → MCU RX |
| 21 | EXT_MCU_RX | Display MCU / ESP32 | Serial5 | 3.3V | Teensy RX ← MCU TX |
| 22 | CAN1_TX | AK45-10 hip motors | CAN1 | 3.3V | Via SN65HVD230 transceiver |
| 23 | CAN1_RX | AK45-10 hip motors | CAN1 | 3.3V | Separate bus from ODrive |
| 24 | SPI1_CS | Display MCU (optional) | SPI1 | 3.3V | High-speed alt to UART |
| 25 | SPI1_MISO | Display MCU (optional) | SPI1 | 3.3V | |
| 26 | SPI1_MOSI | Display MCU (optional) | SPI1 | 3.3V | |
| 27 | SPI1_SCK | Display MCU (optional) | SPI1 | 3.3V | |
| 28–33 | (spare) | — | — | — | Digital / analog / PWM |
| VBAT sense | Analog voltage divider | Battery monitor | A_in | 3.3V max | 24V → 3.3V: use 68k/10k divider |

### Bottom pads (solder only)
| Pad | Signal | Notes |
|-----|--------|-------|
| 34 | Serial8_RX | Spare UART |
| 35 | Serial8_TX | Spare UART |
| 36–39 | SPI1 alt pins | SPI1 MISO=39, usable if needed |
| 54–57 | SDIO | microSD card (built-in slot on Teensy 4.1) |

---

## CAN Bus Architecture

Two separate CAN buses — cleaner isolation, no ID collision risk:

```
Teensy CAN1 (pins 22/23)  →  SN65HVD230  →  CANH/CANL  →  AK45-10 L (0x11)
                                                           →  AK45-10 R (0x12)
                                              120 Ω terminator at each end

Teensy CAN2 (pins  0/ 1)  →  SN65HVD230  →  CANH/CANL  →  ODrive axis 0 (node 0)
                                                           →  ODrive axis 1 (node 1)
                                              120 Ω terminator at each end
```

Each SN65HVD230 is powered from the 3.3V rail. No level shifting needed.

---

## CAN ID Reference

### CAN1 — AK45-10 Hip Motors (MIT Cheetah protocol)

Motor IDs are set in the AK45 firmware (factory default 0x01; ours are configured to **11 = left, 12 = right** — decimal, not hex).
All frames (Teensy → motor and motor → Teensy) use the **same arbitration ID = motor ID**.
The response is distinguished by the first data byte equalling the motor ID.

| ID (decimal) | Direction | Frame content |
|--------------|-----------|---------------|
| 11 | TX → left hip | 8-byte MIT control packet [p_des, v_des, Kp, Kd, t_ff] |
| 11 | RX ← left hip | 8-byte status [motor_id, pos, vel, torque] |
| 12 | TX → right hip | same format |
| 12 | RX ← right hip | same format |

Special command bytes (all other bytes 0xFF):

| Last byte (byte 7) | Command |
|--------------------|---------|
| 0xFC | Enter MIT mode |
| 0xFD | Exit MIT mode |
| 0xFE | Set zero position |

### CAN2 — ODrive Wheel Motors (ODrive CAN v2, fw 0.5.6)

Arbitration ID format: `(node_id << 5) | cmd_id`

ODrive axis 0 = node 0 (base 0x000), axis 1 = node 1 (base 0x020).

| ID (axis 0) | ID (axis 1) | Dir | Command |
|-------------|-------------|-----|---------|
| 0x001 | 0x021 | RX ← ODrive | Heartbeat (axis state + error) |
| 0x006 | 0x026 | TX → ODrive | Clear errors |
| 0x007 | 0x027 | TX → ODrive | Set axis state (e.g. CLOSED_LOOP = 8) |
| 0x009 | 0x029 | TX → (RTR) | Get encoder estimates → pos [turns], vel [turns/s] |
| 0x00B | 0x02B | TX → ODrive | Set controller mode |
| 0x00D | 0x02D | TX → ODrive | Set input velocity [turns/s] + torque_ff [Nm] |
| 0x00E | 0x02E | TX → ODrive | Set input torque [Nm] |
| 0x014 | 0x034 | TX → (RTR) | Get IQ [A] |
| 0x018 | 0x038 | TX → (RTR) | Get bus voltage & current |

---

## IMU SPI (BNO086)

6-pin connection to Teensy SPI0:

| BNO086 pin | Teensy pin | Notes |
|------------|-----------|-------|
| CS | 10 | Active low |
| MOSI (SDA) | 11 | |
| MISO (SA0) | 12 | |
| SCK (SCL) | 13 | |
| INT | 2 | Active low; input with pullup |
| RST | 3 | Active low; drive low 10 ms to reset |
| VCC | 3.3V | |
| GND | GND | |
| PS0 | GND | PS0+PS1 both LOW = SPI mode |
| PS1 | GND | |

No pin collision with CAN — CAN1/CAN2 do not share any SPI0 pins on Teensy 4.1.

---

## FlySky iBUS

- **Protocol:** 115200 baud, single-wire, receiver → controller only (RX pin only needed)
- **Signal voltage:** *Most FlySky receivers (FS-iA6B, FS-X6B) output iBUS at **3.3V logic** when powered from 5V.* Some older/clone units output 5V.
- **Safe wiring:** Add a **1 kΩ series resistor** between receiver iBUS pin and Teensy pin 16 (RX only, no TX needed). This protects against a 5V unit without a full level shifter.
- **Library:** `IBusBM` (Arduino) — non-blocking, interrupt-driven, works on any Teensy Serial port.

---

## External MCU / WiFi Module (UART)

Serial5 (pins 20/21) connects to the ESP32 DevKit V1:

```
Teensy TX (pin 20)  →  ESP32 GPIO16 (RX2)
Teensy RX (pin 21)  ←  ESP32 GPIO17 (TX2)
Teensy GND          →  ESP32 GND   (shared ground required)
5V rail             →  ESP32 VIN   (accepts 5V in)
```

---

## ESP32 DevKit V1 Pinout

Board: `esp32dev` (ESP32-WROOM-32, 3.3V logic, 5V tolerant VIN).

### Teensy Link (Serial2 / UART2)

| ESP32 Pin | GPIO | Signal | Connects to |
|-----------|------|--------|-------------|
| GPIO16 | RX2 | UART2 RX | Teensy pin 20 (Serial5 TX) |
| GPIO17 | TX2 | UART2 TX | Teensy pin 21 (Serial5 RX) |
| VIN | — | Power in | 5V rail (buck output) |
| GND | — | Ground | Shared GND |

Baud rate: **1,200,000** (defined in `esp32/src/config.h` as `TEENSY_UART_BAUD`).
No level shifter needed — both sides are 3.3V logic.

### WiFi Roles

| Role | Protocol | Port |
|------|----------|------|
| Telemetry stream (ESP32 → PC) | UDP | 5005 |
| Command link (PC → ESP32) | TCP | 5006 |
| OTA firmware update | espota | — |

All WiFi I/O is handled internally by the ESP32-WROOM-32's integrated radio — no extra pins required.

---

## Serial Port Allocation Summary

| Port | TX | RX | Assigned to |
|------|----|----|-------------|
| Serial1 | 1 | 0 | **UNAVAILABLE** — pins used by CAN2 |
| Serial2 | 8 | 7 | AK45-1 UART (optional) |
| Serial3 | 14 | 15 | AK45-2 UART (optional) |
| Serial4 | 17 | 16 | FlySky iBUS RX |
| Serial5 | 20 | 21 | External MCU / WiFi |
| Serial6 | 24 | 25 | Spare (or repurpose if SPI1 used) |
| Serial7 | 29 | 28 | Spare |
| Serial8 | 35 | 34 | Spare (bottom pads) |
