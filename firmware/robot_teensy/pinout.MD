# Teensy 4.1 Pinout — Wheeled-Leg Robot

## Power

| Stage | Device | Notes |
|-------|--------|-------|
| 24V → 5V | Buck converter (e.g. Pololu D24V50F5, or LM2596 module) | Must supply ≥1A for Teensy + peripherals |
| 5V → 3.3V | Teensy 4.1 built-in regulator (250 mA) | Powers Teensy logic, IMU, CAN transceivers |
| 5V consumers | FlySky receiver, display MCU (XIAO 5V pin), CAN transceiver VCC (if TJA1050) | |

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
Teensy CAN1 (pins 22/23)  →  SN65HVD230  →  CANH/CANL  →  AK45-10 L (0x41)
                                                           →  AK45-10 R (0x42)
                                              120 Ω terminator at each end

Teensy CAN2 (pins  0/ 1)  →  SN65HVD230  →  CANH/CANL  →  ODrive axis 0 (node 0)
                                                           →  ODrive axis 1 (node 1)
                                              120 Ω terminator at each end
```

Each SN65HVD230 is powered from the 3.3V rail. No level shifting needed.

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

Serial5 (pins 20/21) connects to a co-processor (e.g. XIAO ESP32-S3 for display):

```
Teensy TX (pin 20)  →  MCU RX
Teensy RX (pin 21)  ←  MCU TX
Teensy GND          →  MCU GND   (shared ground required)
5V rail             →  MCU 5V    (XIAO takes 5V in)
```

For higher bandwidth (streaming display frames), use SPI1 (pins 24–27) with a simple
packet protocol — MCU runs as SPI slave.

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
