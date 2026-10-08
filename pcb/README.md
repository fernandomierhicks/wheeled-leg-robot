# Main board — Teensy 4.1 + ESP32 + CAN + JST-XH I/O

KiCad 10 project `wlr_mainboard` — replaces the hand-wired 90 × 70 mm perfboard
(`components/datasheets/custom board/`).

**Status (2026-10-08): schematic done (ERC 0 errors / 0 warnings), PCB fully
routed — DRC 0 violations, 0 unconnected, 0 schematic parity issues.**

> **Before ordering: double-check every pinout.** Connector pin orders were
> copied from the perfboard cables (photo, sketch and answers). The red note at
> the top of the schematic lists every connector and the specific doubts below.
> If a pin order changes, swap the nets in the schematic, update the PCB (F8)
> and re-route that connector.

## Files

| File | What |
|---|---|
| `wlr_mainboard.kicad_pro` | project; holds the OSH Park design rules and net classes |
| `wlr_mainboard.kicad_sch` | single A2 sheet; every pin goes to a net label or power symbol |
| `wlr_mainboard.kicad_pcb` | 4-layer board, routed, In1 GND / In2 +5V planes filled |
| `lib/wlr.kicad_sym` | Teensy 4.1, ESP32 DevKit V1 30-pin, SN65HVD230 module, power symbols `+3V3_TEENSY`, `+3V3_ESP32`, `VIN_TEENSY` |
| `lib/wlr.pretty/` | matching footprints (socket strips only in the courtyard, so parts can sit under the modules) |
| `sym-lib-table`, `fp-lib-table` | project-local libraries via `${KIPRJMOD}` |

These files are the source of truth: edit them in KiCad as usual and keep the
two in sync with *Tools → Update PCB from Schematic* (F8). They were generated
once by a throw-away script; nothing regenerates them.

## Board

- 90 × 70 mm, 4 × M2 (Ø2.2 NPTH) 2 mm in from the corners — same outline and
  holes as the perfboard (`cad/v5 Ai designed/Box/Electronics/CustomBoard`).
- Teensy USB and ESP32 USB-C at the top edge (they overhang it slightly).
- Teensy 4.1 and ESP32 on female sockets. Low parts sit under them:
  R1–R7, D1, D2, Q1 under the Teensy; U5, C2, R8–R12 under the ESP32. Seat
  them flat (keep under ~7 mm).
- CAN modules (U3 AK45 bus, U4 ODrive bus) stand on their 6-pin edge header,
  as on the perfboard; footprint assumes the module body on the +x side.
- Every JST-XH header has its latch side toward the board centre, like the
  perfboard, so the existing cables plug in the same way.
- Back silkscreen prints the pin name next to every connector pad.
- No copper pour or vias under the ESP32 antenna (rule area).
- Fully through-hole.

### OSH Park 4-layer

| Item | Value |
|---|---|
| Stackup | 1 oz / 7.87 mil prepreg / 0.5 oz / 39 mil core / 0.5 oz / 7.87 mil prepreg / 1 oz, FR408-HR (εr 3.61), ENIG, 1.6 mm nominal |
| Layers | L1 signal · L2 **GND plane** · L3 **+5V plane** · L4 signal |
| Rules (Board Setup) | 0.127 mm track / clearance, 0.254 mm drill, 0.102 mm annular ring, 0.381 mm copper-to-edge, 0.254 mm hole-to-copper |
| Net classes | Default 0.2 mm · Power 0.5 mm (`GND`, `+5V`, `VIN_TEENSY`, `+3V3_*`) · CAN 0.3 mm (`CAN?_H/L`) |
| Price | 90 × 70 mm ≈ 9.8 in² ≈ $98 for 3 boards ($10/in²) |

### Routing

Autorouted with Freerouting 2.5.0 (Specctra DSN → SES), then checked in KiCad.

- Signals on L1/L4 only; every GND and +5V pin connects straight to its plane
  through thermal reliefs (no GND/+5V traces).
- Widths: signals 0.2 mm, CAN pairs 0.3 mm, `+3V3_TEENSY` / `+3V3_ESP32` /
  `VIN_TEENSY` 0.5 mm. 17 vias (0.6 / 0.3 mm), ~2.6 m of track.
- No vias or pours in the ESP32 antenna area.
- To re-route with Freerouting: *File → Export → Specctra DSN*, shrink the DSN
  `boundary` ~0.25 mm (Freerouting ignores KiCad's 0.381 mm edge clearance),
  run with `--router.automatic_neckdown=false` (neckdown makes sub-0.127 mm
  traces), then *File → Import → Specctra Session*, refill zones (B), run DRC.
  Export the DSN from the board inside this folder, next to the `.kicad_pro`,
  or the net classes (power/CAN widths) are lost.

## Power

- `+5V` from J10 (buck output) feeds ESP32 VIN, the NeoPixel strip, IMU, ELRS
  receiver, buzzer and U5.
- Teensy VIN through **D1 (1N5817)**: Teensy USB can no longer back-feed the
  5 V rail, so the VUSB–VIN pad can stay uncut. The ESP32 DevKit still
  back-feeds `+5V` from its USB-C through its own diode (as on the perfboard).
- Two 3.3 V rails, never tied: `+3V3_TEENSY` (Teensy regulator) → CAN modules,
  UART6/7, I2C, spare header; `+3V3_ESP32` (DevKit regulator) → TFT, lasers,
  laser INT pull-ups, ESP32 spare header.

## Connectors

Orders are as seen on the board from the top (USB at the top). TX/RX are named
from the MCU (Teensy) side.

| Ref | Name | Footprint | Physical order on the board | Nets (pin 1 first) |
|---|---|---|---|---|
| J1 | BUZZER | XH 2p | top edge, left→right: BZ-, BZ+ | BUZZER_LOW, +5V |
| J2 | RGB_LED | XH 4p | left edge, top→bottom: B, G, GND, R | LED_R, GND, LED_G, LED_B |
| J3 | ELRS_RX | XH 4p | left edge, top→bottom: GND, 5V, RX (→Teensy 16), TX (Teensy 17→) | RC_TX, RC_RX_IN, +5V, GND |
| J4 | IMU_SIG | XH 6p | left edge, top→bottom: RST, INT, CS, AD0, SDA, SCL | IMU_SCK, IMU_MISO, IMU_MOSI, IMU_CS, IMU_INT, IMU_RST |
| J5 | IMU_PWR | XH 2p | left edge, top→bottom: GND, VCC (+5V) | +5V, GND |
| J6 | AK45_UART2 | XH 3p | bottom, inner row, left→right: GND, TX (Teensy 14), RX (Teensy 15) | AK2_RX, AK2_TX, GND |
| J7 | AK45_UART1 | XH 3p | bottom, outer row, left→right: GND, TX (Teensy 8), RX (Teensy 7) | AK1_RX, AK1_TX, GND |
| J8 | AK45_CAN | XH 2p | bottom, left→right: CANH, CANL | CAN2_L, CAN2_H |
| J9 | ODRIVE_CAN | XH 3p | bottom, left→right: GND, CANH, CANL | CAN3_L, CAN3_H, GND |
| J10 | 5V_IN | XH 2p | bottom, left→right: GND, +5V | +5V, GND |
| J11–J14 | LASER1–4 | XH 6p | bottom (1/2 outer row, 3/4 inner row), left→right: XSHUT, INT, SDA, SCL, GND, 3V3 | +3V3_ESP32, GND, TOF_SCL, TOF_SDA, LASERn_INT, LASERn_XSHUT |
| J15 | TFT_PWR | XH 2p | right edge, top→bottom: GND, 3V3 | GND, +3V3_ESP32 |
| J16 | TFT_SIG | XH 6p | right edge, top→bottom: SCL, SDA, RST, DC, CS, BL | TFT_SCK, TFT_MOSI, TFT_RST, TFT_DC, TFT_CS, TFT_BL |
| J17 | NEOPIXEL | XH 3p | right edge, top→bottom: 5V, GND, DATA | +5V, GND, NEO_DATA |
| J18 | LIMIT_L | XH 2p | *new*, right edge, top→bottom: GND, SIG (Teensy 23) | GND, LIMIT_L |
| J19 | LIMIT_R | XH 2p | *new*, right edge, top→bottom: GND, SIG (Teensy 22) | GND, LIMIT_R |
| J20 | UART6 | XH 4p | *new*, inner column, top→bottom: TX (24), RX (25), 3V3, GND | GND, +3V3_TEENSY, UART6_RX, UART6_TX |
| J21 | UART7 | XH 4p | *new*, inner column, top→bottom: RX (28), TX (29), 3V3, GND | GND, +3V3_TEENSY, UART7_TX, UART7_RX |
| J22 | I2C | XH 4p | *new*, inner column, top→bottom: SCL (19), SDA (18), 3V3, GND | GND, +3V3_TEENSY, I2C_SDA, I2C_SCL |
| J23 | TEENSY_GPIO | 1×14 header | *new*, top→bottom: 3V3, 26, 27, 32, GND, 41…33 — pins GND…33 sit level with the matching Teensy pins | — |
| J24 | ESP32_GPIO | 1×5 header | *new*, top→bottom: GND, 3V3, IO19, IO32, IO33 | — |

Lasers 1–4 = firmware sensors 0–3: XSHUT on ESP32 GPIO14/27/26/25, INT on
GPIO34/35/36/39 (10 kΩ pull-ups R9–R12; firmware still polls).

## Open checks before ordering

1. **IMU VCC = +5V** (as told): the BNO08x module must have its own 3.3 V regulator.
2. **IMU MOSI/MISO**: module `SDA` → Teensy 12 (MISO), `AD0` → Teensy 11 (MOSI),
   `SCL` → 13, per the BNO08x datasheet (H_SDA/H_MISO, SA0/H_MOSI).
   `firmware/robot_teensy/pinout.MD` lists SDA/AD0 the other way round —
   verify on the working perfboard.
3. **ELRS RX/TX** naming (J3) assumed from the Teensy side, like the AK45 UARTs.
4. **CAN termination**: the board is one end of each bus — check whether the
   HiLetgo module has a 120 Ω resistor fitted.
5. Confirm the CAN module stands with its body toward the Teensy (+x).
6. Not routed to the module: the IMU `PS0/WAKE` pin (`lib/IMU/README.md` asks
   for it on a future board, but the module only brings it out on solder
   bridges). A spare pin on J23 plus a flying wire would do.

## Parts

Through-hole only.

| Qty | Refs | Part |
|---|---|---|
| 1 | U1 | Teensy 4.1 + 2 × 1×24 female socket (2.54 mm) |
| 1 | U2 | ESP32 DevKit V1 30-pin USB-C + 2 × 1×15 female socket |
| 2 | U3, U4 | HiLetgo SN65HVD230 CAN board |
| 1 | U5 | 74AHCT125 (DIP-14, solder directly — no socket, it sits under the ESP32) |
| 1 | D1 | 1N5817 |
| 1 | D2 | 1N4148 |
| 1 | Q1 | 2N3904 (TO-92) |
| 3 | R1–R3 | 33 Ω (LED) |
| 2 | R4, R5 | 1 kΩ (buzzer base, ELRS RX series) |
| 2 | R6, R7 | 4.7 kΩ (I2C pull-ups) |
| 1 | R8 | 330 Ω (NeoPixel data) |
| 4 | R9–R12 | 10 kΩ (laser INT pull-ups) |
| 1 | C1 | 100 µF 10 V radial, Ø6.3 mm, 2.5 mm pitch |
| 1 | C2 | 100 nF disc, 5 mm pitch |
| 7 | J1, J5, J8, J10, J15, J18, J19 | JST B2B-XH-A |
| 4 | J6, J7, J9, J17 | JST B3B-XH-A |
| 5 | J2, J3, J20, J21, J22 | JST B4B-XH-A |
| 6 | J4, J11–J14, J16 | JST B6B-XH-A |
| 1 | J23 | 1×14 male header 2.54 mm |
| 1 | J24 | 1×5 male header 2.54 mm |
