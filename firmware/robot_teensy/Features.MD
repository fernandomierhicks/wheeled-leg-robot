# Robot Firmware Architecture — T4.1 + ESP32

## Device Part Numbers

| Device | Part Number | Interface |
|--------|------------|-----------|
| MCU (main) | PJRC Teensy 4.1 | — |
| MCU (wireless) | Seeed XIAO ESP32-S3 | UART / SPI |
| IMU | Bosch BNO086 | SPI |
| Hip motors (×2) | CubeMars AK45-10 KV75 | CAN (MIT mode) |
| Wheel motor controller | ODESC 3.6 Dual Drive | CAN (ODrive v0.5.x) |
| RC receiver | FlySky FS-iA6B | iBUS UART |
| CAN transceiver (×2) | TI SN65HVD230 | — |
| ToF sensors (×4) | ST VL53L1X | I2C |

---


- Dual-MCU firmware: **Teensy 4.1** handles real-time control; **ESP32** handles wireless/telemetry
- PlatformIO build with separate environments per firmware flavor and per unit test
- Driver-per-peripheral library structure (`IMU/`, `MotorController/`, `Telemetry/`) each with own `test/` subfolder
- 500 Hz control loop enforced by hardware timer interrupt; loop order: read sensors → compute → write actuators → send telemetry
- T4.1 → ESP32 link: UART + DMA at 1.2 Mbaud, circular buffer, zero CPU overhead
- Framed binary packet protocol (0xFF start, 0xFE end, checksum) defined in a single shared `telemetry_protocol.h`
- Dual telemetry output: same packet sent to ESP32 (wireless) and T4.1 USB simultaneously
- SD card storage for high-res telemetry logging, downloadable to PC for plotting
- Simple state machine: `STARTUP`, `CALIBRATION`, `RUNNING`, `E_STOP`
- Single Python GUI: flash firmware, serial monitor, live telemetry charts, unit test results display
- GUI auto-detects data source: WiFi (UDP) vs USB; USB can be from either ESP32 or Teensy
- Packets include a source identifier field so the GUI knows which MCU sent them
- OLED controlled by ESP32 via SPI
- WiFi telemetry via UDP
- WiFi commands to T4.1 via TCP
- Basic tones with buzzer connected to T4.1
- 4× VL53L1X ToF distance sensors read by T4.1
- RGB LED controlled by T4.1
- iBUS RC receiver (FlySky) read by T4.1 or ESP32