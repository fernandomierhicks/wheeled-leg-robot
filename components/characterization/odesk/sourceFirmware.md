# ODESC (MKS XDRIVE MINI) — Source Firmware Investigation

## Hardware

- **Product:** MKS XDRIVE MINI (single-axis ODESC, ODrive v3.6 clone)
- **Purchase link:** [AliExpress listing](https://www.aliexpress.us/item/3256806293928426.html)
- **MCU:** STM32F405 (same as ODrive v3.6)

## Shipped Firmware

The board ships with **modified ODrive v0.5.1 firmware** (per the
[MKS-XDRIVE-MINI repo](https://github.com/justlovescience/MKS-XDRIVE-MINI) README).

> "Do NOT use the 'upgrade' command in odrivetool. This driver ships with
> modified v0.5.1 firmware; standard upgrades will brick the device."

## Binary Analysis (2026-04-11)

### MKS partial dump

A 1 KB firmware dump is published at:
[MKS_XDRIVE_MINI_original_FW.bin](https://github.com/justlovescience/MKS-XDRIVE-MINI/blob/main/Firmwares/MKS_XDRIVE_MINI_original_FW.bin)

This file is only 1024 bytes — just the first flash sector (vector table + startup).
It was dumped for ST-Link unbricking, not a full firmware image.

### Comparison against official ODrive releases

| Release | Variant | Size | First 1 KB match? |
|---|---|---|---|
| 0.5.1 | v3.5-24V | 234 228 B | No |
| 0.5.1 | v3.5-48V | 234 228 B | No |
| 0.5.1 | v3.6-24V | 234 228 B | No |
| 0.5.1 | v3.6-56V | 234 228 B | No |
| 0.5.2 | v3.6-56V | 339 648 B | No |
| 0.5.2 | v3.6-24V | 339 656 B | No |
| 0.5.3 | v3.6-56V | 310 348 B | No |
| 0.5.3 | v3.6-24V | 310 340 B | No |
| 0.5.4 | v3.6-56V | 310 892 B | No |
| 0.5.4 | v3.6-24V | 310 884 B | No |
| 0.5.5 | — | — | No prebuilt bin |
| 0.5.6 | — | — | No prebuilt bin |

The vector table diverges at byte 5 (reset vector):
- MKS:   initial SP = `0x20020000`, reset = `0x0800C07C`
- Stock 0.5.1 v3.6-56V: initial SP = `0x20020000`, reset = `0x0800B249`

Same MCU (128 KB SRAM), but code is laid out differently in flash — confirming
the firmware is modified, not a straight rebuild of stock ODrive.

### MKS-ODrive source repo (separate product)

The [makerbase-motor/MKS-ODrive](https://github.com/makerbase-motor/MKS-ODrive) repo
ships an ODrive v0.5.1 source zip for the dual-axis "MKS ODrive" board.
A full recursive diff (ignoring CRLF) shows **zero source-level changes** vs
the official ODrive 0.5.1 tag — byte-identical C++ code.

This does NOT apply to the XDRIVE MINI, which has actual binary differences.

## What the modifications could be

Without a full flash dump or source we can only speculate:
- Hardware ID / board variant flags (different shunt resistance, GPIO mapping, single-axis config)
- Bootloader changes (custom DFU or write-protect first sector)
- Compile-time config differences (`HW_VERSION_MINOR`, voltage variant)
- Possible feature backports or removals

## Recommended next steps

1. **Flash stock ODrive 0.5.6** — already confirmed working on this board
   (see [odrive.md](../odrive/odrive.md)). This gives access to all 0.5.2-0.5.6
   features (anticogging, spinout detection, cyclic CAN, vel_integrator_limit, etc.)
2. If curiosity demands it: dump full 512 KB flash via ST-Link SWD and disassemble
   with Ghidra to identify exact MKS modifications.
