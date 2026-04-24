# ODrive Characterization GUI

PySide6 tool for ODrive v3.6 (hw variant 56, fw 0.5.6) characterization, flashing,
calibration, anticogging, and live control. Entry point: `odrive_gui.py`.

---

## Known quirk: AS5048A encoder only works under mode 256, not 257

### Symptom

With an **AS5048A** magnetic encoder on SPI (GPIO 3/4/5, CS = GPIO 3):

- `encoder.config.mode = 257` (**`SPI_ABS_AMS`** — the documented choice for the
  AMS AS504x family) → calibration fails with axis error `0x0100 ENCODER_FAILED`,
  encoder error flags set, `is_ready` never goes true, live readouts unusable.
- `encoder.config.mode = 256` (**`SPI_ABS_CUI`** — nominally for CUI AMT23xx) →
  encoder reads correctly; `pos_estimate` tracks the magnet in the Control tab.
  `encoder.error` still shows intermittent SPI bits, but the position stream is
  usable.

Because the mode-256 encoder errors don't stop the encoder from reading, they are
**suppressed in the status bar** (see `odrive_gui.py` — errors dict filtered to
axis/motor/controller only) so they don't drown out real faults while we
continue developing the GUI. Re-enable that block once the root cause is fixed.

Because of this, the Setup tab exposes **three** encoder-type options:
`SPI Absolute AMS (257)`, `SPI Absolute CUI (256)`, `Incremental (0)` — use 256
for the AS5048A until the root cause is fixed.

### Leading hypothesis — AS5047 vs AS5048A frame layout

ODrive's `SPI_ABS_AMS` driver is written against the **AS5047P** 16-bit SPI
frame. The AS5048A is register-compatible at 0x3FFF but the status bits in the
response word are **swapped**:

| Bit | AS5047P                         | AS5048A                      |
|-----|---------------------------------|------------------------------|
| 15  | even parity over bits 14..0     | error flag (EF)              |
| 14  | error flag (EF)                 | even parity over bits 13..0  |
| 13..0 | 14-bit angle                  | 14-bit angle                 |

If the ODrive 0.5.6 `SPI_ABS_AMS` handler validates parity at bit 15 (AS5047
layout), it will fail ~50% of the time on real AS5048A data — which matches the
`ABS_SPI_COM_FAIL` / `NO_RESPONSE` behavior we see under mode 257. Mode 256
(`SPI_ABS_CUI`) uses a different frame-validation routine (CUI AMT23 K1/K2
checksum), which happens to leave the 14-bit position bits usable even though
the checksum is meaningless against AS5048A data — explaining why position reads
work but SPI error bits still flicker.

### Things to confirm

1. **Read the raw error bits under each mode.** With the GUI connected, check
   `axis0.encoder.error` and `axis0.encoder.spi_error_rate` after a few seconds
   under mode 257 vs mode 256. If 257 shows bit 7 (`ABS_SPI_COM_FAIL`, 0x080)
   and 256 shows something different (or just a low but non-zero
   `spi_error_rate`), that confirms the parity/checksum story.
2. **Read the ODrive 0.5.6 firmware source** at
   `Firmware/MotorControl/encoder.cpp` — specifically the `SPI_ABS_AMS` branch
   of `Encoder::abs_spi_start_transaction()` / `abs_spi_cb()`. Check which bit
   is treated as parity and which as the EF. If it matches AS5047 (parity at
   bit 15), the hypothesis is confirmed.
3. **Sanity check CPOL/CPHA.** Both AS5047 and AS5048A use SPI mode 1
   (CPOL=0, CPHA=1). The ODrive AMS driver should also be mode 1. A mode
   mismatch would break *both* 256 and 257, so this is unlikely to be the
   cause — but worth ruling out.
4. **Log `pos_abs` stability.** Under mode 256, sample `encoder.pos_abs` at
   ~100 Hz while the magnet is stationary. If it's rock-steady, the bottom
   14 bits are being read correctly and only the checksum wrapper is wrong.

### Working fix: error-suppression during calibration

The root cause (parity/EF bit swap) cannot be fixed without patching ODrive
firmware. However, the error only fires every ~150 ms, and the error register
is writable from the host. The workaround:

1. A background thread calls `odrv.clear_errors()` at ~333 Hz, keeping
   `encoder.error` at zero faster than the SPI ISR sets it.
2. Calibration runs as a **split sequence**: motor calibration (state 4)
   first, then encoder offset calibration (state 7). Both complete
   successfully while the suppressor is active.
3. After calibration, `pre_calibrated = True` is saved to NVM so
   subsequent boots skip calibration and the suppressor is not needed
   at runtime.

This is implemented in:
- `encoder_cal_fix.py` — standalone script (close GUI first)
- `tabs/tab_setup.py` — GUI "Calibrate" button uses `_ErrorSuppressor`
  class with the same split-calibration approach

### Important: power-cycle required when switching encoder mode

If an axis was previously calibrated under mode 256 (CUI), changing to mode 257
(AMS) and re-calibrating is **not enough** — the board needs a full power-cycle
(unplug and replug DC power) before the new mode takes effect cleanly. A software
reboot (`odrv.reboot()` / save_configuration) does not fully reset the SPI
encoder state, and errors will persist until a real power-cycle.

### Current status (2026-04-12)

Calibration under mode 257 now completes successfully thanks to the error
suppressor + split calibration. With `pre_calibrated = True` saved to NVM,
subsequent boots skip calibration and the encoder reads position correctly.
Everything *appears* to work — but the underlying parity/EF bit-swap issue
is still present. The SPI error bits still flicker at ~150 ms; they're just
no longer checked after boot because calibration is skipped. This is
acceptable for bench characterization but not for the final robot.

### Permanent fix options

ODrive 0.5.6 has **no setting** to disable SPI error checking. A proposed
`ignore_abs_ams_error_flag` parameter (GitHub PR #563) was never merged into
mainline — it only exists in ODrive Pro/S1 firmware (0.6.x+). The three
realistic paths:

1. **Keep error-suppression workaround (current, $0).** With
   `pre_calibrated = True` saved to NVM, the suppressor is only needed
   during one-time calibration. Normal closed-loop operation works because
   the PLL tracks position despite the flickering SPI error bit. Acceptable
   for bench characterization; for the balancing robot's final control loop
   a silent SPI dropout would be invisible — consider option 2 or 3.

2. **Build custom 0.5.6 firmware (~15 line change).** Modify
   `Firmware/MotorControl/encoder.cpp` `abs_spi_cb()` to swap the
   parity/error-flag bits for AS5048A: read error flag from bit 15 (not 14),
   compute parity over bits 14..0 (not 15..0). Requires ARM GCC toolchain +
   `tup` build system
   ([developer guide](https://docs.odriverobotics.com/v/0.5.6/developer-guide.html)).
   A community fork exists at
   `github.com/grahameth/ODrive_hbuhle2s_0.5.4_abs_ignore` but only adds
   an ignore flag — it does **not** fix the parity calculation, so it would
   still silently discard ~50% of frames. A proper fix must also rewrite
   `ams_parity()`.

3. **Replace AS5048A with AS5047P (~$5–8/board, drop-in).** Pin-compatible,
   same magnetic interface, native ODrive support. Also provides incremental
   ABZ outputs (the AS5048A lacks these). This is the ODrive community
   consensus recommendation.

   Alternative: **MagAlpha MA732 / MA702** (mode 260, `SPI_ABS_MA732`).
   ODrive's driver for these does zero parity validation — just a bit shift —
   making it the most reliable SPI encoder option for ODrive v3.6.

Note: the AS5048A's PWM output mode is **not usable** — ODrive 0.5.6 has no
PWM-input encoder mode. The AS5048A also lacks incremental ABZ outputs, so
incremental mode is not an option either.

---

## Files

- `odrive_gui.py` — main window, connection/poll loop, status bar.
- `tabs/tab_setup.py` — motor + encoder parameters, calibrate, flash, verify.
- `tabs/tab_control.py` — live control (position/velocity/torque).
- `tabs/tab_anticogging.py` — anticogging map capture and flash.
- `core/odrive_operations.py` — low-level ODrive ops (reused by Claude cmd interface).
- `core/constants.py` — enum tables (motor types, encoder types, axis states).
- `core/odrive_errors.py` — bitfield decode tables.
- `core/odrive_manager.py` — USB connection + axis helpers.
- `cmd/cmd_interface.py` — file-based command inbox so Claude can poke the
  ODrive while the GUI is running.
- `encoder_cal_fix.py` — standalone calibration script with error suppression.
