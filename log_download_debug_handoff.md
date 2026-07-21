# Handoff: SD-log download via ESP32 relay still fails

## Problem

`log_download` (GET) over the ESP32-relayed transport — both WiFi mode and ESP32-USB
mode — hangs and never completes. GUI shows "no teensy" for several seconds then
reverts to STANDBY without the file landing. **Direct Teensy-USB (bypassing the ESP32
entirely) works reliably** — every fix attempt so far has used this comparison to
localize the bug to the ESP32 relay path specifically.

## Architecture

Teensy (SD card owner, control loop) --Serial5 UART 4 Mbaud--> ESP32 (WiFi/USB bridge)
--USB-CDC or WiFi--> GUI (PyQt6). LOG_DATA = 480 B chunks + 8 B header
(`LOG_CHUNK_DATA`, `firmware/robot_teensy/shared/comm_protocol.h`). Firmware streams
fire-and-forget, no per-chunk ACK. GUI's `software/gui/tabs/log_transfer.py`
(`LogTransferManager`) restarts the WHOLE file from chunk 0 if it detects a gap/CRC
mismatch at the end — no partial retry.

## Confirmed facts — don't re-derive these

- **WiFi is not the cause.** Reproduced identically with `WIFI_ENABLED=0` fully
  compiled out (`PLATFORMIO_BUILD_FLAGS="-D WIFI_ENABLED=0" pio run` in
  `firmware/robot_teensy/esp32/` — the flag is now guarded with `#ifndef` in
  `config.h` so this works without editing the file).
- Before commit `e0b2f28` ("ESP32 Phase 1: decouple uplink sends from UART parse
  path"), every frame including LOG_DATA was sent directly/synchronously from
  `on_teensy_packet()`. That commit added a shared, drop-oldest `g_uplink_q` for a
  *different*, legitimate reason (blocking WiFi TCP sends stalling the UART parser
  during active telemetry) — but broke downloads as a side effect. Downloads
  "used to work" — this is why.
- Teensy's own 500 Hz loop is healthy during a GET (only trivial ~2-3 ms overruns
  once the transfer is underway). Separately, there ARE large 50-95 ms overruns at
  `log_start`/`log_stop` (SD file open/close latency) — real, but not the download
  hang; don't conflate the two.
- `software/gui/logs/esp32.log` / `teensy.log` are continuous raw serial captures
  (binary + text mixed). Useful post-mortem: `tail -c <N> esp32.log | grep -a -oE
  "[ -~]{6,}"` extracts readable ASCII lines.

## Fixes already applied and still in the tree — don't re-propose these

1. Timeouts bumped (`software/gui/tools/robot_ctl.py:TIMEOUT_S`,
   `software/gui/tabs/remote_control.py:_cmd_log_download`'s wait) 35s/30s → 95s/90s.
   Real fix (500 Hz logging produces multi-MB files) but not the hang.
2. `teensy/src/main.cpp` `loop()`: telemetry (TELEM_A/B) pauses for the whole
   duration of a GET (`sd_logger_transfer_active()`), so 100% of link bandwidth
   goes to the log stream.
3. `esp32/src/main.cpp`: LOG_DATA/LOG_INFO get their own queue (`g_log_uplink_q`)
   AND their own dedicated FreeRTOS task (`log_uplink_task`, separate from
   `uplink_task`) — not sharing the small telemetry queue (eviction) and not
   blocking `on_teensy_packet()` (core 1 stalls — this manifested as "~1s freezes
   on any log command" when I tried direct/synchronous sends instead).
4. Watchdog safety: `log_uplink_task`'s queue wait is bounded
   (`pdMS_TO_TICKS(100)`, not `portMAX_DELAY`) — an earlier version with
   `portMAX_DELAY` caused a permanent 5s panic-reboot loop, since the queue sits
   empty outside active transfers and the task never got back to petting the
   watchdog.
5. `g_usb_mutex` added, guards every `Serial` write (`g_usb.send()`,
   `g_usb_diag.send()`, the debug `Serial.printf` lines) — `log_uplink_task` and
   `uplink_task` can now both write to the same non-thread-safe `Serial` object
   concurrently; an interleaved write corrupts CommLink framing on the wire.
   **This is the most likely current culprit but is UNTESTED** — flashed, GUI
   reconnected, no download attempt reported back yet. Start here.
6. Download/recording UI indicators (NeoPixel blue progress ring for downloads,
   additive blue strobe for recording, TFT "DOWNLOADING X%" banner + REC dot,
   boot rainbow/splash) — all confirmed working by the user. Not suspects.
7. Buzzer chirps for recording-start/download-start/download-complete added to
   `teensy/src/main.cpp` — compiled, **not yet flashed**.

Note: none of this is committed yet (`git status` will show a large diff across
`firmware/robot_teensy/esp32/src/{main.cpp,config.h}`,
`firmware/robot_teensy/teensy/src/main.cpp`,
`software/gui/tools/robot_ctl.py`, `software/gui/tabs/remote_control.py`).

## What's still unknown

- Whether fix #5 (`g_usb_mutex`) actually resolves the hang — verify this first.
- The failure signature has been consistent across attempts: chunk 0 (the
  `WLRLOG` file header) gets through, then it silently stalls — no further
  chunks, no crash, no error logged on the Teensy side.
- If #5 doesn't fix it, the bug is somewhere else in ESP32 UART2 RX (`Serial2`,
  from Teensy) → `CommLink` parse → queue → task → `Serial` TX, or still
  possibly GUI-side (`software/gui/tabs/flash_monitor.py`'s `PacketDecoder`/
  `SerialReader` — reviewed once, looked like a correct incremental parser, but
  not proven innocent under sustained high-throughput binary load specifically).

## Suggested next steps

1. Ask the user to try a download now (mutex fix is already flashed) — get a
   clean pass/fail before changing anything else.
2. If it still fails: pull a fresh log tail immediately after (command above) —
   look for anything past the first `WLRLOG` chunk this time.
3. Add explicit counters — chunks enqueued (`on_teensy_packet`) vs. chunks
   actually sent (`log_uplink_task`) vs. `g_log_uplink_drops` — with a periodic
   debug print while a transfer is active. Proposed but never implemented; would
   conclusively show whether the ESP32 is failing to send vs. the GUI failing to
   receive/assemble what was sent correctly. This is the highest-value next
   diagnostic if the mutex fix doesn't resolve it.
4. If ESP32-side counters show 100% of chunks sent but the GUI still can't
   assemble the file, move the investigation entirely to
   `software/gui/tabs/log_transfer.py` and `flash_monitor.py`'s `PacketDecoder`.
