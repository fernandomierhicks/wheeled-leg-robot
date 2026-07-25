# Teensy unit tests

PlatformIO Unity tests, one per subdirectory. Environment is `test_teensy` in
`../platformio.ini` (`test_filter = test_*`, excludes `src/`), except
`test_comm_usb` and the desktop-only tests, which have dedicated environments.

> **AI maintenance note:** If you find anything here that is stale while
> working in this tree, update this README in the same change.

Hardware tests: `test_buzzer`, `test_comm_usb`, `test_hip_motor`, `test_imu`,
`test_rc`, `test_uart_hw`, and `test_wheel_motor`.

Desktop tests: `test_commlink_robustness` (`native` or `windows_x86`),
`test_command_validation` (`windows_command_validation`), and
`test_param_store` (`windows_param_store`).

## Running from the CLI

```sh
pio test -e test_teensy -f test_imu --upload-port COM12
```

This builds, uploads, then lets `pio test` itself open the serial port to
capture Unity's `PASS`/`FAIL` output and print a summary.

For tests whose `loop()` also streams free-running telemetry after the Unity
assertions finish (e.g. `test_imu`), it's often more useful to skip `pio
test`'s own serial capture and just watch the raw stream yourself (via the
GUI's Flash & Monitor tab, `pio device monitor`, or any serial terminal):

```sh
pio test -e test_teensy -f test_imu --upload-port COM12 --without-testing
```

`--without-testing` makes this behave exactly like `pio run -t upload` —
build + upload only, no serial takeover — which is what the GUI's test
buttons use (`software/gui/tabs/flash_monitor.py`, `FLASH_ACTIONS`).

## Gotcha: "error writing to Teensy" / needs a manual reset

Before `teensy_loader_cli` runs, PlatformIO tries to auto-reboot the board
into its bootloader by briefly opening the currently-running sketch's serial
port. **That only works if nothing else already has the port open** — a
leftover `pio device monitor`/`pio test` process, or the GUI's Flash &
Monitor tab sitting connected. If the port is busy, the auto-reboot step
silently fails and `teensy_loader_cli` falls back to waiting for a manual
press of the board's reset/program button. Close any other serial
connection to the port before flashing from the CLI, and the upload should
go through without touching the board.

## Equivalent GUI buttons

`software/gui/tabs/flash_monitor.py` → Flash & Monitor tab → Teensy panel →
"Tests" grid. Each button runs the same `pio test -f <name>` (or, for
`test_imu`, `--without-testing`) command via `QProcess`, releasing its own
serial connection first so it doesn't self-block the reboot step above.

## Reading output yourself outside the GUI (e.g. from a script)

A plain pyserial read works without any special handshake for most tests —
they write to `Serial` unconditionally:

```python
import serial, time
ser = serial.Serial("COM12", 115200, timeout=0)
end = time.time() + 20
buf = b""
while time.time() < end:
    n = ser.in_waiting
    if n:
        buf += ser.read(n)
    time.sleep(0.02)
print(buf.decode("utf-8", errors="replace"))
```

Two things that will burn time if you don't know them going in:

- **`if (Serial)` on Teensy reflects DTR**, not just "USB enumerated". Code
  gated behind it (see `src/main.cpp`'s `if (Serial) g_comm_usb.send(...)`)
  stays silent until something asserts DTR on the port — pyserial does this
  automatically on `open()` unless you explicitly set `ser.dtr = False`
  first. The GUI's `SerialPortManager.acquire()` deliberately opens with
  DTR/RTS low (to avoid tripping the ESP32's auto-reset circuit) and then
  sets `ser.dtr = True` right after for the Teensy panel specifically —
  mirror that if you're scripting against it. A **blocking** `while
  (!Serial)` in a test's own `setup()` is a trap for exactly this reason: if
  the wait is on the firmware side, you *must* get DTR right on the host
  side or the board hangs forever with zero output and no way to tell why
  from the GUI alone. None of the current tests do this — keep it that way,
  or pair it with a bounded timeout.
- **Zero bytes for 15-20+ seconds means a real hang, not slowness.** Every
  test here prints something within the first couple seconds of `setup()`
  once `Serial` is up. If you've confirmed the upload succeeded (build+
  upload log shows `[SUCCESS]`/`[PASSED]`) and DTR is asserted but you still
  see nothing after a long wait, suspect a blocking call in the driver under
  test (e.g. a sensor `begin()` that never returns because the physical part
  isn't responding) rather than a flashing or serial-reading problem.

## More flashing gotchas

- **After a real "error writing to Teensy," just retry the same command
  immediately — don't reach for the reset button first.** `teensy_loader_cli`
  prints `Soft reboot is not implemented for Win32` on Windows, which sounds
  fatal but usually isn't: PlatformIO's own pre-upload "Rebooting..." step
  (open the port at a magic baud rate to trigger the bootloader) still runs
  before that, and it commonly fails on the *first* attempt right after a
  previous upload/monitor session but succeeds cleanly on the *second* with
  zero manual intervention. Only escalate to "please press the button" after
  a retry has also failed.
- **A board that's genuinely hung (stuck in a blocking call, not just
  running normal firmware) is more likely to need that manual button press**,
  even with the port free — the auto-reboot trick seems to rely on some
  responsiveness from the running sketch. Don't be surprised if a hung test
  needs one physical press to recover even though a healthy one didn't.
- **Flash and read in one script, not two separate commands, if you need the
  first second of output.** Piping a build/upload command and a separate
  pyserial read into two different tool calls loses everything printed in
  the gap between them — in practice that gap can be several seconds, easily
  eating an entire bounded `setup()` sequence. Trigger the upload via
  `subprocess` and start retrying the serial `open()` immediately after, in
  the same Python process, to actually catch early boot output.
- **You can safely patch a vendored library under `.pio/libdeps/<env>/` for
  one-off diagnosis** (e.g. adding a hard timeout to a blocking call, or
  `printf`/`Serial.print` tracing) — it's gitignored and per-environment, so
  it can't leak into the tracked repo and won't affect other envs (like the
  real `teensy41` build). It also doesn't survive a clean reinstall, so any
  fix worth keeping has to be ported into this project's own source
  (`lib/IMU/IMU.cpp`, not the library copy) before it's actually useful.

## `test_imu/` extras (kept as `.txt`, not built)

`test_imu/` has three reference-only subfolders. Each holds a `.cpp`/`.h`
saved as `.cpp.txt`/`.h.txt` on purpose — PlatformIO builds every source file
directly inside a test folder into one firmware image, so a second file with
its own `setup()`/`loop()` next to `test_imu.cpp` would fail to link. To
actually flash one, copy it out to its own `test/<name>/` folder (matching
the pattern of every other test here) and rename back to `.cpp`/`.h`, or
temporarily swap it in for `test_imu.cpp`.

- **`probe/`** — `test_imu_probe.cpp.txt`: a from-scratch SPI probe that
  bypasses the Adafruit_BNO08x/SH2 library entirely (raw reset, INT polling,
  hand-built SHTP reads/writes). Every wait in it is time-bounded, so it's
  always safe to flash unattended even if the sensor is completely dead.
  Useful any time you need to answer "is the IMU electrically alive at all"
  independent of whether the driver's protocol handshake is working.
- **`adafruit_example/`** — `test_imu_adafruit_example.cpp.txt`: the
  library's own bundled `quaternion_yaw_pitch_roll` example, changed only to
  use `begin_SPI()` with this project's pins instead of `begin_I2C()`, and to
  print-and-continue instead of hanging forever on failure. Useful for
  telling apart "something in this project's `IMU.cpp` is wrong" from
  "the sensor/library combo itself doesn't work on this hardware" — run this
  first when in doubt, since it has zero project-specific code in the path.
- **`may2026_reference/`** — a snapshot of `test_imu.cpp` and `lib/IMU/`
  exactly as they were at commit `2af310c` (2026-05-13), the last known-good
  IMU checkpoint before this test/driver pair was rewritten. Kept for future
  A/B comparisons if the IMU ever regresses again.

**2026-07-16 postmortem, for context:** the IMU test hung for an entire
session (identically on this exact May driver, the stock Adafruit example,
and current code) before turning out to be a loose/marginal connector on the
BNO086's SPI header — reseating it fixed everything. Along the way we also
found and fixed a real bug unrelated to the connector: `Adafruit_BNO08x::
_init()` never calls `sh2_close()` on failure, which leaks the SH2 driver's
one-slot connection pool — after any single failed init, every retry after
it fails `sh2_open()` instantly and NOMINAL becomes unreachable forever, even
once the physical problem is fixed. `lib/IMU/IMU.cpp`'s `attempt_init()` now
calls `sh2_close()` on failure to keep retries actually retrying.
