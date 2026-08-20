# Adafruit_BNO08x (vendored, patched)

> **AI maintenance note:** If you find anything here that is stale while
> working in this tree, update this README in the same change.

Vendored copy of `adafruit/Adafruit BNO08x` (upstream docs in
`src/README.md`, license in `license.txt`). Previously pulled via
`platformio.ini`'s `lib_deps`, which meant PlatformIO fetched a fresh,
unpatched copy into `.pio/libdeps/<env>/` per build environment —
gitignored, and silently re-fetched (losing any local patch) on a clean
install. That's exactly how `teensy41` and `test_teensy` ended up running
two different, independently hand-patched copies of `sh2.c` for a while
without anyone noticing, which cost a full debugging session before the
divergence itself turned out to be the bug.

Vendoring it here means every environment builds from this one tracked
copy — no more possibility of that drift. If Adafruit ships a real
upstream update, it has to be pulled and merged by hand; there's no
automatic sync anymore.

## Local patch: `src/sh2.c` `opProcess()` timeout

No `sh2_Op_t` in the upstream file ever sets `timeout_us`, so without a
fallback, `opProcess()` never bails out when a request's response never
arrives (e.g. `getProdIdOp` during `sh2_getProdIds()`) — it busy-loops
forever with no way to recover from a non-responding sensor. This file
adds `DEFAULT_OP_TIMEOUT_US`, used whenever `timeout_us == 0`.

Set to **3,000,000 (3 s)**. Measured on real hardware (`test_imu`'s SH2
debug trace): `sh2_getProdIds()` alone legitimately takes ~690 ms on a
healthy connection. A 100 ms fallback was tried first and was wrong — it
silently timed out every real connection attempt before the sensor
finished responding, which is what looked like "the IMU won't initialize
in the main firmware but the standalone test passes" before the actual
cause (two environments' `.pio/libdeps` copies had different patches) was
found. 3 s gives a healthy sensor comfortable margin while still bounding
a truly dead/disconnected one.

See `lib/IMU/README.md` for how this project's own driver
(`lib/IMU/IMU.cpp`) calls into this library.

## Local patch: lossless, bounded SPI polling

Upstream stores the destination pointer supplied to `getSensorEvent()` and lets
every callback decoded by one `sh2_service()` call overwrite it. An SHTP packet
can contain many reports, so this silently discarded all but the final report
and could starve one of two independently enabled required streams. This copy
queues 48 decoded reports, enough for the maximum 384-byte incoming payload,
and exposes queue-overflow and decode-error counters.

SPI polling also has explicit real-time behavior:

- H_INTN high with no queued event returns `false` immediately;
- the payload handshake is bounded to 5 ms and writes to 100 ms (measured boot
  command traffic needs tens of milliseconds);
- a HAL timeout is returned to the caller and never triggers a hidden reset;
- transfer timestamps and SH2 time use `micros()` rather than millisecond
  quantization;
- the dedicated Gyro RV channel has SPI-aware SHTP sequence-gap diagnostics
  (successful payloads normally advance by two because SPI reads use separate
  header and full-payload transactions);
- failed post-open initialization closes its SHTP instance so retries do not
  exhaust the static instance pool.

The initial H_INTN wait remains 500 ms because the BNO086's boot handshake is
the one place a long wait is legitimate. Runtime reset/reconfiguration policy
lives in `lib/IMU/IMU.cpp`, where the robot can leave its energetic state before
the fixed 30 ms reset pulse occurs.

The robot hardware holds PS0 high. That selects SPI at boot but prevents PS0
from becoming the active-low WAKE signal required by the BNO08x protocol.
`_init()` therefore performs an explicit reset-as-wake transaction before the
product-ID request. `lib/IMU/IMU.cpp` does the same before report setup and
stages later configuration writes only on sensor-originated interrupts. This
replaces the old HAL behavior that silently reset the sensor after a timeout
and could erase reports configured earlier in the same startup sequence.
