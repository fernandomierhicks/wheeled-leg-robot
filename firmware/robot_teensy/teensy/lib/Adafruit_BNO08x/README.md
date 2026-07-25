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
