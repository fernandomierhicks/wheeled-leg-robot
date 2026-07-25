# AK45Uart

> **AI maintenance note:** If you find anything here that is stale while
> working in this tree, update this README in the same change.

AK45-10 encoder readback over UART (CubeMars ASCII terminal protocol, §5.3.1).

This is a **separate bus from the MIT Cheetah CAN** used by HipMotors. The UART is only for reading encoder angle; all torque commands go over CAN.

## Wiring

| Motor | Serial port | TX pin | RX pin |
|---|---|---|---|
| AK45-1 (left) | Serial2 | 8 | 7 |
| AK45-2 (right) | Serial3 | 14 | 15 |

Baud: 921 600 (default; matches `AK45_UART_BAUD` in `config.h`).

## API

```cpp
AK45Uart ak45_left(Serial2);
ak45_left.begin();              // call once in setup()

// in control loop (blocking, up to timeout_ms):
if (ak45_left.poll()) {
    float angle = ak45_left.state().m_angle_rad; // mechanical angle, unlimited
}
```

`state()` returns `{e_angle_rad, m_angle_rad, raw, last_fb_ms, ok}`.

## Gotchas

**Blocking poll** — `poll()` flushes RX, sends the `encoder` request, then blocks up to `timeout_ms` (default 50 ms) waiting for a valid response. Don't call this from an ISR or inside a tight real-time loop without accounting for the latency.

**Two frames per query** — the motor sends a human-readable diagnostic "show encoder now" frame before the actual data frame. The driver drains both frames until one parses successfully; this is normal and expected.

**CRC is pre-computed** — the `CMD_ENCODER` byte array is hardcoded from the datasheet (§5.3.1.1, pp 69–70). If the motor firmware changes major versions, verify the frame against the new manual.

**UART ≠ CAN feedback** — `HipAxisState.pos_rad` (from CAN) and `AK45UartState.m_angle_rad` (from UART) are both rotor angle in rad but sourced differently. The CAN value is fresher (interrupt-driven, updated every command cycle); the UART value is useful for calibration or when CAN feedback is not needed.
