# Esp32Link

UART bridge from Teensy to ESP32. Wraps `CommLink` with a typed API for telemetry TX and command RX.

## Wiring

| Signal | Teensy 4.1 pin |
|---|---|
| TX (to ESP32 RX) | 20 |
| RX (from ESP32 TX) | 21 |

Serial port: `Serial5`. Baud: 1 200 000 (`ESP32_BAUD` in `config.h`).

## API

```cpp
g_esp32.begin();                    // call once in setup()
// control loop:
g_esp32.send_telemetry(payload);    // send TelemetryPayload to ESP32
g_esp32.update();                   // pump inbound RX, fire callbacks
// register command handler (call before begin or loop):
g_esp32.onCommand([](const CommPacket& p) { /* handle */ });
```

`g_esp32` is a global singleton declared in `esp32_link.cpp`.

## Protocol

Packets are framed by `CommLink` (length-prefixed, with type and version fields). See `CommLink/` and `comm_protocol.h` for the wire format. `COMM_SRC_TEENSY` is baked into every outbound frame so the ESP32 can distinguish Teensy packets from its own loopback.

## Gotchas

**`update()` must be called every tick** — inbound command packets accumulate in the UART FIFO until `update()` drains them. Skipping calls means commands are processed late (or not at all if the buffer overflows).

**Baud is 1.2 Mbps** — standard USB-UART adapters top out at 921 600. Use a logic analyser or the ESP32's native UART peripheral for debugging; do not lower the baud without also changing `ESP32_BAUD` on both sides.

**No ACK / flow control** — `send_telemetry()` fires and forgets. If the ESP32 is booting or busy, frames are silently dropped; `tx_count()` increments regardless.
