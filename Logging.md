# High-Datarate Teensy Logging → Retrieval → GUI Playback — Execution Plan

> **For the executing agent (Sonnet, fresh session):** This is a self-contained
> build plan. Work **phase by phase**. Each phase ends with a **CHECKPOINT** —
> stop, run the stated check, and confirm it passes before starting the next
> phase. Do **not** batch phases. If a checkpoint fails, fix it before moving on.
> Follow the repo rule in `CLAUDE.md`: *ask before assuming, simplest thing that
> works, don't touch unrelated code, compile/verify after each change.*

---

## 1. Context & goal

Live telemetry is capped at **50 Hz** because the 235-byte `TelemetryPayload` is
throttled by the Teensy→ESP32→PC bridge (`send_telemetry()` fires every 10th tick
of the 500 Hz loop, `teensy/src/main.cpp:594`). That is too coarse to troubleshoot
fast dynamics (balance oscillations, jumps, torque saturation, landing transients).

The Teensy 4.1 has a built-in SDIO microSD slot but **no SD code exists yet** (only
an empty `teensy/lib/SDLogger/.gitkeep`). We are adding:

1. **500 Hz SD logging** on the Teensy that never breaks the 2 ms control loop.
2. **Over-the-air retrieval** (WiFi TCP + USB serial) reusing the existing
   `CommLink` framing — no card pulling required for normal use.
3. **Rosbag-style playback** that replays a log through the existing PyQt GUI
   (every tab + 3D visualizer renders unchanged), plus a **CSV/pandas export** so
   logs are directly analyzable offline by a human or by Claude.

### Decisions locked with the user
- **Record** = full `TelemetryPayload` **+ a `uint32 micros` timestamp** per sample.
- **Triggers** = GUI button + timed duration now; **RC switch = wired stub, no
  channel assigned yet** (see §9).
- **Retrieval** = over-the-air, works over **WiFi TCP and serial** (chunked GET).
- **Playback** = replay into the existing GUI (rosbag-style) + CSV export.

---

## 2. Existing architecture (grounding — don't re-explore)

**Three deployments:**
- `firmware/robot_teensy/teensy` — Teensy 4.1, 500 Hz control loop.
- `firmware/robot_teensy/esp32` — ESP32 bridge (UART↔Teensy, USB+WiFi↔PC).
- `software/gui` — PyQt6 desktop app.
- `firmware/robot_teensy/shared` — `comm_protocol.h`, `CommLink/`, `udp_stream.h`
  (shared by all three; `lib_extra_dirs = ../shared`).

**Key facts and anchors:**

| Thing | Location | Notes |
|---|---|---|
| Control loop | `teensy/src/main.cpp:582-602` `loop()` | 500 Hz, busy-waits `while (micros()-t_start < 2000){}`. **No IntervalTimer, no overrun catch-up.** |
| Telemetry fill+send | `teensy/src/main.cpp:277-351` `send_telemetry()` | Fills local `TelemetryPayload` from `g_state`/sensors, sends `TELEM_A`+`TELEM_B`. Called at 50 Hz via `telem_div`. |
| Command handler | `teensy/src/main.cpp:77-184` `on_command()` | Registered on both `g_comm` (Serial5→ESP32) and `g_comm_usb` (Serial→PC). Dispatches on `cmd_id`. |
| Radio handling | `teensy/src/main.cpp:436-553` `radio_update()` | iBUS switches: CH5 calib, CH6 jump, CH7 pitch trim, CH9 profile, CH10 arm. Edge-detect pattern lives here. |
| Comm helpers | `teensy/src/main.cpp:31-73` `comm_log()`, `send_param_report()` | Pattern for sending Teensy→PC frames on both links. |
| Telemetry struct | `shared/comm_protocol.h:181-251` `TelemetryPayload` | 235 B, packed, `TELEM_VERSION 8`. **Do NOT change — keep version 8.** |
| Command IDs | `shared/comm_protocol.h:285-297` | `CMD_ID_*`. Next free id = `0x12`. |
| Packet types | `shared/comm_protocol.h:24-33` | `COMM_TYPE_*`. Next free = `0x12`, `0x13`. |
| Payload cap | `shared/CommLink/CommLink.h:24` `COMM_MAX_PAYLOAD 512` | Hard ceiling for any frame payload. |
| Teensy build | `teensy/platformio.ini` | `board=teensy41`, `framework=arduino`, `lib_extra_dirs=../shared`. **No SD lib yet.** |
| ESP32 relay | `esp32/src/main.cpp:517-556` `on_teensy_packet()` | Fans Teensy packets to USB (always) + TCP (if client) + UDP (telem only). PC→Teensy forward is `forward_to_teensy()` (~`main.cpp:501-507`), forwards only `COMM_TYPE_COMMAND`. |
| GUI packet decode | `software/gui/flash_monitor.py:310-404` `PacketDecoder._parse()` | Byte-stream state machine; emits dict to `TelemetryBus`. |
| Split telem decoders | `software/gui/flash_monitor.py:211-307` `_decode_telem_a/_b` | Module-level, importable. Reuse for log playback. |
| Telemetry bus | `software/gui/telemetry_bus.py` `TelemetryBus` | Singleton, `pyqtSignal(dict)`. All tabs subscribe. |
| Command senders | `software/gui/comm_commands.py` | `build_frame()`, `send_frame()` (WiFi TCP + serial), `send_*()` helpers. |
| WiFi transport | `software/gui/wifi_transport.py` | UDP recv :5005, TCP send :5006. Has its own `PacketDecoder("wifi")`. |
| Source arbitration | `software/gui/source_manager.py` | `is_active(device)` gates who emits to `TelemetryBus` (`flash_monitor.py:402-403`). |
| GUI tabs | `software/gui/main.py:699-710` | `QTabWidget` assembly; add new tab here. |

**Bridge data-flow reminder:**
- PC→Teensy commands: PC → (TCP :5006 or USB) → ESP32 `forward_to_teensy()` →
  Serial5 → Teensy. Only `COMM_TYPE_COMMAND` is forwarded. **Our log control uses
  `CMD_ID_LOG` under `COMM_TYPE_COMMAND`, so this path needs no ESP32 change.**
- Teensy→PC: Teensy → Serial5 → ESP32 `on_teensy_packet()` → USB + TCP (all types)
  + UDP (telem only). **Our replies (`LOG_INFO`/`LOG_DATA`) must ride USB/TCP
  (reliable), never UDP (lossy).** Phase 4 verifies the ESP32 forwards all types.

---

## 3. Design overview

```
Teensy 500 Hz loop ──fill LogRecord every tick──► SdFat RingBuf (RAM, DMAMEM)
                                                        │ writeOut() 1 sector/tick
                                                        ▼
                                        LOGxxxx.WLOG on microSD (preallocated)
                                                        │  (GET request)
   PC GUI ──CMD_ID_LOG(GET)──► ESP32 ──UART──► Teensy ──LOG_DATA chunks──► ESP32 ──► PC
                                                        │
                              .wlog file on PC ──► LogReader ──► TelemetryBus ──► all tabs
                                                             └─► wlog_to_csv.py ──► CSV/pandas
```

**Why wrap `TelemetryPayload` instead of adding `micros` to it:** keeps the live
50 Hz wire format (`TELEM_A`/`TELEM_B`, version 8) **completely unchanged** — no
ripple through the ESP32 relay or the GUI split-frame decoder / propagation
checklist. The PC log reader reads `t_micros`, then feeds the embedded 235-byte
`telem` blob straight into the **existing** `_decode_telem_a`/`_decode_telem_b` —
zero schema duplication.

---

## 4. Wire/file formats (add these verbatim to `shared/comm_protocol.h`)

Add packet types after `COMM_TYPE_TELEM_B` (line 33):

```c
#define COMM_TYPE_LOG_INFO     0x12  // Teensy→PC: SD-log directory / transfer metadata (LogInfoPayload)
#define COMM_TYPE_LOG_DATA     0x13  // Teensy→PC: SD-log file chunk (LogDataHeader + raw bytes)
```

Add command id + sub-commands after `CMD_ID_PARAM_GET` (line 297):

```c
#define CMD_ID_LOG        0x12  // payload: uint8_t sub_cmd [, args] — high-datarate SD logging

// Log sub-commands (CMD_ID_LOG payload byte 1)
#define LOG_SUB_START     0x01  // + uint32_t duration_ms (0 = log until STOP)
#define LOG_SUB_STOP      0x02  // no args — close the active log file
#define LOG_SUB_LIST      0x03  // no args — reply: one LOG_INFO ENTRY per file, then LIST_END
#define LOG_SUB_GET       0x04  // + uint16_t file_index, uint32_t start_chunk — stream LOG_DATA
#define LOG_SUB_DELETE    0x05  // + uint16_t file_index — erase one .wlog file
```

Add the record/header/transfer structs after the `TelemetryPayload` static_asserts
(after line 262). **`WlogHeader` and `LogRecord` reference `TelemetryPayload`, so
they must appear after its definition.** Keep `TELEM_VERSION` = 8, unchanged.

```c
// ── High-datarate SD log (.wlog) ──────────────────────────────────────────────
//
// The Teensy logs one LogRecord per 500 Hz control tick to a preallocated .wlog
// file on the built-in microSD. A LogRecord WRAPS the unchanged TelemetryPayload
// (so the live 50 Hz wire format is untouched) and prepends a micros() timestamp.
// The PC reads t_micros, then decodes the embedded 235-byte telem blob with the
// SAME split-telemetry decoder used for live data. See software/gui/log_playback.py.
//
#define WLOG_FORMAT_V1  1
#define WLOG_SAMPLE_HZ  500      // control/log tick rate [Hz]
#define LOG_CHUNK_DATA  480      // max file bytes per COMM_TYPE_LOG_DATA frame

typedef struct __attribute__((packed)) {
    char     magic[8];         // "WLRLOG\0" (7 used + 1 pad)
    uint8_t  format_version;   // WLOG_FORMAT_V1
    uint8_t  telem_version;    // == TELEM_VERSION at capture time (decode key)
    uint16_t record_size;      // == sizeof(LogRecord)
    uint16_t sample_rate_hz;   // WLOG_SAMPLE_HZ
    uint32_t start_millis;     // millis() at sd_logger_start()
    uint8_t  reserved[14];
} WlogHeader;                  // 32 bytes — file header

typedef struct __attribute__((packed)) {
    uint32_t         t_micros; // micros() at capture — sub-ms inter-tick timing
    TelemetryPayload telem;    // the exact 235-byte struct, TELEM_VERSION 8
} LogRecord;                   // 239 bytes — one per control tick

// ── Payload: COMM_TYPE_LOG_INFO (Teensy→PC) ──────────────────────────────────
#define LOG_INFO_ENTRY       0x01  // one directory entry (reply to LIST): file_index, file_size
#define LOG_INFO_LIST_END    0x02  // end of directory listing
#define LOG_INFO_XFER_BEGIN  0x03  // start of a GET: file_index, file_size, total_chunks
#define LOG_INFO_XFER_END    0x04  // end of a GET: file_index, total_chunks, crc32
#define LOG_INFO_STATUS      0x05  // START/STOP/DELETE ack: file_index, status (0=ok)

typedef struct __attribute__((packed)) {
    uint8_t  info_type;    // LOG_INFO_*
    uint16_t file_index;   // LOGxxxx index (0xFFFF = n/a)
    uint32_t file_size;    // bytes (ENTRY / XFER_BEGIN)
    uint32_t total_chunks; // XFER_BEGIN / XFER_END
    uint32_t crc32;        // XFER_END — CRC32 of the whole file
    uint8_t  status;       // STATUS: 0=ok, non-zero=error code
} LogInfoPayload;          // 16 bytes

// ── Payload: COMM_TYPE_LOG_DATA (Teensy→PC) ──────────────────────────────────
// Frame payload = LogDataHeader (8 B) + data_len raw file bytes (≤ LOG_CHUNK_DATA).
typedef struct __attribute__((packed)) {
    uint16_t file_index;
    uint32_t chunk_index;  // 0-based; file byte offset = chunk_index * LOG_CHUNK_DATA
    uint16_t data_len;     // raw file bytes following this header (≤ LOG_CHUNK_DATA)
} LogDataHeader;           // 8 bytes

#ifdef __cplusplus
static_assert(sizeof(WlogHeader) == 32, "WlogHeader must be 32 bytes");
static_assert(sizeof(LogRecord) == 239, "LogRecord must be sizeof(uint32)+sizeof(TelemetryPayload)");
static_assert(sizeof(LogInfoPayload) == 16, "LogInfoPayload must be 16 bytes");
static_assert(sizeof(LogDataHeader) == 8, "LogDataHeader must be 8 bytes");
static_assert(LOG_CHUNK_DATA + sizeof(LogDataHeader) <= COMM_MAX_PAYLOAD,
    "LOG_DATA frame exceeds COMM_MAX_PAYLOAD");
#endif
```

**CRC32:** use the standard IEEE 802.3 polynomial (`0xEDB88320` reflected). The
Teensy computes it while streaming; the PC verifies with Python's
`zlib.crc32(data) & 0xFFFFFFFF`. Make sure both use the same convention — validate
against a known vector in the CHECKPOINT.

---

## 5. Phase 0 — Shared protocol

**Files:** `shared/comm_protocol.h` only.

1. Add the packet types, command ids, and structs from §4.
2. Confirm `COMM_MAX_PAYLOAD` (512) already covers `LOG_DATA` (488 B) and
   `TELEM_*` — no change needed to `CommLink.h`.

### ✅ CHECKPOINT 0 — protocol compiles everywhere
- Build native/unit test env to exercise the `static_assert`s:
  `cd firmware/robot_teensy/teensy && pio run -e test_teensy` *(or whichever env is
  fastest to compile the shared header — `test_comm_usb` also pulls it in).*
- Also compile the ESP32 (`cd firmware/robot_teensy/esp32 && pio run`) since it
  includes `comm_protocol.h`. **All three deployments must still compile with the
  new structs before writing any logic.**
- **Pass criteria:** no `static_assert` failures, all envs build.

---

## 6. Phase 1 — Teensy SDLogger library

**Files:** create `teensy/lib/SDLogger/sd_logger.h` and `sd_logger.cpp`
(the dir exists with only `.gitkeep`).

Use **SdFat** (`greiman/SdFat`) with the canonical low-latency logger pattern
(preallocated contiguous file + `RingBuf` in RAM). Reference SdFat's
`examples/TeensySdioLogger` / `RingBuf` API.

**Public API (`sd_logger.h`):**
```c
bool     sd_logger_begin();               // init SDIO card. false = no card (non-fatal)
bool     sd_logger_available();           // card present & initialised
bool     sd_logger_start(uint32_t duration_ms);  // 0 = until stop. false on error
void     sd_logger_stop();
bool     sd_logger_is_active();
void     sd_logger_write(const LogRecord* rec);  // called EVERY tick when active — RAM memcpy
void     sd_logger_service();             // drain 1 sector/tick + periodic flush + auto-stop
uint16_t sd_logger_active_index();        // current LOGxxxx index (for status reporting)

// Retrieval (driven by on_command in main.cpp; all Teensy→PC replies via callbacks)
void     sd_logger_list();                        // emit LOG_INFO ENTRY per file + LIST_END
void     sd_logger_begin_get(uint16_t idx, uint32_t start_chunk); // arm a streaming transfer
void     sd_logger_service_transfer();            // pace 1-2 LOG_DATA chunks/tick; emit XFER_END
bool     sd_logger_transfer_active();
void     sd_logger_delete(uint16_t idx);          // erase + LOG_INFO STATUS ack
```

**Implementation notes:**
- Filenames: `LOG%04u.WLOG` (e.g. `LOG0001.WLOG`). On `start`, scan root for the
  next free index.
- Buffer: `static RingBuf<FsFile, 32768> rb;` backed conceptually by a `DMAMEM`
  buffer to keep it out of fast RAM1 (~270 ms cushion at 119 KB/s). Follow the
  exact SdFat RingBuf idiom for the installed version.
- `sd_logger_start`: `preAllocate()` a large contiguous file (e.g. 64 MB), write
  `WlogHeader` (fill `telem_version = TELEM_VERSION`, `record_size = sizeof(LogRecord)`,
  `sample_rate_hz = WLOG_SAMPLE_HZ`, `start_millis = millis()`), then `rb.begin(&file)`.
  Store the auto-stop deadline if `duration_ms != 0`.
- `sd_logger_write`: `rb.write(rec, sizeof(LogRecord))`. If `rb.getWriteError()`
  (overrun — RAM buffer full because SD stalled too long), set a sticky overflow
  flag surfaced via `comm_log(WARN)` and in status. **Never block here.**
- `sd_logger_service`: `if (rb.bytesUsed() >= 512 && !file_full) rb.writeOut(512);`
  `file.flush()` ~once/second; auto-stop when `millis() >= deadline`.
- `sd_logger_stop`: `rb.sync()`/drain remaining, `file.truncate()` to actual bytes
  written, `file.close()`, then emit a `LOG_INFO STATUS` ack.
- **Transfer** (`begin_get`/`service_transfer`): open the file read-only, seek to
  `start_chunk * LOG_CHUNK_DATA`, and on each `service_transfer()` read + send
  **1–2** `LOG_DATA` frames (header + up to 480 B). Accumulate CRC32 across the
  whole file. When EOF: emit `LOG_INFO XFER_END` with `total_chunks` + `crc32`,
  close, clear active flag. Pacing keeps per-tick work bounded (SDIO read is µs;
  the 512 B `Serial5` TX buffer returns immediately — see `main.cpp:190-193`).
- The library must not directly own the CommLink instances. Give it a **send
  callback** it calls to emit frames, wired from `main.cpp` (which owns `g_comm` /
  `g_comm_usb`). E.g. `void sd_logger_set_sender(void (*fn)(uint8_t type, uint8_t ver, const void* p, uint16_t n));`
  and `main.cpp` sets a function that sends on both links (mirroring `comm_log`).

**Do not** wire it into `main.cpp` yet — this phase is the library in isolation.

### ✅ CHECKPOINT 1 — library compiles
- Add `greiman/SdFat` to `teensy/platformio.ini` `lib_deps` (Phase 3 also touches
  this file; doing it here is fine).
- `cd firmware/robot_teensy/teensy && pio run` — must compile with the new lib even
  though nothing calls it yet (reference it from a throwaway `#include` if the
  linker strips it, or just confirm the TU compiles).
- **Pass criteria:** clean build, no SdFat API mismatches. Resolve any RingBuf API
  differences against the actually-installed SdFat version **now**.

---

## 7. Phase 2 — Teensy main.cpp wiring

**File:** `teensy/src/main.cpp` (+ include `sd_logger.h`).

1. **Refactor `send_telemetry()` (`main.cpp:277-351`)** into two functions:
   - `static void fill_telemetry(TelemetryPayload& t)` — the ~65 field assignments
     (lines 278-343), operating on the reference.
   - `static void send_telemetry()` — declares a local `TelemetryPayload`, calls
     `fill_telemetry()`, then does the two `g_comm.send()` / `g_comm_usb.send()`
     splits (lines 344-350) exactly as today.
   - *This is the one existing function meaningfully restructured — it is directly
     required to populate a record every tick without duplicating field wiring.
     Behaviour of the 50 Hz path must be byte-for-byte identical.*

2. **Boot init:** call `sd_logger_begin()` in `setup()` (after `Serial5`/CommLink
   are up so `comm_log` works). Wire `sd_logger_set_sender(...)` to a static
   function that sends on `g_comm` and (if `Serial`) `g_comm_usb`, mirroring the
   pattern in `comm_log()` (`main.cpp:31-44`).

3. **Loop hook (`loop()`, `main.cpp:582-602`):** after the existing per-tick work,
   before the telemetry divider:
   ```c
   if (sd_logger_is_active()) {
       static LogRecord rec;
       fill_telemetry(rec.telem);
       rec.t_micros = micros();
       sd_logger_write(&rec);
   }
   sd_logger_service();            // 1 sector/tick + auto-stop
   sd_logger_service_transfer();   // paced chunk streaming during a GET
   ```
   Keep all of this inside the existing 2 ms budget; the trailing spin-wait absorbs
   the bounded SD work. **Do not** add a second `fill` for the 50 Hz path — it keeps
   its own local as today (fine to fill twice per 10 ticks; cheap).

4. **`on_command()` (`main.cpp:77-184`):** add a `CMD_ID_LOG` branch (mirror the
   existing `CMD_ID_PARAM_*` branches):
   ```c
   if (cmd_id == CMD_ID_LOG && len >= 2) {
       uint8_t sub = payload[1];
       if (sub == LOG_SUB_START) {
           uint32_t dur = 0; if (len >= 6) memcpy(&dur, payload + 2, 4);
           sd_logger_start(dur);
       } else if (sub == LOG_SUB_STOP)   { sd_logger_stop(); }
       else if (sub == LOG_SUB_LIST)     { sd_logger_list(); }
       else if (sub == LOG_SUB_GET && len >= 8) {
           uint16_t idx; uint32_t start;
           memcpy(&idx, payload + 2, 2); memcpy(&start, payload + 4, 4);
           sd_logger_begin_get(idx, start);
       } else if (sub == LOG_SUB_DELETE && len >= 4) {
           uint16_t idx; memcpy(&idx, payload + 2, 2);
           sd_logger_delete(idx);
       }
       return;
   }
   ```
   Add `comm_log(INFO, ...)` lines to match the existing command logging style.

5. **RC-switch STUB (`radio_update()`, `main.cpp:436-553`):** add an edge-detected
   hook but **leave the channel unassigned** (see §9). It must be a no-op until a
   channel is chosen. Example:
   ```c
   // TODO(user): assign a spare iBUS channel for log start/stop.
   // CH5/6/7/9/10 are taken (calib/jump/trim/profile/arm). Set to a real
   // channel index (1-based) to enable; LOG_SWITCH_CH == 0 keeps this disabled.
   static constexpr uint8_t LOG_SWITCH_CH = 0;   // 0 = unassigned (stub)
   if (LOG_SWITCH_CH != 0) {
       bool on = g_ibus.channel(LOG_SWITCH_CH) > 1500;
       static bool prev = false;
       if (on && !prev) { if (!sd_logger_is_active()) sd_logger_start(0); }
       if (!on && prev) { sd_logger_stop(); }
       prev = on;
   }
   ```
   Keep the edge-detect shape consistent with the existing jump/arm switches.

### ✅ CHECKPOINT 2 — Teensy builds + 50 Hz telemetry unaffected
- `cd firmware/robot_teensy/teensy && pio run` — clean build.
- Flash to hardware. Connect the GUI. **Confirm live 50 Hz telemetry still works
  exactly as before** (no version-mismatch banner, all tabs animate). The refactor
  must not have changed the wire output.
- **Pass criteria:** clean build + unchanged live telemetry.

---

## 8. Phase 3 — CSV tool + first real logging test (DE-RISK EARLY)

> Rationale: validate the **on-SD file format** by pulling the card, *before*
> building the whole retrieval stack. If the format is wrong, we find out now.

**File:** create `software/gui/tools/wlog_to_csv.py` — a **standalone, no-Qt**
script.

- Reads a `.wlog`: parse `WlogHeader`, then iterate `LogRecord`s (`t_micros` +
  235 B telem blob).
- Reuse the telem field layout. Import the format strings from `flash_monitor.py`
  if clean to do so **without pulling in Qt** (they are module-level `_FMT_TELEM_A`
  / `_FMT_TELEM_B` + `_decode_telem_a/_b`). If importing `flash_monitor` drags Qt,
  instead copy the two `struct` format strings into a tiny shared helper
  `software/gui/telem_format.py` and import from there in **both** places (single
  source of truth). Prefer the shared-helper route if there's any Qt coupling.
- Output: pandas DataFrame → CSV (one row per sample; columns = `t_micros` +
  all telemetry fields). CLI: `python wlog_to_csv.py LOG0001.WLOG [out.csv]`.
- Handle the self-describing header: assert `telem_version == 8` (warn + best-effort
  if not), use `record_size` to stride.

### ✅ CHECKPOINT 3 — real 500 Hz log captured & decodes
- On hardware: trigger a log via a **temporary** hard-coded `sd_logger_start(...)`
  or a quick serial CLI hook, OR wait until Phase 5 gives a GUI button. Simplest
  now: temporarily start logging for ~10 s using a timed `sd_logger_start(10000)`
  called once from `setup()` behind a `#if 0` you flip on, then revert.
- Log ~30 s while the robot balances (or on the bench moving the IMU/legs by hand).
- **Also capture loop timing:** temporarily record max & 95th-pctile `loop()`
  duration while logging and print via `comm_log`. Confirm overruns beyond 2 ms are
  rare and small (this is the key safety check for the logger).
- Pull the microSD, run `wlog_to_csv.py LOG0001.WLOG`.
- **Pass criteria:** ~500 rows/s, monotonically increasing `t_micros` (~2000 µs
  steps), sane field values, CSV opens in pandas. Loop overruns rare/small. Revert
  the temporary logging trigger and timing instrumentation afterward.

---

## 9. Phase 4 — ESP32 forwarding check

**File:** `esp32/src/main.cpp` (change only if needed).

- Read `on_teensy_packet()` (`main.cpp:517-556`). Confirm it forwards **all** packet
  types to USB (`g_usb.send`) and TCP (`g_comm_tcp->send`) — not a type-gated
  switch. The exploration says it does; **verify in the actual code.**
- If it is type-gated (only forwards `TELEM_*`/`ACK`/`LOG`/etc.), add
  `COMM_TYPE_LOG_INFO` and `COMM_TYPE_LOG_DATA` to the USB + TCP forward path.
  **Do not** add them to the UDP path (UDP is lossy; transfers must be reliable).
- `CMD_ID_LOG` rides `COMM_TYPE_COMMAND`, which `forward_to_teensy()` already
  forwards — no change for the command direction.

### ✅ CHECKPOINT 4 — ESP32 relays log frames
- `cd firmware/robot_teensy/esp32 && pio run` — clean build; flash.
- Manually trigger `LOG_SUB_LIST` from the Teensy (temporary hook) and confirm the
  PC receives `LOG_INFO` frames **over both USB and WiFi TCP** (a raw hex dump in
  the GUI serial monitor or a `PacketDecoder` print is enough).
- **Pass criteria:** `LOG_INFO`/`LOG_DATA` frames arrive on the PC over USB and TCP.

---

## 10. Phase 5 — GUI command senders

**File:** `software/gui/comm_commands.py`.

Add (mirror the existing `send_*` helpers; add `CMD_ID_LOG = 0x12` + `LOG_SUB_*`
constants near the other `CMD_ID_*`):
```python
def send_log_start(duration_ms: int = 0): ...   # <BBI: CMD_ID_LOG, LOG_SUB_START, duration_ms
def send_log_stop(): ...                          # <BB
def send_log_list(): ...                          # <BB
def send_log_get(file_index: int, start_chunk: int = 0): ...  # <BBHI
def send_log_delete(file_index: int): ...         # <BBH
```
All go through the existing `send_frame()` (WiFi TCP + serial).

### ✅ CHECKPOINT 5 — start/stop from Python works
- From a Python REPL or a temporary button, call `send_log_start(5000)` with the
  robot connected. Confirm the Teensy creates a `LOGxxxx.WLOG` (pull card or check
  a `LOG_INFO STATUS` ack). `send_log_stop()` closes it.
- **Pass criteria:** logging starts/stops on command over the live link.

---

## 11. Phase 6 — GUI decode + transfer manager

**File:** `software/gui/flash_monitor.py` (+ maybe a new `log_transfer.py`).

1. In `PacketDecoder._parse()` add branches for `ptype == 0x12` (`LOG_INFO`) and
   `0x13` (`LOG_DATA`), unpacking `LogInfoPayload` / `LogDataHeader`. Put decoded
   info into the emitted dict (so a manager can consume it) — **do not** route these
   onto `TelemetryBus` as telemetry.
2. New `LogTransferManager` (`log_transfer.py`) — a `QObject` with signals for
   `directory_updated(list)`, `transfer_progress(idx, got, total)`,
   `transfer_complete(idx, path, crc_ok)`:
   - Subscribes to `PacketDecoder.packet_decoded` (both serial + wifi decoders).
   - `LOG_INFO ENTRY` → build directory list; `LIST_END` → emit `directory_updated`.
   - `LOG_INFO XFER_BEGIN` → allocate a buffer / sparse dict of chunks for
     `total_chunks`.
   - `LOG_DATA` → store chunk by `chunk_index`, update progress.
   - `LOG_INFO XFER_END` → assemble in order, verify `zlib.crc32` vs `crc32`; if
     gaps or CRC mismatch, re-issue `send_log_get(idx, first_missing_chunk)`; on
     success write `software/gui/logs/LOGxxxx.wlog` and emit `transfer_complete`.
   - Timeout/retry: if no chunk for N ms mid-transfer, re-request from the first
     missing chunk.

### ✅ CHECKPOINT 6 — over-the-air download matches card-pull
- From a temporary "download" call, LIST then GET the file captured in Phase 3.
- Download over **WiFi**, then over **USB**.
- Compare the downloaded `.wlog` byte-for-byte with the card-pulled copy from
  Phase 3 (`fc` / `Compare-Object` / hashes). Verify CRC32 reported OK.
- **Pass criteria:** downloaded file == card-pulled file, over both transports.
  Note transfer time for a ~1-min log (expect ~15–30 s).

---

## 12. Phase 7 — Logs tab + rosbag playback

**Files:** `software/gui/log_playback.py` (new), `software/gui/telemetry_bus.py`,
`software/gui/main.py`.

1. `telemetry_bus.py`: add a `playback_active: bool = False` attribute on the
   `TelemetryBus` singleton.
2. `flash_monitor.py:402-403`: gate the live emit with
   `and not TelemetryBus.instance().playback_active` so a live source doesn't fight
   playback.
3. `log_playback.py`:
   - `LogReader` — opens a `.wlog`, reads `WlogHeader`, yields per-record dicts by
     reading `t_micros` and decoding the embedded 235 B telem blob with the **same**
     `_decode_telem_a`/`_decode_telem_b` (via the shared `telem_format.py` from
     Phase 3), setting `ptype=0x01`, `type_name="TELEM"` so consumers treat it like
     live telemetry.
   - A **"Logs" tab** (`QWidget`) with two panels:
     - **Retrieve:** Refresh (`send_log_list`), file table (index/size from
       `LogTransferManager.directory_updated`), Download (progress bar), Delete,
       plus **Start/Stop/Timed** logging buttons (`send_log_start/stop`).
     - **Playback:** open a `.wlog`, transport (play/pause, scrub `QSlider`,
       speed 0.1×–4×, step). A `QTimer` drives replay; on each step set
       `TelemetryBus.playback_active = True` and emit the next record onto
       `TelemetryBus.instance().packet`. Clearing playback re-enables live sources.
     - **Export CSV** button → call the Phase 3 converter on the open file.
   - Register the tab in `main.py:699-710` (`QTabWidget`), e.g. after "Flash &
     Monitor".

### ✅ CHECKPOINT 7 — full playback in the GUI
- Open a downloaded `.wlog` in the Logs tab, press Play.
- **Pass criteria:** the 3D visualizer, IMU charts, and Raw Data tab all animate
  from the log; scrub + speed work; live sources are suppressed while playing and
  resume after. Export CSV produces the same result as Phase 3.

---

## 13. Final checkpoint — end-to-end acceptance

Run the whole loop with **no temporary hooks** (all instrumentation reverted):

1. GUI **Start** → robot balances ~30 s → **Stop** (and a **Timed 10 s** run that
   auto-stops).
2. GUI **Refresh** → **Download** the newest file over WiFi (and once over USB).
3. **Play** it back in the Logs tab; scrub/speed.
4. **Export CSV**; open in pandas; sanity-check a jump or torque-saturation event.
5. Confirm live telemetry is unaffected before/after (no version mismatch).
6. RC-switch stub present but disabled (`LOG_SWITCH_CH == 0`), documented for the
   user to assign a channel later.

---

## 14. Files touched (summary)

**Create:**
- `firmware/robot_teensy/teensy/lib/SDLogger/sd_logger.h`
- `firmware/robot_teensy/teensy/lib/SDLogger/sd_logger.cpp`
- `software/gui/tools/wlog_to_csv.py`
- `software/gui/telem_format.py` *(only if needed to avoid Qt coupling — Phase 3)*
- `software/gui/log_transfer.py`
- `software/gui/log_playback.py`

**Modify:**
- `firmware/robot_teensy/shared/comm_protocol.h` — packet types, cmd ids, structs.
- `firmware/robot_teensy/teensy/src/main.cpp` — `fill_telemetry` split, loop hook,
  `on_command` `CMD_ID_LOG`, RC stub, boot init + sender wiring.
- `firmware/robot_teensy/teensy/platformio.ini` — add `greiman/SdFat`.
- `firmware/robot_teensy/esp32/src/main.cpp` — only if forwarding is type-gated.
- `software/gui/comm_commands.py` — log command senders.
- `software/gui/flash_monitor.py` — decode `LOG_INFO`/`LOG_DATA`; `playback_active`
  gate on the live emit.
- `software/gui/telemetry_bus.py` — `playback_active` flag.
- `software/gui/main.py` — register the Logs tab.

---

## 15. Risks & mitigations

| Risk | Mitigation |
|---|---|
| SD write stall makes one control tick > 2 ms (loop has no catch-up) | Preallocated contiguous file (no FAT alloc stalls); aligned 1-sector `writeOut`; 32 KB RAM `RingBuf` cushion (~270 ms). **CHECKPOINT 3 measures actual overruns.** Use a good/high-endurance card. |
| RingBuf overrun (SD too slow to keep up sustained) | Sticky overflow flag → `comm_log(WARN)` + status; visible in GUI. 119 KB/s is well within SDIO write bandwidth, so only pathological cards overflow. |
| Transfer starves/ jitters the loop | Pace 1–2 chunks/tick; SDIO read is µs; `Serial5` has a 512 B TX buffer that returns immediately. Transfers normally run while disarmed. |
| UART Teensy↔ESP32 drops during transfer | Reliable TCP/USB end-to-end from ESP32→PC; CRC32 over whole file + chunk-gap re-request handle the lossy UART hop. |
| Schema drift PC vs firmware | `.wlog` header stores `telem_version`; single source of truth for the telem format string (`telem_format.py`); keep `TELEM_VERSION` = 8. |
| ESP32 forwarding assumption wrong | CHECKPOINT 4 verifies before the GUI transfer stack is built. |

---

## 16. Open item for the user

- **RC log switch channel:** the stub is wired (`LOG_SWITCH_CH == 0`, disabled).
  Assign a spare iBUS channel (CH5/6/7/9/10 are taken) and set `LOG_SWITCH_CH` to
  enable start/stop from the transmitter.
