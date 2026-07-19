# UARTplat.md — Teensy ↔ ESP32 UART Link: Phase 7 Hardware Campaign

Executor: Claude Sonnet. This file was rewritten on 2026-07-18 to hand Phase 7
off to a fresh conversation — Phases 1-6 (all pure code/firmware/GUI work)
are done and committed. Read this whole file before touching anything.
Follow CLAUDE.md rules (simplest solution first, compile after every change,
don't touch unrelated code). Commit after each completed step; ask before
pushing (not yet pushed as of this handoff — see §0).

**Start at §2.** §0 and §1 are context to understand what's already in place
and why; §2 is the actual task.

---

## 0. What's already done (Phases 1-6 — do not redo)

Everything below is implemented, compiled, and committed to `main` (local
only — not pushed). Commits in order (`git log --oneline` to confirm):

1. `ESP32 Phase 1: decouple uplink sends from UART parse path` — the
   root-cause fix: `on_teensy_packet()` now only enqueues; a new `uplink_task`
   (core 0) is the sole writer to Serial/TCP/UDP. TCP client mutex. Unified
   `TEENSY_LINK_TIMEOUT_MS` (1500 ms). "No Teensy" Neopixel changed to a dim
   red pulse.
2. `Teensy Phase 2: UART hardening + stale artifact cleanup` — Serial5 RX
   ring 64 B → 2048 B. Paced PARAM_GET/RESET_DEFAULTS dumps (4/tick, not a
   burst). Deleted `teensy/lib/Esp32Link/`, `test/test_esp32_link/`, and
   `test/test_telemetry/` (all stale/broken relative to the current
   protocol — see that commit message for why `test_telemetry` was also
   removed beyond the original plan's literal scope).
3. `Phase 3: ESP32<->Teensy link supervision, TELEM V9, WIFI_DIAG V2` — new
   `COMM_TYPE_ESP32_STATUS` (0x16) heartbeat, ESP32→Teensy at 5 Hz.
   `TelemetryPayload` 235→242 bytes (`TELEM_VERSION` 9). `WifiDiagPayload`
   26→38 bytes (V2). GUI decode updated and verified end-to-end with
   synthetic frames (including V1-length backward-compat).
4. `Phase 4: GUI tri-state header, link-health panel, TCP backoff` —
   Connected/ESP32-ONLY-NO-TEENSY/Disconnected header (250 ms periodic
   re-evaluation, not per-packet-restarted). New `LinkHealthWidget` on the
   Dashboard tab (9 monitored fields, thresholds in one constants block).
   TCP reconnect exponential backoff in `wifi_transport.py`.
5. `Add calibration-bypass param...` — **not mine**, a pre-existing
   uncommitted change from an unrelated earlier session, split into its own
   commit rather than bundled into Phase 5.
6. `Phase 5: effect-confirmed command retry via ReliableCommand` — GUI-side
   retry against observed telemetry effects (no protocol change). Wired into
   the ESTOP/Reset/Reboot buttons and the Params tab's per-row Set. **Caught
   and fixed a real bug via testing, not review**: the returned
   `ReliableCommand` wasn't kept alive by callers (matching its own
   fire-and-forget API), so it was garbage-collected before its retry timer
   could ever fire — a signal connection alone does not keep a plain QObject
   alive in PyQt6. Fixed with a class-level `_in_flight` keep-alive set.
7. `Phase 6: wifi_load_sim.py load generator + native CommLink robustness
   test` — new headless, pyserial-free load generator mirroring the GUI's
   network load (ping/param-dump/TCP-churn), for the campaign in §2 below.
   Also a native (`platform = native`) CommLink parser robustness test —
   **written and manually traced against `CommLink.cpp`'s exact state
   machine, but never actually compiled**: no native C/C++ toolchain (gcc/
   g++/clang/MSVC) is installed on this machine. Run
   `pio test -e native` (from `firmware/robot_teensy/teensy/`) once a
   toolchain is available, before trusting that test.
8. `Update firmware README for Phases 1-3's UART/telemetry changes` — fixed
   stale byte counts and the deleted-folder reference.

**Verified so far** (code-level only — no hardware has been flashed with any
of this fix yet):
- `pio run` clean for both `firmware/robot_teensy/esp32` and
  `firmware/robot_teensy/teensy`.
- GUI decode verified end-to-end with hand-built synthetic CommLink frames
  (not just import/size checks) for TELEM_A/B V9 and WIFI_DIAG V2.
- `LinkHealthWidget` + `StatusBar` tri-state transitions verified with a
  real `QApplication` and synthetic packets.
- `ReliableCommand` retry/confirm/reject verified with a real Qt event loop.
- `wifi_load_sim.py`'s frame builder verified byte-identical CRC-8 to the
  real implementation; IP auto-detect and TCP reconnect verified against
  local mock sockets.

**Known pre-existing, unrelated dirty files — leave alone, do not bundle
into any Phase 7 commit** (confirmed by diffing each against HEAD before
touching anything nearby; this is other in-progress work, a per-motor
bench-test/calibration-bypass effort, not part of this plan):
- `firmware/robot_teensy/teensy/src/state_machine.cpp`
- `firmware/robot_teensy/teensy/lib/ParamRegistry/param_ids.h`
- `firmware/robot_teensy/teensy/lib/ParamRegistry/param_registry.cpp`
- `software/gui/tabs/controllers_tab.py`
- `software/gui/tabs/imu_tab.py`
- `software/gui/tabs/robot_visualizer_tab.py`
- `software/gui/tabs/theme.py`

These currently make `test_hip_motor` and `test_imu` fail under
`pio test -e test_teensy` (a linker gap from the in-progress state-machine
work, and a missing `Adafruit_BusIO_Register.h` library dependency,
respectively) — confirmed pre-existing and unrelated to Phases 1-6. If
`git status` ever shows something unexpected dirty beyond this known list,
stop and ask rather than assuming it's safe to touch or bundle in.

**Commits are not pushed.** Ask the user before pushing, per standing
instructions — don't assume earlier permission to flash/test hardware also
covers pushing to a remote.

---

## 1. Context — why this work exists (condensed)

Symptom: the ESP32 intermittently declared **NO TEENSY** on the TFT,
especially when the Python GUI was open in WiFi mode. Root cause (fixed in
Phase 1): `on_teensy_packet()` ran USB/TCP/UDP sends inline inside the core-1
UART parse loop; a slow blocking TCP/UDP write under WiFi load stalled the
parser, making the link look dead even though UART bytes kept arriving. A
prior WiFi capture campaign (evidence: `software/gui/logs/wifi_campaign_summary.csv`)
showed **zero CRC failures / malformed frames** across 7 runs (ruling out
electrical corruption) but a 20-minute unicast soak degraded to 31.6%
datagram loss with `udp_send_fail_count = 37,175` — confirming the ESP32
send path, not the wire, was the bottleneck.

Phases 2-3 hardened the UART path (bigger RX ring, paced param dumps) and
added bidirectional link-health telemetry so both firmwares and the GUI can
see link health precisely, independent of each other. Phases 4-6 built GUI
visibility, command-reliability retry, and the test tooling this phase uses.
Full original root-cause writeup and CommLink protocol details, if ever
needed: `git log -p -- UARTplat.md` (this file's history) or the Phase 1-3
commit messages above.

**Decisions already made with the user** (still binding for Phase 7):
- Keep 4 Mbaud UART (no evidence of electrical corruption).
- Both boards USB-connected for flashing/monitoring; load tests reproduce
  the failure over WiFi (`wifi_capture.py` + `wifi_load_sim.py`).
- Motors are disabled via params (all 4 `*_enable` = 0) — safe to reflash
  and test freely.
- No new faults / no ESTOP behavior changes — link supervision is
  telemetry-only.
- **Run a short sanity check before any long campaign** (decided
  2026-07-18, see §2.1): flash the fix, run a short (~2-3 min) combined
  capture/load-sim pass, look at the numbers, and check in with the user
  before committing to the full 70+ minute campaign in §2.2. Do not launch
  the long (15-30 min) runs without checking in first.

---

## 2. Phase 7 — Hardware test campaign (the actual task)

Motors are param-disabled; both boards are safe to reflash and test freely.
**The GUI app must be closed during `pio` uploads** (it holds the COM
ports). Identify ports with `pio device list`: Teensy 4.1 = VID 16C0, ESP32
CP2102 = VID 10C4:EA60 (same signatures as
`software/gui/tabs/port_manager.py SIGNATURES`).

**Before doing anything else**, confirm with the user that both boards are
connected via USB right now and the GUI is closed — as of this handoff,
`pio device list` returned nothing (no boards attached), so don't assume
they're connected. If a board doesn't come back up after flashing, ask the
user to physically reset/replug it rather than attempting a programmatic
reset.

### 2.0 Tool usage reference

`python software/gui/tools/wifi_load_sim.py --duration <secs> [--ping-hz N]
[--dump-period N] [--reboot-teensy-at N] [--esp32-ip A.B.C.D]` — headless
load generator; auto-detects the ESP32's IP from the first UDP telemetry
datagram (needs the ESP32 already on WiFi and past STARTUP). See its module
docstring for full behavior.

`python software/gui/tools/wifi_capture.py --duration <secs> --out
logs/<name>.jsonl [--esp32-ip A.B.C.D]` — captures one JSON line per UDP
datagram (decoded TELEM/WIFI_DIAG fields + arrival timestamp/seq/crc_ok).

`python software/gui/tools/analyze_wifi_capture.py logs/<name>.jsonl` —
computes loss/jitter/clumping/reorder metrics from a capture (check its
`--help`; not modified in Phases 1-6 — confirmed it has no hardcoded payload
sizes that needed updating, unlike `wifi_capture.py` itself).

Run capture + load-sim **concurrently** (two terminals or background
processes) for any of the steps below.

### 2.1 Short sanity check — do this first

1. Flash the current `main` (Phases 1-6) to both boards via the GUI's Flash
   & Monitor tab or `pio run -t upload` in each of
   `firmware/robot_teensy/esp32` and `firmware/robot_teensy/teensy` (GUI
   closed first). No need to separately reproduce the old bug on the
   pre-fix commit first — the user wants to see the fix work, not re-prove
   the symptom (that's already documented in §1/commit history; §2.2 covers
   a true side-by-side if it's ever wanted later).
2. Run for ~2-3 minutes concurrently:
   ```
   python software/gui/tools/wifi_capture.py --duration 150 --out logs/uartfix_sanity.jsonl
   python software/gui/tools/wifi_load_sim.py --duration 150
   ```
3. Analyze: `python software/gui/tools/analyze_wifi_capture.py logs/uartfix_sanity.jsonl`.
   Check against the same criteria as the full campaign (§2.2's table):
   telemetry gaps > 1 s (want 0), UART CRC drops/seq gaps both directions
   (want 0/0 — these are the new V9/V2 fields, confirm the analyzer surfaces
   them), `uplink_queue_drops` (want ~0), `udp_send_fail_count` (want low).
4. Report the numbers to the user in plain terms (pass/fail against the
   table) and ask whether to proceed to §2.2's full campaign, stop here, or
   investigate anything that looks off.

### 2.2 Full campaign (only if the user asks for it after §2.1)

**7.1 — Optional true baseline on the pre-fix commit.** Current `main`
already has the fix, so reproducing the *old* buggy behavior requires
temporarily checking out the commit before this work: `6d4292a` (the commit
immediately before "ESP32 Phase 1" in the log above) — i.e.
`git log --oneline` and confirm `6d4292a` is still the right parent before
using it. `git checkout 6d4292a`, flash both boards, run the same
capture+load-sim pair for 15 min
(`--out logs/uartfix_baseline.jsonl`), analyze, **then `git checkout main`
and reflash the fix before continuing** — don't leave the boards on old
firmware or the tree checked out on a detached HEAD. Only do this if the
user specifically wants numbers to compare against, since §1 already has
baseline numbers from the original campaign.

**7.2 — Unit-level checks.** Already done throughout Phases 1-6 (`pio run`
after every change; see §0). Nothing further needed here unless new changes
are made during Phase 7 itself, in which case: `pio run -d
firmware/robot_teensy/teensy` and `pio run -d firmware/robot_teensy/esp32`
after each.

**7.3 — Fixed-firmware verification.** Repeat §2.1's load for the full 15
minutes instead of ~2-3. Pass criteria:

| Metric | Requirement |
|---|---|
| Telemetry gaps > 1 s (NO TEENSY events) | **0** |
| UART CRC drops / seq gaps (both directions) | **0 / 0** |
| `uplink_queue_drops` | 0 (a handful acceptable only during param-dump bursts) |
| `udp_send_fail_count` | < 10 total |
| Tick loss | < 0.5 % |
| Inter-tick jitter p95 | < 50 ms |

**7.4 — Soak + stress:**
1. **30-min soak**: same as 7.3 with `--duration 1800`. Same pass criteria —
   this is the test the pre-fix firmware failed catastrophically (31.6%
   loss at 20 min, per the original campaign).
2. **Param-dump stress**: `wifi_load_sim.py --dump-period 2 --duration 300`
   — dump every 2 s under load. Criteria: no CRC/gap increments, telemetry
   gaps 0, Teensy `HEALTH_LOOP_OVERRUN` not set (visible in `health_flags`).
3. **Teensy-dead test** (validates the ESP32-alive feature from Phase 3/4):
   while a capture runs, `wifi_load_sim.py --reboot-teensy-at <N>` — confirm
   WIFI_DIAG datagrams keep arriving through the Teensy reboot window,
   `wifi_teensy_link_up` goes False→True, and telemetry resumes after. In
   the GUI this is the amber "ESP32 ONLY — NO TEENSY" header state — verify
   it live in the GUI at the end (not automatable headlessly; the user can
   watch).
4. **TCP client churn**: start/stop `wifi_load_sim.py` 5× mid-capture — no
   telemetry gaps, no counter growth (validates the Phase 1 TCP mutex +
   non-blocking accept path).

**7.5 — Report.** Write a results table (baseline vs. fixed, if 7.1 was
run; otherwise just the fixed-firmware numbers) to `logs/uartfix_report.md`.
Prompt to commit + push at each green milestone (per standing preference).

---

## 3. If anything here turns out stale

This file was accurate as of commit `cab031d` / the README update
immediately after it. If `git log` shows commits past that under a
different subject line, someone (or a prior Phase-7 session) has already
made progress beyond this handoff — read those commit messages first rather
than assuming this file is still the frontier.
