# GUIlog.md — unify GUI-side (.jsonl) capture with SD (.wlog) logs in the Log Analyzer

Status: **plan only, not implemented**. Written after adding host-side capture
(`tabs/host_logger.py`, "GUI:" button in the status bar) so `LogAnalyzerTab`
can open either flavor of log and get identical metrics/plots out of it.

## Background

Two logging paths now exist:

- **SD (Teensy)** — `tabs/log_transfer.py` / firmware `sd_logger.cpp`. Fixed-size
  binary `LogRecord` (4-byte `t_micros` header + full `TelemetryPayload`)
  written once per control tick, gap-free by construction. Read by
  `analysis/wlog_metrics.py::decode_wlog()`.
- **GUI (host)** — `tabs/host_logger.py`. One JSON line per `TelemetryBus`
  packet as it's received (TELEM, robot LOG, CALIB, WIFI_DIAG), each prefixed
  with `host_ts`. Arrival is whatever the transport delivers — no fixed rate,
  can have real gaps (dropped WiFi packets), unlike the SD path.

`tabs/log_analyzer_tab.py` currently only opens `.wlog`, via
`analysis/wlog_metrics.decode_wlog()` → `DecodedRun` → `compute_metrics()` →
plotting code keyed off `run.fields`/`run.t_s`.

## Design: unify at the `DecodedRun` layer, not the file format

`DecodedRun` (dataclass in `analysis/wlog_metrics.py`: `path`, `telem_version`,
`sample_rate_hz`, `count`, `t_s: np.ndarray`, `fields: dict[str, np.ndarray]`,
`has_gain_sched_alpha`) is already the shared interface every consumer
(`compute_metrics`, `check_safety`, the fitness functions, all of
`LogAnalyzerTab`'s plot flavors, `tools/analyze_hw_run.py`) is built on.
`decode_wlog()` is just one producer of it.

**Plan: add a second producer, `decode_hostlog(path) -> DecodedRun`,** that
reads a `.jsonl` capture and emits the same shape. Nothing downstream needs to
know which producer ran — that's the whole point, and it's why extending
`log_analyzer_tab.py` to accept both formats is *not* separate work from this:
`_load()` just needs to pick a decoder by file extension.

### `decode_hostlog()` — new function in `analysis/wlog_metrics.py`

1. Read the `.jsonl` file line by line, `json.loads` each line.
2. Keep only `ptype == 1` (TELEM) records — these already carry the exact
   field names `decode_telem_full()` produces, since `HostLogger` wrote the
   `TelemetryBus` packet dict verbatim. Non-TELEM lines (robot LOG, CALIB,
   WIFI_DIAG) are skipped for `DecodedRun` purposes — see "Future" below.
3. `t_s`: use each record's own `timestamp_ms` field (already present —
   it's in `_SCALAR_FIELDS`), not `host_ts`. Confirmed on the firmware side
   (`teensy/src/main.cpp:1189`) that `.wlog`'s `t_micros` header is
   `micros()` stamped right next to `fill_telemetry()` — i.e. functionally
   the same instant as the payload's own `timestamp_ms`. Using
   `timestamp_ms` for both keeps the time base consistent regardless of
   source, and avoids host-side wall-clock jitter (`host_ts`) leaking into
   the physics.
4. `sample_rate_hz`: compute an effective rate for metadata/display
   (`round(1 / np.median(np.diff(t_s)))`), but see the dt-handling change
   below — this value stops being load-bearing for the actual math.
5. `telem_version` / `has_gain_sched_alpha`: read off the first TELEM
   record's own fields, same check `decode_wlog()` already does against
   `TELEM_VERSION`.
6. Raise the same kind of `ValueError` as `decode_wlog()` on: empty file, no
   TELEM records found, or a `telem_version` mismatch — `LogAnalyzerTab`
   already has a catch block for this (`_load()`'s `except (ValueError,
   OSError)`).

### `compute_metrics()` — real per-sample dt instead of a constant

Chosen approach (over "keep constant dt, flag as approximate"): **use the
real per-sample dt**, because this module's own docstring states its numbers
drive the automated hardware accept/reject decision — a WiFi drop silently
skewing a fitness score is not an acceptable tradeoff for saving a small
diff here.

- ISE integration: replace `ise_pitch = sum(err^2) * dt` (constant `dt`) with
  `dt_arr = np.diff(t_s, prepend=t_s[0] - 1/sample_rate_hz)` and
  `ise_pitch = sum(err^2 * dt_arr)`. For `.wlog` runs `dt_arr` is uniform
  (real dt == nominal dt), so this is a no-op there — only jsonl runs with
  real gaps get a different number, and it's the *more correct* one.
- Oscillation FFT (`_oscillation_check`): `np.fft.rfft` requires uniform
  sampling. Resample the post-settle pitch segment onto a uniform grid via
  `np.interp(t_uniform, seg_t, seg)` (`t_uniform = np.arange(seg_t[0],
  seg_t[-1], 1/sample_rate_hz)`) before the FFT. For `.wlog` runs this
  resample is close to a no-op (already near-uniform); for jsonl runs it's
  what makes the FFT meaningful at all.
- `_settle_time()` already walks `t` directly (not index-gridded), so no
  change needed there.

### `log_analyzer_tab.py` — branch on extension, nothing else changes

```python
def _load(self, path: Path):
    decode = decode_hostlog if path.suffix.lower() == ".jsonl" else decode_wlog
    try:
        run = decode(path)
    except (ValueError, OSError) as e:
        ...  # unchanged
```

- Widen the `_on_open()` file dialog filter to
  `"Log files (*.WLOG *.wlog *.jsonl)"`.
- No changes needed to `_redraw()`, any plot flavor, `_set_meta_rows()`, or
  the safety label — all already only touch `run`/`m`.

## Future extension (not in scope here)

Robot LOG (ptype 0x04) and CALIB (0x05) lines in a `.jsonl` capture carry
information a `.wlog` never had (free-text fault/event messages with exact
timestamps). Once the base unification above is in, a natural follow-up is
overlaying those as vertical markers/annotations on the Log Analyzer plot —
but only for jsonl-sourced runs, since `.wlog` has no equivalent data to draw
from. Not needed for `compute_metrics()`/fitness parity, so left out of this
plan.

## Files touched (when implemented)

- `software/gui/analysis/wlog_metrics.py` — add `decode_hostlog()`; change
  `compute_metrics()`'s dt handling (ISE + FFT) as above.
- `software/gui/tabs/log_analyzer_tab.py` — branch `_load()` on file
  extension; widen the open-dialog filter.
