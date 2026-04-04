"""probe_firmware.py — Dump ODrive attribute tree and fingerprint firmware version.

Run with:   python probe_firmware.py
Output:     probe_results.txt  (in same folder)
"""

import odrive
import sys, io, os

# Force UTF-8 on stdout (Windows console often defaults to cp1252)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUT = []
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_results.txt")
_out_file = open(out_path, "w", encoding="utf-8", errors="replace")

def log(s=""):
    line = str(s)
    print(line, flush=True)
    OUT.append(line)
    _out_file.write(line + "\n")
    _out_file.flush()   # write incrementally so a crash never truncates


def safe_repr(val):
    try:
        return repr(val)
    except Exception as e:
        return f"<repr-ERR: {e}>"


def safe_get(obj, attr):
    try:
        return getattr(obj, attr)
    except Exception as e:
        return f"ERR({e})"


_visited = set()

def dump_tree(obj, path, depth=0, max_depth=8):
    """Recursively walk every attribute of obj and log it."""
    obj_id = id(obj)
    if obj_id in _visited:
        log(f"{'  '*depth}  {path}  [already visited, skipping]")
        return
    _visited.add(obj_id)

    try:
        attrs = [a for a in dir(obj) if not a.startswith("_")]
    except Exception as e:
        log(f"{'  '*depth}  {path}  [dir() ERR: {e}]")
        return

    for a in attrs:
        full = f"{path}.{a}"
        try:
            val = getattr(obj, a)
        except Exception as e:
            log(f"{'  '*depth}  {full} = ERR({e})")
            continue

        if isinstance(val, (int, float, str, bool, bytes, type(None))):
            log(f"{'  '*depth}  {full} = {safe_repr(val)}")
            continue

        # Check if it has sub-attributes (is a compound object)
        try:
            sub_attrs = [x for x in dir(val) if not x.startswith("_")]
        except Exception:
            sub_attrs = []

        is_compound = bool(sub_attrs) and not isinstance(val, type)

        if callable(val) and not is_compound:
            log(f"{'  '*depth}  {full}()  [callable]")
        elif is_compound and depth < max_depth:
            n = len(sub_attrs)
            if callable(val):
                log(f"{'  '*depth}  {full}()  [callable+object, {n} attrs]")
            else:
                log(f"{'  '*depth}  {full}  [object, {n} attrs]")
            dump_tree(val, full, depth + 1, max_depth)
        else:
            log(f"{'  '*depth}  {full} = {safe_repr(val)}")


# ── Connect ────────────────────────────────────────────────────────────────────
log("=" * 70)
log("ODrive Firmware Probe — full attribute tree + fingerprints")
log("=" * 70)
log()
log("Connecting...")
try:
    odrv = odrive.find_any(timeout=10)
except Exception as e:
    log(f"FAILED to connect: {e}")
    _out_file.close()
    sys.exit(1)
log("Connected.")
log()

# ── Version info ───────────────────────────────────────────────────────────────
log("── Version Attributes " + "─" * 48)
for attr in ["hw_version_major", "hw_version_minor", "hw_version_variant",
             "fw_version_major", "fw_version_minor", "fw_version_revision",
             "fw_version_unreleased", "serial_number", "user_config_loaded"]:
    log(f"  odrv0.{attr} = {safe_get(odrv, attr)!r}")
log()

# ── Full recursive tree ────────────────────────────────────────────────────────
log("── Full attribute tree (depth ≤ 8) " + "─" * 34)
log()
dump_tree(odrv, "odrv0", depth=0, max_depth=8)
log()

# ── Version fingerprint tests ──────────────────────────────────────────────────
log("── Version Fingerprint Tests " + "─" * 41)
fingerprints = [
    ("get_adc_voltage(4)                        [new in 0.5.6]",
     lambda: odrv.get_adc_voltage(4)),
    ("odrv.can                                  [new in 0.5.4+]",
     lambda: dir(odrv.can)),
    ("axis0.controller.config.anticogging       [0.5.4+]",
     lambda: dir(odrv.axis0.controller.config.anticogging)),
    ("axis0.config.startup_homing               [0.5.4+]",
     lambda: odrv.axis0.config.startup_homing),
    ("axis0.trap_traj.config                    [0.5.0+]",
     lambda: dir(odrv.axis0.trap_traj.config)),
    ("axis0.controller.config.inertia           [0.5.1+]",
     lambda: odrv.axis0.controller.config.inertia),
    ("axis0.controller.config.input_filter_bandwidth [0.5.1+]",
     lambda: odrv.axis0.controller.config.input_filter_bandwidth),
    ("axis0.config.general_lockin               [0.5.1+]",
     lambda: dir(odrv.axis0.config.general_lockin)),
    ("axis0.sensorless_estimator                [0.5.0+]",
     lambda: dir(odrv.axis0.sensorless_estimator)),
    ("odrv.config.enable_brake_resistor         [0.5.0+]",
     lambda: odrv.config.enable_brake_resistor),
    ("axis0.acim_estimator                      [ACIM, 0.5.x]",
     lambda: dir(odrv.axis0.acim_estimator)),
    ("odrv.config.dc_bus_overvoltage_trip_level [0.5.3+]",
     lambda: odrv.config.dc_bus_overvoltage_trip_level),
    ("odrv.config.dc_bus_undervoltage_trip_level [0.5.3+]",
     lambda: odrv.config.dc_bus_undervoltage_trip_level),
    ("axis0.motor.config.current_control_bandwidth [0.5.1+]",
     lambda: odrv.axis0.motor.config.current_control_bandwidth),
    ("axis0.encoder.config.enable_phase_interpolation [0.5.4+]",
     lambda: odrv.axis0.encoder.config.enable_phase_interpolation),
    ("axis0.encoder.config.use_index_offset     [0.5.5+]",
     lambda: odrv.axis0.encoder.config.use_index_offset),
    ("odrv.config.gpio_modes                    [0.5.5+ style]",
     lambda: odrv.config.gpio_modes),
    ("axis0.controller.move_to_pos()            [pre-0.5.2 only]",
     lambda: odrv.axis0.controller.move_to_pos),
    ("axis0.controller.move_incremental()       [0.5.2+]",
     lambda: odrv.axis0.controller.move_incremental),
]

for desc, fn in fingerprints:
    try:
        result = fn()
        log(f"  OK     {desc}")
        if isinstance(result, list) and len(result) < 30:
            log(f"         → {result}")
    except Exception as e:
        log(f"  ABSENT {desc}  ({e})")

log()
log("=" * 70)
log("Done.")
_out_file.close()
print(f"\nResults saved to {out_path}")
