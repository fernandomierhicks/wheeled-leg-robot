#!/usr/bin/env python3
"""Generate radio/sdcard/WIDGETS/WLRHUD/robotdef.lua from the firmware schema.

Single source of truth is firmware/robot_teensy/protocol/schema.json (states,
faults) plus two tables that live in C++ and have no schema representation yet:

  * fault severity tiers   -> shared/comm_protocol.h  fault_severity()
  * canonical state colour -> esp32/src/main.cpp      mode_color()

Both are mirrored here and CHECKED against their C++ source, so a firmware edit
that this file has not caught up with fails the generator instead of silently
producing a radio that annunciates the wrong recovery action.

Usage:
    python gen_robotdef_lua.py            # write robotdef.lua
    python gen_robotdef_lua.py --check    # exit 1 if the file is stale
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCHEMA = REPO / "firmware" / "robot_teensy" / "protocol" / "schema.json"
COMM_H = REPO / "firmware" / "robot_teensy" / "shared" / "comm_protocol.h"
ESP_CPP = REPO / "firmware" / "robot_teensy" / "esp32" / "src" / "main.cpp"
OUT = REPO / "radio" / "sdcard" / "WIDGETS" / "WLRHUD" / "robotdef.lua"

# -- Mirrors of the two C++ tables --------------------------------------------
# Severity tier -> annunciation policy on the radio. Tier names match
# fault_severity_t in comm_protocol.h.
SEVERITY = {
    "FAULT_HUMAN_ESTOP": "SOFT",
    "FAULT_WHEEL_RUNAWAY": "SOFT",
    "FAULT_PITCH_WATCHDOG": "REPOSITION",
    "FAULT_ROLL_WATCHDOG": "REPOSITION",
    "FAULT_CALIBRATION_TIMEOUT": "REPOSITION",
    "FAULT_STANDUP_FAILED": "REPOSITION",
    "FAULT_JUMP_TIMEOUT": "REPOSITION",
    "FAULT_HIP_LARGE_POS_CMD": "GUI_FIX",
}
SEVERITY_DEFAULT = "REBOOT"

# mode_color() in esp32/src/main.cpp, as 0xRRGGBB. The ESP32 writes these
# through tft.color565()/TFT_* constants; the radio keeps 24-bit and converts
# with lcd.RGB() at use.
STATE_COLOR = {
    "STATE_STARTUP":     0xFFFFFF,  # TFT_WHITE
    "STATE_CALIBRATION": 0x0000FF,  # TFT_BLUE
    "STATE_STANDBY":     0xFFFF00,  # TFT_YELLOW
    "STATE_RUNNING":     0x00FF00,  # TFT_GREEN
    "STATE_ESTOP":       0xFF0000,  # TFT_RED
    "STATE_MANUAL":      0x00FFFF,  # TFT_CYAN
    "STATE_CMD_REJECT":  0xFF6400,  # color565(255, 100, 0)
    "STATE_JUMPING":     0xC800FF,  # color565(200, 0, 255)
    "STATE_STANDING_UP": 0xFF3C00,  # color565(255, 60, 0)
    "STATE_DISARMING":   0xFFB400,  # color565(255, 180, 0)
}

# Short display names. The HUD state chip is narrow and the CRSF FLIGHT_MODE
# frame is a short string, so long state names get an abbreviation.
STATE_SHORT = {
    "STATE_CALIBRATION": "CALIB",
    "STATE_CMD_REJECT":  "REJECT",
    "STATE_STANDING_UP": "STANDUP",
}

# Spoken-callout wav basenames under /SOUNDS/en/WLR/. Kept to 8 chars or less
# so the EdgeTX Play-Track picker can browse them if you also wire Special
# Functions to them.
STATE_WAV = {
    "STATE_STARTUP":     "st_boot",
    "STATE_CALIBRATION": "st_calib",
    "STATE_STANDBY":     "st_stby",
    "STATE_RUNNING":     "st_run",
    "STATE_ESTOP":       "st_estop",
    "STATE_MANUAL":      "st_man",
    "STATE_CMD_REJECT":  "st_rej",
    "STATE_JUMPING":     "st_jump",
    "STATE_STANDING_UP": "st_up",
    "STATE_DISARMING":   "st_disar",
}

FAULT_WAV = {
    "FAULT_IMU_ERROR":           "f_imuerr",
    "FAULT_HIP_INIT_TIMEOUT":    "f_hipini",
    "FAULT_HIP_FEEDBACK_LOST":   "f_hipfb",
    "FAULT_HIP_LARGE_POS_CMD":   "f_hipjmp",
    "FAULT_CALIBRATION_TIMEOUT": "f_calib",
    "FAULT_HUMAN_ESTOP":         "f_human",
    "FAULT_PITCH_WATCHDOG":      "f_pitchw",
    "FAULT_WHEEL_RUNAWAY":       "f_runawy",
    "FAULT_IMU_LOST":            "f_imulos",
    "FAULT_WHEEL_FEEDBACK_LOST": "f_whlfb",
    "FAULT_WHEEL_INIT_TIMEOUT":  "f_whlini",
    "FAULT_STANDUP_FAILED":      "f_stndup",
    "FAULT_ROLL_WATCHDOG":       "f_rollw",
    "FAULT_JUMP_TIMEOUT":        "f_jumpto",
}

# Short fault labels for the HUD banner (schema descriptions are sentences).
FAULT_SHORT = {
    "FAULT_NONE":                "NONE",
    "FAULT_IMU_ERROR":           "IMU ERROR",
    "FAULT_HIP_INIT_TIMEOUT":    "HIP INIT TIMEOUT",
    "FAULT_HIP_FEEDBACK_LOST":   "HIP FEEDBACK LOST",
    "FAULT_HIP_LARGE_POS_CMD":   "HIP POS JUMP",
    "FAULT_CALIBRATION_TIMEOUT": "CALIB TIMEOUT",
    "FAULT_HUMAN_ESTOP":         "HUMAN ESTOP",
    "FAULT_PITCH_WATCHDOG":      "PITCH WATCHDOG",
    "FAULT_WHEEL_RUNAWAY":       "WHEEL RUNAWAY",
    "FAULT_IMU_LOST":            "IMU LOST",
    "FAULT_WHEEL_FEEDBACK_LOST": "WHEEL FEEDBACK LOST",
    "FAULT_WHEEL_INIT_TIMEOUT":  "WHEEL INIT TIMEOUT",
    "FAULT_STANDUP_FAILED":      "STANDUP FAILED",
    "FAULT_ROLL_WATCHDOG":       "ROLL WATCHDOG",
    "FAULT_JUMP_TIMEOUT":        "JUMP TIMEOUT",
}

JUMP_PHASE = ["CROUCH", "EXTEND", "RETRACT", "LANDING", "HANDOFF"]
STANDUP_PHASE = ["CROUCH", "RECOVER", "PAUSE", "STIFFEN"]

# The radio's stock fonts do not carry the typographic characters the schema
# descriptions use, and a stray multi-byte glyph renders as a box on a screen
# you are reading mid-fall. Fold to ASCII rather than trusting the font.
ASCII_FOLD = {
    "×": "x", "°": "deg", "·": ".", "‑": "-", "–": "-",
    "—": "-", "‘": "'", "’": "'", "“": "'", "”": "'",
    "…": "...", "µ": "u", "→": "->", "±": "+/-",
}


def to_ascii(text):
    for src, dst in ASCII_FOLD.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


def verify_against_cpp(states, faults):
    """Fail loudly if the C++ tables have moved on without this file."""
    problems = []

    comm = COMM_H.read_text(encoding="utf-8", errors="replace")
    body = comm.split("fault_severity(uint8_t code)", 1)
    if len(body) < 2:
        problems.append("could not find fault_severity() in comm_protocol.h")
    else:
        block = body[1].split("}", 2)[0]
        # Walk the switch: symbols accumulate until a `return TIER;` line.
        pending, found = [], {}
        for line in block.splitlines():
            m = re.search(r"case\s+(FAULT_\w+)\s*:", line)
            if m:
                pending.append(m.group(1))
            m = re.search(r"return\s+FAULT_SEVERITY_(\w+)\s*;", line)
            if m and pending:
                for sym in pending:
                    found[sym] = m.group(1)
                pending = []
        if found != SEVERITY:
            problems.append(
                "fault_severity() in comm_protocol.h no longer matches SEVERITY here.\n"
                "      C++ : %s\n      here: %s" % (found, SEVERITY)
            )

    esp = ESP_CPP.read_text(encoding="utf-8", errors="replace")
    cases = re.findall(r"case\s+RS_(\w+)\s*:\s*return", esp)
    if cases:
        want = set(s["symbol"][len("STATE_"):] for s in states)
        missing = want - set(cases)
        if missing:
            problems.append(
                "mode_color() in esp32/src/main.cpp has no case for: %s" % sorted(missing)
            )
    else:
        problems.append("could not find mode_color() cases in esp32/src/main.cpp")

    known_faults = set(f["symbol"] for f in faults)
    stray = set(SEVERITY) - known_faults
    if stray:
        problems.append("SEVERITY names faults absent from schema.json: %s" % sorted(stray))
    unwaved = known_faults - set(FAULT_WAV) - {"FAULT_NONE"}
    if unwaved:
        problems.append("no wav mapped for faults: %s" % sorted(unwaved))
    unlabelled = known_faults - set(FAULT_SHORT)
    if unlabelled:
        problems.append("no short label for faults: %s" % sorted(unlabelled))

    if problems:
        print("gen_robotdef_lua.py: firmware and radio definitions have diverged:\n",
              file=sys.stderr)
        for p in problems:
            print("  - %s\n" % p, file=sys.stderr)
        sys.exit(1)


def render(states, faults):
    L = []
    w = L.append
    w("-- GENERATED by radio/tools/gen_robotdef_lua.py -- DO NOT EDIT BY HAND.")
    w("-- Source: firmware/robot_teensy/protocol/schema.json")
    w("--         firmware/robot_teensy/shared/comm_protocol.h  (fault_severity)")
    w("--         firmware/robot_teensy/esp32/src/main.cpp      (mode_color)")
    w("-- Regenerate after any schema change; --check catches staleness in CI.")
    w("")
    w("local M = {}")
    w("")
    w("-- Robot states, keyed by the wire value of TelemetryPayload.robot_state.")
    w("-- colour is 0xRRGGBB and matches the ESP32 TFT / Neopixel table exactly,")
    w("-- so radio, robot lights and GUI never disagree about what state means.")
    w("M.states = {")
    for s in sorted(states, key=lambda s: s["id"]):
        sym = s["symbol"]
        short = STATE_SHORT.get(sym, s["name"])
        w('  [%d] = { name="%s", short="%s", colour=0x%06X, wav="%s" },'
          % (s["id"], s["name"], short, STATE_COLOR[sym], STATE_WAV[sym]))
    w("}")
    w("")
    w("-- Fault codes. tier drives how loudly the radio annunciates:")
    w("--   SOFT       single beep      -- ESTOP -> STANDBY, no re-init needed")
    w("--   REPOSITION two-tone + voice -- robot fell / calib failed; go pick it up")
    w("--   GUI_FIX    voice, no alarm  -- bad param, fix it before reset")
    w("--   REBOOT     siren + haptic   -- hardware dropout, power-cycle required")
    w("M.faults = {")
    for f in sorted(faults, key=lambda f: f["id"]):
        sym = f["symbol"]
        if sym == "FAULT_NONE":
            w('  [0] = { name="NONE", tier="SOFT", wav=nil, desc="" },')
            continue
        tier = SEVERITY.get(sym, SEVERITY_DEFAULT)
        desc = to_ascii(f.get("description", "")).replace('"', "'").replace("\n", " ")
        w('  [%d] = { name="%s", tier="%s", wav="%s",'
          % (f["id"], FAULT_SHORT[sym], tier, FAULT_WAV[sym]))
        w('        desc="%s" },' % desc)
    w("}")
    w("")
    w("M.jumpPhase    = { " + ", ".join('"%s"' % p for p in JUMP_PHASE) + " }")
    w("M.standupPhase = { " + ", ".join('"%s"' % p for p in STANDUP_PHASE) + " }")
    w("")
    w("-- health_flags bits, comm_protocol.h HEALTH_*")
    w("M.health = {")
    for i, (bit, label) in enumerate([
        ("HIP_L_OK", "hip L"), ("HIP_R_OK", "hip R"),
        ("WM_L_OK", "whl L"), ("WM_R_OK", "whl R"),
        ("HIP_LIMITS_VALID", "limits"), ("IMU_NOMINAL", "IMU"),
        ("LQR_ACTIVE", "LQR"), ("VEL_PI_SAT", "velSat"),
        ("YAW_PI_SAT", "yawSat"), ("WM_L_VEL_LIMITED", "govL"),
        ("WM_R_VEL_LIMITED", "govR"), ("LOOP_OVERRUN", "overrun"),
    ]):
        w('  %s = { bit=%d, label="%s" },' % (bit, 1 << i, label))
    w("}")
    w("")
    w("function M.state(id)")
    w('  return M.states[id] or { name="?", short="?", colour=0x808080, wav=nil }')
    w("end")
    w("")
    w("function M.fault(code)")
    w("  return M.faults[code] or")
    w('         { name=string.format("FAULT %d", code), tier="REBOOT", wav=nil, desc="" }')
    w("end")
    w("")
    w("return M")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if robotdef.lua is stale instead of writing it")
    args = ap.parse_args()

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    states, faults = schema["states"], schema["faults"]
    verify_against_cpp(states, faults)
    text = render(states, faults)

    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else None
        if current != text:
            print("%s is stale -- rerun gen_robotdef_lua.py" % OUT, file=sys.stderr)
            sys.exit(1)
        print("%s up to date" % OUT.name)
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print("wrote %s (%d states, %d faults)" % (OUT, len(states), len(faults)))


if __name__ == "__main__":
    main()
