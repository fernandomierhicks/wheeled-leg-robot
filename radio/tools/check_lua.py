#!/usr/bin/env python3
"""Run the staged Lua against a mock EdgeTX API.

Syntax-checking Lua proves almost nothing about a widget: every interesting
code path lives inside a property function that only the firmware ever calls.
This harness loads the widget through radio/tools/edgetx_stub.lua, builds it at
several zone sizes, and drives it through the scenarios that actually happen --
no link, link but no robot telemetry, running normally, and each fault tier --
calling every property function on every frame.

It also enforces the firmware's own strictness: lua_lvgl_widget.cpp raises
"Invalid property" for any unrecognised key, so a typo that would kill the
script on the radio fails here instead.

Needs `lupa` (pip install lupa).

Usage:
    python check_lua.py
"""
import sys
from pathlib import Path

try:
    from lupa import LuaRuntime
except ImportError:
    sys.exit("check_lua.py needs lupa:  pip install lupa")

REPO = Path(__file__).resolve().parents[2]
SD = REPO / "radio" / "sdcard"
STUB = Path(__file__).resolve().parent / "edgetx_stub.lua"

# Sensor sets. "native" is what EdgeTX creates on its own from the standard
# CRSF frames; "robot" is the custom frame that firmware does not emit yet.
NATIVE = {
    "Ptch": -6.1, "Roll": 3.0, "Yaw": 12.0,
    "RxBt": 24.1, "Curr": 4.2, "Bat%": 82.0,
    "RQly": 99.0, "1RSS": -71.0, "TPWR": 100.0,
}
ROBOT = {
    "Stat": 3.0, "Flt": 0.0, "Alph": 0.42,
    "HipL": 2.7, "HipR": 2.9, "WVel": 0.55,
    "Jump": 0.0, "SUp": 0.0, "Hlth": 127.0,
    "Glch": 0.0, "E32": 1.0, "Prof": 1.0,
}

ZONES = [
    ("full screen 480x320", 0, 0, 480, 320, True),
    ("half screen 480x150", 0, 34, 480, 150, False),
    ("quarter 236x150", 0, 34, 236, 150, False),
    ("tile 120x70", 0, 34, 120, 70, False),
]

# Floor on how much each layout must draw. Catches the failure where a layout
# branch quietly produces an empty screen -- which looks exactly like a working
# one until you are on the bench wondering why the HUD is blank.
MIN_OBJECTS = {
    "full screen 480x320": 60,
    "half screen 480x150": 20,
    "quarter 236x150": 10,
    "tile 120x70": 3,
}


def lua_setup():
    lua = LuaRuntime(unpack_returned_tuples=True)
    lua.execute("S = nil")
    stub_src = STUB.read_text(encoding="utf-8")
    chunk = lua.eval("load([==[%s]==], 'edgetx_stub.lua')" % stub_src)
    if chunk is None:
        sys.exit("edgetx_stub.lua failed to parse")
    S = chunk()
    S.root = str(SD).replace("\\", "/")
    lua.globals().S = S
    return lua, S


def syntax_check(lua):
    # load() returns one value on success and two on failure, so normalise to
    # a single error string rather than unpacking a variable-length tuple.
    checker = lua.eval(
        "function(s, n)"
        "  local fn, err = load(s, n)"
        "  if fn then return '' end"
        "  return tostring(err)"
        "end")
    problems = []
    for p in sorted(SD.rglob("*.lua")):
        err = checker(p.read_text(encoding="utf-8"), p.name)
        if err:
            problems.append("%s: %s" % (p.relative_to(SD), err))
    return problems


def set_sensors(lua, S, table):
    lua.execute("S.sensors = {}")
    for k, v in table.items():
        lua.globals().S.sensors[k] = v


def run_widget(lua, S, scenario, sensors, rssi, zones=ZONES, frames=6):
    """Load main.lua fresh, build at each zone, and run several frames."""
    problems = []
    src = (SD / "WIDGETS" / "WLRHUD" / "main.lua").read_text(encoding="utf-8")
    chunk = lua.eval("load([==[%s]==], 'WLRHUD/main.lua')" % src)
    if chunk is None:
        return ["WLRHUD/main.lua: failed to load"]
    widget = chunk()

    for name in ("name", "create", "update", "refresh", "background"):
        if widget[name] is None:
            problems.append("main.lua does not return %r" % name)
    if problems:
        return problems
    if len(widget["name"]) > 10:
        problems.append("widget name %r is over the 10-char limit"
                        % widget["name"])

    set_sensors(lua, S, sensors)
    S.rssi = rssi
    built = {}
    played = set()

    for label, x, y, w, h, fullscreen in zones:
        S.reset()
        S.fullScreen = fullscreen
        S.time = 0
        zone = lua.table(x=x, y=y, w=w, h=h)
        opts = lua.table(Sounds=1, Accent=0x00A8FF)

        try:
            wgt = widget["create"](zone, opts)
            widget["update"](wgt, opts)
            for f in range(frames):
                S.time = S.time + 20          # 200 ms per frame
                widget["refresh"](wgt, None, None)
                widget["background"](wgt)
                S.callRefs()
        except Exception as exc:                      # noqa: BLE001
            problems.append("%s / %s: %s" % (scenario, label, exc))
            continue

        # A layout that silently builds nothing looks identical to one that
        # works, until you are staring at a blank screen on the bench.
        if S.objectCount < MIN_OBJECTS[label]:
            problems.append("%s / %s: built %d objects, expected at least %d"
                            % (scenario, label, S.objectCount,
                               MIN_OBJECTS[label]))
        built[label] = S.objectCount

        # Every wav the annunciator reached for must actually be on the card.
        # A missing file is silent on the radio -- the worst possible failure
        # mode for the one subsystem whose entire job is to be heard.
        for i in range(1, len(S.played) + 1):
            f = S.played[i]["file"]
            if f is not None:
                played.add(f)

        for i in range(1, len(S.errors) + 1):
            problems.append("%s / %s: %s" % (scenario, label, S.errors[i]))
        lua.execute("S.errors = {}")

    for f in sorted(played):
        if not (SD / f.lstrip("/")).exists():
            problems.append("%s: played missing sound %s" % (scenario, f))

    return problems, built, played


def short(path):
    return path.rsplit("/", 1)[-1].replace(".wav", "")


# A scripted flight. The annunciator's whole job lives in TRANSITIONS -- state
# changes, a fault firing, the link dropping -- and a scenario that holds one
# frozen sensor set never exercises any of them. This walks the robot through a
# session and asserts the radio said the right things at the right moments.
FLIGHT = [
    # (label, sensor overrides, sounds that must be heard during this step)
    ("boot into standby", {"Stat": 2.0, "Flt": 0.0}, []),
    ("calibrate",         {"Stat": 1.0}, ["st_calib"]),
    ("arm and run",       {"Stat": 3.0}, ["st_run"]),
    ("jump",              {"Stat": 7.0, "Jump": 1.0}, ["st_jump"]),
    ("back to running",   {"Stat": 3.0, "Jump": 0.0}, ["st_run"]),
    # Pitch watchdog: REPOSITION tier, so two-tone plus the spoken fault.
    ("pitch watchdog",    {"Stat": 4.0, "Flt": 8.0}, ["t_repos", "f_pitchw"]),
    ("rescue to standby", {"Stat": 2.0, "Flt": 0.0}, ["st_stby"]),
    # Hardware dropout: REBOOT tier, siren.
    ("hip feedback lost", {"Stat": 4.0, "Flt": 3.0}, ["t_siren", "f_hipfb"]),
    ("esp32 drops",       {"Stat": 4.0, "E32": 0.0}, ["w_esp"]),
    ("battery critical",  {"RxBt": 19.1}, ["w_batcrt"]),
]


def run_flight(lua, S):
    problems, spoken = [], set()
    src = (SD / "WIDGETS" / "WLRHUD" / "main.lua").read_text(encoding="utf-8")
    widget = lua.eval("load([==[%s]==], 'WLRHUD/main.lua')" % src)()

    sensors = dict(NATIVE, **ROBOT)
    set_sensors(lua, S, sensors)
    S.rssi = 71
    S.reset()
    S.fullScreen = True
    S.time = 0

    wgt = widget["create"](lua.table(x=0, y=0, w=480, h=320),
                           lua.table(Sounds=1, Accent=0x00A8FF))
    widget["update"](wgt, lua.table(Sounds=1, Accent=0x00A8FF))

    # Two settling frames so the annunciator adopts the initial state instead
    # of announcing it -- loading a widget should not shout at you.
    for _ in range(2):
        S.time = S.time + 20
        widget["refresh"](wgt, None, None)

    for label, overrides, expect in FLIGHT:
        sensors.update(overrides)
        set_sensors(lua, S, sensors)
        lua.execute("S.played = {}")
        for _ in range(3):
            S.time = S.time + 30
            widget["refresh"](wgt, None, None)
            S.callRefs()

        heard = set()
        for i in range(1, len(S.played) + 1):
            f = S.played[i]["file"]
            if f is not None:
                heard.add(short(f))
                spoken.add(f)
        for want in expect:
            if want not in heard:
                problems.append(
                    "transition %r: expected to hear %s, heard %s"
                    % (label, want, sorted(heard) or "nothing"))

    for i in range(1, len(S.errors) + 1):
        problems.append("transitions: %s" % S.errors[i])
    lua.execute("S.errors = {}")

    for f in sorted(spoken):
        if not (SD / f.lstrip("/")).exists():
            problems.append("transitions: played missing sound %s" % f)

    return problems, spoken


def main():
    lua, S = lua_setup()

    problems = syntax_check(lua)
    if problems:
        print("Lua syntax errors:", file=sys.stderr)
        for p in problems:
            print("  - %s" % p, file=sys.stderr)
        return 1
    print("syntax ok: %d files" % len(list(SD.rglob("*.lua"))))

    scenarios = [
        ("no link at all", {}, 0),
        ("link up, no robot telemetry", dict(NATIVE), 71),
        ("running normally", dict(NATIVE, **ROBOT), 71),
    ]
    # One scenario per fault tier, so every branch of the banner and of the
    # annunciator's tier table gets executed.
    for code, label in ((6, "SOFT"), (8, "REPOSITION"), (4, "GUI_FIX"),
                        (3, "REBOOT"), (99, "unknown fault code")):
        s = dict(NATIVE, **ROBOT)
        s["Stat"], s["Flt"] = 4.0, float(code)
        scenarios.append(("fault %d (%s)" % (code, label), s, 71))
    # Extremes: the arithmetic paths that clamp, and the states that switch
    # the phase line on.
    edge = dict(NATIVE, **ROBOT)
    edge.update({"Ptch": -47.0, "Roll": 61.0, "Alph": 1.4, "WVel": -3.3,
                 "HipL": 9.9, "HipR": -0.0, "RxBt": 19.2, "RQly": 12.0,
                 "Hlth": 0.0, "Glch": 812.0, "E32": 0.0, "Stat": 7.0,
                 "Jump": 4.0, "Prof": 2.0})
    scenarios.append(("saturated / degraded", edge, 3))
    up = dict(NATIVE, **ROBOT)
    up.update({"Stat": 8.0, "SUp": 3.0})
    scenarios.append(("standing up", up, 71))

    failures = []
    all_played = set()
    for label, sensors, rssi in scenarios:
        found, built, played = run_widget(lua, S, label, sensors, rssi)
        all_played |= played
        status = "FAIL" if found else "ok  "
        shape = "  ".join("%s=%d" % (k.split()[0], v)
                          for k, v in built.items())
        print("%s  %-28s  objects: %s  sounds: %d"
              % (status, label, shape, len(played)))
        failures.extend(found)

    found, spoken = run_flight(lua, S)
    failures.extend(found)
    all_played |= spoken
    print("%s  %-28s  sounds: %s"
          % ("FAIL" if found else "ok  ", "state/fault transitions",
             ", ".join(sorted(short(s) for s in spoken))))

    if failures:
        print("\n%d problem(s):" % len(failures), file=sys.stderr)
        for f in failures:
            print("  - %s" % f, file=sys.stderr)
        return 1

    print("\nall scenarios clean: %d property callbacks exercised, "
          "%d distinct sounds played, all present on the card"
          % (S.propCalls, len(all_played)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
