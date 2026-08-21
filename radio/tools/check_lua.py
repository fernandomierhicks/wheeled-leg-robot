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
# EdgeTX creates the ATTITUDE sensors with unit RADIANS, so getValue() returns
# radians and the widget converts. Feeding degrees here would test the wrong
# scale and hide a real bug.
NATIVE = {
    "Ptch": -0.1064, "Roll": 0.0524, "Yaw": 0.2094,   # -6.1, 3.0, 12.0 degrees
    "RxBt": 24.1, "Bat%": 82.0,   # no Curr: firmware sends CRSF "no data"
    "RQly": 99.0, "1RSS": -71.0, "TPWR": 100.0,
}
# The robot-specific numerics no longer arrive as sensors. They ride a private
# CRSF frame (type 0x24) that EdgeTX hands to Lua, so the harness has to encode
# and push real bytes -- exactly the path the widget decodes.
WLR_FRAME_ID = 0x24

ROBOT = {
    "state": 3, "fault": 0, "jump": 0, "standup": 0, "alpha": 0.42,
    "profile": 1, "health": 127, "hip_l": 2.7, "hip_r": 2.9,
    "wheel": 0.55, "esp32": 1, "glitch": 0,
}


def _be16(v):
    v = int(round(v))
    if v < 0:
        v += 65536
    return [(v >> 8) & 0xFF, v & 0xFF]


def encode_wlr(f):
    """Mirror of crsf_build_wlr_state() in teensy/src/crsf_protocol.h."""
    d = [
        int(f["state"]) & 0xFF,
        int(f["fault"]) & 0xFF,
        int(f["jump"]) & 0xFF,
        int(f["standup"]) & 0xFF,
        max(0, min(255, int(round(f["alpha"] * 200)))),
        int(f["profile"]) & 0xFF,
        (int(f["health"]) >> 8) & 0xFF,
        int(f["health"]) & 0xFF,
    ]
    d += _be16(f["hip_l"] * 100)
    d += _be16(f["hip_r"] * 100)
    d += _be16(f["wheel"] * 100)
    g = min(65535, int(f["glitch"]))
    d += [int(f["esp32"]) & 0xFF, (g >> 8) & 0xFF, g & 0xFF]
    assert len(d) == 17, len(d)
    return d


def flight_mode_text(f, faults):
    """What crsf_telemetry_tick() would put in the FLIGHT_MODE frame."""
    code = int(f["fault"])
    if code:
        return "!" + faults.get(code, "FAULT")
    return STATE_NAMES.get(int(f["state"]), "?")


STATE_NAMES = {
    0: "STARTUP", 1: "CALIBRATION", 2: "STANDBY", 3: "RUNNING", 4: "ESTOP",
    5: "MANUAL", 6: "CMD_REJECT", 7: "JUMPING", 8: "STANDING_UP", 9: "DISARMING",
}
# Matches FAULT_FLIGHTMODE_NAMES in protocol/generate_protocol.py.
FAULT_SHORT = {
    1: "IMUERR", 2: "HIPINIT", 3: "HIPFB", 4: "HIPJUMP", 5: "CALTIME",
    6: "HUMAN", 8: "PITCHWD", 9: "RUNAWAY", 10: "IMULOST", 11: "WHLFB",
    12: "WHLINIT", 13: "STANDUP", 14: "ROLLWD", 15: "JUMPTO",
}


def push_wlr(lua, S, fields, with_frame=True):
    """Queue one 0x24 frame and set the matching FLIGHT_MODE text."""
    S.sensors["FM"] = flight_mode_text(fields, FAULT_SHORT)
    if not with_frame:
        return
    payload = lua.table(*encode_wlr(fields))
    lua.globals().crossfireTelemetryPush(WLR_FRAME_ID, payload)

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


def run_widget(lua, S, scenario, sensors, rssi, robot=None, with_frame=True,
               zones=ZONES, frames=6):
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
                if robot is not None:
                    push_wlr(lua, S, robot, with_frame)
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
    # (label, robot overrides, native overrides, expected sounds,
    #  needs_private_frame)
    ("boot into standby", {"state": 2, "fault": 0}, {}, [], False),
    ("calibrate",         {"state": 1}, {}, ["st_calib"], False),
    ("arm and run",       {"state": 3}, {}, ["st_run"], False),
    ("jump",              {"state": 7, "jump": 1}, {}, ["st_jump"], False),
    ("back to running",   {"state": 3, "jump": 0}, {}, ["st_run"], False),
    # Pitch watchdog: REPOSITION tier, so two-tone plus the spoken fault.
    ("pitch watchdog",    {"state": 4, "fault": 8}, {}, ["t_repos", "f_pitchw"], False),
    ("rescue to standby", {"state": 2, "fault": 0}, {}, ["st_stby"], False),
    # Hardware dropout: REBOOT tier, siren.
    ("hip feedback lost", {"state": 4, "fault": 3}, {}, ["t_siren", "f_hipfb"], False),
    # esp32_link_ok rides only the private frame -- there is nowhere to put it
    # in FLIGHT_MODE -- so this callout is expected to be silent without it.
    ("esp32 drops",       {"state": 4, "esp32": 0}, {}, ["w_esp"], True),
    ("battery critical",  {}, {"RxBt": 19.1}, ["w_batcrt"], False),
]


def run_flight(lua, S, with_frame=True):
    problems, spoken = [], set()
    src = (SD / "WIDGETS" / "WLRHUD" / "main.lua").read_text(encoding="utf-8")
    widget = lua.eval("load([==[%s]==], 'WLRHUD/main.lua')" % src)()

    sensors = dict(NATIVE)
    robot = dict(ROBOT)
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
        push_wlr(lua, S, robot, with_frame)
        widget["refresh"](wgt, None, None)

    for label, r_over, n_over, expect, needs_frame in FLIGHT:
        robot.update(r_over)
        sensors.update(n_over)
        set_sensors(lua, S, sensors)
        lua.execute("S.played = {}")
        for _ in range(3):
            S.time = S.time + 30
            push_wlr(lua, S, robot, with_frame)
            widget["refresh"](wgt, None, None)
            S.callRefs()

        heard = set()
        for i in range(1, len(S.played) + 1):
            f = S.played[i]["file"]
            if f is not None:
                heard.add(short(f))
                spoken.add(f)
        if needs_frame and not with_frame:
            continue          # this callout has no FLIGHT_MODE equivalent
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

    # (label, native sensors, rssi, robot fields or None, custom frame relays?)
    scenarios = [
        ("no link at all", {}, 0, None, True),
        ("link up, no robot telemetry", dict(NATIVE), 71, None, True),
        ("running normally", dict(NATIVE), 71, dict(ROBOT), True),
        # The degradation that matters: ExpressLRS relays ATTITUDE, BATTERY and
        # FLIGHT_MODE for certain, but whether it relays our private 0x24 frame
        # is unverified. If it does not, state and fault must still reach the
        # HUD through the FLIGHT_MODE text.
        ("custom frame does not relay", dict(NATIVE), 71, dict(ROBOT), False),
    ]
    # One scenario per fault tier, so every branch of the banner and of the
    # annunciator's tier table gets executed.
    for code, label in ((6, "SOFT"), (8, "REPOSITION"), (4, "GUI_FIX"),
                        (3, "REBOOT"), (99, "unknown fault code")):
        r = dict(ROBOT, state=4, fault=code)
        scenarios.append(("fault %d (%s)" % (code, label),
                          dict(NATIVE), 71, r, True))
        # And the same fault arriving only as FLIGHT_MODE text.
        scenarios.append(("fault %d (%s), text only" % (code, label),
                          dict(NATIVE), 71, r, False))
    # A run of the flight with the private frame suppressed, proving the
    # annunciator still speaks off the FLIGHT_MODE text alone.
    # Extremes: the arithmetic paths that clamp, and the states that switch
    # the phase line on.
    edge_native = dict(NATIVE)
    # Attitude arrives in RADIANS on the wire; the widget converts. Feeding
    # degrees here would silently test the wrong scale.
    edge_native.update({"Ptch": -0.82, "Roll": 1.06, "RxBt": 19.2, "RQly": 12.0})
    edge_robot = dict(ROBOT, alpha=1.4, wheel=-3.3, hip_l=9.9, hip_r=-0.0,
                      health=0, glitch=812, esp32=0, state=7, jump=4, profile=2)
    scenarios.append(("saturated / degraded", edge_native, 3, edge_robot, True))
    scenarios.append(("standing up", dict(NATIVE), 71,
                      dict(ROBOT, state=8, standup=3), True))

    failures = []
    all_played = set()
    for label, sensors, rssi, robot, with_frame in scenarios:
        found, built, played = run_widget(lua, S, label, sensors, rssi,
                                          robot, with_frame)
        all_played |= played
        status = "FAIL" if found else "ok  "
        shape = "  ".join("%s=%d" % (k.split()[0], v)
                          for k, v in built.items())
        print("%s  %-28s  objects: %s  sounds: %d"
              % (status, label, shape, len(played)))
        failures.extend(found)

    for label, with_frame in (("state/fault transitions", True),
                              ("transitions, FM text only", False)):
        found, spoken = run_flight(lua, S, with_frame)
        failures.extend(found)
        all_played |= spoken
        print("%s  %-28s  sounds: %s"
              % ("FAIL" if found else "ok  ", label,
                 ", ".join(sorted(short(x) for x in spoken))))

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
