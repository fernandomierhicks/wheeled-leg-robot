#!/usr/bin/env python3
"""Generate radio/sdcard/MODELS/model90.yml in EdgeTX's own YAML dialect.

WHY THIS EXISTS
---------------
EdgeTX's YAML parser is hand-rolled (radio/src/storage/yaml/yaml_parser.cpp)
and **has no comment support whatsoever**. `#` is not a token it knows: a
comment line is read as an attribute name, `find_node()` fails, and at top
level `toParent()` returning false makes the parser `return DONE_PARSING` --
it stops dead and silently keeps only what it read before that point.

That is not theoretical. The first hand-written version of this model was
17312 bytes of well-commented YAML. The radio loaded it, parsed `header`,
`inputNames` and `expoData`, hit the comments above `mixData`, stopped, and
rewrote the file as a 4336-byte default -- losing every mix, the CRSF module,
the switch warnings, the checklist flag and both custom screens. The only
symptom was that selecting the model showed a stock main view.

So: the model is authored HERE, in Python, where comments are free, and
emitted as comment-free YAML in the exact style the radio itself writes.
check_sdcard.py enforces that no staged model contains a comment, so this
cannot regress.

Blocks the radio authors for itself (topbar layout, function-switch config)
are copied verbatim from model_base.yml, which is a capture of the radio's own
output -- ground truth for the dialect rather than a guess at it.

Usage:
    python build_model.py            # write the model
    python build_model.py --check    # exit 1 if the staged file is stale
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent / "model_base.yml"
OUT = REPO / "radio" / "sdcard" / "MODELS" / "model90.yml"
MODELS_DIR = OUT.parent
CHECKLIST_SRC = MODELS_DIR / "WLR ROBOT.txt"

MODEL_NAME = "WLR ROBOT"

# Blocks we let the radio own. Copied verbatim from its own output: they
# describe hardware config (the six RGB function switches) and the top bar,
# neither of which this package has an opinion about yet.
VERBATIM = ["topbarData", "topbarWidgetWidth"]


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
#
# POLARITY CONVENTION, and it holds for every switch below:
#   switch UP   -> -100%  -> ~1000 us -> firmware sees the channel LOW  -> SAFE
#   switch DOWN -> +100%  -> ~2000 us -> firmware sees the channel HIGH -> ACTIVE
# So "all switches up" is always inert, which is what switchWarning enforces
# at power-on. radio_channels.md says things like "C5 up = start recording",
# meaning the CHANNEL is high -- that is this transmitter's switch-DOWN
# position. Same signal, opposite word. Do not fix the clash by flipping a mix
# weight; see radio/CHANNELS.md.

# Inputs, in the radio's native RETA order so the Inputs screen looks like
# every other model on this transmitter. Mode 2 throughout: hip height MUST
# sit on the non-centring throttle stick, because leg height is a held pose.
#
# All linear. Hip height carried 30% expo briefly; removed on request. Stick
# position now maps straight to leg height, which also means what you see on
# the Outputs screen is what the firmware receives -- worth keeping in mind if
# you ever add a curve back, since hip_cmd_rate_lim already shapes the command
# on the robot side and two shaping stages compound.
#   (chn, srcRaw, name, expo)
INPUTS = [
    (0, "Rud", "Yaw",  0),
    (1, "Ele", "Vel",  0),
    (2, "Thr", "Hip",  0),   # linear: no expo
    (3, "Ail", "Roll", 0),
]

# (destCh, srcRaw, mix name, channel name)
# destCh is 0-based; CH1 is destCh 0.
#
# SHOULDER SWITCHES: the TX15 ships with SF as the momentary button and SE as
# the latching one (RadioMaster's own manual diagram, back view). The
# integration plan proposed the opposite lettering, which was arbitrary -- what
# actually matters is that JUMP gets a momentary control and ARM gets a
# latching one. Following the factory fit instead means no hardware swap at
# all, and removes a whole class of "did I fit the right switch" error.
#
#   jump  (CH6)  needs momentary -> SF: rising-edge trigger, and a latching
#                switch here could sit latched where you cannot see it
#   reset (CH12) wants latching  -> SE: an edge-triggered fault clear. The
#                firmware requires a debounced release before another edge
#                counts, so the switch resting down is harmless.
#
# ARM is on SD, the rightmost top toggle. Configure SD as 2POS in
# Radio > Hardware: the mix maps a 3-position switch linearly, so the middle
# position reads as disarmed, but a binary function deserves a binary switch.
#
# There is deliberately NO hard-ESTOP channel. Dropping ARM disarms, which
# goes through DISARMING -- a controlled hip-gain taper rather than an
# immediate torque cut. If you ever want the immediate version back, it was
# implemented on CH11 and is in git history.
MIXES = [
    (0,  "I3", "Roll",  "ROLL"),   # right stick H -- self-centring: release levels
    (1,  "I1", "Vel",   "VEL"),    # right stick V -- release means stop
    (2,  "I2", "Hip",   "HIP"),    # left stick V, the throttle stick
    (3,  "I0", "Yaw",   "YAW"),    # left stick H
    (4,  "SB", "Log",   "LOG"),    # top row, 2nd from left
    (5,  "SF", "Jump",  "JUMP"),   # momentary shoulder -- rising edge
    (6,  "S1", "TuneA", "TUNEA"),  # left dial, live tune slot 0
    (7,  "S2", "TuneB", "TUNEB"),  # right dial, live tune slot 1
    (8,  "SC", "Prof",  "PROF"),   # top row, right of the dials
    (9,  "SD", "Arm",   "ARM"),    # top row, rightmost
    (10, "SA", "Calib", "CALIB"),  # top row, leftmost
    (11, "SE", "Reset", "RESET"),  # latching shoulder -- clear fault / ESTOP
    # Live tuning, on the RGB push buttons under the gimbals. Group select
    # lives here rather than on CH5/CH6 so a tuning session no longer gives up
    # SD logging and the jump trigger -- and a tuning session is exactly the
    # one you most want a log of.
    #
    # All three group buttons share ONE channel. They are already mutually
    # exclusive (switch group 1), so three channels would be three ways to say
    # one thing, and every channel is spoken for. Each mixes in with weight w
    # and offset w so OFF contributes 0 and ON contributes 2w -- see
    # ENCODED_TUNE_GROUP below and the band decode in main.cpp.
    (13, "SJ", "Latch", "LATCH"),   # RGB button 4: commit tuned gains -> CH14
    (14, "SK", "Lean",  "LEAN"),    # RGB button 5: coordinated-turn lean
]

# Three exclusive buttons encoded onto CH13 as distinct levels:
#   none 0% (1500 us) / SG 32% (1660) / SH 66% (1830) / SI 100% (2000)
# EdgeTX computes source*weight/100 + offset, and a switch source is -100 when
# off, so weight == offset puts OFF at exactly 0 and ON at 2*weight.
ENCODED_TUNE_GROUP = [("SG", 16), ("SH", 33), ("SI", 50)]

# Function-switch configuration, which the radio stores per model.
#
#   2POS   = latching, push-on/push-off  (switches.cpp toggles the logical
#            state only on press: `SWITCH_2POS && physicalState`)
#   TOGGLE = momentary, on while held    (toggles on every physical change,
#            and is forced to start OFF)
#
# SG/SH/SI share group 1, which makes them mutually exclusive. cfsGroupOn for
# that group stays 0 ("not always on"), so all-off is a legal state and is
# what "tuning inactive" looks like. SJ is a momentary commit in no group.
#
# Colours: amber for a selected gain group (matching the HUD's tune accent),
# green for the commit press.
FUNCTION_SWITCHES = {
    "SW1": ("TUNE G0", "2POS",   1, (255, 180, 0)),
    "SW2": ("TUNE G1", "2POS",   1, (255, 180, 0)),
    "SW3": ("TUNE G2", "2POS",   1, (255, 180, 0)),
    "SW4": ("LATCH",   "TOGGLE", 0, (46, 230, 138)),
    # Lean: latched, and cyan because that is the HUD's "this is live" colour.
    "SW5": ("LEAN",    "2POS",   0, (0, 229, 255)),
}

# Channel travel is left at the full +/-100% default on purpose. The firmware
# thresholds are absolute microseconds (>1990, <1010) and both stick combos
# need genuine full deflection, so clamping any of CH1-CH4 or CH10 here would
# quietly break arming, the rescue combo and the calibration combo.

# Logical switches. Channel-based only: telemetry sensors do not exist until
# the robot is bound and emitting, and a logical switch pointed at a sensor
# that was never discovered is one that silently never fires.
#   L1 armed, L2 jump requested, L3 calibration requested
# ch(n) is 0-based (yaml_datastructs_funcs.cpp: n + MIXSRC_FIRST_CH).
LOGICAL_SW = [
    ("FUNC_VPOS", "ch(9),50"),
    ("FUNC_VPOS", "ch(5),50"),
    ("FUNC_VPOS", "ch(10),50"),
]

# Special functions: annunciation and logging only, never control. Firmware
# stays authoritative -- TX-side logic that tried to auto-disarm would fight
# the interlocks and produce behaviour nobody could reconstruct afterwards.
#
# Tracks are the flat wlr*-prefixed copies in /SOUNDS/en/; the Play Track
# picker cannot reliably reach into /SOUNDS/en/WLR/, and the stock pack
# already ships armed.wav and disarm.wav that other models use.
CUSTOM_FN = [
    ("L1",     "PLAY_TRACK", "wlrarm,1,1x"),
    ("!L1",    "PLAY_TRACK", "wlrdis,1,1x"),
    ("L3",     "PLAY_TRACK", "wlrcal,1,1x"),
    # Radio-side telemetry log at 1 Hz on the same switch that starts the
    # robot's own .wlog. Two records with different clocks catch a whole class
    # of "the log lied" bugs -- but only if they start together.
    ("SBd",    "LOGS",       "10,1"),
    ("SBd",    "PLAY_TRACK", "wlrlog,1,1x"),
    ("!SBd",   "PLAY_TRACK", "wlrlgf,1,1x"),
]

# All six up at boot. That is inert for every one of them: no calibration
# request, no logging, profile 1 (slowest), disarmed, no reset edge, no jump.
SWITCH_WARNING = ["SA", "SB", "SC", "SD", "SE", "SF"]


# carryTrim: 0 is TRIM_ON, 1 is TRIM_OFF (myeeprom.h; mixer.cpp applies the
# trim when carryTrim == 0). Every stick mix here uses TRIM_OFF.
#
# Trims are a second authority over the robot's setpoints, and this robot
# already has its own: pitch_trim_rad, and the whole balance-point story in
# CLAUDE.md. A bumped trim rocker would put a permanent creep on velocity, a
# standing lean on roll, a slow turn on yaw, or an offset on leg height -- and
# because the custom screens turn EdgeTX's trim display off, it would be
# invisible while doing it. Radios get bumped in bags.
#
# Set TRIM_ON deliberately, per channel, if you ever actually want it.
TRIM_OFF = 1


def mix(dest_ch, src, name, weight=100, offset=0):
    return {
        "destCh": dest_ch, "srcRaw": q(src), "carryTrim": TRIM_OFF, "mixWarn": 0,
        "mltpx": "ADD", "delayPrec": 0, "speedPrec": 0,
        "flightModes": Raw("000000000"), "weight": weight, "offset": offset,
        "swtch": q("NONE"), "delayUp": 0, "delayDown": 0,
        "speedUp": 0, "speedDown": 0, "name": q(name),
    }


def expo(chn, src, name, expo_pct):
    return {
        "mode": 3, "scale": 0, "trimSource": 0, "srcRaw": q(src),
        "weight": 100, "offset": 0, "swtch": q("NONE"),
        "curve": {"type": 1, "value": expo_pct},
        "chn": chn, "flightModes": Raw("000000000"), "name": q(name),
    }


def limit(name):
    return {"min": 0, "max": 0, "ppmCenter": 0, "offset": 0, "symetrical": 0,
            "revert": 0, "curve": 0, "name": q(name)}


def widget(name, options):
    w = {"widgetName": q(name)}
    if options:
        w["widgetData"] = {"options": {i: o for i, o in enumerate(options)}}
    return w


def colour(idx):
    return {"type": "Color", "value": {"color": Raw("COLIDX%d" % idx)}}


def boolean(v):
    return {"type": "Bool", "value": {"boolValue": v}}


def signed(v):
    return {"type": "Signed", "value": {"signedValue": v}}


def build():
    m = {}

    m["header"] = {
        "name": q(MODEL_NAME),
        "modelId": {0: {"val": 1}},
        "bitmap": q("wlrrobot.png"),
        "labels": q("Robot"),
    }

    mixes = [mix(ch, src, nm) for ch, src, nm, _ in MIXES]
    # CH13 (destCh 12): the three exclusive group buttons summed.
    for src, w in ENCODED_TUNE_GROUP:
        mixes.append(mix(12, src, "G" + src[1], weight=w, offset=w))
    mixes.sort(key=lambda m: m["destCh"])
    m["mixData"] = mixes
    m["limitData"] = {ch: limit(cn) for ch, _, _, cn in MIXES}
    m["limitData"][12] = limit("TUNEG")
    m["expoData"] = [expo(*e) for e in INPUTS]
    m["inputNames"] = {i: {"val": q(n)} for i, (_, _, n, _) in enumerate(INPUTS)}

    m["logicalSw"] = {i: {"func": Raw(f), "def": q(d), "andsw": q("NONE"),
                          "delay": 0, "duration": 0}
                      for i, (f, d) in enumerate(LOGICAL_SW)}

    m["customFn"] = {i: {"swtch": q(sw), "func": Raw(fn), "def": q(d)}
                     for i, (sw, fn, d) in enumerate(CUSTOM_FN)}

    m["timers"] = {0: {"start": 0, "swtch": q("L1"), "value": 0, "mode": Raw("ON"),
                       "countdownBeep": 0, "minuteBeep": 1, "persistent": 0,
                       "countdownStart": 0, "showElapsed": 0, "extraHaptic": 0,
                       "name": q("ARMED")}}

    m["switchWarning"] = {sw: {"pos": Raw("up")} for sw in SWITCH_WARNING}

    # The internal module is the LR1121 ELRS radio speaking CRSF.
    #
    # crsfArmingMode 0 keeps ELRS arming tied to the channel rather than to a
    # switch it picks for itself: the robot's arm interlock lives in firmware
    # on CH10, and a second arming authority inside the radio link is exactly
    # the two-sources-of-truth problem this project already decided to avoid.
    #
    # failsafeMode NOT_SET leaves the receiver's own failsafe in charge. Do not
    # set HOLD -- the firmware's safety story depends on a dead link looking
    # dead (alive() false, channel() returning 0), and a receiver holding the
    # last frame would present a stale ARM as a live one.
    m["moduleData"] = {0: {
        "type": Raw("TYPE_CROSSFIRE"), "subType": 0,
        "channelsStart": 0, "channelsCount": 16,
        "failsafeMode": Raw("NOT_SET"),
        "mod": {"crsf": {"telemetryBaudrate": 0, "crsfArmingMode": 0,
                         "crsfArmingTrigger": q("NONE")}},
    }}

    # Screen 0: the HUD, full screen, chrome off so the widget owns 480x320.
    # Screen 1: the Outputs monitor -- the screen the CRSF bring-up gate needs
    # for verifying every channel with torque off.
    m["screenData"] = {
        0: {"LayoutId": q("Layout1x1"),
            "layoutData": {
                "zones": {0: widget("WLR HUD", [boolean(1), colour(4)])},
                "options": {i: boolean(0) for i in range(5)}}},
        1: {"LayoutId": q("Layout1x1"),
            "layoutData": {
                "zones": {0: widget("Outputs", [signed(1), boolean(0),
                                                colour(5), colour(0), colour(3)])},
                "options": {0: boolean(1), 1: boolean(0), 2: boolean(0),
                            3: boolean(1), 4: boolean(0)}}},
    }

    # Scalars. displayChecklist shows MODELS/"WLR ROBOT".txt on model select;
    # disableThrottleWarning 0 keeps the "hip stick must be retracted at boot"
    # guard live.
    cs = {}
    for i in range(1, 7):
        key = "SW%d" % i
        name, typ, grp, on = FUNCTION_SWITCHES.get(key, ("", "NONE", 0, (255, 255, 255)))
        cs[key] = {
            "name": q(name),
            "type": Raw(typ),
            "group": grp,
            "start": Raw("START_OFF"),
            "state": 0,
            "onColorLuaOverride": Raw("OFF"),
            "offColorLuaOverride": Raw("OFF"),
            "onColor": {"r": on[0], "g": on[1], "b": on[2]},
            "offColor": {"r": 0, "g": 0, "b": 0},
        }
    m["customSwitches"] = cs
    # Group 1 holds the mutually exclusive gain-group buttons. 0 = not
    # "always on", so all-off is legal and means tuning is inactive.
    m["cfsGroupOn"] = {1: {"v": 0}, 2: {"v": 0}, 3: {"v": 0}}

    m["displayChecklist"] = 1
    m["checklistInteractive"] = 0
    m["disableThrottleWarning"] = 0
    m["thrTraceSrc"] = Raw("Thr")
    m["view"] = 0

    return m


# --------------------------------------------------------------------------
# EdgeTX-dialect emitter
# --------------------------------------------------------------------------

class Raw(str):
    """A scalar emitted without quotes (enums, bitfields, COLIDX)."""


class Quoted(str):
    """A scalar emitted with quotes, the way the radio writes strings."""


def q(s):
    return Quoted(s)


IND = "   "   # the radio uses three spaces per level


def emit(value, depth, out):
    pad = IND * depth
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                out.append("%s%s: " % (pad, k))     # trailing space, as the radio writes
                emit(v, depth + 1, out)
            else:
                out.append("%s%s: %s" % (pad, k, scalar(v)))
    elif isinstance(value, list):
        for item in value:
            out.append("%s-" % (IND * (depth - 1) + " "))
            emit(item, depth, out)
    else:
        out.append("%s%s" % (pad, scalar(value)))


def scalar(v):
    if isinstance(v, Quoted):
        return '"%s"' % v
    if isinstance(v, Raw):
        return str(v)
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def verbatim_blocks(text):
    """Pull the blocks we let the radio own out of its own output."""
    blocks, current, name = {}, [], None
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_]\w*):", line)
        if m:
            if name in VERBATIM:
                blocks[name] = current
            name, current = m.group(1), [line]
        elif name is not None:
            current.append(line)
    if name in VERBATIM:
        blocks[name] = current
    missing = [b for b in VERBATIM if b not in blocks]
    if missing:
        sys.exit("model_base.yml is missing verbatim blocks: %s\n"
                 "Re-capture it from the radio after it has written a model."
                 % missing)
    return blocks


def render():
    base = BASE.read_text(encoding="utf-8")
    keep = verbatim_blocks(base)

    out = ["semver: 3.0.0"]
    for key, value in build().items():
        if isinstance(value, (dict, list)):
            out.append("%s: " % key)
            emit(value, 1, out)
        else:
            out.append("%s: %s" % (key, scalar(value)))
    for name in VERBATIM:
        out.extend(keep[name])

    text = "\n".join(out) + "\n"
    if "#" in text:
        sys.exit("emitted model contains a '#'. EdgeTX's YAML parser has no "
                 "comment support and will stop parsing at it.")
    return text


def checklist_aliases():
    """Every filename EdgeTX might look for the checklist under.

    view_text.cpp's getModelNotesFile() tries, in order: the model name, the
    model name with spaces replaced by underscores, the model FILE name with
    .yml swapped for .txt, and that with underscores. Which of those the
    boot-time gate actually uses is not something I could establish for
    EdgeTX 3.0 -- the gate calls modelHasNotes(), whose definition is not in
    the 2.12 tree I can read.

    So rather than guess, ship all of them. They are 3.5 kB each and copied
    from one source at build time, so they cannot drift apart.
    """
    stem = OUT.stem                     # "model90"
    names = [MODEL_NAME + ".txt",
             MODEL_NAME.replace(" ", "_") + ".txt",
             stem + ".txt"]
    # Preserve order, drop duplicates.
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(MODELS_DIR / n)
    return out


def sync_checklist(check_only=False):
    if not CHECKLIST_SRC.exists():
        sys.exit("checklist source %s is missing" % CHECKLIST_SRC)
    body = CHECKLIST_SRC.read_text(encoding="utf-8")
    stale = []
    for path in checklist_aliases():
        if path == CHECKLIST_SRC:
            continue
        if check_only:
            if not path.exists() or path.read_text(encoding="utf-8") != body:
                stale.append(path.name)
        else:
            path.write_text(body, encoding="utf-8", newline="\n")
    return stale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    text = render()
    if args.check:
        stale = sync_checklist(check_only=True)
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print("%s is stale -- rerun build_model.py" % OUT, file=sys.stderr)
            return 1
        if stale:
            print("checklist copies stale: %s -- rerun build_model.py"
                  % ", ".join(stale), file=sys.stderr)
            return 1
        print("%s up to date" % OUT.name)
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    sync_checklist()
    print("wrote %s (%d bytes, %d mixes, %d special functions)"
          % (OUT.relative_to(REPO), len(text), len(MIXES), len(CUSTOM_FN)))
    print("checklist under %d names: %s"
          % (len(checklist_aliases()),
             ", ".join(p.name for p in checklist_aliases())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
