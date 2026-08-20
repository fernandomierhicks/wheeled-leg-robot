#!/usr/bin/env python3
"""Cross-check everything staged in radio/sdcard/ before it goes on the radio.

The failure mode this exists to prevent is a silent one: a model that loads
fine but references a widget name that does not exist, a Special Function
pointing at a sound file nobody copied, or a checklist whose filename does not
match the model name so it never appears. None of those announce themselves --
they just quietly do nothing, on a radio you are trusting with a robot that
can fall over.

Everything here is checked against the EdgeTX 2.12 YAML schema
(radio/src/storage/yaml/yaml_datastructs_tx15.cpp) and against the other
staged files, not against assumptions.

Usage:
    python check_sdcard.py
"""
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("check_sdcard.py needs pyyaml:  pip install pyyaml")

REPO = Path(__file__).resolve().parents[2]
SD = REPO / "radio" / "sdcard"

# TX15 analog and switch inventory, from radio/src/targets/tx15/hal.h:
# four stick axes, two pots (POT1/POT2 -> S1/S2), six switches SA-SF, and six
# RGB function switches SG-SL.
STICKS = {"Ail", "Ele", "Thr", "Rud"}
POTS = {"S1", "S2"}
SWITCHES = {"SA", "SB", "SC", "SD", "SE", "SF"}
FUNCTION_SWITCHES = {"SG", "SH", "SI", "SJ", "SK", "SL"}

# The contract from firmware/robot_teensy/radio_channels.md, 0-based destCh.
EXPECTED_CHANNELS = {
    0: "roll setpoint",
    1: "forward velocity",
    2: "hip height",
    3: "yaw rate",
    4: "SD log",
    5: "jump trigger",
    6: "live tune A",
    7: "live tune B",
    8: "speed profile",
    9: "ARM",
}

THEME_COLORS = [
    "PRIMARY1", "PRIMARY2", "PRIMARY3",
    "SECONDARY1", "SECONDARY2", "SECONDARY3",
    "FOCUS", "EDIT", "ACTIVE", "WARNING", "DISABLED",
]

problems = []
notes = []


def bad(fmt, *a):
    problems.append(fmt % a if a else fmt)


def note(fmt, *a):
    notes.append(fmt % a if a else fmt)


def check_model():
    models = sorted((SD / "MODELS").glob("*.yml"))
    if not models:
        bad("no model .yml staged in MODELS/")
        return
    for path in models:
        # storage/modelslist.cpp only opens files whose name is the literal
        # prefix "model" followed by digits and ".yml". Anything else in
        # /MODELS/ is skipped without a word -- the model simply never appears
        # in the model list, and nothing tells you why.
        if not re.fullmatch(r"model\d+\.yml", path.name, re.IGNORECASE):
            bad("%s: EdgeTX only reads model<digits>.yml from /MODELS/; this "
                "file would be ignored silently", path.name)
            continue
        try:
            m = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            bad("%s: not valid YAML: %s", path.name, exc)
            continue
        check_one_model(path, m)


def check_one_model(path, m):
    name = m.get("header", {}).get("name")
    if not name:
        bad("%s: header.name missing", path.name)
        return
    if len(name) > 15:
        bad("%s: model name %r is over EdgeTX's 15-char limit", path.name, name)

    # Checklist. EdgeTX (gui/common/stdlcd/model_notes.cpp) builds the path
    # from the model NAME, spaces preserved, then falls back to the name with
    # spaces stripped. Ship at least one of the two or the checklist is a
    # setting that does nothing.
    if m.get("displayChecklist"):
        primary = SD / "MODELS" / ("%s.txt" % name)
        fallback = SD / "MODELS" / ("%s.txt" % name.replace(" ", ""))
        if not primary.exists() and not fallback.exists():
            bad("%s: displayChecklist is on but neither %r nor %r exists",
                path.name, primary.name, fallback.name)

    bitmap = m.get("header", {}).get("bitmap")
    if bitmap and not (SD / "IMAGES" / bitmap).exists():
        bad("%s: header.bitmap %r not staged in IMAGES/", path.name, bitmap)

    # Mixes: one per channel, no duplicates, every source real.
    seen = {}
    for mix in m.get("mixData", []):
        ch, src = mix.get("destCh"), mix.get("srcRaw")
        if ch in seen:
            note("%s: CH%d has more than one mix (%s, %s) -- intentional?",
                 path.name, ch + 1, seen[ch], src)
        seen[ch] = src
        if src is None:
            bad("%s: mix on CH%s has no srcRaw", path.name, ch)
        elif re.fullmatch(r"I\d+", src):
            idx = int(src[1:])
            if not any(e.get("chn") == idx for e in m.get("expoData", [])):
                bad("%s: CH%d uses input %s but no expoData defines chn %d",
                    path.name, ch + 1, src, idx)
        elif src in FUNCTION_SWITCHES:
            note("%s: CH%d uses function switch %s -- confirm it is assigned "
                 "in Radio > Hardware", path.name, ch + 1, src)
        elif src not in SWITCHES | POTS | STICKS:
            bad("%s: CH%d source %r is not on a TX15", path.name, ch + 1, src)

    for ch, what in EXPECTED_CHANNELS.items():
        if ch not in seen:
            bad("%s: CH%d (%s) has no mix -- the firmware expects it",
                path.name, ch + 1, what)

    for e in m.get("expoData", []):
        if e.get("srcRaw") not in STICKS:
            bad("%s: input chn %s reads %r, expected a stick",
                path.name, e.get("chn"), e.get("srcRaw"))

    for idx, lim in (m.get("limitData") or {}).items():
        nm = lim.get("name", "")
        if len(nm) > 6:
            bad("%s: channel name %r is over the 6-char limit", path.name, nm)
        if idx not in seen:
            note("%s: CH%d is named %r but has no mix", path.name, idx + 1, nm)

    # Logical switches must point at channels that exist.
    for idx, ls in (m.get("logicalSw") or {}).items():
        d = str(ls.get("def", ""))
        for ref in re.findall(r"ch\((\d+)\)", d):
            # ch(n) is 0-based (yaml_datastructs_funcs.cpp: n + MIXSRC_FIRST_CH)
            if int(ref) not in seen:
                bad("%s: L%d references ch(%s) = CH%d, which has no mix",
                    path.name, idx + 1, ref, int(ref) + 1)

    # Special functions: every track must be a file the picker can reach, i.e.
    # flat in /SOUNDS/en/ with a 6-char-or-shorter name.
    for idx, fn in (m.get("customFn") or {}).items():
        if fn.get("func") != "PLAY_TRACK":
            continue
        track = str(fn.get("def", "")).split(",")[0]
        if len(track) > 6:
            bad("%s: SF%d track %r is over the 6-char Play Track limit",
                path.name, idx + 1, track)
        wav = SD / "SOUNDS" / "en" / ("%s.wav" % track)
        if not wav.exists():
            bad("%s: SF%d plays %r but SOUNDS/en/%s.wav is not staged",
                path.name, idx + 1, track, track)

    warn = m.get("switchWarningState", "")
    if warn:
        if not re.fullmatch(r"(?:S?[A-L][ud-])+", warn):
            bad("%s: switchWarningState %r is malformed; expected pairs like "
                "'AuBuCu'", path.name, warn)
        else:
            for sw, pos in re.findall(r"([A-L])([ud-])", warn):
                if ("S" + sw) not in SWITCHES | FUNCTION_SWITCHES:
                    bad("%s: switchWarningState names S%s, not a TX15 switch",
                        path.name, sw)

    mod = (m.get("moduleData") or {})
    if 0 not in mod and 1 in mod:
        bad("%s: module configured on index 1 (external). The TX15's ELRS is "
            "the INTERNAL module -- use index 0", path.name)
    for idx, md in mod.items():
        if md.get("type") == "TYPE_CROSSFIRE" and md.get("channelsCount") != 16:
            note("%s: CRSF module has channelsCount %s; 16 is the usual choice",
                 path.name, md.get("channelsCount"))

    check_screens(path, m)


def widget_names():
    """Names the staged Lua widgets register under, plus EdgeTX built-ins."""
    builtin = {"Outputs", "ModelBmp", "Timer", "Value", "Gauge", "BigGauge",
               "Text", "Sliders", "Bitmap"}
    found = set(builtin)
    for main in (SD / "WIDGETS").glob("*/main.lua"):
        src = main.read_text(encoding="utf-8")
        m = re.search(r'name\s*=\s*"([^"]+)"', src)
        if m:
            found.add(m.group(1))
            if len(m.group(1)) > 10:
                bad("%s: widget name %r is over the 10-char limit",
                    main.parent.name, m.group(1))
        else:
            bad("%s/main.lua does not return a name", main.parent.name)
        # The Lua docs claim an 8-character limit on widget folder names, but
        # the TX15 ships with "MicroValues" (11) working fine, so this is a
        # note rather than an error.
        if len(main.parent.name) > 8:
            note("widget folder %r is over the documented 8-char limit "
                 "(the radio tolerates longer, but keep it short)",
                 main.parent.name)
    return found


def check_screens(path, m):
    known = widget_names()
    for idx, screen in (m.get("screenData") or {}).items():
        zones = screen.get("layoutData", {}).get("zones", {}) or {}
        for zidx, zone in zones.items():
            wn = zone.get("widgetName")
            if wn and wn not in known:
                bad("%s: screen %s zone %s uses widget %r, which is neither a "
                    "staged widget nor a known built-in",
                    path.name, idx, zidx, wn)


def check_theme():
    themes = [p for p in (SD / "THEMES").iterdir() if p.is_dir()] \
        if (SD / "THEMES").exists() else []
    if not themes:
        bad("no theme staged in THEMES/")
    for t in themes:
        yml = t / "theme.yml"
        if not yml.exists():
            bad("%s: theme.yml missing", t.name)
            continue
        data = yaml.safe_load(yml.read_text(encoding="utf-8"))
        colors = data.get("colors", {}) or {}
        for key in THEME_COLORS:
            if key not in colors:
                bad("%s: theme.yml is missing colour %s", t.name, key)
        for key, val in colors.items():
            if not isinstance(val, int):
                bad("%s: colour %s is %r, expected 0xRRGGBB", t.name, key, val)
            elif not 0 <= val <= 0xFFFFFF:
                bad("%s: colour %s = %#x is out of range", t.name, key, val)
        for key in ("QM_BG", "QM_FG"):
            if key not in colors:
                note("%s: no %s (EdgeTX 2.12 quick menu); harmless on 2.11",
                     t.name, key)
        # The theme browser wants a banner and three previews; the radio wants
        # a background at its own resolution. The TX15 is 480x320 (hal.h).
        for f in ("logo.png", "screenshot1.png", "screenshot2.png",
                  "screenshot3.png", "background_480x320.png"):
            if not (t / f).exists():
                bad("%s: %s missing", t.name, f)


def check_lua_tree():
    for main in (SD / "WIDGETS").glob("*/main.lua"):
        src = main.read_text(encoding="utf-8")
        for ref in re.findall(r'loadScript\("([^"]+)"', src):
            if not (SD / ref.lstrip("/")).exists():
                bad("%s: loadScript(%r) has no staged file",
                    main.parent.name, ref)
        for ref in re.findall(r'playFile\((?:SND\s*\.\.\s*)?"([^"]*)"', src):
            if ref.startswith("/") and not (SD / ref.lstrip("/")).exists():
                bad("%s: playFile(%r) has no staged file", main.parent.name, ref)

    # Every sound folder the Lua reaches for must exist.
    for lua in (SD / "WIDGETS").rglob("*.lua"):
        src = lua.read_text(encoding="utf-8")
        m = re.search(r'local SND = "([^"]+)"', src)
        if m and not (SD / m.group(1).lstrip("/")).is_dir():
            bad("%s: sound folder %s is not staged",
                lua.relative_to(SD), m.group(1))


def main():
    check_model()
    check_theme()
    check_lua_tree()

    files = [p for p in SD.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)

    for n in notes:
        print("note: %s" % n)
    if problems:
        print("\n%d problem(s):" % len(problems), file=sys.stderr)
        for p in problems:
            print("  - %s" % p, file=sys.stderr)
        return 1

    print("\nsdcard staging OK: %d files, %.1f MB" % (len(files), total / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
