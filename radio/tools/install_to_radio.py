#!/usr/bin/env python3
"""Copy radio/sdcard/ onto the TX15's USB mass-storage volume.

Deliberately conservative, because the thing on the other end of this cable
holds the only copy of your gimbal calibration:

  * dry run by default -- nothing is written until you pass --apply
  * refuses to run against a volume that does not look like an EdgeTX card
  * refuses a board mismatch unless you say --force
  * NEVER touches /RADIO/radio.yml. That file holds the stick and pot
    calibration, and re-calibrating a set of hall gimbals by hand is an
    afternoon you will not get back
  * NEVER touches labels.yml. EdgeTX rebuilds it and will pick the new model
    up on its own
  * every file it would overwrite is copied into a timestamped folder under
    /BACKUP/ on the card first
  * picks a free model<NN>.yml number instead of clobbering an existing model

Usage:
    python install_to_radio.py                    # find the card, show a plan
    python install_to_radio.py --apply            # actually copy
    python install_to_radio.py --drive F: --apply
    python install_to_radio.py --apply --only THEMES WIDGETS SOUNDS
"""
import argparse
import filecmp
import re
import shutil
import string
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SD = REPO / "radio" / "sdcard"

# Trees we install. Everything else on the card is left alone.
TREES = ["THEMES", "WIDGETS", "SOUNDS", "IMAGES", "MODELS"]

# Files that must never be written, whatever else happens.
FORBIDDEN = {"radio/radio.yml", "models/labels.yml", "models/models.yml"}


def looks_like_edgetx(root):
    return ((root / "RADIO").is_dir() and (root / "MODELS").is_dir()
            and (root / "SOUNDS").is_dir())


def find_cards():
    found = []
    for letter in string.ascii_uppercase:
        root = Path("%s:/" % letter)
        try:
            if root.exists() and looks_like_edgetx(root):
                found.append(root)
        except OSError:
            continue
    return found


def radio_info(root):
    """Read board and firmware version out of RADIO/radio.yml, read-only."""
    info = {}
    p = root / "RADIO" / "radio.yml"
    if not p.exists():
        return info
    # Scan the whole file: the stick calibration block sits near the top and
    # pushes stickMode well past any small line budget.
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*(board|semver|stickMode|internalModule|selectedTheme)"
                     r"\s*:\s*(\S.*?)\s*$", line)
        if m and m.group(1) not in info:
            info[m.group(1)] = m.group(2).strip('"')
    return info


# radio.yml stores stickMode 0-indexed; the UI calls them Mode 1..4.
STICK_MODES = {"0": "Mode 1", "1": "Mode 2", "2": "Mode 3", "3": "Mode 4"}


def free_model_number(root, preferred):
    used = set()
    for p in (root / "MODELS").glob("*.yml"):
        m = re.fullmatch(r"model(\d+)\.yml", p.name, re.IGNORECASE)
        if m:
            used.add(int(m.group(1)))
    if preferred not in used:
        return preferred
    n = preferred
    while n in used:
        n += 1
    return n


def plan_copy(root, trees, model_rename):
    """Return [(src, dst, action)] without touching anything."""
    actions = []
    for tree in trees:
        src_root = SD / tree
        if not src_root.is_dir():
            continue
        for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
            rel = src.relative_to(SD)
            if str(rel).replace("\\", "/").lower() in FORBIDDEN:
                continue
            dst = root / rel
            if model_rename and rel.parts[0] == "MODELS" and \
                    re.fullmatch(r"model\d+\.yml", rel.name, re.IGNORECASE):
                dst = root / "MODELS" / model_rename
            if not dst.exists():
                action = "new"
            elif filecmp.cmp(str(src), str(dst), shallow=False):
                action = "same"
            else:
                action = "overwrite"
            actions.append((src, dst, action))
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", help="card root, e.g. F: (default: autodetect)")
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without it this is a dry run")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if radio.yml says a different board")
    ap.add_argument("--only", nargs="+", metavar="TREE", choices=TREES,
                    help="install only these trees (default: all)")
    ap.add_argument("--model-number", type=int, default=90,
                    help="preferred model<NN>.yml slot (default 90)")
    args = ap.parse_args()

    if args.drive:
        root = Path(args.drive if args.drive.endswith(("/", "\\"))
                    else args.drive + "/")
        if not looks_like_edgetx(root):
            sys.exit("%s does not look like an EdgeTX card "
                     "(expected RADIO/, MODELS/ and SOUNDS/)" % root)
        cards = [root]
    else:
        cards = find_cards()

    if not cards:
        print("No EdgeTX card found.")
        print("Plug the TX15 in and put it in USB storage mode "
              "(Radio -> SD-HC card / USB, choose 'USB Storage'), then re-run.")
        print("If it is mounted somewhere unusual, pass --drive.")
        return 1
    if len(cards) > 1:
        sys.exit("More than one EdgeTX card found (%s) -- pass --drive to "
                 "pick one." % ", ".join(str(c) for c in cards))

    root = cards[0]
    info = radio_info(root)
    print("Card:     %s" % root)
    print("Board:    %s" % info.get("board", "unknown"))
    print("Firmware: %s" % info.get("semver", "unknown"))
    mode = info.get("stickMode")
    print("Stick mode: %s" % STICK_MODES.get(mode, "unknown"))
    print("Internal module: %s" % info.get("internalModule", "unknown"))
    print("Current theme: %s" % info.get("selectedTheme", "unknown"))

    board = info.get("board", "").lower()
    if board and board != "tx15" and not args.force:
        sys.exit(
            "\nradio.yml says board '%s', but this package is built for the "
            "TX15 (480x320, SA-SF, S1/S2).\nThe model file in particular "
            "assumes that hardware. Re-run with --force if you know what you "
            "are doing." % info["board"])

    semver = info.get("semver", "")
    if semver and not semver.startswith(("2.11", "2.12", "3.")):
        print("\nWarning: the LVGL Lua API used by the HUD needs EdgeTX 2.11 "
              "or newer; this card reports %s. The widget will show a message "
              "rather than a HUD." % semver)

    if mode is not None and mode != "1":
        print("\nWarning: this radio is in %s. The model assumes Mode 2 -- hip "
              "height has to sit on the non-centring throttle stick, and in "
              "another mode that stick is not where the model thinks it is."
              % STICK_MODES.get(mode, "an unknown mode"))

    internal = info.get("internalModule", "")
    if internal and internal != "TYPE_CROSSFIRE":
        print("\nWarning: radio.yml internalModule is %s, not TYPE_CROSSFIRE. "
              "The model configures the internal module as CRSF; set the "
              "internal module type in Radio settings -> Hardware first."
              % internal)

    trees = args.only or TREES
    slot = free_model_number(root, args.model_number)
    model_rename = "model%d.yml" % slot
    if slot != args.model_number:
        print("\nmodel%d.yml is taken on this card; using %s instead."
              % (args.model_number, model_rename))

    actions = plan_copy(root, trees, model_rename)
    new = [a for a in actions if a[2] == "new"]
    over = [a for a in actions if a[2] == "overwrite"]
    same = [a for a in actions if a[2] == "same"]

    print("\nPlan for %s:" % ", ".join(trees))
    print("  %4d new" % len(new))
    print("  %4d overwrite" % len(over))
    print("  %4d already identical (skipped)" % len(same))
    for src, dst, _ in over:
        print("    overwrite  %s" % dst.relative_to(root))
    if new:
        print("    new files include:")
        for src, dst, _ in new[:8]:
            print("      %s" % dst.relative_to(root))
        if len(new) > 8:
            print("      ... and %d more" % (len(new) - 8))

    if not args.apply:
        print("\nDry run. Nothing was written. Re-run with --apply to copy.")
        return 0

    backup = root / "BACKUP" / time.strftime("wlr-%Y%m%d-%H%M%S")
    if over:
        backup.mkdir(parents=True, exist_ok=True)
        for src, dst, _ in over:
            b = backup / dst.relative_to(root)
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(dst), str(b))
        print("\nBacked up %d file(s) to %s" % (len(over), backup))

    written = 0
    for src, dst, action in actions:
        if action == "same":
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        written += 1

    print("Wrote %d file(s)." % written)
    print("\nOn the radio:")
    print("  1. Eject the USB volume, then exit USB mode on the radio.")
    print("  2. Radio settings -> Themes -> RoboBlue -> Set as default.")
    print("  3. Model select -> WLR ROBOT. The checklist should appear.")
    print("  4. Confirm SE is momentary and SF is 2-position latching in")
    print("     Radio settings -> Hardware, then re-check the switch warning.")
    print("  5. Screen 2 is the Outputs monitor -- use it for the CRSF")
    print("     bring-up gate with all four motor-enable params at 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
