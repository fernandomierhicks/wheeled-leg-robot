"""Single source of truth for where things live.

Import this first from any recipe; it also puts lib/ on sys.path so the other
modules import cleanly no matter which directory you run from.
"""
import os, sys

LIB   = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(LIB)                 # cad/aesthetics
CAD   = os.path.dirname(ROOT)                # cad
if LIB not in sys.path:
    sys.path.insert(0, LIB)

INPUT   = os.path.join(ROOT, "input")
SPECS   = os.path.join(ROOT, "specs")
OUT     = os.path.join(ROOT, "out")
BOARDS  = os.path.join(OUT, "boards")
RENDERS = os.path.join(OUT, "renders")
PRINT   = os.path.join(OUT, "print")
STYLED  = os.path.join(OUT, "styled")

# Where to look for source STEP exports, in priority order.  Drop new exports in
# input/parts; the live CAD folder is searched as a fallback so nothing has to be
# duplicated just to be found.
# "STEP exports/Middle multiple parts" is the 2026-09-19 re-export: one STEP per
# assembly node, and the folder the Femur and Tibia now resolve to.  It is
# searched AFTER input/parts so a hand-placed override still wins -- which is how
# Coupler and "Side panel" get found at all: the export lost both to a Windows
# case collision with the assembly nodes COUPLER.STEP and "SIDE PANEL.STEP", and
# they were recovered into input/parts by XCAF extraction.  See DECISIONS.md 15.
EXPORTS = os.path.join(CAD, "v4 Larger Ball bearings", "STEP exports")
SOURCE_DIRS = [
    os.path.join(INPUT, "parts"),
    os.path.join(EXPORTS, "Middle multiple parts"),
    EXPORTS,
    os.path.join(CAD, "v4 Larger Ball bearings"),
    os.path.join(CAD, "v4 Larger Ball bearings", "Links"),
]
ASSEMBLY_DIR = os.path.join(INPUT, "assembly")


def part_step(name):
    """Resolve a part name ('Tibia') to a STEP file, case-insensitively."""
    want = name.lower()
    if not want.endswith((".step", ".stp")):
        cands = [want + ".step", want + ".stp"]
    else:
        cands = [want]
    for d in SOURCE_DIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower() in cands:
                return os.path.join(d, f)
    raise FileNotFoundError(
        f"No STEP for '{name}'. Looked in:\n  " + "\n  ".join(SOURCE_DIRS))


def assembly_steps():
    """Every assembly STEP dropped in input/assembly, sorted by name."""
    if not os.path.isdir(ASSEMBLY_DIR):
        return []
    return [os.path.join(ASSEMBLY_DIR, f) for f in sorted(os.listdir(ASSEMBLY_DIR))
            if f.lower().endswith((".step", ".stp"))]


OTHER = "other colors"     # anything that is not the locked concept


def print_dir(part, tag="glacier", make=False):
    """Per-part output folder.  One flat folder held 17 concepts x 3 parts x 4
    files and was unnavigable; everything for one part now sits together, and
    anything that is not the locked concept goes under `other colors/`."""
    base = PRINT if tag == "glacier" else os.path.join(PRINT, OTHER)
    d = os.path.join(base, part)
    if make:
        os.makedirs(d, exist_ok=True)
    return d


def styled_dir(part, tag="glacier", make=False):
    base = STYLED if tag == "glacier" else os.path.join(STYLED, OTHER)
    d = os.path.join(base, part)
    if make:
        os.makedirs(d, exist_ok=True)
    return d


def styled_step(part, tag="glacier"):
    """The fused single body -- geometry reference, ONE colour by nature."""
    return os.path.join(styled_dir(part, tag), f"{part}_{tag}_styled.step")


def colour_step(part, tag="glacier"):
    """The three filament bodies in one file, each with its colour.  This is
    the one to open to LOOK at a part; `styled_step` cannot show colour because
    it is a single fused solid."""
    return os.path.join(styled_dir(part, tag), f"{part}_{tag}_colour.step")


def ensure_out():
    for d in (OUT, BOARDS, RENDERS, PRINT, STYLED, SPECS):
        os.makedirs(d, exist_ok=True)
