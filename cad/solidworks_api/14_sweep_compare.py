"""Step 14: the collision sweep, styling ON vs OFF, one phase per run.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/14_sweep_compare.py on   [step_deg]
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/14_sweep_compare.py off  [step_deg]
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/14_sweep_compare.py compare

The sweep half of 10_verify_styled.py, split so each phase is short and its
result is on disk: a long SolidWorks session (6 GB after a night of automation)
threw "The server threw an exception" half way through a combined run.

  on       styled parts exactly as saved (reloaded from disk first)
  off      every GL_* feature suppressed -> the original parts; reloads after
  compare  every (pose, pair) whose interference the styling made NEW or
           bigger by more than 1 mm3 -- the answer to "is it collision free"
"""
import os
import sys
import json
import time
from importlib import import_module

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib

V = import_module("10_verify_styled")
OUT = os.path.join(HERE, "out")
TOL = 1.0


def path(phase):
    return os.path.join(OUT, f"sweep_{phase}.json")


def reload_all(sw, model):
    for rel in V.STYLED:
        if V._doc(sw, rel) is not None:
            swlib.reload_from_disk(sw, os.path.join(swlib.V5, rel))
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
    model.ForceRebuild3(False)


def run(phase, step):
    sw, _ = swlib.connect()
    model = swlib.open_v5(sw)
    reload_all(sw, model)
    if phase == "off":
        V.set_styling(sw, model, False)
    hip = swlib.HipDriver(model)
    n = int(round((V.HI - V.LO) / step)) + 1
    angles = [V.LO + i * (V.HI - V.LO) / (n - 1) for i in range(n)]
    res, t = [], time.time()
    for a in angles:
        got = hip.set(a)
        res.append({"hip": got, "pairs": {" x ".join(k): v for k, v in V.volumes(model, blocks=True).items()}})
    json.dump(res, open(path(phase), "w"), indent=1)
    print(f"  styling {phase.upper()}: {n} poses in {time.time() - t:.0f} s -> {path(phase)}")
    if phase == "off":
        reload_all(sw, model)


def compare():
    on, off = json.load(open(path("on"))), json.load(open(path("off")))
    worst = {}
    for a, b in zip(on, off):
        for k, v in a["pairs"].items():
            d = v - b["pairs"].get(k, 0.0)
            if d > TOL and (k not in worst or d > worst[k][1]):
                worst[k] = (a["hip"], d, b["pairs"].get(k, 0.0), v)
    print(f"SWEEP {len(on)} poses, -28..+57 deg: {len(worst)} pair(s) the styling made new or bigger")
    for k, (a, d, b, v) in sorted(worst.items(), key=lambda t: -t[1][1]):
        print(f"   +{d:8.2f} mm3 at hip {a:+6.1f} ({b:.2f} -> {v:.2f})  {k}")
    # The other way round: a contact of a STYLED part that got smaller or went
    # away -- the limit switch and the retract hard stop are designed contacts
    # (17_contact_keepout.py), and a removal over one deletes it silently.
    # Only pairs touching a styled component: the suppressed left leg is
    # missing from a right-leg-only ON for that reason alone.
    styled = import_module("13_collision_keepout").STYLED_COMPONENTS
    lost = {}
    for a, b in zip(on, off):
        for k, v in b["pairs"].items():
            if not any(n in styled for n in k.split(" x ")):
                continue
            d = v - a["pairs"].get(k, 0.0)
            if d > max(0.01, 0.02 * v) and (k not in lost or d > lost[k][1]):
                lost[k] = (a["hip"], d, v, a["pairs"].get(k, 0.0))
    print(f"CONTACTS of styled parts that shrank or vanished: {len(lost)}")
    for k, (a, d, b, v) in sorted(lost.items(), key=lambda t: -t[1][1]):
        print(f"   -{d:8.3f} mm3 at hip {a:+6.1f} ({b:.3f} -> {v:.3f})  {k}")
    return worst


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if mode == "compare":
        compare()
    else:
        run(mode, float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
