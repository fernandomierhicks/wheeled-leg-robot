"""Step 31: one command per part -- build + recolour + mirror + save in ONE SolidWorks session.

    cadenv/python 31_part.py <Part> [--quick]     (Part: any 28 PARTS name, e.g. "Side panel")

Replaces the sequence `28 build` -> 30 (whiten) -> `25 mirror-colours` -> `16 --chain <Mirror part>`
(each its own process: reconnect, reopen, a full chain save each time).  Needs the part's
`28 export` + `28 preview` done.  Stages, each timed:
  build    28's sw_build (RL_ features, checks, probe) -- nothing saved yet
  hook     part-specific recolour (HOOKS: Side panel -> 30's whiten)
  mirror   25's mirror_colours for THIS part's mirror only
  save     16's chain of the part + its mirror part (one chain, sub-assemblies first)
--quick: the chain saves only the assemblies that directly hold the parts and skips ROBOT
(`16 --chain --quick`): fast while iterating; ROBOT stays dirty, so run the plain command
(no --quick) -- or `16 --chain <part>` -- at the end of the part.
"""
import importlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
R = importlib.import_module("28_relief")
F25 = importlib.import_module("25_fasteners")
F16 = importlib.import_module("16_check_and_save")


def hook_sidepanel(sw):
    importlib.import_module("30_sidepanel").whiten(sw)


HOOKS = {"Side panel": hook_sidepanel}


def main():
    args = sys.argv[1:]
    quick = "--quick" in args
    parts = [a for a in args if not a.startswith("--")]
    if len(parts) != 1 or parts[0] not in R.PARTS:
        raise SystemExit(f"usage: 31_part.py <Part> [--quick]; Part in {list(R.PARTS)}")
    part = parts[0]
    t0 = time.time()

    def stage(name):
        nonlocal t0
        now = time.time()
        print(f"  [{name}: {now - t0:.0f} s]", flush=True)
        t0 = now

    sw, done = R.sw_build([part])
    stage("build")
    if part in HOOKS:
        HOOKS[part](sw)
        stage("hook")
    rel = R.PARTS[part]
    mirrors = [m for m, s in F25.MIRROR_PARTS.items() if s == rel]
    if mirrors:
        F25.mirror_colours(only=[part])
        stage("mirror")
    F16.chain(sw, [rel] + mirrors, quick=quick)
    stage("save")


if __name__ == "__main__":
    main()
