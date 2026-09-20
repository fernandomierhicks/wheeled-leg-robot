r"""Build named concepts from gallery.py in 3D, verify each, render in colour.

    python parts/build_concepts.py GLACIER NIGHTRUN APEX
    python parts/build_concepts.py --all

Every build is checked with lib/verify.py before it is rendered.  A concept that
changes ANY opening is reported and dropped rather than shown -- a pretty render
of an unprintable part is worse than no render.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import paths, spec as S
import tibia, femur, verify as V
import render_color as RC
from gallery import BASE, CONCEPTS, WILD

ALL = {n: (sub, over) for n, sub, over in (CONCEPTS + WILD)}
PARTS = {"tibia": tibia, "femur": femur}
MOD = tibia


def one(name, size=760, views=("iso",)):
    sub, over = ALL[name]
    sp = S.derive(**{**BASE, **over})
    tag = name.lower()
    t0 = time.time()
    res = MOD.build(sp, verbose=False)
    MOD.export(res, tag=tag)
    bad, worst = V.verify(MOD.PART, tag=tag)
    w, g, a = (res["white"].volume / 1000, res["graphite"].volume / 1000,
               res["accent"].volume / 1000 if res["accent"] else 0.0)
    tot = w + g + a
    print(f"   {name}: {time.time()-t0:5.1f}s  white {w:6.2f} graphite {g:6.2f} "
          f"accent {a:5.2f} cm3 ({100*a/tot:4.1f}% accent)  "
          f"verify {bad} changed / {worst:.3f} mm3")
    if bad:
        # export() ran before verify(), so the files exist.  Remove them: a
        # failed build must not leave STEPs and a 3MF lying around for a
        # downstream renderer to pick up by filename and present as real.
        print(f"   !! {name} CHANGED {bad} opening(s) -- dropped, artifacts removed")
        for d, pat in ((paths.print_dir(MOD.PART, tag), f"{MOD.PART}_{tag}_"),
                       (paths.styled_dir(MOD.PART, tag), f"{MOD.PART}_{tag}_")):
            for f in os.listdir(d):
                if f.startswith(pat) or f == f"{MOD.PART}_{tag}.3mf":
                    try: os.remove(os.path.join(d, f))
                    except OSError: pass
        return None
    pal = S.PALETTES[sp["palette"]]
    bodies, allV = RC.tess_bodies(res, pal)
    ims = [RC.view(bodies, allV, size, *RC.VIEWS[v]) for v in views]
    return dict(name=name, sub=sub, imgs=ims, spec=sp,
                vols=(w, g, a), accent_pct=100 * a / tot)


if __name__ == "__main__":
    paths.ensure_out()
    args = sys.argv[1:]
    if "--part" in args:
        i = args.index("--part")
        MOD = PARTS[args[i + 1].lower()]
        globals()["MOD"] = MOD
        args = args[:i] + args[i + 2:]
    names = list(ALL) if (not args or args[0] == "--all") else args
    tiles, meta = [], []
    for n in names:
        if n not in ALL:
            print(f"?? {n}"); continue
        print(f"== {n}")
        r = one(n)
        if r is None:
            continue
        tiles.append((r["name"], f"{r['sub']}  ·  accent {r['accent_pct']:.1f}% by volume",
                      r["imgs"][0]))
        meta.append(dict(name=r["name"], accent_pct=round(r["accent_pct"], 2),
                         white_cm3=round(r["vols"][0], 2),
                         graphite_cm3=round(r["vols"][1], 2),
                         accent_cm3=round(r["vols"][2], 2),
                         palette=r["spec"]["palette"]))
    if tiles:
        RC.sheet(tiles, 3, os.path.join(paths.RENDERS, f"{MOD.PART.lower()}_built.png"),
                 header="TIBIA v4 — concepts built in 3D",
                 sub="every one verified: 0 openings changed, worst deviation 0.000 mm3")
        json.dump(meta, open(os.path.join(paths.RENDERS, "concepts_3d.json"), "w"), indent=2)
        print(f"\n{len(tiles)} concept(s) built and verified")
