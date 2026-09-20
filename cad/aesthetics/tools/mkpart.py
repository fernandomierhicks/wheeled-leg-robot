r"""Generate a part recipe from parts/femur.py, the general one.

Re-run after ANY change to femur.py so the derived recipes do not drift.
Every datum below was measured off the part's own openings and cross-checked
against CLAUDE.md where CLAUDE.md has a number for it.
"""
import io, os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PARTS = {
    "Coupler": dict(
        module="coupler", show="-Z",
        datums=[("F", (-84.77, 0.0), 16.00), ("E", (84.77, 0.0), 21.02)],
        doc='''Coupler v4 -- the third 4-bar link.

DATUMS, DERIVED (they match CLAUDE.md exactly):

    F = (-84.770, 0)   4-bolt cross, 7.0 mm arms -- the FIXED BODY PIVOT.  The
                       same 7 mm cross appears on the Side panel at
                       (-36.420, +37.540), CLAUDE.md's F-relative-to-A, so this
                       is the end that bolts to the body.
    E = (+84.770, 0)   D26 bearing bore, 4 bolts on a D38.05 circle -- the
                       TIBIA end.
    |EF| = 169.540 mm, the coupler length in CLAUDE.md.

The ends are NOT the same size -- F lobe 16.00, E lobe 21.02 -- so bosses,
collars and joint accents are sized from RAD[p], never one shared radius.

SHOW FACE local -Z: local +Z points inboard.  Five D11 lightening holes sit on
the spine, 30 mm apart, which the Femur and Tibia do not have; they are
openings like any other and `kb` keeps added material off them.'''),

    "Side panel": dict(
        module="side_panel", show="+Z",
        datums=[("A", (0.0, 0.0), 22.5), ("F", (-36.42, 37.54), 7.0)],
        doc='''Side panel v4 -- the plate the whole leg module bolts to.

A PLATE, not a link: 116.92 x 104.25 x 20.50, nearly square in plan, where the
Femur is 224 x 38.8.  The feature bands are fractions of the part's own extent
in BOTH axes, so they adapt; that is why femur.py's absolute Y bounds had to go.

DATUMS, DERIVED -- and this part is where the repo's canonical geometry lives:

    A = (0, 0)              THE ORIGIN IS THE HIP AXIS.  Five holes sit on a
                            D45 circle about it -- (+-19.487, +-11.25) and
                            (0, -22.5), all at r = 22.5 -- which is the
                            AK45-10 stator bolt pattern.
    F = (-36.420, +37.540)  the 7 mm-arm bolt cross, the fixed body pivot.

    |AF| = 52.30 mm, which is CLAUDE.md's |AF| verbatim.  CLAUDE.md warns that
    conflating F's body-centre Z with the A->F offset once cost a whole robot;
    these are the A-relative numbers, read off the part itself.

SHOW FACE local +Z: the assembly walk puts this face outboard, toward the leg.'''),

    "RobotMount": dict(
        module="robotmount", show="+Z",
        datums=[("B", (151.0, 33.0), 13.45), ("S", (100.653, 44.544), 8.30)],
        doc='''RobotMount v4 -- the plate that carries the leg module into the box.

A PLATE: 200.00 x 97.50 x 19.00, and the most lopsided part here -- its
openings all sit at x > 100, leaving the first half of the plate bare.  The
band fractions therefore spread features across ground that has no features of
its own, which is the point.

DATUMS, DERIVED:

    B = (+151.000, +33.000)   the principal bore, counterbored (two concentric
                              openings, r_equiv 13.45 and 11.31).
    S = (+100.653, +44.544)   the secondary bore, r_equiv 8.30.

    It also carries a D6.8 pair at x = 139.794 and 160.198, spacing 20.404 mm.
    The Side panel has the same pair at x = +-10.2, spacing 20.405 -- that is
    the mating feature by which the two plates locate to each other, so neither
    may grow over it.  They are openings, so `kb` already protects them.

SHOW FACE local +Z.  This is the innermost part in the stack (global Z 80..99),
so much of it is hidden by the Side panel; only the region beyond the Side
panel's footprint actually shows.'''),
}


def generate(name):
    cfg = PARTS[name]
    s = io.open("parts/femur.py", encoding="utf-8").read()
    s = 'r"""' + cfg["doc"] + '''

Generated from parts/femur.py by scratchpad/mkpart.py -- edit the generator,
not this file, or the two drift.
"""''' + s[s.index('"""', 3) + 3:]

    ds = cfg["datums"]
    names = ", ".join(d[0] for d in ds)
    coords = ", ".join(repr(d[1]) for d in ds)
    rad = ", ".join(f"{d[0]}: {d[2]}" for d in ds)
    tup = "(" + ", ".join(d[0] for d in ds) + ")"
    for old, new in [
        ('PART = "Femur"\nA, C = (-93.79, 0.0), (93.79, 0.0)      # derived; see module docstring',
         f'PART = "{name}"\n{names} = {coords}      # derived; see module docstring\n'
         f'RAD = {{{rad}}}   # each datum\'s own feature radius'),
        ('    # circular bosses at the two pivots\n'
         '    r_end = 20.0 + sp["knee_grow"]\n'
         '    bosses = unary_union([ShPoint(*A).buffer(r_end, 96), ShPoint(*C).buffer(r_end, 96)])',
         '    # circular bosses at the datums, each sized from ITS OWN radius\n'
         '    bosses = unary_union([ShPoint(*p).buffer(RAD[p] + 3.5 + sp["knee_grow"], 96)\n'
         f'                          for p in {tup}])'),
        ('        joints=[(A, 19.0), (C, 19.0)])',
         f'        joints=[(p, RAD[p] + 2.5) for p in {tup}])'),
        ('    collars = unary_union([ShPoint(*p).buffer(17.0 + ct, 96).difference(ShPoint(*p).buffer(17.0, 96))\n'
         '                           for p in (A, C)])',
         '    collars = unary_union([ShPoint(*p).buffer(RAD[p] + 0.5 + ct, 96)\n'
         '                           .difference(ShPoint(*p).buffer(RAD[p] + 0.5, 96))\n'
         f'                           for p in {tup}])'),
        ('sf = os.path.join(paths.SPECS, "femur.json")',
         f'sf = os.path.join(paths.SPECS, "{name.lower()}.json")'),
    ]:
        assert old in s, f"{name}: pattern not found -> {old[:60]}"
        s = s.replace(old, new, 1)
    out = f"parts/{cfg['module']}.py"
    io.open(out, "w", encoding="utf-8").write(s)
    print(f"{out}  ({name}, show face {cfg['show']}, datums {names})")


if __name__ == "__main__":
    for n in (sys.argv[1:] or PARTS):
        generate(n)
