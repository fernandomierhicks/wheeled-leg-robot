"""Hand-laid additions to a recipe's plan(), for ground the recipe leaves bare.

08_style_part.py calls EXTRA[part](P, M) right after plan(): each adds to
P["grey"] (graphite inlay, through the plate on a thin part) and P["strip"]
(blue accent, engraved 1.2 mm and coloured, exactly like the recipe's own
trace), so everything downstream -- engraving, colour split, checks, 3MF --
treats them as the recipe's.  Coordinates are the recipe's part-local mm.
Every addition is kept off the fastener/bearing seats M (+0.6 mm, as 08's
additions are).

RobotMount (his markup, 2026-10-03): the plate's left half, x 0..85, is a
flat 5 mm field that the Side panel does not cover.  The recipe's features
there are rails and pads that never clear the plate (skipped as buried, in
the locked OCC look too), so it came out bare.  Added:
  * the graphite step-bar (x 76..135) carried on leftwards across the field
    with a mirrored 45-degree step, ending in a wide head cut by white vent
    slashes;
  * two blue traces leaving the left-edge screw seats, 45-degree doglegs,
    ticks on the upper one, pads at the ends;
  * a blue three-pin "header" between the two lower-left screws.
"""
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

TRACE_W = 2.2      # mm, the recipe's blue strip as built on this part


def _trace(pts, w=TRACE_W):
    return LineString(pts).buffer(w / 2, cap_style=2, join_style=2, mitre_limit=2.0)


def _bar(p0, p1, w):
    return LineString([p0, p1]).buffer(w / 2, cap_style=2)


def robotmount(P, M):
    # graphite: y 22..29.2 is the recipe bar's left end (x >= 76); overlap it
    grey = Polygon([(80, 22.0), (70, 22.0), (63, 29.0), (38, 29.0), (31, 36.0),
                    (17.5, 36.0), (17.5, 45.5), (32, 45.5), (41.3, 36.2), (63, 36.2),
                    (70, 29.2), (80, 29.2)])
    vents = unary_union([_bar((x, 37.4), (x + 6.2, 44.6), 1.2) for x in (19.5, 22.4, 25.3, 28.2)])
    blue = unary_union([
        # upper: from the (10, 55) seat, dogleg down, ticks, pad
        _trace([(15.0, 55.0), (24.0, 55.0), (30.0, 49.5), (45.0, 49.5)]),
        Point(45.0, 49.5).buffer(1.8, quad_segs=16),
        *[_bar((x, 50.0), (x + 3.4, 57.0), 1.0) for x in (33.0, 36.0, 39.0)],
        # lower: from the (10, 10) seat, dogleg up, pad short of the (65, 15) seat
        _trace([(17.0, 11.0), (39.0, 11.0), (45.5, 17.5), (55.0, 17.5)]),
        Point(55.0, 17.5).buffer(1.6, quad_segs=16),
        # pin header between the two lower-left screws
        *[_bar((3.5, y), (13.5, y), 1.0) for y in (19.6, 21.9, 24.2)],
    ])
    K = M.buffer(0.6, 48)
    grey = grey.difference(vents).difference(K)
    blue = blue.difference(K).difference(grey)
    P["grey"] = unary_union([P["grey"], grey])
    P["strip"] = unary_union([P["strip"], blue])
    return f"graphite +{grey.area:.0f} mm2, blue +{blue.area:.0f} mm2 on the left field"


EXTRA = {"RobotMount": robotmount}
