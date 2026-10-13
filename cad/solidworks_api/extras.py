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


# the lobe: 1 mm of it overlaps the plate (y > 0) so it merges; part-local mm, plate z 0..5
LOBE = Polygon([(33, 1), (33, -9), (37, -13), (68, -13), (71.5, -9.5), (78, -9.5), (82, -5.5), (82, 1)])
LOBE_CUT = Polygon([(40, -6.0), (45.5, -3.6), (67.5, -3.6), (67.5, -8.8), (50.5, -8.8)])
# the corrugated hose on the show face, his green mark (2026-10-12): under the graphite step-bar, left field.
# 24_pipes.PIPES["RobotMount"] is this line (u = x, v = y on the show face) -- keep the two in step
ROBOTMOUNT_HOSE = [(19.5, 30.0), (26.0, 28.4), (33.0, 26.0), (41.0, 24.6), (49.0, 24.4), (56.0, 24.0)]


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
    # the growth below the field's bottom edge (his orange outline, 2026-10-12): not structural, so it
    # carries a through window and its own accents; 08 adds it as GL_Lobe / GL_LobeCut before the colour split
    # the recipe has stray graphite / blue in the ground the lobe now covers (outside its outline, never inlaid
    # before): cleared, the lobe gets only its own accents
    lobe_new = LOBE.difference(P["OUT"]).buffer(0.6)
    P["grey"], P["strip"] = P["grey"].difference(lobe_new), P["strip"].difference(lobe_new)
    allowed = LOBE.intersection(box(0, -30, 200, -0.2)).buffer(-0.6, join_style=2).difference(LOBE_CUT.buffer(1.0, join_style=2)).difference(K)
    lgrey = unary_union([box(41.0, -11.9, 66.0, -10.3),
                         *[_bar((x, -12.5), (x + 4.5, -4.5), 2.0) for x in (73.0, 76.5)]]).intersection(allowed)
    lblue = unary_union([*[_bar((x, -12.5), (x + 3.4, -1.6), 1.1) for x in (34.0, 36.2, 38.4)],
                         _trace([(45.0, -1.7), (62.0, -1.7)], 1.2), Point(64.8, -1.7).buffer(1.2, quad_segs=16),
                         _bar((69.0, -12.0), (71.0, -10.0), 1.0)]).intersection(allowed)
    lgrey, lblue = _clean(lgrey, 2.0, 0.8), _clean(lblue.difference(lgrey.buffer(0.8)), 1.0, 0.8)
    # his call (2026-10-12): the blue ring round the bearing boss sits under the femur plate, never seen -> no colour there
    P["ring"] = Polygon()
    P["lobe"] = (LOBE.buffer(0), LOBE_CUT)
    P["grey"] = unary_union([P["grey"], grey, lgrey])
    P["strip"] = unary_union([P["strip"], blue, lblue])
    return (f"graphite +{grey.area:.0f} mm2, blue +{blue.area:.0f} mm2 on the left field; lobe {LOBE.area:.0f} mm2 "
            f"(window {LOBE_CUT.area:.0f}), graphite +{lgrey.area:.0f}, blue +{lblue.area:.0f} on it")


def _clean(g, min_area=2.0, w=0.4):
    """Drop slivers: shapes thinner than w, polygons under min_area (hairlines left by a
    difference leave degenerate loops that break the white body's face)."""
    g = g.buffer(-w / 2, join_style=2).buffer(w / 2, join_style=2)
    return unary_union([p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and p.area >= min_area])


# the corrugated hose on the Coupler's show face (his green line): part-local plan centre line.
# 24_pipes.PIPES["Coupler"] is this line converted to the face frame -- keep the two in step
COUPLER_HOSE = [(46.0, 0.5), (38.0, -0.5), (30.0, 0.5), (22.0, -0.5), (14.0, 0.0)]


def coupler(P, M):
    """His markup (2026-10-10): no accent near the E bearing (the blue ring, the graphite
    blocks and the blue trace ends inside r 38 of E), and more colour variety on the raised
    frame along the -Y edge -- graphite raked bars and blue slashes across it.  Frame accents
    are colour only: 0.6 mm deep from the frame top (P["frame_grey"] / P["frame_blue"], read by 08)."""
    E = Point(84.77, 0.0).buffer(38.0, 96)
    P["ring"] = _clean(P["ring"].difference(E))
    # the blue trace makes room for the hose (R 2.5 + 1.5 clear): trimmed back off its corridor
    corridor = LineString(COUPLER_HOSE).buffer(4.0, cap_style=2)
    P["strip"] = _clean(P["strip"].difference(corridor).difference(E), 2.0, 0.3)
    P["grey"] = _clean(P["grey"].difference(E), 2.0, 0.3)
    K = M.buffer(0.6, 48)
    room = P["frame"].buffer(-1.2, join_style=2).difference(P["wins"].buffer(1.2, join_style=2))
    room = room.difference(P["pock"].buffer(1.0)).difference(K).difference(E.buffer(1.0))
    def slash(x, y0, y1, w, rake=0.7):
        return Polygon([(x, y0), (x + w, y0), (x + w + rake * (y1 - y0), y1), (x + rake * (y1 - y0), y1)])
    grey = unary_union([
        slash(-55, -21, -9, 6), slash(-47, -21, -9, 2.2),
        slash(-3, -21, -9, 5), slash(27, -21, -9, 7), slash(37, -21, -9, 2.2),
        box(8, -19.0, 24, -17.6), box(44, -12.4, 62, -11.0)])
    blue = unary_union([
        slash(-40, -21, -9, 1.4), slash(-37, -21, -9, 1.4), slash(-34, -21, -9, 1.4),
        slash(10, -21, -9, 1.4), slash(13, -21, -9, 1.4),
        slash(46, -21, -9, 1.6), slash(50, -21, -9, 1.6),
        box(-20, -17.6, 3, -16.4), box(25, -18.4, 42, -17.2)])
    grey = grey.intersection(room).difference(P["strip"].buffer(0.8)).difference(P["grey"].buffer(0.8))
    blue = blue.intersection(room).difference(grey.buffer(0.8)).difference(P["strip"].buffer(0.8))
    grey, blue = _clean(grey, 2.0, 0.8), _clean(blue, 2.0, 0.8)
    P["frame_grey"], P["frame_blue"] = grey, blue
    return f"E-end accents removed; frame graphite +{grey.area:.0f} mm2, blue +{blue.area:.0f} mm2"


# the corrugated hose on Femur_inside's +Z (tube) face (his green mark, 2026-10-11): part-local plan centre line,
# along the old blue trace toward the knee end.  24_pipes.PIPES["Femur_inside"] is this line in the face frame -- keep in step
FI_HOSE = [(33.0, 0.0), (40.0, 1.5), (46.3, 6.5), (54.0, 8.2), (61.8, 8.4)]
FI_A = (-93.79, 0.0)      # hip axis (the D26 bearing bore)


def femur_inside(P, M):
    """His marks (2026-10-11): (1) white disc r 20 round the hip bearing bore, both faces (no accent touches it);
    (2) the blue trace gives way to the hose along FI_HOSE; (3) more recessed relief on the +Z (tube) face -- raked
    slots, a via bus, chevrons -- added to P["back_eng"] (the same shallow groove as the recipe's circuit)."""
    disc = Point(*FI_A).buffer(20.0, 96)
    corridor = LineString(FI_HOSE).buffer(4.0, cap_style=2)
    bcorr = LineString(FI_HOSE + [(74.0, 10.3)]).buffer(6.0, cap_style=2)
    for k in ("grey", "strip", "blocks", "ring"):
        if P.get(k) is not None and not P[k].is_empty:
            g = P[k].difference(disc)
            if k in ("strip", "blocks", "ring"):
                g = g.difference(bcorr)
            P[k] = _clean(g, 2.0 if k == "grey" else 10.0, 0.3)
    OUT = P["OUT"]
    keep = unary_union([g for g in (P["cutouts"], P["cut_pockets"], P["pock"]) if not g.is_empty]).buffer(3.6)
    allowed = (OUT.buffer(-8.6, join_style=2).difference(M.buffer(1.0)).difference(keep)
               .difference(unary_union([P["strip"], P["blocks"], P["ring"]]).buffer(1.5))
               .difference(corridor.buffer(1.5)).difference(P["back_eng"].buffer(2.0)).difference(disc.buffer(1.0)))

    def slot(x, y0, y1, w, rake=0.5):
        return Polygon([(x, y0), (x + w, y0), (x + w + rake * (y1 - y0), y1), (x + rake * (y1 - y0), y1)])

    new = unary_union([
        *[slot(x, -3.0, 9.0, 2.4) for x in (-72.0, -68.0, -64.0)],             # raked slot group, hip side
        _trace([(-47.0, -6.5), (-38.0, -6.5), (-32.0, -0.5), (-24.0, -0.5)], 1.4),   # via bus
        Point(-47.0, -6.5).buffer(2.0, 24), Point(-24.0, -0.5).buffer(2.0, 24),
        _trace([(-44.0, 6.0), (-36.0, 6.0), (-30.0, 10.0), (-20.0, 10.0)], 1.4),
        Point(-44.0, 6.0).buffer(2.0, 24),
        *[slot(x, -8.5, 3.0, 2.0, 0.7) for x in (-16.0, -12.0, -8.0)],         # slots, mid
        _trace([(14.0, 5.5), (20.0, 5.5), (25.0, 0.5)], 1.4),
        Point(14.0, 5.5).buffer(2.0, 24), Point(25.0, 0.5).buffer(2.0, 24),
        *[slot(x, -9.0, -3.0, 2.2, 0.9) for x in (30.0, 34.5, 39.0, 43.5)],    # short slashes under the hose
        *[slot(x, -9.0, 4.0, 2.2, 0.6) for x in (52.0, 56.0, 60.0)],
        box(64.0, -9.0, 74.0, -7.0), box(66.0, -4.0, 74.0, -2.5),
    ])
    new = _clean(new.intersection(allowed), 3.0, 1.0)
    P["back_eng"] = unary_union([P["back_eng"], new])
    P["hose"], P["back_new"] = LineString(FI_HOSE).buffer(2.5, cap_style=2), new
    # the +Z face accents become shallow recesses (08: back_recess_d) on a 0.6 mm colour floor; the hose corridor stays
    # flush (the hose lies on flat ground), the knee ring stays a flush inlay
    clear = LineString(FI_HOSE).buffer(6.0, cap_style=1)
    P["back_rec_blue"] = _clean(unary_union([P["strip"], P["blocks"]]).difference(clear), 3.0, 0.9)
    P["back_rec_grey"] = _clean(P["grey"].difference(clear).difference(P["back_rec_blue"].buffer(0.8)), 3.0, 0.9)
    # the back grooves get a coloured floor, alternating blue / graphite along x
    pieces = sorted(getattr(P["back_eng"], "geoms", [P["back_eng"]]), key=lambda g: g.centroid.x)
    P["groove_blue"] = unary_union(pieces[0::2])
    P["groove_grey"] = unary_union(pieces[1::2])
    return (f"hip disc cleared, blue trimmed off the hose, +{new.area:.0f} mm2 of back relief; back recess blue "
            f"{P['back_rec_blue'].area:.0f} / graphite {P['back_rec_grey'].area:.0f} mm2, {len(pieces)} coloured grooves")


EXTRA = {"RobotMount": robotmount, "Coupler": coupler, "Femur_inside": femur_inside}
