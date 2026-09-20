"""The style spec: one dict that drives BOTH the 2D board and the 3D build.

Choosing a cell on a board means choosing a spec.  Nothing is reinterpreted
between the picture and the part, so there is no translation loss.

LANGUAGE picks the design idea.  The three axes then tune it:

  aggression 0-10  how far the part departs from the source.
  density    0-2   how MANY features.
  relief     0-2   how TALL/DEEP those features are.
  scale      0-2   how BIG each feature is in plan.  Independent of density:
                   "the count is right, the size could be bigger" is a real
                   note and density could not express it.
  accent     0-2   how much blue, and how big the blue blocks are.
  organic    0-2   rounds the whole silhouette into tangent curves, and necks
                   the mid-shaft in, so the part reads as a waisted link with
                   circular bosses -- the concept art's "Leg Link (Lower)".
                   Opposes `outline`, which makes hard trapezoid creases.
  outline    0-2   how much the SILHOUETTE departs all the way around.  The
                   keel only ever grew downward; this grows the upper edge and
                   the ends too, so the part stops reading as the original
                   outline with a belly added.  Defaults to 0 so every spec
                   locked before it existed still reproduces exactly.

Language is the categorical choice and must be settled first -- tuning axes
within the wrong language just produces nine things that look alike.
"""
import numpy as np

PALETTES = {
    # The ten explorations from `artistic concepts/` -- sheet "5. COLOR PALETTE
    # EXPLORATIONS", hex codes taken straight off it.  "white" is the main body
    # colour, "dark" the structural/mechanism colour, "accent" the vivid third.
    # Their own note: accent goes on light strips, joint rings, small panels and
    # functional highlights; the main body stays neutral.
    "arctic":     {"white": (0xF2, 0xF4, 0xF7), "dark": (0x2B, 0x2F, 0x36), "accent": (0x1E, 0x7B, 0xFF)},
    "stealth":    {"white": (0x3A, 0x3F, 0x46), "dark": (0x0D, 0x0D, 0x12), "accent": (0x00, 0xE5, 0xFF)},
    "industrial": {"white": (0x7A, 0x7F, 0x87), "dark": (0x1E, 0x21, 0x26), "accent": (0xFF, 0x6A, 0x00)},
    "precision":  {"white": (0xEB, 0xEC, 0xEF), "dark": (0x11, 0x13, 0x18), "accent": (0xE6, 0x39, 0x46)},
    "ocean":      {"white": (0xA7, 0xB1, 0xBB), "dark": (0x0E, 0x2A, 0x3D), "accent": (0x18, 0xD1, 0xC9)},
    "forest":     {"white": (0x6E, 0x8F, 0x6E), "dark": (0x3B, 0x2F, 0x2A), "accent": (0xD9, 0xC7, 0xA1)},
    "nebula":     {"white": (0x5A, 0x5F, 0x6B), "dark": (0x1A, 0x1D, 0x24), "accent": (0x7B, 0x61, 0xFF)},
    "titanium":   {"white": (0xC2, 0xC7, 0xCE), "dark": (0x20, 0x24, 0x2A), "accent": (0xFF, 0xC8, 0x57)},
    "desert":     {"white": (0xD6, 0xB8, 0x90), "dark": (0x2A, 0x2D, 0x33), "accent": (0xFF, 0x8A, 0x3D)},
    "cyber":      {"white": (0x2B, 0x1A, 0x3D), "dark": (0x0B, 0x0D, 0x10), "accent": (0xFF, 0x2D, 0x9A)},
    # off the build-guide sheets: lighter secondary than 01 Arctic's near-black
    "concept":    {"white": (0xE6, 0xE6, 0xE6), "dark": (0x6B, 0x72, 0x80), "accent": (0x00, 0xA8, 0xFF)},
    "concept_d":  {"white": (0xEE, 0xF0, 0xF3), "dark": (0x8A, 0x92, 0xA0), "accent": (0x00, 0xA8, 0xFF)},
    # mine: 01 Arctic's body with the lighter structural grey that reads softer
    "arctic_lt":  {"white": (0xF2, 0xF4, 0xF7), "dark": (0x7E, 0x87, 0x95), "accent": (0x1E, 0x7B, 0xFF)},
    # mine: the yellow HUD sheet in artistic concepts/circuit
    "amber":      {"white": (0xE9, 0xEB, 0xEE), "dark": (0x33, 0x36, 0x3C), "accent": (0xFF, 0xC1, 0x07)},
    # mine: monochrome, accent carried by value alone -- the most restrained option
    "mono":       {"white": (0xF0, 0xF2, 0xF5), "dark": (0x4A, 0x51, 0x5C), "accent": (0x9A, 0xA4, 0xB2)},
    # mine: warm-white body, cool structure -- reads machined rather than printed
    "bone":       {"white": (0xF0, 0xEC, 0xE4), "dark": (0x5C, 0x60, 0x68), "accent": (0x2E, 0x6F, 0xDB)},
}

# Names taken from the "DESIGN LANGUAGE STUDIES" panel of the concept art.
LANGUAGES = {
    "angular": dict(
        blurb="hard facets, chamfered blade edge, trapezoid panels",
        facet=True, chamfer_k=1.0, frame_k=1.0, rail_k=1.0,
        pad_n=1.0, pad_size=1.0, pock_n=1.0, pock_size=1.0,
        side_n=1.0, side_k=1.0, cut_n=0, keel_k=1.0),
    "exposed": dict(
        blurb="skeletal -- material removed, ribs and voids left showing",
        facet=True, chamfer_k=0.5, frame_k=0.0, rail_k=0.6,
        pad_n=0.3, pad_size=0.8, pock_n=0.3, pock_size=1.0,
        side_n=1.4, side_k=1.3, cut_n=5, keel_k=1.15),
    "armored": dict(
        blurb="few large stepped plates, thick overlapping shells",
        facet=True, chamfer_k=1.5, frame_k=1.7, rail_k=1.3,
        pad_n=0.45, pad_size=2.0, pock_n=0.3, pock_size=1.8,
        side_n=0.5, side_k=1.4, cut_n=0, keel_k=1.1),
    "curved": dict(
        blurb="smooth silhouette, minimal features, one sweeping accent",
        facet=False, chamfer_k=1.4, frame_k=0.45, rail_k=0.9,
        pad_n=0.4, pad_size=1.6, pock_n=0.45, pock_size=1.5,
        side_n=0.4, side_k=0.7, cut_n=0, keel_k=0.85),
    # From the concept art: the limb is one or two large unbroken cover panels,
    # not a field of small detail.  Very low counts, very large sizes.
    "shell": dict(
        blurb="one large cover panel, a single flowing accent, mechanism at the joints",
        facet=True, chamfer_k=1.2, frame_k=1.5, rail_k=0.45,
        pad_n=0.18, pad_size=2.8, pock_n=0.14, pock_size=2.6,
        side_n=0.35, side_k=1.1, cut_n=0, keel_k=1.05),
    "greebled": dict(
        blurb="dense small detail -- looks like machinery underneath",
        facet=True, chamfer_k=0.8, frame_k=0.9, rail_k=0.9,
        pad_n=2.4, pad_size=0.45, pock_n=2.4, pock_size=0.45,
        side_n=2.2, side_k=0.8, cut_n=2, keel_k=1.0),
}
LV = [0, 3, 6, 10]          # calibration levels for the aggression curve


def blend(a, b, t=0.5):
    """Mix two languages.  t=0 is all `a`, t=1 is all `b`.

    Languages are plain dicts of numbers, so a crossbreed is just a weighted
    average -- which is why "something between B and C" is a one-liner and not
    a new language that has to be authored by hand."""
    A, B = LANGUAGES[a], LANGUAGES[b]
    out = {}
    for k, va in A.items():
        vb = B[k]
        if k == "blurb":       out[k] = f"{va}  ×  {vb}"
        elif isinstance(va, bool): out[k] = va or vb
        else:                  out[k] = va * (1 - t) + vb * t
    return out


def _c(a, v):
    return float(np.interp(a, LV, v))


def derive(aggression=6.0, density=1.0, relief=1.0, scale=1.0, accent=1.0,
           outline=0.0, organic=0.0, waist=0.0,
           language="angular", palette="arctic", mix=None, **over):
    """Build a spec from a language plus the axes.  `over` pins fields.

    `mix=("exposed","armored",0.5)` crossbreeds two languages instead of
    using one; `language` then just names the result for the label."""
    a = float(np.clip(aggression, 0, 10))
    d, r = float(density), float(relief)
    sc, ac, ol = float(scale), float(accent), float(outline)
    og, wa = float(organic), float(waist)
    if mix is not None:
        L = blend(mix[0], mix[1], float(mix[2]))
        language = f"{mix[0][:3]}+{mix[1][:3]} {float(mix[2]):.2f}"
    else:
        L = LANGUAGES[language]
    s = {
        "aggression": a, "density": d, "relief": r, "scale": sc, "accent": ac,
        "outline": ol, "organic": og, "waist": wa,
        # smoothing radius, and how far the mid-shaft necks in
        "org_r":     (3.5 + 5.5 * og) if og > 0.05 else 0.0,
        "waist_d":   _c(a, [0.0, 4.0, 7.0, 10.0]) * wa,
        "mix": list(mix) if mix is not None else None,
        "language": language, "palette": palette, "blurb": L["blurb"],
        "facet": L["facet"],
        # silhouette
        "chamfer":    _c(a, [0.0, 2.5, 4.0, 5.5]) * L["chamfer_k"],
        "keel_depth": _c(a, [0.0, 3.0, 9.5, 17.0]) * L["keel_k"],
        "knee_grow":  _c(a, [0.0, 0.4, 0.8, 2.2]),
        "wheel_grow": _c(a, [0.0, 1.0, 2.0, 4.0]),
        # Trapezoidal edge growth: straight plateaus joined at sharp vertices.
        # The first attempt grew a smooth profile AND inflated the round knee /
        # wheel lobes; the verdict was "too round ... just look fat".  So the
        # lobes are left alone and the growth is now hard-cornered only.
        "out_h":     _c(a, [0.0, 3.0, 5.0, 7.5]) * ol,
        "out_steps": 2,
        "out_sides": "both",
        "out_ramp":  14.0,
        # circles: collar thickness and whether the outer edge is faceted / tabbed
        "collar_t":   3.4,
        "collar_seg": 96,        # 96 = round; 2 = octagon, i.e. straight creases
        "collar_tabs": 0,
        # the thin-ring motif reused as a surface feature away from the bores
        "n_rings":   0,
        "ring_r":    7.0 * max(0.7, sc),
        "ring_t":    1.7,
        "ring_band": "mid",
        "notches":    a >= 5.0 and L["facet"],
        # relief heights / depths
        "frame_h":  _c(a, [0.0, 3.0, 8.0, 13.0]) * r * L["frame_k"],
        "rail_h":   _c(a, [0.0, 2.0, 4.0,  6.5]) * r * L["rail_k"],
        "pad_h":    _c(a, [0.0, 1.4, 2.6,  4.2]) * r,
        "pocket_d": _c(a, [0.0, 1.0, 1.8,  3.0]) * r,
        "win_d":    _c(a, [0.0, 2.0, 3.6,  5.5]) * r,
        "side_d":   _c(a, [0.0, 2.0, 3.6,  5.0]) * r * L["side_k"],
        # feature counts and sizes
        "n_pads":    int(round(_c(a, [0, 2, 4, 7]) * d * L["pad_n"])),
        "n_pockets": int(round(_c(a, [0, 3, 5, 9]) * d * L["pock_n"])),
        "n_wins":    int(round(_c(a, [0, 2, 3, 5]) * d)),
        "n_side_lo": int(round(_c(a, [0, 3, 5, 9]) * d * L["side_n"])),
        "n_side_hi": int(round(_c(a, [0, 2, 4, 7]) * d * L["side_n"])),
        "n_cuts":    int(round(L["cut_n"] * d * min(1.0, a / 6.0))),
        "pad_size":  L["pad_size"] * sc, "pock_size": L["pock_size"] * sc,
        # gaps shrink as features grow, so a bigger feature really is bigger
        # in X and not just taller in Y
        "gap_k":     1.0 / max(0.35, sc),
        # accent placement.  "spine" follows the perimeter contour; "axial" runs
        # one tapered stripe down the limb's own midline, which is what the
        # concept art does and what reads as flowing rather than scattered.
        "accent_style":   "spine",
        # elongated-trapezoid accent: same vocabulary as the cutouts
        "n_traps":   4, "trap_h": 6.5 * ac, "trap_skew": 10.0, "trap_grad": 0.0,
        # circuit trace: long straight runs, occasional trapezoidal level change,
        # optionally shadowed by a wider grey trace over part of its length
        "n_jogs":  3, "jog_amp": 4.5, "jog_w": 7.0,
        "trace_t": 3.0 * max(0.6, ac), "comp_w": 0.0, "comp_gap": 1.7,
        "n_hatch": 0,
        "accent_channel": 0.0,     # graphite border width around the stripe
        "axial_w0": 4.6 * ac, "axial_w1": 1.5 * ac,
        "axial_off": 0.0,
        # accent: a continuous spine with blocks strung along it
        "acc_spine": _c(a, [0.0, 1.6, 2.2, 2.8]) * ac,
        "acc_block": _c(a, [0.0, 5.0, 7.0, 9.0]) * ac * max(0.6, sc),
        "n_acc":     int(round(_c(a, [0, 3, 4, 6]) * d)),
        # colour
        "split_z": -5.0,
    }
    s.update(over)
    return s


def label(s):
    return (f"{s['language'].upper()}   agg {s['aggression']:.0f} · "
            f"dens {s['density']:.2f} · rel {s['relief']:.2f} · "
            f"scale {s.get('scale', 1.0):.2f} · blue {s.get('accent', 1.0):.2f} · "
            f"outline {s.get('outline', 0.0):.2f} · org {s.get('organic', 0.0):.2f}"
            + (f" · waist {s.get('waist', 0.0):.2f}" if s.get('waist', 0.0) else ""))
