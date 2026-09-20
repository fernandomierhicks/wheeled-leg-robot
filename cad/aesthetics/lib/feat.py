"""Trapezoidal feature language shared by the styling recipes.

Everything is an elongated trapezoid: raised pads, recessed pockets, side-wall
cuts and the interlocking colour breaks all use the same primitive so the part
reads as one design instead of a pile of unrelated details.
"""
import numpy as np
from build123d import *
from shapely.geometry import Polygon as ShPoly
from shapely.ops import unary_union
from shputil import _clean_ring, prism, geoms

def trap_xz(x0, x1, z0, z1, skew=5.0, taper_end=True):
    """Elongated trapezoid in the X-Z plane (for side-wall features)."""
    if taper_end:
        return [(x0+skew, z0), (x0, z1), (x1-skew, z1), (x1, z0)]
    return [(x0, z0), (x0+skew, z1), (x1, z1), (x1-skew, z0)]

def side_solid(pts, side, reach=90.0):
    """Solid swept through the part in Y from a X-Z profile.
    side=-1 sweeps the -Y (keel) wall, +1 the +Y wall."""
    f = Plane.XZ * make_face(Polyline(*[(x, z, 0) for x, z in pts], close=True))
    return extrude(f, amount=reach if side < 0 else -reach)

def wall_shell(poly, z0, z1, t=3.6):
    """Thin shell hugging the outer wall, used to clip side cuts to a depth."""
    return prism(poly, z0, z1) - prism(poly.buffer(-t, join_style=2), z0, z1)

def trap_plan(x0, x1, y0, y1, skew=7.0, drop=0.0):
    """Elongated trapezoid in plan, optionally sloping along its length."""
    return ShPoly([(x0+skew, y0), (x0, y1), (x1-skew, y1+drop), (x1, y0+drop)])

def band_along(poly, inset0, inset1, xlo, xhi, upper=None):
    """Contour-following band between two insets, clipped in X."""
    from shapely.geometry import box as shbox
    b = poly.buffer(-inset0, join_style=2).difference(poly.buffer(-inset1, join_style=2))
    b = b.intersection(shbox(xlo, -200, xhi, 200))
    if upper is not None:
        b = b.intersection(upper) if upper is not False else b
    return b
