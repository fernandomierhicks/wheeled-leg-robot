"""geometry.py — collision shapes and tests.

Each link's collision body is the convex hull of the discs at its nodes (the
"tapered slot" plate shape).  Because a link is rigid, that hull is built ONCE
in the link's local frame and then merely rotated/translated per pose.

The polygon that gets tested is the polygon that gets drawn, so there is no
hidden discrepancy between the picture and the constraint.

Tests used:
  link  vs link   -> SAT on convex polygons (exact for the polygonal hull)
  circle vs link  -> exact point/segment distance
  circle vs circle-> centre distance
A bounding-circle broad phase runs first and rejects most pairs outright.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull

from model import (LinkageSpec, FEMUR, TIBIA, COUPLER, FEMUR_SHAFT, MOTOR,
                   WHEEL, WHEEL_MOTOR, MOVING_LINKS, pair_key)
from kinematics import local_nodes, trace_world

# Polygon resolution: coarse for optimization, fine for drawing.
RES_FAST = 8
RES_DRAW = 32


# ---------------------------------------------------------------------------
# Shape construction
# ---------------------------------------------------------------------------
def disc_hull(nodes, n_per_disc: int = RES_DRAW) -> np.ndarray:
    """Convex hull of the node discs.  nodes = [(name, x, z, r), ...].
    Returns (N,2) vertices in CCW order, in the same frame as the input."""
    th = np.linspace(0.0, 2.0 * np.pi, n_per_disc, endpoint=False)
    cs = np.stack([np.cos(th), np.sin(th)], axis=1)      # (n,2)
    pts = np.vstack([np.array([x, z]) + r * cs for _, x, z, r in nodes])
    hull = ConvexHull(pts)
    return pts[hull.vertices]                            # scipy gives CCW in 2D


def bounding_circle(poly: np.ndarray) -> tuple[np.ndarray, float]:
    """Cheap conservative bounding circle (centroid + max radius)."""
    c = poly.mean(axis=0)
    r = float(np.max(np.linalg.norm(poly - c, axis=1)))
    return c, r


class ShapeSet:
    """Local-frame hulls for the three moving links, built once per geometry."""

    def __init__(self, spec: LinkageSpec, resolution: int = RES_DRAW):
        self.resolution = resolution
        ln = local_nodes(spec)
        self.local: dict[str, np.ndarray] = {
            k: disc_hull(v, resolution) for k, v in ln.items()
        }
        self.bc: dict[str, tuple[np.ndarray, float]] = {
            k: bounding_circle(v) for k, v in self.local.items()
        }

    def world(self, pose, link: str) -> np.ndarray:
        ang, org = pose.frames[link]
        return transform(self.local[link], ang, org)

    def world_bc(self, pose, link: str) -> tuple[np.ndarray, float]:
        ang, org = pose.frames[link]
        c, r = self.bc[link]
        ca, sa = math.cos(ang), math.sin(ang)
        wc = np.array([ca * c[0] - sa * c[1] + org[0],
                       sa * c[0] + ca * c[1] + org[1]])
        return wc, r


def transform(poly: np.ndarray, angle: float, origin) -> np.ndarray:
    ca, sa = math.cos(angle), math.sin(angle)
    R = np.array([[ca, -sa], [sa, ca]])
    return poly @ R.T + np.asarray(origin)


# ---------------------------------------------------------------------------
# Primitive overlap tests
# ---------------------------------------------------------------------------
def _axes(poly: np.ndarray) -> np.ndarray:
    e = np.roll(poly, -1, axis=0) - poly
    n = np.stack([-e[:, 1], e[:, 0]], axis=1)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln < 1e-15] = 1.0
    return n / ln


def sat_overlap(A: np.ndarray, B: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons.  True if they overlap."""
    for ax in (_axes(A), _axes(B)):
        pa = A @ ax.T                # (nA, nAxes)
        pb = B @ ax.T
        if np.any((pa.max(axis=0) < pb.min(axis=0)) |
                  (pb.max(axis=0) < pa.min(axis=0))):
            return False
    return True


def point_in_convex(p: np.ndarray, poly: np.ndarray) -> bool:
    e = np.roll(poly, -1, axis=0) - poly
    v = p - poly
    cr = e[:, 0] * v[:, 1] - e[:, 1] * v[:, 0]
    return bool(np.all(cr >= -1e-12) or np.all(cr <= 1e-12))


def point_poly_distance(p: np.ndarray, poly: np.ndarray) -> float:
    """Distance from p to the polygon boundary (0 if inside)."""
    if point_in_convex(p, poly):
        return 0.0
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.einsum("ij,ij->i", p - a, ab) / np.maximum(
        np.einsum("ij,ij->i", ab, ab), 1e-18)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(p - proj, axis=1)))


def circle_poly_overlap(c, r: float, poly: np.ndarray) -> bool:
    return point_poly_distance(np.asarray(c, dtype=float), poly) < r


def circles_overlap(c1, r1: float, c2, r2: float) -> bool:
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1]) < (r1 + r2)


# ---------------------------------------------------------------------------
# Whole-mechanism collision query
# ---------------------------------------------------------------------------
def body_shapes(spec: LinkageSpec, pose, shapes: ShapeSet) -> dict:
    """World-space shape for every body present at this pose.

    Links map to ('poly', vertices); motor/wheel/wheel_motor map to
    ('circle', centre, r).
    """
    out = {k: ("poly", shapes.world(pose, k)) for k in MOVING_LINKS}
    out[MOTOR] = ("circle", np.array([0.0, 0.0]), spec.motor_r)
    # Knee shaft: rides with the femur at C, so it needs no local hull of its
    # own — the node already tracks the pose.
    out[FEMUR_SHAFT] = ("circle", np.asarray(pose.nodes["C"], dtype=float),
                        spec.femur_shaft_r)
    wc = np.asarray(trace_world(pose, spec.primary_trace()), dtype=float)
    if spec.wheel_enabled:
        out[WHEEL] = ("circle", wc, spec.wheel_r)
    # The hub motor is always present — it is bolted to the leg whether or not
    # the tyre outline is being drawn.
    out[WHEEL_MOTOR] = ("circle", wc, spec.wheel_motor_r)
    return out


def _overlap(sa, sb) -> bool:
    if sa[0] == "poly" and sb[0] == "poly":
        return sat_overlap(sa[1], sb[1])
    if sa[0] == "circle" and sb[0] == "circle":
        return circles_overlap(sa[1], sa[2], sb[1], sb[2])
    circ, poly = (sa, sb) if sa[0] == "circle" else (sb, sa)
    return circle_poly_overlap(circ[1], circ[2], poly[1])


def _bcirc(spec: LinkageSpec, pose, shapes: ShapeSet, name: str, shape):
    if shape[0] == "circle":
        return shape[1], shape[2]
    return shapes.world_bc(pose, name)


def collisions(spec: LinkageSpec, pose, shapes: ShapeSet) -> list[tuple[str, str]]:
    """All enabled body pairs that overlap at this pose."""
    sh = body_shapes(spec, pose, shapes)
    names = list(sh.keys())
    bcs = {n: _bcirc(spec, pose, shapes, n, sh[n]) for n in names}

    hits = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not spec.collides(a, b):
                continue
            (ca, ra), (cb, rb) = bcs[a], bcs[b]
            if not circles_overlap(ca, ra, cb, rb):     # broad phase
                continue
            if _overlap(sh[a], sh[b]):
                hits.append(pair_key(a, b))
    return hits


def any_collision(spec: LinkageSpec, pose, shapes: ShapeSet) -> bool:
    return len(collisions(spec, pose, shapes)) > 0
