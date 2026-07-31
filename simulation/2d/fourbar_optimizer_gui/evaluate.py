"""evaluate.py — collision-free range of motion and path metrics.

The reachable range is the CONTIGUOUS interval containing the seed pose: a
mechanism can only get to a configuration by moving there continuously, so a
collision-free island on the far side of an interference is unreachable.

March outward from the seed in both directions; stop each direction at the
first singularity, collision, or user limit; bisect to pin the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from model import LinkageSpec, MOVING_LINKS
from kinematics import solve_pose, trace_world
from geometry import ShapeSet, collisions, RES_FAST, RES_DRAW

DQ_COARSE = math.radians(1.0)     # march step
BISECT_TOL = math.radians(0.05)   # boundary resolution
N_METRIC_SAMPLES = 240


@dataclass
class RangeResult:
    valid: bool
    q_lo: float = 0.0
    q_hi: float = 0.0
    stop_lo: str = ""
    stop_hi: str = ""
    reason: str = ""
    # First BLOCKED angle just past each end (None when the end is a user
    # limit).  q_lo/q_hi are the last *good* poses, so without these there is
    # no way to see what actually interferes.
    q_block_lo: float | None = None
    q_block_hi: float | None = None
    q_seed_used: float = 0.0     # may differ from spec.q_seed when auto_seed

    @property
    def stroke_deg(self) -> float:
        return math.degrees(self.q_hi - self.q_lo)


@dataclass
class Metrics:
    valid: bool
    reason: str = ""
    travel_mm: float = 0.0        # vertical travel of the primary traced point
    max_dev_mm: float = 0.0       # max |x - mean x| (best-fit vertical)
    rms_dev_mm: float = 0.0
    stroke_deg: float = 0.0
    mean_x_mm: float = 0.0
    q_lo: float = 0.0
    q_hi: float = 0.0
    stop_lo: str = ""
    stop_hi: str = ""
    q_block_lo: float | None = None
    q_block_hi: float | None = None
    q_seed_used: float = 0.0     # differs from spec.q_seed when auto_seed moved it
    path: np.ndarray = field(default=None, repr=False)   # (N,2) traced path
    q_samples: np.ndarray = field(default=None, repr=False)
    alphas: np.ndarray = field(default=None, repr=False)  # branch at each sample

    def alpha_at(self, q: float):
        """Assembly-branch alpha nearest to q — pass as alpha_prev when solving
        an arbitrary pose so it stays on the same branch as the sweep."""
        if self.alphas is None or self.q_samples is None or len(self.alphas) == 0:
            return None
        return float(self.alphas[int(np.argmin(np.abs(self.q_samples - q)))])

    def summary(self) -> str:
        if not self.valid:
            return f"INVALID — {self.reason}"
        return (f"travel {self.travel_mm:.1f} mm | dev {self.max_dev_mm:.2f} mm | "
                f"stroke {self.stroke_deg:.2f}deg | "
                f"[{math.degrees(self.q_lo):.2f}, {math.degrees(self.q_hi):.2f}]deg")


def _blocked(spec, q, alpha_prev, shapes, check_collisions):
    """Return (pose, reason) — reason empty if the pose is usable."""
    p = solve_pose(spec, q, alpha_prev)
    if p is None:
        return None, "singularity"
    if check_collisions:
        hits = collisions(spec, p, shapes)
        if hits:
            return p, "collision:" + ",".join(f"{a}/{b}" for a, b in hits)
    if spec.enforce_wheel_below_hip:
        # The wheel's UPPER tangent — not its centre — must stay below the hip
        # motor axis.  Once the top of the tyre passes the axis the chassis is
        # already down on the ground and there is no jump left.
        top = trace_world(p, spec.primary_trace())[1] + spec.wheel_r
        if top > spec.wheel_below_hip_margin:
            return p, f"wheel top above hip ({top*1000:+.1f} mm)"
    if spec.enforce_W_below_E:
        # W above E => tibia leaned back past horizontal, far end toward ground.
        dz = p.nodes["W"][1] - p.nodes["E"][1]
        if dz > 0.0:
            return p, f"W above E ({dz*1000:+.1f} mm)"
    if check_collisions and spec.enforce_ground:
        gz = ground_z(spec, p)
        for link in MOVING_LINKS:
            lo = shapes.world(p, link)[:, 1].min()
            if lo < gz:
                return p, f"ground:{link} ({(gz - lo)*1000:.1f} mm below)"
        if -spec.motor_r < gz:
            return p, f"ground:motor ({(gz + spec.motor_r)*1000:.1f} mm below)"
    return p, ""


def ground_z(spec, pose) -> float:
    """Height of the ground plane at this pose: tangent to the wheel bottom."""
    return trace_world(pose, spec.primary_trace())[1] - spec.wheel_r


def find_seed(spec: LinkageSpec, shapes: ShapeSet,
              check_collisions: bool = True, step_deg: float = 1.5):
    """Nearest usable hip angle to spec.q_seed, or None.

    During optimization the nominal seed often lands in a collision for a
    perturbed geometry, which would reject an otherwise fine mechanism.  This
    scans the allowed span and returns the valid angle closest to the nominal
    seed, so the candidate is judged on its best assembly rather than on one
    arbitrary pose.
    """
    step = math.radians(step_deg)
    n = max(2, int((spec.q_max - spec.q_min) / step) + 1)
    best, best_d = None, None
    for q in np.linspace(spec.q_min, spec.q_max, n):
        # Same predicate as the range march, so no check can be enforced in
        # one path and forgotten in the other.
        _, reason = _blocked(spec, float(q), None, shapes, check_collisions)
        if reason:
            continue
        d = abs(q - spec.q_seed)
        if best_d is None or d < best_d:
            best, best_d = float(q), d
    return best


def find_range(spec: LinkageSpec, shapes: ShapeSet | None = None,
               check_collisions: bool = True,
               dq: float = DQ_COARSE,
               auto_seed: bool = False) -> RangeResult:
    """Largest contiguous usable interval containing the seed pose.

    auto_seed=True relocates the seed to the nearest usable angle instead of
    failing outright — used by the optimizer.
    """
    if shapes is None:
        shapes = ShapeSet(spec, RES_FAST)

    q_seed = spec.q_seed
    seed, reason = _blocked(spec, q_seed, None, shapes, check_collisions)
    if reason and auto_seed:
        alt = find_seed(spec, shapes, check_collisions)
        if alt is not None:
            q_seed = alt
            seed, reason = _blocked(spec, q_seed, None, shapes, check_collisions)

    if reason:
        return RangeResult(False, reason=f"seed pose unusable: {reason}")

    bounds, stops, blocks = {}, {}, {}
    for direction, limit in ((+1, spec.q_max), (-1, spec.q_min)):
        q_good, alpha = q_seed, seed.alpha
        stop, q_block = "limit", None
        while True:
            q_next = q_good + direction * dq
            if (direction > 0 and q_next > limit) or (direction < 0 and q_next < limit):
                q_next, at_limit = limit, True
            else:
                at_limit = False

            pose, reason = _blocked(spec, q_next, alpha, shapes, check_collisions)
            if reason:
                # Bisect between q_good (usable) and q_next (blocked).
                lo, hi = q_good, q_next
                a_lo = alpha
                while abs(hi - lo) > BISECT_TOL:
                    mid = 0.5 * (lo + hi)
                    p_mid, r_mid = _blocked(spec, mid, a_lo, shapes, check_collisions)
                    if r_mid:
                        hi = mid
                    else:
                        lo, a_lo = mid, p_mid.alpha
                # `hi` is now the first blocked angle; re-read its reason there
                # rather than reusing the coarse-step one.
                _, r_hi = _blocked(spec, hi, a_lo, shapes, check_collisions)
                q_good, stop, q_block = lo, (r_hi or reason), hi
                break

            q_good, alpha = q_next, pose.alpha
            if at_limit:
                stop = "limit"
                break

        bounds[direction], stops[direction] = q_good, stop
        blocks[direction] = q_block

    return RangeResult(True, q_lo=bounds[-1], q_hi=bounds[+1],
                       stop_lo=stops[-1], stop_hi=stops[+1],
                       q_block_lo=blocks[-1], q_block_hi=blocks[+1],
                       q_seed_used=q_seed)


def evaluate(spec: LinkageSpec, shapes: ShapeSet | None = None,
             check_collisions: bool = True,
             n_samples: int = N_METRIC_SAMPLES,
             auto_seed: bool = False) -> Metrics:
    """Full evaluation: reachable range + traced-path metrics."""
    if shapes is None:
        shapes = ShapeSet(spec, RES_FAST)

    rng = find_range(spec, shapes, check_collisions, auto_seed=auto_seed)
    if not rng.valid:
        return Metrics(False, reason=rng.reason)
    if rng.q_hi - rng.q_lo < 1e-6:
        return Metrics(False, reason="zero-width range")

    tp = spec.primary_trace()
    qs = np.linspace(rng.q_lo, rng.q_hi, n_samples)

    # Walk from the seed outward in each direction so branch tracking is
    # seeded correctly, then reassemble in ascending q order.
    seed = solve_pose(spec, rng.q_seed_used)
    i_seed = int(np.searchsorted(qs, rng.q_seed_used))
    pts: list[tuple[float, float] | None] = [None] * n_samples
    als: list[float | None] = [None] * n_samples

    alpha = seed.alpha
    for i in range(min(i_seed, n_samples - 1), n_samples):
        p = solve_pose(spec, qs[i], alpha)
        if p is None:
            break
        alpha = p.alpha
        pts[i], als[i] = trace_world(p, tp), p.alpha

    alpha = seed.alpha
    for i in range(min(i_seed, n_samples - 1) - 1, -1, -1):
        p = solve_pose(spec, qs[i], alpha)
        if p is None:
            break
        alpha = p.alpha
        pts[i], als[i] = trace_world(p, tp), p.alpha

    keep = [i for i, v in enumerate(pts) if v is not None]
    if len(keep) < 3:
        return Metrics(False, reason="path solve failed across range")

    path = np.array([pts[i] for i in keep], dtype=float)
    q_kept = qs[keep]
    a_kept = np.array([als[i] for i in keep], dtype=float)

    x, z = path[:, 0], path[:, 1]
    mean_x = float(x.mean())
    dev = x - mean_x

    return Metrics(
        valid=True,
        travel_mm=float(z.max() - z.min()) * 1000.0,
        max_dev_mm=float(np.abs(dev).max()) * 1000.0,
        rms_dev_mm=float(np.sqrt(np.mean(dev ** 2))) * 1000.0,
        stroke_deg=rng.stroke_deg,
        mean_x_mm=mean_x * 1000.0,
        q_lo=rng.q_lo, q_hi=rng.q_hi,
        stop_lo=rng.stop_lo, stop_hi=rng.stop_hi,
        q_block_lo=rng.q_block_lo, q_block_hi=rng.q_block_hi,
        q_seed_used=rng.q_seed_used,
        path=path, q_samples=q_kept, alphas=a_kept,
    )
