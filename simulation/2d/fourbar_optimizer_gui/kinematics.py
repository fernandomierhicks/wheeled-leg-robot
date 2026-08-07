"""kinematics.py — closed-form 4-bar IK with assembly-branch continuity.

Topology (A at origin, +X forward, +Z up):

    A ---- femur ----> C          A = hip motor output (driven)
    F ---- coupler --> E          F = body-fixed passive pivot
    E - C - W  are one rigid body (the tibia); W may be offset off-axis.

The closed loop is A-C-E-F.  W rides on the tibia body and does not enter the
loop, which is why a dogleg tibia costs nothing in solve time.

Derived from `simulation/mujoco/archive/baseline1_leg_analysis/physics.py`
(`solve_ik`, lines 78-112) with two changes:

  1. A_Z is dropped (A is the origin here).
  2. Branch selection.  The archive picks the branch minimising |q_knee| at
     every angle *independently*, which can silently flip assembly mode partway
     through a sweep.  For range-of-motion work that is fatal, so here the
     branch is resolved once at the seed pose and then tracked by continuity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from model import LinkageSpec, FEMUR, TIBIA, COUPLER

SINGULARITY_LIM = 0.95      # |K|/R must stay below this (archive value)


def wrap(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Pose:
    """One solved configuration."""
    q_hip: float
    alpha: float                 # tibia absolute angle (archive convention)
    q_knee: float
    kr: float                    # |K|/R singularity measure
    nodes: dict                  # 'A','C','E','F','W' -> (x, z) world
    frames: dict                 # link name -> (angle_rad, (ox, oz)) world

    def closure_error(self) -> float:
        """|E - F| - Lc should be ~0; a self-test of the solution."""
        ex, ez = self.nodes["E"]
        fx, fz = self.nodes["F"]
        return math.hypot(ex - fx, ez - fz)


def _branches(spec: LinkageSpec, q_hip: float):
    """Return (C, alpha1, alpha2, kr) or None at a singularity."""
    L_f, L_s, Lc = spec.L_femur, spec.L_stub, spec.Lc
    F_X, F_Z = spec.F_X, spec.F_Z

    C_x = -L_f * math.cos(q_hip)
    C_z = L_f * math.sin(q_hip)

    dx, dz = C_x - F_X, C_z - F_Z
    R = math.hypot(dx, dz)
    if R < 1e-9:
        return None

    K = (Lc * Lc - dx * dx - dz * dz - L_s * L_s) / (2.0 * L_s)
    kr = abs(K) / R
    if kr >= SINGULARITY_LIM:
        return None

    phi = math.atan2(dz, dx)
    asinv = math.asin(max(-1.0, min(1.0, K / R)))
    a1 = wrap(asinv - phi)
    a2 = wrap(math.pi - asinv - phi)
    return (C_x, C_z), a1, a2, kr


def solve_pose(spec: LinkageSpec, q_hip: float,
               alpha_prev: float | None = None) -> Pose | None:
    """Solve one configuration.

    alpha_prev=None  -> seed behaviour: pick the branch with the smaller |q_knee|
                        (matches the archive, so cross-checks line up).
    alpha_prev given -> pick the branch closest to it, keeping the sweep on one
                        assembly mode.
    """
    br = _branches(spec, q_hip)
    if br is None:
        return None
    (C_x, C_z), a1, a2, kr = br

    if alpha_prev is None:
        qk1, qk2 = a1 - q_hip, a2 - q_hip
        alpha = a1 if abs(qk1) <= abs(qk2) else a2
    else:
        alpha = a1 if abs(wrap(a1 - alpha_prev)) <= abs(wrap(a2 - alpha_prev)) else a2

    sa, ca = math.sin(alpha), math.cos(alpha)
    E_x = C_x + spec.L_stub * sa
    E_z = C_z + spec.L_stub * ca

    # Tibia local frame: origin C, +x_local toward E.
    t_ang = math.atan2(ca, sa)
    W_x, W_z = local_to_world((-spec.L_tibia, spec.w_perp), t_ang, (C_x, C_z))

    nodes = {
        "A": (0.0, 0.0),
        "C": (C_x, C_z),
        "E": (E_x, E_z),
        "F": (spec.F_X, spec.F_Z),
        "W": (W_x, W_z),
    }
    frames = {
        FEMUR: (math.atan2(C_z, C_x), (0.0, 0.0)),
        TIBIA: (t_ang, (C_x, C_z)),
        COUPLER: (math.atan2(E_z - spec.F_Z, E_x - spec.F_X), (spec.F_X, spec.F_Z)),
    }
    return Pose(q_hip=q_hip, alpha=alpha, q_knee=alpha - q_hip, kr=kr,
                nodes=nodes, frames=frames)


def wheel_jacobian(spec: LinkageSpec, pose: Pose) -> tuple[float, float] | None:
    """d(W)/d(q_hip) in world axes, [m/rad].  None at a singularity.

    Closed form, not a finite difference: the loop constraint |E - F| = Lc is
    differentiated directly, so there is no step-size to tune and no extra IK
    solve per pose.

        C  = (-Lf cos q, Lf sin q)            dC/dq = (Lf sin q, Lf cos q)
        E  = C + L_stub (sin a, cos a)        u = E - F,  |u| = Lc
        u . dE/dq = 0   ->   a' = -(u . dC/dq) / (L_stub (u_x cos a - u_z sin a))

    W rides on the tibia, whose frame angle is t = pi/2 - a, so dt/dq = -a'.
    The denominator vanishing IS the 4-bar singularity — the same event
    SINGULARITY_LIM guards — so it returns None rather than a huge number.
    """
    q, a = pose.q_hip, pose.alpha
    dC_x = spec.L_femur * math.sin(q)
    dC_z = spec.L_femur * math.cos(q)

    u_x = pose.nodes["E"][0] - spec.F_X
    u_z = pose.nodes["E"][1] - spec.F_Z
    ca, sa = math.cos(a), math.sin(a)

    den = spec.L_stub * (u_x * ca - u_z * sa)
    if abs(den) < 1e-12:
        return None
    da = -(u_x * dC_x + u_z * dC_z) / den

    t = math.atan2(ca, sa)                    # tibia frame angle = pi/2 - a
    st, ct = math.sin(t), math.cos(t)
    p_x, p_z = -spec.L_tibia, spec.w_perp     # W in the tibia's local frame
    dt = -da
    return (dC_x + dt * (-st * p_x - ct * p_z),
            dC_z + dt * (ct * p_x - st * p_z))


def local_to_world(pt, angle: float, origin) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    x, z = pt
    return (c * x - s * z + origin[0], s * x + c * z + origin[1])


def trace_world(pose: Pose, tp) -> tuple[float, float]:
    """World position of a TracePoint at this pose."""
    ang, org = pose.frames[tp.body]
    return local_to_world((tp.local_x, tp.local_z), ang, org)


def local_nodes(spec: LinkageSpec) -> dict[str, list]:
    """Each link's node list in its own local frame: [(name, x, z, radius), ...]."""
    return {
        FEMUR: [("A", 0.0, 0.0, spec.femur_r_A),
                ("C", spec.L_femur, 0.0, spec.femur_r_C)],
        TIBIA: [("E", spec.L_stub, 0.0, spec.tibia_r_E),
                ("C", 0.0, 0.0, spec.tibia_r_C),
                ("W", -spec.L_tibia, spec.w_perp, spec.tibia_r_W)],
        COUPLER: [("F", 0.0, 0.0, spec.coupler_r_F),
                  ("E", spec.Lc, 0.0, spec.coupler_r_E)],
    }


def sweep(spec: LinkageSpec, q_values) -> list:
    """Solve a list of hip angles keeping one assembly branch, seeded at
    q_seed.  Returns a list of Pose|None, aligned with q_values."""
    seed = solve_pose(spec, spec.q_seed)
    prev = seed.alpha if seed else None
    out = []
    for q in q_values:
        p = solve_pose(spec, q, prev)
        if p is not None:
            prev = p.alpha
        out.append(p)
    return out


def branch_is_continuous(poses, tol_deg: float = 15.0) -> bool:
    """True if alpha never jumps more than tol_deg between consecutive poses."""
    tol = math.radians(tol_deg)
    prev = None
    for p in poses:
        if p is None:
            prev = None
            continue
        if prev is not None and abs(wrap(p.alpha - prev)) > tol:
            return False
        prev = p.alpha
    return True
