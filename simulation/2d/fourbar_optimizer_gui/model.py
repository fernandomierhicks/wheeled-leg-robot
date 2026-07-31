"""model.py — LinkageSpec: every dimension of the 4-bar in one place.

Frame convention (canonical, per CLAUDE.md):
    +X = robot forward,  +Z = up.
    The hip motor A is FIXED AT THE ORIGIN (0, 0).

Frame-shift note vs the mujoco archive
--------------------------------------
`simulation/mujoco/archive/baseline1_leg_analysis/sim_config.py` expresses F
relative to the *body centre*, with the motor at A_Z = -0.0235 m:

    F_X = -0.05887,  F_Z = -0.01821,  A_Z = -0.0235

With A at the origin the coupler pivot becomes

    F_X = -0.05887              (unchanged, A was already at x = 0)
    F_Z = -0.01821 - (-0.0235) = +0.00529      <-- NOT -0.01821

That +5.29 mm is the true hip-to-coupler *height* offset.  Everything in this
package uses the A-at-origin frame; `to_body_frame_fz()` converts back.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Body names used for collision pairs and trace-point attachment.
FEMUR = "femur"
TIBIA = "tibia"
COUPLER = "coupler"
MOTOR = "motor"
WHEEL = "wheel"

MOVING_LINKS = (FEMUR, TIBIA, COUPLER)
ALL_BODIES = (FEMUR, TIBIA, COUPLER, MOTOR, WHEEL)

# A_Z used by the archive; kept only for frame conversion + cross-checking.
ARCHIVE_A_Z = -0.0235


def pair_key(a: str, b: str) -> tuple[str, str]:
    """Order-independent key for a collision pair."""
    return (a, b) if a <= b else (b, a)


# Pairs that are physically incapable of colliding because the two bodies are
# rigidly joined or share a pin (hence sit on different plates):
#   femur/tibia  share the knee pivot C
#   tibia/coupler share the pivot E
#   femur/motor   the femur rotates on the motor's own output shaft at A
#   tibia/wheel   the wheel is bolted to the tibia at W
# Plus, per the build: the wheel is checked against the hip motor and the
# coupler.  It still runs clear of the femur plate in Y.
_DEFAULT_OFF = {
    pair_key(FEMUR, TIBIA),
    pair_key(TIBIA, COUPLER),
    pair_key(FEMUR, MOTOR),
    pair_key(TIBIA, WHEEL),
    pair_key(FEMUR, WHEEL),
}


def default_collision_matrix() -> dict[tuple[str, str], bool]:
    m = {}
    for i, a in enumerate(ALL_BODIES):
        for b in ALL_BODIES[i + 1:]:
            k = pair_key(a, b)
            m[k] = k not in _DEFAULT_OFF
    return m


@dataclass
class TracePoint:
    """A point rigidly attached to one body, in that body's local frame."""
    name: str
    body: str
    local_x: float
    local_z: float
    primary: bool = False       # the point the optimizer scores
    color: str = "#40ff80"


@dataclass
class LinkageSpec:
    """Complete description of one candidate mechanism.  All lengths in metres."""

    # ── Bar lengths ──────────────────────────────────────────────────────────
    L_femur: float = 0.174       # A -> C
    L_stub: float = 0.035        # C -> E   (tibia stub, above the knee)
    L_tibia: float = 0.130       # C -> W   (knee to wheel centre)
    Lc: float = 0.150            # F -> E   (coupler)

    # ── Body-fixed passive pivot F, relative to A at origin ──────────────────
    F_X: float = -0.05887
    F_Z: float = +0.00529        # = archive -0.01821 - A_Z(-0.0235)

    # ── Tibia dogleg: W offset perpendicular to the E-C axis ─────────────────
    w_perp: float = 0.0

    # ── Collision radii, PER LINK PER END.  NOT optimized. ───────────────────
    # Two links meeting at one pin are separate parts and may have different
    # widths there (e.g. the femur is 15 mm at C while the tibia is 20 mm), so
    # each link owns its own end radii rather than sharing a per-node value.
    femur_r_A: float = 0.020     # motor side
    femur_r_C: float = 0.015     # tapers toward the knee
    tibia_r_E: float = 0.020     # 20 mm all around
    tibia_r_C: float = 0.020
    tibia_r_W: float = 0.020
    coupler_r_F: float = 0.020   # body side
    coupler_r_E: float = 0.015   # tibia side

    # ── Circular bodies ──────────────────────────────────────────────────────
    motor_r: float = 0.0265      # AK45-10 housing, dia 53 mm
    wheel_enabled: bool = True
    wheel_r: float = 0.056       # 112 mm diameter wheel

    # ── Ride-height constraint ───────────────────────────────────────────────
    # The wheel's UPPER horizontal tangent (W_z + wheel_r) may never rise above
    # the hip motor axis.  By the time the top of the tyre reaches the axis the
    # chassis is already on the ground and there is no jump left.  This binds
    # even when nothing is colliding, so it is separate from the collision set.
    enforce_wheel_below_hip: bool = True
    wheel_below_hip_margin: float = 0.0   # W_z + wheel_r must stay <= this [m]

    # The wheel pivot may never rise above the tibia's stub end.  W above E
    # means the tibia has leaned back past horizontal and its far end swings
    # toward the ground.  Unlike the ride-height limit this is measured against
    # a moving point, so it cannot be expressed as a fixed height.
    enforce_W_below_E: bool = True

    # Ground plane: horizontal, tangent to the bottom of the wheel, so it
    # tracks the wheel as the leg moves.  Nothing — no link, not the motor —
    # may reach it.  A fixed z would be wrong: the chassis rides up and down.
    enforce_ground: bool = True

    # ── Hip sweep ────────────────────────────────────────────────────────────
    q_seed: float = -0.67498     # neutral stance (30% ret->ext), rad
    q_min: float = -2.50         # hard search limits, rad
    q_max: float = +0.50

    # ── Collision enable matrix ──────────────────────────────────────────────
    collide: dict = field(default_factory=default_collision_matrix)

    # ── Traced points ────────────────────────────────────────────────────────
    trace_points: list = field(default_factory=lambda: [
        TracePoint(name="W (wheel centre)", body=TIBIA,
                   local_x=0.0, local_z=0.0, primary=True),
    ])

    # -----------------------------------------------------------------------
    def __post_init__(self):
        # W's local coords are derived from L_tibia/w_perp, so keep the default
        # trace point pinned to the real W rather than a stale copy.
        for tp in self.trace_points:
            if tp.primary and tp.body == TIBIA and tp.name.startswith("W"):
                tp.local_x, tp.local_z = -self.L_tibia, self.w_perp

    def sync_primary_to_W(self):
        """Re-pin the default 'W' trace point after lengths change."""
        for tp in self.trace_points:
            if tp.primary and tp.body == TIBIA and tp.name.startswith("W"):
                tp.local_x, tp.local_z = -self.L_tibia, self.w_perp

    def collides(self, a: str, b: str) -> bool:
        return bool(self.collide.get(pair_key(a, b), False))

    def node_radii(self) -> dict[str, dict[str, float]]:
        """Radii keyed by link, then by node."""
        return {
            FEMUR: {"A": self.femur_r_A, "C": self.femur_r_C},
            TIBIA: {"E": self.tibia_r_E, "C": self.tibia_r_C, "W": self.tibia_r_W},
            COUPLER: {"F": self.coupler_r_F, "E": self.coupler_r_E},
        }

    def link_spans(self) -> dict[str, float]:
        """End-to-end span of each link, radius-centre to radius-centre [m].

        The tibia is a three-node bar, so its span is the E→W distance — not
        just one of its lengths, and not their plain sum either once w_perp
        bends it off-axis.  That is why a per-variable bound cannot express a
        link-length cap on its own.
        """
        return {
            FEMUR: self.L_femur,
            TIBIA: math.hypot(self.L_stub + self.L_tibia, self.w_perp),
            COUPLER: self.Lc,
        }

    def max_link_span(self) -> float:
        return max(self.link_spans().values())

    def c_offset(self) -> float:
        """Perpendicular distance from the knee C to the straight line E–W [m].

        Two ways to describe the same dogleg, and they are NOT equal:

          w_perp    how far W sits off the C→E axis  (the free variable — it is
                    what the optimizer moves, because C and E define the tibia's
                    local frame and W is the thing being displaced)
          c_offset  how far C sits off the E–W line  (the derived, more
                    intuitive reading: how much the knee bulges out of the
                    straight run from stub end to wheel)

        Geometrically  c_offset = L_stub * |w_perp| / |EW|, so it is always the
        smaller of the two and both vanish together.
        """
        ew = math.hypot(self.L_tibia + self.L_stub, self.w_perp)
        if ew < 1e-12:
            return 0.0
        return abs(self.L_stub * self.w_perp) / ew

    def primary_trace(self) -> TracePoint:
        for tp in self.trace_points:
            if tp.primary:
                return tp
        return self.trace_points[0]

    def to_body_frame_fz(self, a_z: float = ARCHIVE_A_Z) -> float:
        """F_Z expressed in the archive's body-centre frame."""
        return self.F_Z + a_z

    def copy(self) -> "LinkageSpec":
        return LinkageSpec.from_dict(self.to_dict())

    # ── Serialisation ────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        d["collide"] = {f"{a}|{b}": v for (a, b), v in self.collide.items()}
        return d

    @staticmethod
    def from_dict(d: dict) -> "LinkageSpec":
        d = dict(d)
        col = d.pop("collide", None)
        tps = d.pop("trace_points", None)
        spec = LinkageSpec(**d)
        if col is not None:
            spec.collide = {pair_key(*k.split("|")): bool(v) for k, v in col.items()}
        if tps is not None:
            spec.trace_points = [TracePoint(**t) for t in tps]
        return spec

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "LinkageSpec":
        return LinkageSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def baseline1() -> LinkageSpec:
    """Current working geometry: baseline-1 rounded to the dimensions actually
    being built (femur 174, tibia 130, stub 35, coupler 150 mm), with the real
    per-link end radii."""
    return LinkageSpec()


def archive_baseline1() -> LinkageSpec:
    """Exact baseline-1 as optimized: run_id 51167 of the 70,818-run
    evolutionary search (`4bar_optimization_with_balancing/results_balanced.csv`,
    jump height 282.65 mm), translated into the A-at-origin frame.

    Kept for the IK cross-check and for before/after comparisons."""
    return LinkageSpec(L_femur=0.17378, L_stub=0.03513,
                       L_tibia=0.12939, Lc=0.15081)


# Archive reference values, for the cross-check test.
ARCHIVE_Q_RET = -0.35071
ARCHIVE_Q_EXT = -1.43161
ARCHIVE_STROKE_DEG = 61.93
