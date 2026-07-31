"""optimize.py — search for link lengths and pivot offsets.

Objective: maximise the VERTICAL TRAVEL of the traced point over its
collision-free range, with path straightness handled one of two ways
(selectable at run time):

    constrained : maximise travel, reject anything whose max |deviation from
                  the best-fit vertical| exceeds a tolerance.
    weighted    : maximise  w_travel*travel - w_vert*deviation.

Free variables are bar lengths, the F pivot offsets, and the tibia dogleg.
The hip motor stays at the origin and every collision radius is a fixed input,
per the build constraints.

No Qt or matplotlib here — the GUI drives this through `optimize(...)` with a
callback, and `api.py opt` runs the identical code headless.
"""

from __future__ import annotations

import csv
import math
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution

from model import LinkageSpec
from geometry import ShapeSet, RES_FAST
from evaluate import evaluate, Metrics

# Everything the optimizer is allowed to move.  Motor position and all
# collision radii are deliberately absent: they are fixed by the build.
VARS = ("L_femur", "L_stub", "L_tibia", "Lc", "F_X", "F_Z", "w_perp")

# No link may span more than this, radius-centre to radius-centre.  For the
# tibia that is E->W = L_stub + L_tibia, so the cap is also enforced as a
# constraint in `score()` — bounds alone cannot express the sum.
MAX_LINK_M = 0.200

# Wide-open defaults: let the search explore, and let the collision/validity
# checks and the link-span cap do the rejecting rather than narrow bounds.
DEFAULT_BOUNDS = {
    "L_femur": (0.001, MAX_LINK_M),
    "L_stub":  (0.001, MAX_LINK_M),
    "L_tibia": (0.001, MAX_LINK_M),
    "Lc":      (0.001, MAX_LINK_M),
    "F_X":     (-0.200, 0.000),
    "F_Z":     (-0.200, 0.200),
    "w_perp":  (-0.050, 0.050),
}


@dataclass
class OptConfig:
    free: tuple = VARS
    bounds: dict = field(default_factory=lambda: dict(DEFAULT_BOUNDS))

    mode: str = "constrained"     # 'constrained' | 'weighted'
    tol_mm: float = 5.0           # constrained: allowed |deviation|
    penalty: float = 20.0         # constrained: fitness lost per mm over tol
    w_travel: float = 1.0         # weighted
    w_vert: float = 2.0           # weighted

    min_travel_mm: float = 5.0    # below this a design is not a mechanism
    min_stroke_deg: float = 10.0
    max_link_mm: float = MAX_LINK_M * 1000.0   # hard cap on any link span

    algo: str = "de"              # 'de' | 'es'
    budget: int = 6000            # max evaluations
    seed: int = 0

    popsize: int = 12             # DE population multiplier
    lam: int = 8                  # ES children per generation
    sigma_frac: float = 0.15      # ES initial step, fraction of each range

    csv_path: str | None = None

    def bounds_list(self):
        return [self.bounds.get(k, DEFAULT_BOUNDS[k]) for k in self.free]


@dataclass
class OptResult:
    best_spec: LinkageSpec
    best_metrics: Metrics
    best_fitness: float
    n_evals: int
    history: list           # (n_evals, best_fitness) samples
    elapsed_s: float
    csv_path: str | None = None
    stopped_early: bool = False
    stall_evals: int = 0    # evaluations since the last improvement

    @property
    def stall_frac(self) -> float:
        """Fraction of the budget spent without improving.  High = settled."""
        return self.stall_evals / max(1, self.n_evals)

    def convergence_note(self) -> str:
        f = self.stall_frac
        if f > 0.6:
            return f"settled ({self.stall_evals} evals with no gain, {f*100:.0f}% of run)"
        if f > 0.25:
            return f"likely settled ({self.stall_evals} evals stalled, {f*100:.0f}%)"
        return (f"STILL IMPROVING at cutoff ({self.stall_evals} evals stalled, "
                f"{f*100:.0f}%) — raise the budget")


@dataclass
class MultiResult:
    """Several independent runs.  The SPREAD across restarts is the actual
    evidence of convergence: a stochastic search has no optimality proof, so
    agreement between runs started from different seeds is what you get."""
    runs: list
    best: OptResult

    def fitnesses(self) -> list:
        return [r.best_fitness for r in self.runs]

    def spread_pct(self) -> float:
        fs = [f for f in self.fitnesses() if f > -1e6]
        if len(fs) < 2 or max(fs) <= 0:
            return float("nan")
        return (max(fs) - min(fs)) / abs(max(fs)) * 100.0

    def verdict(self) -> str:
        s = self.spread_pct()
        if math.isnan(s):
            return "too few valid runs to judge"
        if s < 2.0:
            return f"CONVERGED — restarts agree within {s:.1f}%"
        if s < 10.0:
            return f"probably near-optimal — restarts spread {s:.1f}%"
        return (f"NOT CONVERGED — restarts spread {s:.1f}%; "
                f"raise budget or restarts")


# ---------------------------------------------------------------------------
def apply_vector(base: LinkageSpec, free, x) -> LinkageSpec:
    s = base.copy()
    for k, v in zip(free, x):
        setattr(s, k, float(v))
    s.sync_primary_to_W()
    return s


def vector_of(spec: LinkageSpec, free) -> list:
    return [getattr(spec, k) for k in free]


def score(spec: LinkageSpec, cfg: OptConfig) -> tuple[float, Metrics]:
    """Fitness (higher is better) plus the metrics behind it."""
    # Manufacturing cap, checked before the (more expensive) sweep.  Graded
    # rather than flat so the search is pushed back inside the feasible region
    # instead of wandering on a plateau.
    over_mm = (spec.max_link_span() * 1000.0) - cfg.max_link_mm
    if over_mm > 0.0:
        return -1e7 - over_mm, Metrics(
            False, reason=f"link span {spec.max_link_span()*1000:.1f} mm "
                          f"> {cfg.max_link_mm:.0f} mm cap")

    m = evaluate(spec, ShapeSet(spec, RES_FAST), check_collisions=True,
                 auto_seed=True)
    if not m.valid:
        return -1e9, m

    # Auto-seeding may have relocated the seed pose to make this candidate
    # work.  Bake that back into the spec, otherwise the winner re-evaluates
    # as INVALID everywhere auto_seed is off — in the GUI panels and, worse,
    # when the saved preset is reloaded.
    spec.q_seed = m.q_seed_used
    if m.travel_mm < cfg.min_travel_mm or m.stroke_deg < cfg.min_stroke_deg:
        return -1e6 + m.travel_mm, m

    if cfg.mode == "weighted":
        f = cfg.w_travel * m.travel_mm - cfg.w_vert * m.max_dev_mm
    else:
        over = max(0.0, m.max_dev_mm - cfg.tol_mm)
        f = m.travel_mm - cfg.penalty * over
    return float(f), m


# ---------------------------------------------------------------------------
class _Run:
    """Shared bookkeeping: best-so-far, CSV log, callback, cancellation."""

    def __init__(self, base, cfg, callback=None, should_stop=None):
        self.base, self.cfg = base, cfg
        self.callback, self.should_stop = callback, should_stop
        self.n = 0
        self.last_gain = 0
        self.best_f = -math.inf
        self.best_spec = base.copy()
        self.best_m = None
        self.history = []
        self.stopped = False
        self._w = None
        self._fh = None
        if cfg.csv_path:
            self._fh = open(cfg.csv_path, "w", newline="", encoding="utf-8")
            self._w = csv.writer(self._fh)
            self._w.writerow(["eval", "fitness", "travel_mm", "max_dev_mm",
                              "stroke_deg", "valid", "reason", *cfg.free])

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def evaluate(self, x) -> float:
        if self.should_stop and self.should_stop():
            self.stopped = True
            # Return a neutral-bad value; the driver checks .stopped to exit.
            return 1e9
        spec = apply_vector(self.base, self.cfg.free, x)
        f, m = score(spec, self.cfg)
        self.n += 1

        if self._w:
            self._w.writerow([self.n, round(f, 4), round(m.travel_mm, 3),
                              round(m.max_dev_mm, 3), round(m.stroke_deg, 3),
                              int(m.valid), m.reason,
                              *[round(v, 6) for v in x]])

        if f > self.best_f:
            self.best_f, self.best_spec, self.best_m = f, spec, m
            self.last_gain = self.n
            self.history.append((self.n, f))
            if self.callback:
                self.callback(self.n, f, spec, m)
        elif self.callback and self.n % 200 == 0:
            self.callback(self.n, self.best_f, self.best_spec, self.best_m)
        return -f            # scipy minimises

    def budget_left(self) -> bool:
        return self.n < self.cfg.budget and not self.stopped


# ---------------------------------------------------------------------------
def _run_de(run: _Run):
    cfg = run.cfg
    bounds = cfg.bounds_list()
    maxiter = max(1, cfg.budget // (cfg.popsize * len(cfg.free)))

    def cb(xk, convergence=None):
        return (not run.budget_left())

    differential_evolution(
        run.evaluate, bounds,
        strategy="best1bin", maxiter=maxiter, popsize=cfg.popsize,
        tol=1e-8, mutation=(0.4, 1.0), recombination=0.8,
        seed=cfg.seed, polish=False, init="sobol",
        callback=cb, workers=1, updating="immediate",
    )


def _run_es(run: _Run):
    """(1+lambda) ES with 1/5-success sigma adaptation — the scheme that
    produced baseline-1 in `4bar_optimization_with_balancing/optimize_balanced.py`
    (LAMBDA=8, SIGMA_FRAC 0.15), kept for continuity."""
    cfg = run.cfg
    rng = random.Random(cfg.seed)
    bounds = cfg.bounds_list()
    spans = [hi - lo for lo, hi in bounds]

    parent = vector_of(run.base, cfg.free)
    parent_f = -run.evaluate(parent)
    sigma = cfg.sigma_frac
    wins, gens = 0, 0

    while run.budget_left():
        gens += 1
        improved = False
        for _ in range(cfg.lam):
            if not run.budget_left():
                break
            child = [min(hi, max(lo, p + rng.gauss(0.0, sigma * sp)))
                     for p, sp, (lo, hi) in zip(parent, spans, bounds)]
            f = -run.evaluate(child)
            if f > parent_f:
                parent, parent_f, improved = child, f, True
        wins += int(improved)

        if gens % 10 == 0:                       # 1/5 success rule
            rate = wins / 10.0
            sigma *= 1.22 if rate > 0.2 else 1.0 / 1.22
            sigma = min(0.50, max(1e-4, sigma))
            wins = 0


def optimize(spec: LinkageSpec, cfg: OptConfig | None = None,
             callback=None, should_stop=None) -> OptResult:
    """Run a search.  `callback(n_evals, best_fitness, best_spec, best_metrics)`
    is called on every improvement (and periodically otherwise);
    `should_stop()` is polled so a GUI can cancel."""
    cfg = cfg or OptConfig()
    run = _Run(spec, cfg, callback, should_stop)
    t0 = time.perf_counter()

    # Seed the incumbent with the starting geometry so the result can never be
    # worse than what you already had.
    f0, m0 = score(spec, cfg)
    run.best_f, run.best_spec, run.best_m = f0, spec.copy(), m0
    run.history.append((0, f0))

    try:
        (_run_es if cfg.algo == "es" else _run_de)(run)
    finally:
        run.close()

    return OptResult(
        best_spec=run.best_spec, best_metrics=run.best_m,
        best_fitness=run.best_f, n_evals=run.n, history=run.history,
        elapsed_s=time.perf_counter() - t0, csv_path=cfg.csv_path,
        stopped_early=run.stopped, stall_evals=run.n - run.last_gain,
    )


def optimize_multi(spec: LinkageSpec, cfg: OptConfig | None = None,
                   restarts: int = 4, callback=None,
                   should_stop=None) -> MultiResult:
    """Independent runs from different seeds.

    Neither DE nor the ES can prove optimality — they are stochastic global
    searches.  Running several and comparing the results is the practical
    convergence test: tight agreement means the budget is sufficient, wide
    scatter means it is not.
    """
    import copy as _copy
    cfg = cfg or OptConfig()
    runs = []
    for i in range(max(1, restarts)):
        c = _copy.copy(cfg)
        c.seed = cfg.seed + i
        if cfg.csv_path:
            root, ext = os.path.splitext(cfg.csv_path)
            c.csv_path = f"{root}_r{i}{ext}"
        runs.append(optimize(spec, c, callback=callback, should_stop=should_stop))
        if should_stop and should_stop():
            break
    best = max(runs, key=lambda r: r.best_fitness)
    return MultiResult(runs=runs, best=best)
