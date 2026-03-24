"""
SearchSpace — defines parameter bounds, log/linear scaling, and ES sampling.

Each parameter is sampled in log10 space (all robot gains span orders of
magnitude, so log-space exploration is natural). Parameters whose optimal
value can be exactly zero are marked via `zero_ok` and snapped to 0.0 when
their value drops below ZERO_FLOOR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple

import numpy as np


# Values below this threshold are snapped to 0.0 for zero_ok parameters
ZERO_FLOOR: float = 1e-6


@dataclass(frozen=True)
class ParamSpec:
    """Specification for a single search-space parameter."""
    lo: float                   # lower bound (linear space, must be > 0)
    hi: float                   # upper bound (linear space)
    zero_ok: bool = False       # if True, values < ZERO_FLOOR snap to 0.0


@dataclass(frozen=True)
class SearchSpace:
    """Immutable definition of an ES search space.

    All sampling and mutation happen in log10 space.
    """
    params: Dict[str, ParamSpec]

    # -- convenience accessors ------------------------------------------------

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self.params.keys())

    @property
    def dim(self) -> int:
        return len(self.params)

    # -- sampling -------------------------------------------------------------

    def random_init(self, rng: np.random.Generator | None = None) -> Dict[str, float]:
        """Uniform random sample in log10 space, mapped back to linear."""
        rng = rng or np.random.default_rng()
        candidate: Dict[str, float] = {}
        for name, spec in self.params.items():
            log_lo, log_hi = math.log10(spec.lo), math.log10(spec.hi)
            log_val = rng.uniform(log_lo, log_hi)
            candidate[name] = 10.0 ** log_val
        return candidate

    def sample_offspring(
        self,
        parent: Dict[str, float],
        sigmas: Dict[str, float],
        rng: np.random.Generator | None = None,
    ) -> Dict[str, float]:
        """Gaussian mutation of *parent* in log10 space.

        Parameters
        ----------
        parent : dict
            Current best parameter vector (linear-space values).
        sigmas : dict
            Per-parameter step sizes in log10 space.
        rng : numpy Generator, optional

        Returns
        -------
        dict  — clamped child parameter vector (linear space).
        """
        rng = rng or np.random.default_rng()
        child: Dict[str, float] = {}
        for name, spec in self.params.items():
            # parent value → log10 (guard against zero)
            p_val = parent[name]
            log_val = math.log10(max(1e-9, p_val)) if p_val > 0 else math.log10(spec.lo)
            # Gaussian perturbation
            log_val += rng.normal(0.0, sigmas[name])
            # clamp to log-space bounds
            log_lo = math.log10(spec.lo)
            log_hi = math.log10(spec.hi)
            log_val = max(log_lo, min(log_hi, log_val))
            val = 10.0 ** log_val
            # snap near-zero for parameters that allow it
            if spec.zero_ok and val < ZERO_FLOOR:
                val = 0.0
            child[name] = val
        return child

    def clamp(self, candidate: Dict[str, float]) -> Dict[str, float]:
        """Clamp *candidate* to bounds (linear space), respecting zero_ok."""
        out: Dict[str, float] = {}
        for name, spec in self.params.items():
            val = candidate[name]
            if spec.zero_ok and val < ZERO_FLOOR:
                out[name] = 0.0
            else:
                out[name] = max(spec.lo, min(spec.hi, val))
        return out

    def init_sigmas(self, sigma_init: float) -> Dict[str, float]:
        """Create per-parameter sigma dict initialised to *sigma_init*."""
        return {name: sigma_init for name in self.params}

    def in_bounds(self, candidate: Dict[str, float]) -> bool:
        """Check whether every value in *candidate* is within bounds."""
        for name, spec in self.params.items():
            val = candidate[name]
            if spec.zero_ok and val == 0.0:
                continue
            if val < spec.lo or val > spec.hi:
                return False
        return True


# ── Pre-built search spaces for each optimizer ──────────────────────────────

LQR_SPACE = SearchSpace(params={
    "Q_PITCH":      ParamSpec(0.01,  50.0),
    "Q_PITCH_RATE": ParamSpec(1e-3,  10.0,   zero_ok=True),
    "Q_VEL":        ParamSpec(1e-7,  1.0),
    "R":            ParamSpec(0.01,  100.0),
})

VELOCITY_PI_SPACE = SearchSpace(params={
    "KP_V":  ParamSpec(1e-3, 10.0),
    "KI_V":  ParamSpec(1e-3, 10.0),
    "KFF_V": ParamSpec(0.1,  0.15),
})

YAW_PI_SPACE = SearchSpace(params={
    "KP_YAW": ParamSpec(1e-3, 100.0),
    "KI_YAW": ParamSpec(1e-4, 100.0, zero_ok=True),
})

SUSPENSION_SPACE = SearchSpace(params={
    "LEG_K_S":    ParamSpec(0.1,  100.0),
    "LEG_B_S":    ParamSpec(0.01, 100.0,  zero_ok=True),
    "LEG_K_ROLL": ParamSpec(1e-3, 500.0,  zero_ok=True),
    "LEG_D_ROLL": ParamSpec(1e-4, 100.0,  zero_ok=True),
})

INTEGRATED_SPACE = SearchSpace(params={
    **LQR_SPACE.params,
    **VELOCITY_PI_SPACE.params,
    **YAW_PI_SPACE.params,
    **SUSPENSION_SPACE.params,
})

# Maps scenario group → search space.  Used by optimize_integrated to restrict
# the search to only the gains that the chosen scenario actually exercises.
SPACE_BY_GROUP: Dict[str, SearchSpace] = {
    "lqr":         LQR_SPACE,
    "velocity_pi": VELOCITY_PI_SPACE,
    "yaw_pi":      YAW_PI_SPACE,
    "suspension":  SUSPENSION_SPACE,
    "integrated":  INTEGRATED_SPACE,
}
