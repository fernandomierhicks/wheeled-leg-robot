"""defaults.py — DEFAULT_PARAMS instance.

params.py is the single source of truth for all simulation parameters
including gains. The optimizer writes best gains directly into params.py.
"""
from v4_twin_279mm_baseline.params import SimParams

DEFAULT_PARAMS: SimParams = SimParams()
