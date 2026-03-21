"""defaults.py — DEFAULT_PARAMS instance with Phase 6 (latency-retuned) values.

All default values are baked into the dataclass definitions in params.py,
so this file simply instantiates the top-level SimParams.  If a future
pipeline writes optimized gains to logs/baseline_gains.json, this file
can load and overlay them.
"""
from master_sim.params import SimParams

DEFAULT_PARAMS: SimParams = SimParams()
