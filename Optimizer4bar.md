# 4-Bar Geometry Optimizer — Step-by-Step Implementation

Optimizes link lengths to maximize jump height using the existing ES optimizer infrastructure.

---

## Overview

3 files to touch, in order:

| # | File | Change |
|---|------|--------|
| 1 | `physics.py` | Add `auto_stroke_angles()` |
| 2 | `sim_loop.py` | Add `peak_body_z_m` metric |
| 3 | `optimizer/optimize_geometry.py` | New file — 6D ES optimizer |

---

## Step 1 — `auto_stroke_angles()` in `physics.py`

**File:** `simulation/mujoco/master_sim_jump/physics.py`

Add this function directly after `solve_ik()`:

```python
def auto_stroke_angles(robot: "RobotGeometry") -> tuple[float, float] | None:
    """Find valid hip stroke range for an arbitrary 4-bar geometry.

    Sweeps q_hip from -0.05 to -2.5 rad and calls solve_ik() at each step.
    Returns (q_ret, q_ext) — crouch and full-extension angles — or None if
    the geometry is infeasible (too singular, too short a stroke, etc.).
    """
    import dataclasses
    p = dataclasses.asdict(robot)

    KNEE_LIMIT = math.pi / 3          # ±60° physical joint limit
    N          = 500
    q_sweep    = [(-0.05 - 2.45 * i / (N - 1)) for i in range(N)]

    valid = []   # list of (q_hip, W_z)
    for q in q_sweep:
        r = solve_ik(q, p)
        if r is None:
            continue
        if abs(r['q_knee']) > KNEE_LIMIT:
            continue
        valid.append((q, r['W_z']))

    if len(valid) < 50:
        return None   # not enough valid range

    # Trim 5% at each end to stay away from singularities
    margin = max(1, len(valid) // 20)
    valid  = valid[margin:-margin]
    if len(valid) < 20:
        return None

    # Q_RET = where wheel is highest (most retracted)
    # Q_EXT = where wheel is lowest  (most extended)
    q_ret = max(valid, key=lambda x: x[1])[0]
    q_ext = min(valid, key=lambda x: x[1])[0]

    if abs(q_ext - q_ret) < 0.30:    # need at least ~17° usable stroke
        return None

    return (q_ret, q_ext)
```

### Verify Step 1

Run this one-liner from the repo root:

```bash
python -c "
from simulation.mujoco.master_sim_jump.params import RobotGeometry
from simulation.mujoco.master_sim_jump.physics import auto_stroke_angles
r = RobotGeometry()
print(auto_stroke_angles(r))
# Expected: close to (-0.7330, -1.4317)
"
```

---

## Step 2 — `peak_body_z_m` in `sim_loop.py`

**File:** `simulation/mujoco/master_sim_jump/sim_loop.py`

### 2a — Add accumulator

Find the block where metric accumulators are initialized (search for `wheel_liftoff_s = 0.0`).
Add one line below it:

```python
peak_body_z_m     = 0.0
```

### 2b — Track in main loop

Find the inner per-tick block (search for `mujoco.mj_step(model, data)`).
Add after the physics step:

```python
_bz = data.body('body').xpos[2]
if _bz > peak_body_z_m:
    peak_body_z_m = _bz
```

### 2c — Add to return dict

Find the `return dict(` at the end of `run()`. Add one entry:

```python
peak_body_z_m           = round(peak_body_z_m,      4),
```

### Verify Step 2

```bash
python -c "
from simulation.mujoco.master_sim_jump.defaults import DEFAULT_PARAMS
from simulation.mujoco.master_sim_jump.scenarios import evaluate
m = evaluate(DEFAULT_PARAMS, 's10_jump')
print('peak_body_z_m =', m.get('peak_body_z_m'))
# Expected: somewhere in 0.10–0.40 m
"
```

---

## Step 3 — `optimizer/optimize_geometry.py` (new file)

**File:** `simulation/mujoco/master_sim_jump/optimizer/optimize_geometry.py`

Create this file from scratch:

```python
"""optimize_geometry.py — 4-bar link length optimizer for jump height.

Searches over 6 parameters: L_femur, L_tibia, Lc, L_stub, max_torque,
crouch_time. Uses the existing ES engine and multiprocessing infrastructure.

Usage:
    python -m master_sim_jump.optimizer.optimize_geometry --hours 4
    python -m master_sim_jump.optimizer.optimize_geometry --iters 100 --workers 4
    python -m master_sim_jump.optimizer.optimize_geometry --iters 20 --no-baseline  # dry run
"""
import argparse
import multiprocessing

from master_sim_jump.optimizer.search_space import SearchSpace, ParamSpec


# ── 6-dimensional search space ───────────────────────────────────────────────

GEOM_SPACE = SearchSpace(params={
    "L_FEMUR":     ParamSpec(0.140, 0.220),   # [m] femur A→C      140–220 mm
    "L_TIBIA":     ParamSpec(0.100, 0.170),   # [m] tibia C→W      100–170 mm
    "LC":          ParamSpec(0.120, 0.190),   # [m] coupler F→E    120–190 mm
    "L_STUB":      ParamSpec(0.025, 0.050),   # [m] tibia stub C→E  25–50 mm
    "MAX_TORQUE":  ParamSpec(4.0,   7.0),     # [N·m] peak hip torque
    "CROUCH_TIME": ParamSpec(0.10,  0.35),    # [s] crouch duration
})


# ── Picklable eval callable (safe for Windows multiprocessing spawn) ─────────

class EvalWithGeometry:
    """Evaluate one candidate geometry by running s10_jump."""

    def __call__(self, candidate: dict) -> dict:
        import dataclasses
        from dataclasses import replace
        from master_sim_jump.defaults import DEFAULT_PARAMS
        from master_sim_jump.scenarios import evaluate
        from master_sim_jump.physics import auto_stroke_angles

        p = DEFAULT_PARAMS

        # Build candidate RobotGeometry
        new_geom = replace(p.robot,
                           L_femur=candidate["L_FEMUR"],
                           L_tibia=candidate["L_TIBIA"],
                           Lc=candidate["LC"],
                           L_stub=candidate["L_STUB"])

        # Auto-compute stroke angles for this geometry
        stroke = auto_stroke_angles(new_geom)
        if stroke is None:
            return {
                "fitness": 500.0, "fell": True,
                "peak_body_z_m": 0.0, "status": "FAIL",
                "fail_reason": "infeasible_geometry",
            }
        q_ret, q_ext = stroke
        new_geom = replace(new_geom, Q_RET=q_ret, Q_EXT=q_ext)

        # Update jump gains
        new_jump   = replace(p.gains.jump,
                             max_torque=candidate["MAX_TORQUE"],
                             crouch_time=candidate["CROUCH_TIME"])
        new_gains  = replace(p.gains, jump=new_jump)
        params     = replace(p, robot=new_geom, gains=new_gains)

        metrics = evaluate(params, "s10_jump")

        # Geometry fitness (minimise):
        #   200 * fell             — heavy penalty for crashing
        #   (1 - z / 0.30)         — reward jump height (negative when z > 0.30 m)
        #   0.5 * (settle / 6.0)   — mild penalty for slow recovery
        fell = metrics.get("fell", True)
        z    = metrics.get("peak_body_z_m", 0.0)
        st   = metrics.get("settle_time_s", 6.0)
        metrics["fitness"] = 200.0 * fell + (1.0 - z / 0.30) + 0.5 * (st / 6.0)
        return metrics


# ── Baseline saver — patches params.py directly ──────────────────────────────

_GEOM_MAP = {
    "L_FEMUR":     ("RobotGeometry", "L_femur"),
    "L_TIBIA":     ("RobotGeometry", "L_tibia"),
    "LC":          ("RobotGeometry", "Lc"),
    "L_STUB":      ("RobotGeometry", "L_stub"),
    "MAX_TORQUE":  ("JumpGains",     "max_torque"),
    "CROUCH_TIME": ("JumpGains",     "crouch_time"),
}


def save_baseline_geometry(best_params: dict, stroke: tuple,
                            best_fitness: float) -> None:
    """Write best geometry + stroke angles into params.py."""
    from master_sim_jump.optimizer.baseline import _backup_params, _patch_field, PARAMS_PY

    backup = _backup_params()
    text   = PARAMS_PY.read_text(encoding="utf-8")

    for key, value in best_params.items():
        if key not in _GEOM_MAP:
            continue
        class_name, field_name = _GEOM_MAP[key]
        text = _patch_field(text, class_name, field_name, value)

    # Patch auto-computed stroke angles
    q_ret, q_ext = stroke
    text = _patch_field(text, "RobotGeometry", "Q_RET", q_ret)
    text = _patch_field(text, "RobotGeometry", "Q_EXT", q_ext)

    PARAMS_PY.write_text(text, encoding="utf-8")
    print(f"[GEOM BASELINE] fitness={best_fitness:.4f}")
    print(f"[GEOM BASELINE] Q_RET={q_ret:.6f}  Q_EXT={q_ext:.6f}")
    print(f"[GEOM BASELINE] backup → {backup.name}")


# ── Seed from current params.py defaults ─────────────────────────────────────

def _default_seed() -> dict:
    from master_sim_jump.defaults import DEFAULT_PARAMS
    r = DEFAULT_PARAMS.robot
    j = DEFAULT_PARAMS.gains.jump
    return {
        "L_FEMUR":     r.L_femur,
        "L_TIBIA":     r.L_tibia,
        "LC":          r.Lc,
        "L_STUB":      r.L_stub,
        "MAX_TORQUE":  j.max_torque,
        "CROUCH_TIME": j.crouch_time,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    from master_sim_jump.optimizer.es_engine import ESOptimizer, ESConfig
    from master_sim_jump.optimizer.run_log import get_scenario_csv_path

    ap = argparse.ArgumentParser(description="4-bar geometry optimizer")
    ap.add_argument("--hours",       type=float, default=None)
    ap.add_argument("--iters",       type=int,   default=None)
    ap.add_argument("--workers",     type=int,   default=None)
    ap.add_argument("--patience",    type=int,   default=300)
    ap.add_argument("--tol",         type=float, default=1e-4)
    ap.add_argument("--no-baseline", action="store_true",
                    help="Skip writing best geometry to params.py")
    args = ap.parse_args()

    if args.hours is None and args.iters is None:
        args.hours = 4.0

    seed     = _default_seed()
    eval_fn  = EvalWithGeometry()
    csv_path = get_scenario_csv_path("s10_jump_geometry")

    print(f"4-bar geometry optimizer  |  {GEOM_SPACE.dim}D  "
          f"({', '.join(GEOM_SPACE.names)})")
    print(f"Seed: { {k: round(v*1000,1) if 'L_' in k or k=='LC' else round(v,3) for k, v in seed.items()} }")

    def _progress(info: dict):
        it = info.get("iteration", 0)
        best = info.get("best_fitness", float("nan"))
        z    = info.get("best_metrics", {}).get("peak_body_z_m", 0.0)
        if it % 10 == 0:
            print(f"[iter {it:4d}] fitness={best:.4f}  peak_z={z*1000:.0f}mm")

    cfg = ESConfig(patience=args.patience, tol=args.tol, n_workers=args.workers)
    opt = ESOptimizer(
        search_space=GEOM_SPACE,
        eval_fn=eval_fn,
        csv_path=csv_path,
        config=cfg,
        progress_fn=_progress,
    )
    result = opt.run(hours=args.hours, max_iters=args.iters, seed_params=seed)

    if not args.no_baseline and result and result.get("best_params"):
        from master_sim_jump.physics import auto_stroke_angles
        from master_sim_jump.defaults import DEFAULT_PARAMS
        from dataclasses import replace

        bp = result["best_params"]
        geom = replace(DEFAULT_PARAMS.robot,
                       L_femur=bp["L_FEMUR"], L_tibia=bp["L_TIBIA"],
                       Lc=bp["LC"],           L_stub=bp["L_STUB"])
        stroke = auto_stroke_angles(geom)
        if stroke:
            save_baseline_geometry(bp, stroke, result["best_fitness"])
        else:
            print("[GEOM BASELINE] WARNING: best geometry stroke invalid, skipping save")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
```

### Verify Step 3

Dry-run with 20 iterations (no changes to params.py):

```bash
cd "c:\Dropbox\Personal Projects\Robotics\wheeled-leg-robot\simulation\mujoco"
python -m master_sim_jump.optimizer.optimize_geometry --iters 20 --workers 4 --no-baseline
```

Expected output: candidates printed every 10 iters, `peak_z` values varying, no crashes.

---

## Running for Real

```bash
# 4-hour run, all CPU cores, auto-baseline on completion
python -m master_sim_jump.optimizer.optimize_geometry --hours 4

# Overnight
python -m master_sim_jump.optimizer.optimize_geometry --hours 8 --patience 500
```

After completion, replay best result in visualizer:

```bash
python -m master_sim_jump.viz
```

---

## Possible Issues

| Symptom | Fix |
|---------|-----|
| `AttributeError: 'SimParams' has no attribute 'robot'` | Check `defaults.py` — attribute may be named `geometry` or similar; update `EvalWithGeometry` accordingly |
| `auto_stroke_angles` returns values far from baseline | Tune sweep range or margin % in Step 1 |
| All candidates return `infeasible_geometry` | Widen `GEOM_SPACE` bounds or reduce `KNEE_LIMIT` |
| `_patch_field` fails on `RobotGeometry` | That class may use a different field name — check `params.py` |
