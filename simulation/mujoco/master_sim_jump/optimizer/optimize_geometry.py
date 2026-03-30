"""optimize_geometry.py — 4-bar link length optimizer for jump height.

Searches over 5 parameters: L_femur, L_tibia, Lc, L_stub, crouch_time.
max_torque is fixed at the motor limit (7.0 N·m) — more torque always helps,
so there is no benefit in optimizing it.

Fitness = peak body CoM height only (no settle-time term). The fell penalty
eliminates geometries that destabilize the existing LQR controller.

On completion, best geometry is saved to optimizer/best_jump_params.py
(NOT auto-written into params.py — copy manually when satisfied).

Usage:
    python -m master_sim_jump.optimizer.optimize_geometry --hours 4
    python -m master_sim_jump.optimizer.optimize_geometry --iters 100 --workers 4
    python -m master_sim_jump.optimizer.optimize_geometry --iters 20 --no-save   # dry run
"""
import argparse
import multiprocessing

from master_sim_jump.optimizer.search_space import SearchSpace, ParamSpec


# ── 5-dimensional search space ───────────────────────────────────────────────
# max_torque is fixed at 7.0 N·m (motor limit) — always maximum is optimal.

GEOM_SPACE = SearchSpace(params={
    "L_FEMUR":     ParamSpec(0.140, 0.280),   # [m] femur A→C      140–280 mm  (prev upper: 220)
    "L_TIBIA":     ParamSpec(0.100, 0.220),   # [m] tibia C→W      100–220 mm  (prev upper: 170)
    "LC":          ParamSpec(0.120, 0.240),   # [m] coupler F→E    120–240 mm  (prev upper: 190)
    "L_STUB":      ParamSpec(0.025, 0.050),   # [m] tibia stub C→E  25–50 mm
    "CROUCH_TIME": ParamSpec(0.05,  1.00),    # [s] crouch duration            (prev: 0.10–0.35)
})

MAX_TORQUE_FIXED = 7.0   # [N·m] fixed at motor limit


# ── Picklable eval callable (safe for Windows multiprocessing spawn) ─────────

class EvalWithGeometry:
    """Evaluate one candidate geometry by running s10_jump."""

    def __call__(self, candidate: dict) -> dict:
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

        # Fix max_torque at motor limit; use candidate crouch_time
        new_jump  = replace(p.gains.jump,
                            max_torque=MAX_TORQUE_FIXED,
                            crouch_time=candidate["CROUCH_TIME"])
        new_gains = replace(p.gains, jump=new_jump)
        params    = replace(p, robot=new_geom, gains=new_gains)

        metrics = evaluate(params, "s10_jump")

        # Fitness (minimise):
        #   200 * fell         — heavy penalty for crashing
        #   (1 - z / 0.30)     — reward jump height (goes negative above 300 mm)
        fell = metrics.get("fell", True)
        z    = metrics.get("peak_body_z_m", 0.0)
        metrics["fitness"] = 200.0 * fell + (1.0 - z / 0.30)
        return metrics


# ── Save best result to a separate file (never touches params.py) ────────────

def save_best_geometry(best_params: dict, stroke: tuple, best_fitness: float) -> None:
    """Write best geometry to optimizer/best_jump_params.py.

    Does NOT modify params.py. Copy values manually when satisfied.
    """
    import pathlib, datetime
    out = pathlib.Path(__file__).parent / "best_jump_params.py"
    q_ret, q_ext = stroke
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f'"""best_jump_params.py — best geometry found by optimize_geometry.py',
        f"",
        f"Generated: {ts}",
        f"Fitness:   {best_fitness:.6f}  (lower is better; negative = peak_z > 300 mm)",
        f"Peak body Z: {(1.0 - best_fitness) * 0.30 * 1000:.1f} mm  (approx, assumes fell=False)",
        f'"""',
        f"",
        f"# Copy these values into params.py when satisfied.",
        f"# RobotGeometry fields:",
        f"L_femur    = {best_params['L_FEMUR']:.6f}   # [m]",
        f"L_tibia    = {best_params['L_TIBIA']:.6f}   # [m]",
        f"Lc         = {best_params['LC']:.6f}   # [m]",
        f"L_stub     = {best_params['L_STUB']:.6f}   # [m]",
        f"Q_RET      = {q_ret:.6f}   # [rad]  auto-computed by auto_stroke_angles()",
        f"Q_EXT      = {q_ext:.6f}   # [rad]  auto-computed by auto_stroke_angles()",
        f"",
        f"# JumpGains fields:",
        f"crouch_time = {best_params['CROUCH_TIME']:.6f}   # [s]",
        f"max_torque  = {MAX_TORQUE_FIXED:.1f}              # [N·m]  fixed at motor limit",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[GEOM BEST] fitness={best_fitness:.6f}")
    print(f"[GEOM BEST] Q_RET={q_ret:.6f}  Q_EXT={q_ext:.6f}")
    print(f"[GEOM BEST] Saved -> {out}")


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
        "CROUCH_TIME": j.crouch_time,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    from master_sim_jump.optimizer.es_engine import ESOptimizer, ESConfig
    from master_sim_jump.optimizer.run_log import get_scenario_csv_path

    ap = argparse.ArgumentParser(description="4-bar geometry optimizer")
    ap.add_argument("--hours",    type=float, default=None)
    ap.add_argument("--iters",    type=int,   default=None)
    ap.add_argument("--workers",  type=int,   default=None)
    ap.add_argument("--patience", type=int,   default=300)
    ap.add_argument("--tol",      type=float, default=1e-4)
    ap.add_argument("--no-save",  action="store_true",
                    help="Skip writing best_jump_params.py")
    args = ap.parse_args()

    if args.hours is None and args.iters is None:
        args.hours = 4.0

    seed     = _default_seed()
    eval_fn  = EvalWithGeometry()
    csv_path = get_scenario_csv_path("s10_jump_geometry")

    print(f"4-bar geometry optimizer  |  {GEOM_SPACE.dim}D  "
          f"({', '.join(GEOM_SPACE.names)})")
    print(f"max_torque fixed at {MAX_TORQUE_FIXED} N·m")
    print(f"Seed: { {k: round(v*1000,1) if 'L_' in k or k=='LC' else round(v,3) for k, v in seed.items()} }")

    from master_sim_jump.optimizer.progress_ui import ProgressUI
    ui = ProgressUI(
        f"4-Bar Geometry Optimizer — s10_jump ({GEOM_SPACE.dim}D)",
        all_defaults=seed,
        active_names=set(GEOM_SPACE.names),
    )

    cfg = ESConfig(patience=args.patience, tol=args.tol, n_workers=args.workers)
    opt = ESOptimizer(
        search_space=GEOM_SPACE,
        eval_fn=eval_fn,
        csv_path=csv_path,
        config=cfg,
        progress_fn=ui.update_from_progress,
        pause_fn=ui.wait_if_paused,
    )
    try:
        result = opt.run(hours=args.hours, max_iters=args.iters, seed_params=seed)
    finally:
        ui.finish()

    if not args.no_save and result and result.get("best_params"):
        from master_sim_jump.physics import auto_stroke_angles
        from master_sim_jump.defaults import DEFAULT_PARAMS
        from dataclasses import replace

        bp   = result["best_params"]
        geom = replace(DEFAULT_PARAMS.robot,
                       L_femur=bp["L_FEMUR"], L_tibia=bp["L_TIBIA"],
                       Lc=bp["LC"],           L_stub=bp["L_STUB"])
        stroke = auto_stroke_angles(geom)
        if stroke:
            save_best_geometry(bp, stroke, result["best_fitness"])
        else:
            print("[GEOM BEST] WARNING: best geometry stroke invalid, skipping save")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
