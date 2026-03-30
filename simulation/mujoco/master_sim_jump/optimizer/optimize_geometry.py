"""optimize_geometry.py — 4-bar link length optimizer for jump height.

Searches over 5 parameters: L_femur, L_tibia, Lc, L_stub, crouch_time.
max_torque is fixed at the motor limit (7.0 N·m) — more torque always helps,
so there is no benefit in optimizing it.

Fitness (minimise) = weighted sum of:
  - Jump height reward:  (1 - peak_wheel_z / 0.30)  — goes negative above 300 mm
  - Fell penalty:        200 * fell
  - Pitch stability:     W_PITCH_DELTA * |eq_pitch(Q_RET) - eq_pitch(Q_EXT)| / REF
                         Penalises geometries where leg cycling tilts the body.
  - Leg length:          W_LEG_LENGTH * (L_femur + L_tibia) / REF
                         Prefers compact robots that jump high.

On completion, best geometry is saved to optimizer/best_jump_params.py
(NOT auto-written into params.py — copy manually when satisfied).

Usage:
    python -m master_sim_jump.optimizer.optimize_geometry --hours 4
    python -m master_sim_jump.optimizer.optimize_geometry --iters 100 --workers 4
    python -m master_sim_jump.optimizer.optimize_geometry --iters 20 --no-save   # dry run
"""
import argparse
import math
import multiprocessing

from master_sim_jump.optimizer.search_space import SearchSpace, ParamSpec


# ── Fitness weights for new penalty terms ─────────────────────────────────────
W_PITCH_DELTA = 0.3     # pitch stability: how much eq. pitch changes across stroke
REF_PITCH_DELTA = math.radians(15.0)  # [rad] reference: 15° variation = 1× penalty

W_LEG_LENGTH = 0.2      # compactness: penalise long legs
REF_LEG_LENGTH = 0.40   # [m] reference: 400 mm total = 1× penalty


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
        from master_sim_jump.physics import (auto_stroke_angles,
                                              get_equilibrium_pitch,
                                              check_mechanical_constraints)

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
                "peak_body_z_m": 0.0, "peak_wheel_z_m": 0.0,
                "delta_pitch_eq_deg": 0.0, "leg_length_m": 0.0,
                "status": "FAIL", "fail_reason": "infeasible_geometry",
            }
        q_ret, q_ext = stroke
        new_geom = replace(new_geom, Q_RET=q_ret, Q_EXT=q_ext)

        # ── Mechanical constraint check (analytical, no sim) ─────────────
        mech_fail = check_mechanical_constraints(new_geom, q_ret, q_ext)
        if mech_fail is not None:
            return {
                "fitness": 500.0, "fell": True,
                "peak_body_z_m": 0.0, "peak_wheel_z_m": 0.0,
                "delta_pitch_eq_deg": 0.0, "leg_length_m": 0.0,
                "status": "FAIL", "fail_reason": mech_fail,
            }

        # ── Geometry-only penalties (zero sim cost) ───────────────────────
        # Pitch stability: how much equilibrium pitch changes across stroke
        pitch_ret = get_equilibrium_pitch(new_geom, q_ret)
        pitch_ext = get_equilibrium_pitch(new_geom, q_ext)
        delta_pitch_eq = abs(pitch_ret - pitch_ext)

        # Leg length: prefer compact robots
        leg_length = candidate["L_FEMUR"] + candidate["L_TIBIA"]

        # Fix max_torque at motor limit; use candidate crouch_time
        new_jump  = replace(p.gains.jump,
                            max_torque=MAX_TORQUE_FIXED,
                            crouch_time=candidate["CROUCH_TIME"])
        new_gains = replace(p.gains, jump=new_jump)
        params    = replace(p, robot=new_geom, gains=new_gains)

        metrics = evaluate(params, "s10_jump")

        # Fitness (minimise):
        #   200 * fell                    — heavy penalty for crashing
        #   (1 - z / 0.30)               — reward wheel height (negative above 300 mm)
        #   W_PITCH_DELTA * delta/REF     — penalise body tilt from leg cycling
        #   W_LEG_LENGTH  * length/REF    — penalise long legs
        fell = metrics.get("fell", True)
        z    = metrics.get("peak_wheel_z_m", 0.0)

        fit  = 200.0 * fell + (1.0 - z / 0.30)
        fit += W_PITCH_DELTA * delta_pitch_eq / REF_PITCH_DELTA
        fit += W_LEG_LENGTH  * leg_length / REF_LEG_LENGTH

        metrics["fitness"] = fit
        metrics["delta_pitch_eq_deg"] = math.degrees(delta_pitch_eq)
        metrics["leg_length_m"] = leg_length
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
        f"Fitness:   {best_fitness:.6f}  (lower is better)",
        f"Weights:   W_PITCH_DELTA={W_PITCH_DELTA}, W_LEG_LENGTH={W_LEG_LENGTH}",
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
