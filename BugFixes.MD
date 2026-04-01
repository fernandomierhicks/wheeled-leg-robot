# BugFixes.MD — Pending Improvements

Found during static analysis of `simulation/mujoco/master_sim_jump/` on 2026-03-31.

---

## Bugs / Correctness

### [Medium] Matrix predictor unpacking — `sim_loop.py:456–461`
`x_pred` from `self._Ad_n @ x_del` may be a 2D column vector. Unpacking it directly on line 461 could silently produce wrong values or fail.
**Fix:** Use `x_pred = x_pred.ravel()` before unpacking, or extract with `x_pred[0], x_pred[1], x_pred[2]`.

### [Medium] Fragile `prev_q_nom_L` initialization — `sim_loop.py:789, 877`
`prev_q_nom_L` is initialized to `None` and guarded with `is not None`. Correct behavior, but if someone refactors to initialize it to a real value, the first-iteration metric will be wrong.
**Fix:** Initialize to `robot.Q_NOM` and drop the `is not None` guard, or add an explicit `n_samples > 1` gate.

### [Medium] Silent IK fallback — `sim_loop.py:636–637, 495–496`
If `solve_ik()` fails, `_l_eff_nom` is used silently. A physics bug (leg out of bounds) would go undetected and corrupt feedforward terms.
**Fix:** Log a warning on IK failure, or add an `ik_fail_count` field to the telemetry dict.

---

## Dead Code

### [Low] `smith_predict()` never called — `sim_loop.py:165–184`
Function is defined but superseded by the matrix predictor (lines 456–461). Dead code.
**Fix:** Remove or move to a diagnostic/test module.

### [Low] `self._B1_nom` unused — `sim_loop.py:324`
Computed and stored with a "kept for legacy reference" comment, but never read.
**Fix:** Remove entirely.

### [Low] `command_queue=None` unused parameter — `sim_loop.py:722`
Accepted by `run()` with a docstring mention of "sandbox mode" but never used.
**Fix:** Remove the parameter, or implement the feature.

---

## Magic Numbers

### [Medium] Hardcoded `0.01` m leg length threshold — `sim_loop.py:497, 644, 655`
The minimum effective leg length threshold `0.01` appears three times in feedforward paths (FF1, FF2, FF3) with no named constant.
**Fix:** Add `min_l_eff_ff: float = 0.01` to `FeedforwardParams` in `params.py` and reference it by name.

---

## Performance (minor)

### [Low] Redundant `float()` casts in control loop — `sim_loop.py:448–450, 662–666`
Values extracted from the delay buffer and `data.ctrl[]` are already floats. Wrapping them in `float()` is a no-op.
**Fix:** Remove the casts.

### [Low] Duplicate IK calls for FF1/FF2 vs FF3 — `sim_loop.py:495–496, 636`
When both FF3 and FF1/FF2 are enabled, IK is solved twice per tick for the same hip angle.
**Fix:** Consolidate into a single `solve_ik()` call and reuse the result.
