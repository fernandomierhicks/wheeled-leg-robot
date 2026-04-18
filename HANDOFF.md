# HANDOFF — Tube-sizing bending analysis

## Context

User asked whether all leg tubes can be made 16×1.0 mm Al (strength-wise) and
what the mass impact would be. We already made a what-if swap:

- `params.py` — `m_femur`, `m_coupler` updated to 16×1 values
  (femur 19.2 g → 22.1 g; coupler 11.5 g → 19.2 g; tibia unchanged 18.3 g)
- `viz/visualizer.py` `_robot_geom_dict` — all three `F_yield_*` set to 2912 N
  (16×1 axial yield, same as the current tibia)

Jump sim ran — on the Mechanical tab the axial-load lines were barely visible
against the yield threshold. **Axial is not the binding constraint.**

## Why axial is misleading

`_estimate_forces` at [viz/visualizer.py:2410](simulation/mujoco/master_sim_jump/viz/visualizer.py#L2410)
computes axial-only loads. But:

- **Femur & tibia**: cantilever-loaded — transverse force at knee / wheel
  creates bending moment at the root. For thin Al tubes, `σ_bend = M·c/I`
  is typically the binding constraint.
- **Coupler**: actual two-force member (pinned both ends, no transverse
  load) → axial IS the right check. High axial SF (5.79 → ~12 at 16×1)
  reflects that.

## Rough bending bounds (6061-T6, σ_y = 276 MPa, S = π(OD⁴−ID⁴)/(32·OD))

| Tube | S [mm³] | M_yield [N·m] | F_tip @ yield |
|---|---|---|---|
| Femur 14×1 (L=174 mm) | 124 | 34.2 | 197 N |
| Femur 16×1 (L=174 mm) | 166 | 45.9 | 264 N |
| Tibia 16×1 (L=129 mm) | 166 | 45.9 | 356 N |

Bending capacity 14→16 mm femur: **+34%** for the +2.9 g mass penalty.
Real "can we use all 16×1?" answer requires comparing these to the *actual*
bending moments in sim — which we don't compute yet.

## Plan — add bending to Mechanical tab

### Step 1 — extend `_estimate_forces` (viz/visualizer.py:2410)

Return two new keys:

- `M_fem`: bending moment at femur root (pivot A)
  = |F_knee ⊥ femur_axis| · L_femur
- `M_tib`: bending moment at tibia root (knee C)
  = |F_wheel ⊥ tibia_axis| · L_tibia

The transverse components come from resolving the knee-reaction and
wheel-reaction vectors (already computed) perpendicular to each tube's
axis, which is defined by the IK pose (hip angle → knee pos → tibia
direction via stub geometry).

### Step 2 — add 2 panels to Mechanical tab

In the "Row 2 / Row 3" grid block around
[viz/visualizer.py:2153](simulation/mujoco/master_sim_jump/viz/visualizer.py#L2153):

- **Femur bending moment** [N·m], red line at M_yield_femur (34.2 or 45.9)
- **Tibia bending moment** [N·m], red line at M_yield_tibia (45.9)

Pass `M_yield_femur`, `M_yield_tibia` through `_robot_geom_dict`
(around [viz/visualizer.py:2823](simulation/mujoco/master_sim_jump/viz/visualizer.py#L2823)),
computed from OD/wall/length so it auto-updates when the tube is changed.

### Step 3 — rerun the S10 jump scenario

Verdict is simple:

- If bending panels stay well below yield with **14×1 femur**: keep current
  baseline (cheaper, lighter).
- If bending panels only pass with **16×1 femur**: the +2.9 g upsize is
  justified for the strength margin.
- Coupler stays 10×1 regardless — axial-only, and well within yield.

## Files currently modified (what-if state, not committed)

- `simulation/mujoco/master_sim_jump/params.py` — m_femur, m_coupler at 16×1
- `simulation/mujoco/master_sim_jump/viz/visualizer.py` — all F_yield=2912 N

Revert both if we decide bending analysis should be done with the true
baseline (14×1 femur, 10×1 coupler) for an apples-to-apples strength
comparison. Recommended order:

1. Revert what-if edits (back to 14×1 femur / 10×1 coupler baseline).
2. Implement bending panels.
3. Rerun S10 — read peak bending moments.
4. Decide: keep baseline, upsize to 16×1 femur, or go further.
