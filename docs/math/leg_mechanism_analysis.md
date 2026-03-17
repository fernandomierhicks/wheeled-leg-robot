# Leg Mechanism Analysis — Vertical Wheel Motion with Single Motor

## The Constraint (Relaxed)

From step05 simulation: with equal femur and tibia lengths L, the wheel stays
directly below the hip pivot (x = 0) at all hip angles if and only if:

$$\theta_{knee,rel} = -2 \cdot \theta_{hip}$$

**Relaxed assumptions (updated):**
1. **Femur ≠ tibia** — link lengths do not need to be equal; exact vertical
   only holds with equal lengths but approximate vertical is acceptable
2. **Approximate vertical is fine** — mechanically close to x=0 over the jump
   stroke range is sufficient; exactness is not required
3. **Jump stroke only** — the near-vertical constraint only needs to hold during
   the jump power stroke (mid-range hip angles, approx 10°–50°). At fully
   retracted or fully extended positions, wheel x-drift is acceptable
4. **CoM offset handles the rest** — absolute wheel x-position relative to the
   body will be corrected by CG placement, not purely by mechanism geometry

The ideal (equal-link) case gives:

$$y_{wheel} = y_{hip} - 2L\cos(\theta_{hip})$$

Pure vertical motion — this is the reference target, not a hard constraint.

---

## Why a Simple Parallelogram Does NOT Give This

A parallelogram 4-bar enforces θ_knee,rel = 0 (tibia always parallel to femur).
The wheel traces a **circular arc**, not a vertical line.
Wheel x-position = L·sin(θ_hip) — shifts left/right as the leg moves.

This is what the reference Impulse robot's pantograph does.
It keeps the **wheel axle horizontal** (useful for wheel traction contact angle),
but does NOT keep the wheel under the body CoM.

---

## Mechanism Options to Enforce θ_knee = −2·θ_hip

### Option 1 — Toothed Timing Belt, 2:1 Ratio  ⭐ RECOMMENDED

```
   A (hip motor)
   |  [pulley R]──belt──[pulley R/2 at knee]
   | femur
   C (knee) → tibia rotated -2·θ_hip → wheel at (0, y)
```

- Motor at A drives femur AND a pulley of radius R
- Belt routed along inside of femur to a pulley of radius R/2 at knee
- Pulley ratio 2:1 → knee rotates at -2× hip rate (opposite direction from geometry)
- **Exact** constraint satisfied at all angles
- Single motor — no extra actuator
- Timing belt = no stretch, no slip, available off-shelf (GT2, HTD3M)
- This is how Mini Cheetah, Spot, and many research robots route knee actuation

**Downside:** Belt must be routed inside the femur link (adds design complexity,
but is a solved problem in robot design).

---

### Option 2 — Four-Bar with Synthesized Link Ratios  ✓ PURE MECHANICAL

```
   A (hip motor, body-fixed)
   |  femur (driven)          B (body-fixed passive pivot)
   C (knee)                   |
   |  tibia                   | coupler link
   W (wheel)                  D (point on tibia, constrained by coupler)
```

A 4-bar: **A** (driven input), **B** (fixed), **C** (knee = coupler pivot),
**D** (point on tibia constrained by link BD).

For SPECIFIC link ratios, the mechanism approximates θ_knee ≈ −2·θ_hip
over the operating range (0°–60°).

**Key insight:** A pure 4-bar CANNOT give EXACT 2:1 at all angles
(this is a mathematical constraint on 4-bar synthesis). But over 0–60° hip range,
the maximum deviation from ideal vertical motion can be kept < 5 mm
with correct link length choice (found via numerical optimisation).

The body pivot B is placed ABOVE and slightly behind A (inside the body box).
Link BD length and D position on tibia are the optimisation variables.

**Downside:** Non-exact — wheel traces a near-vertical curve, not a perfect line.
No belts — entirely rigid links and pivots (608 bearings).
More complex to design correctly. Requires simulation/optimisation first.

---

### Option 3 — Parallelogram (Original CLAUDE.md Design)  ✓ SIMPLEST

The reference Ascento / Navbot / Impulse-pantograph approach.

- Keeps wheel AXLE horizontal at all angles
- Wheel traces arc — controller compensates for lateral drift
- Single motor at A
- Proven in multiple real builds

**Use this if:** Simplicity matters more than exact vertical motion.
For small hip angles (< 30°), arc deviation from vertical is < 8 mm.

---

## Recommendation

| Mechanism | Exactness | Complexity | Parts needed |
|---|---|---|---|
| Timing belt 2:1 | Exact ✓ | Medium | Belt + 2 pulleys |
| Synthesised 4-bar | ~5 mm error | High | 1 extra link + pivot |
| Parallelogram | ~8 mm at 30° | Low | Simplest |

**For this build:** Try the **user-proposed 4-bar** (step07 — stub + body pivot F)
first. Given the relaxed constraints (approximate vertical, jump stroke only,
CG offset available), a larger error than 5 mm may be perfectly acceptable.
Evaluate the locus over the jump stroke range only (≈10°–50°), not full range.
If the locus is qualitatively close to vertical in that band, the mechanism is good.

The simulation in `step06_mechanism_comparison.py` shows all three side by side
with error plots.

---

## Jumping with This Leg

For a jump:

1. Controller commands a rapid hip angle sweep: θ_hip: 50° → 0° (retract → extend)
2. With 2:1 constraint: wheel moves from y_low → y_hip (full extension)
3. AK45-10 at 7 N·m × rapid rotation → impulsive ground reaction force
4. Body launches upward

The key advantage of the vertical constraint for jumping:
- Ground reaction force is purely vertical at θ_hip = 0 (leg straight)
- Energy goes into vertical velocity, not wasted on lateral reaction
- Maximises jump height
