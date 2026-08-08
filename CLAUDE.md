# CLAUDE.md — Wheeled-Leg Robot




## Working Rules (always follow these)

- Ask, don't assume. If something is unclear, ask before writing a single line. Never make silent assumptions about intent, architecture, or requirements.

- Simplest solution first. Always implement the simplest thing that could work. Do not add abstractions or flexibility that weren't explicitly requested.

- Don't touch unrelated code. If a file or function is not directly part of the current task, do not modify it, even if you think it could be improved.

- When done changing code always run file or try to compile to look for errors and fix them yourself. 

- Flag uncertainty explicitly. If you are not confident about an approach or technical detail, say so before proceeding. Confidence without certainty causes more damage than admitting a gap.

- I'm always open to ideas on better ways to do things. Please don't hesitate to suggest a better way, or one that has long lasting impact over a tactical change. (as a few examples)"
---

## Control Architecture

**See `docs/Control.MD` for:**

---

## Coordinate System (canonical, applies everywhere)

| Axis | Direction |
|---|---|
| **+X** | Robot forward |
| **+Y** | Robot left |
| **+Z** | Robot up |

This matches MuJoCo's default world frame. All simulation files, firmware, and CAD must use this convention.

---

## Project Overview

Home-built two-wheeled bipedal balancing robot. Balances as an inverted pendulum on two wheels,
with 4-bar linkage legs for terrain clearance and jumping.

---

## Architecture — v4 leg (2026-08-07)

Mechanical baseline drawing: **`components/2N_10mm_279mm.pdf`**. Machine-readable
form of the same geometry (agrees to 0.01 mm):
`simulation/2d/fourbar_optimizer_gui/presets/2N_10mm_279mm.json`.
See `components/COMPONENTS.md` for the full geometry table, BOM, and mass
breakdown, and `firmware/robot_teensy/README.md` → "Leg geometry" for the
firmware constants derived from it.

### 4-bar Leg Topology

```
  [  body box  ]
     F─────────A  ← AK45-10 hip motor
     |         |
     | coupler | femur (187.58 mm)
     | 169.54  |
     E─────────C  ← knee pivot
      (red)  (white)
              \
               \  tibia: |C→W| 185.91 mm, |E→W| 224.49 mm.
                \        NOT straight — C sits 5.28 mm off the E–W line,
                 \       away from the hip motor (C_offset).
                  W  ← wheel centre (Ø112 mm direct drive)
```

**All coordinates below are relative to A**, the hip motor output shaft.
+X forward, +Z up.

- A = hip motor output shaft (femur origin), the frame origin
- F = fixed body pivot at **(−36.42 mm X, +37.54 mm Z) from A** — `|AF|` = 52.30 mm
- C = knee pivot (femur tip), 187.58 mm from A
- E = tibia stub end, **39.01 mm** from C (`EC`), connects to the coupler at F
- W = wheel centre, on the tibia body, 224.49 mm from E, 185.91 mm from C
- `C_offset` = **5.28 mm** — C's perpendicular offset from the E–W line. The
  preset stores the equivalent kink in the tibia's local frame instead (W at
  183.41 mm along C→E, 30.35 mm perpendicular; a 9.4° bend at C).

**Hard stops: `Q_RET` = +28°, `Q_EXT` = −57°** (0° = femur horizontal, positive
retracts) — **85° of stop-to-stop travel**. The extended stop sits essentially
on the 4-bar singularity, so there is no travel to recover past it.

**Vertical wheel stroke: 276.62 mm over the hard stops.** The drawing's headline
**279.95 mm** is over the optimizer's evaluated band (86.29°), which runs
slightly past both stops — a design-envelope figure, not a commandable one.

> **Express F relative to A, and say so.** The single most expensive mistake in
> this repo's history was conflating F's *body-centre* Z coordinate with the
> A→F height offset: the wrong one went into CAD and a whole robot was built to
> it (see `AngleRetractedExt.md`). The two differ by `A_Z`, the hip-axis height
> below body centre. `A_Z = −23.5 mm` is inherited from baseline-1 and **has not
> been re-measured on the v4 box** — so any body-centre F_Z quoted for v4 is
> provisional, while the A-relative numbers above are the measured ones.
>
> Everything derived from the pre-v4 geometry is invalid: link lengths, the
> baseline-1 optimisation result, and the old `Q_RET`/`Q_EXT`. `L_EFF_RET`/
> `L_EFF_EXT` in `control_loop.cpp` have been recomputed for v4; `M_BODY` has
> not.

**The wheel is now Ø112 mm, not Ø150.** `WHEEL_R` = 0.056 m. This rescales every
speed in m/s by 0.747×, so velocity-loop tuning predating the change is stale.

---

## Components

Shopping list and best-estimate specs: `components/COMPONENTS.md`.


