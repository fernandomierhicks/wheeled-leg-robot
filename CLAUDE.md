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

## Architecture — Baseline 1

Geometry from evolutionary optimisation (
See `components/COMPONENTS.md` for full geometry table, BOM, and mass breakdown.

### 4-bar Leg Topology

```
  [  body box  ]
  F────────────A  ← AK45-10 hip motor
  |  coupler   |
  |  link      | femur (173.78 mm)
  |            |
  E────────────C  ← knee pivot
  (red pin)  (white pin)
               |
               | tibia (129.39 mm down to W, 35.13 mm stub up to E)
               |
               W  ← wheel centre (Ø150 mm, Maytech MTO5065-70-HA-C direct drive)
```

- A = hip motor output shaft (femur origin)
- F = fixed body pivot at (−58.87 mm X, **−5.5 mm Z**) from body origin —
  i.e. **F sits 18 mm above the hip axis A** (A_Z = −23.5 mm). **As built.**

> **Do not use −18.21 mm for F_Z.** That was the value here until 2026-08-03 and
> it is the single most common source of wrong leg kinematics in this repo.
> −18.21 mm is F's body-centre Z from the baseline-1 optimisation, in which the
> hip-to-coupler *height* offset (A→F) was **5.29 mm**. The two were conflated,
> the wrong one went into CAD, and **the robot was physically built with an
> 18 mm A→F offset**. The build is the reference; the old numbers are not.
>
> Everything derived from the 5.29 mm geometry is invalid — the baseline-1
> optimisation result, `Q_RET`/`Q_EXT`, and `L_EFF_RET`/`L_EFF_EXT` in
> `control_loop.cpp`. Total hip travel is limited to **66°** by deliberate hard
> stops (the 4-bar goes singular at ~75.9° with the as-built geometry).
- C = knee pivot (femur tip)
- E = tibia stub end (35.13 mm above C), connects to coupler at F

---

## Components

Shopping list and best-estimate specs: `components/COMPONENTS.md`.


