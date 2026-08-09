# Balance point vs leg height (v4 leg)

The balance point is the pitch the robot must hold to stand still at a given
leg height (`gain_sched_alpha`) -- the pitch that puts the CG over the contact
patch. It feeds `lqr_pitch_trim_ret` (alpha=0) and `lqr_pitch_trim_ext`
(alpha=1), which the firmware interpolates linearly by alpha.

## What is measured

Simply `pitch_rad`, averaged over a window where the robot is genuinely in
equilibrium. It does **not** depend on what `lqr_pitch_trim_*` happened to be
set to during the run.

The part that needs care is establishing equilibrium, and that is tested on the
physics rather than assumed. A robot that is not accelerating needs no wheel
torque:

```
tau_sym ~ 0  AND  wheels not drifting   =>   pitch IS the balance point
```

The same falls out of the control law --
`tau_sym = -k_pitch*(pitch - theta_ref - trim)`, so `tau_sym = 0` implies
`pitch = theta_ref + trim`. The tool reports that sum as a free cross-check.
On the 2026-08-09 run the two agreed to **0.09 deg** on the window that had
equilibrated and disagreed by **0.7 deg** on the one that had not -- which is
exactly why windows are gated on `tau_sym` and wheel drift rather than on
elapsed time.

This implicitly requires `vel_pi_en = 1`: with the velocity PI off nothing
stops the wheels running away, the drift gate never passes, and the run yields
no points.

## Collecting a point

1. Command the leg to the target height (CH3) and let it settle.
2. **Stand still -- no velocity, no yaw command -- for at least 60 s.**
3. Run the extractor over the capture:

```
python software/gui/tools/balance_trim_sweep.py \
    data/logs/runs/<bundle>/host.jsonl --csv components/characterization/balance_trim/balance_trim.csv
```

The 60 s is not padding. Settling is slow: on the 2026-08-09 run the robot was
still creeping after 12.8 s of standing still, and that window was rejected by
the equilibrium gate. Only a 4.5 s tail out of 55 s of RUNNING qualified. Short
stand-stills mostly produce rejected windows, not wrong answers -- but they
also produce no answers.

## Data

See `balance_trim.csv`. Key columns: `alpha`, `balance_deg` / `balance_rad`
(the answer), `xcheck_deg` (the `theta_ref + trim` cross-check), `tau_nm` and
`drift_tps` (equilibrium quality), `spread_deg` (scatter across independent
windows -- the honest uncertainty once there is more than one), and `run` for
provenance.

| alpha | balance point | confidence |
|---|---|---|
| 0.020 | **-7.44 deg (-0.130 rad)** | single 4.5 s window; cross-check agrees to 0.09 deg. Pre-v4 robot measured about -8 deg, so this is plausible but wants a dedicated 60 s stand-still. |

Everything else is unmeasured.

## Before extending the legs: audit the `_ext` anchors

Every alpha-scheduled pair fades from its `_ret` anchor to its `_ext` anchor.
All v4 balancing so far has been at alpha ~ 0.02, where the `_ext` anchor gets
only 2% weight, so a wrong one has been invisible. That stops being true
immediately as the legs come up.

- **`lqr_pitch_trim_ext` is 0.0 and unmeasured.** The trim walks toward zero as
  the legs extend, when the measured value at alpha=0.02 is -7.4 deg.
- **`hip_running_tff_ext` is 0.0.** The hip feedforward is alpha-scheduled too.
  `hip_running_tff_ret` is what nulls the static sag and holds the leg clear of
  the retract hard stop -- the fix that made v4 balance at all. With `_ext` at
  zero the sag returns proportionally as the legs extend, and with it the
  stop contact and the rattle.
- **`lqr_k_pitch_ext` / `lqr_k_rate_ext`** default to their `_ret` values (flat,
  no fade). If they have been zeroed, available gain falls as `k_ret*(1-alpha)`
  while the required gain *rises* with alpha (the CG climbs further above the
  axle, `l` 0.099 -> 0.363 m). With `k_ret ~ 0.47` those cross near
  **alpha ~ 0.06** -- a few degrees of leg above the current operating point.
  The crossing is model-dependent (`M_BODY` was never recomputed for v4), but
  the direction is not.

Safe posture: set each `_ext` gain anchor equal to its `_ret` value before any
ride-height work, so the schedule is flat, then tune the extended end
deliberately in small alpha steps with this sweep running.

## Do we need a lookup table?

Not yet, and possibly not ever. The firmware already interpolates linearly
between two anchors; if the balance point is linear in alpha -- plausible,
since it is a CG-geometry effect -- then two measured anchors are exact and a
table adds nothing but parameters. Collect 4-5 points across the range first;
the extractor prints a linearity check against the two-anchor prediction and
says whether the deviation is large enough to justify a table.
