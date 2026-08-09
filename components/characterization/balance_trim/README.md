# Balance point vs leg height (v4 leg)

The balance point is the pitch the robot must hold to stand still at a given
leg height (`gain_sched_alpha`). It feeds `lqr_pitch_trim_ret` (alpha=0) and
`lqr_pitch_trim_ext` (alpha=1), which the firmware interpolates linearly by
alpha.

## What is measured, and why it is not just "the trim"

`lqr_pitch_trim_*` on its own tells you nothing about whether it is *right*.
The LQR hold is proportional-only, so when the trim is wrong the **velocity PI
integrator quietly makes up the difference** by biasing `theta_ref`, and the
robot stands there looking perfectly trimmed while permanently spending lean
budget on a DC offset. The integrator is therefore the instrument — what it
converges to *is* the trim error:

```
balance_point(alpha) = pitch_trim_rad + theta_ref      (settled, stationary)
```

Both terms are in telemetry, so a run measures its own correction. Feed the
result back into the trim and the integrator should relax to ~0.

## Collecting a point

1. Command the leg to the target height (CH3) and let it settle.
2. **Stand still — no velocity, no yaw command — for at least 60 s.**
3. Run the extractor over the capture:

```
python software/gui/tools/balance_trim_sweep.py \
    data/logs/runs/<bundle>/host.jsonl --csv components/characterization/balance_trim/balance_trim.csv
```

Requires `vel_pi_en = 1`. With the velocity PI off, `theta_ref` is pinned at 0,
nothing measures the trim error, and the tool refuses the run.

### The 60 s is not padding

The integrator is slow. On the 2026-08-09 run it was **still walking after
12.8 s** of standing still (`theta_ref` -4.62 -> -3.54 deg; balance point
-6.07 -> -6.54 deg). Averaging a whole stretch biases the answer toward
wherever the integrator started, which is why the tool uses only the tail of
each stationary stretch and prints the residual `drift` across it. **A row with
a `STILL MOVING` flag is a lower bound on the magnitude, not a measurement.**

## Data

See `balance_trim.csv`. Columns: `alpha`, `balance_deg` / `balance_rad` (the
answer), `sd_deg`, `drift_deg` (convergence check), plus the `trim_deg` and
`theta_deg` it was decomposed from, and `run` for provenance.

| alpha | balance point | status |
|---|---|---|
| 0.020 | -6.95 deg (-0.1213 rad) | **not converged** (drift -1.25 deg) — true value is beyond this, and the pre-v4 robot measured about -8 deg |

Everything else is unmeasured. In particular **`lqr_pitch_trim_ext` is still at
its 0.0 default**, so the interpolation currently walks the trim toward zero as
the legs extend, on an anchor nobody has ever measured. That has not bitten yet
only because all v4 balancing so far has been at alpha ~ 0.02. Measure the
extended anchor before doing any ride-height work.

## Do we need a lookup table?

Not yet, and possibly not ever. The firmware already interpolates linearly
between two anchors; if the balance point is linear in alpha — plausible, since
it is a CG-geometry effect — then two measured anchors are exact and a table
adds nothing but parameters. Collect 4-5 points across the range first; the
extractor prints a linearity check against the two-anchor prediction and says
whether the deviation is large enough to justify a table.
