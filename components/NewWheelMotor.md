# Wheel Motor Sizing — is the MTO5065 oversized?

**Date:** 2026-08-03 · **Question:** the Maytech MTO5065-70-HA-C feels oversized;
is it, and what replaces it? · **Evidence:** 574 s of RUNNING telemetry across
the 5 most recent balancing runs, plus physics design cases.

**Verdict: yes, substantially.** The worst wheel-torque sample in ten minutes of
balancing is **3.5 % of the motor's peak**, and the hardest case the firmware
will ever ask for is **26 %**. The motor is also 15× overspeed. The two wheel
motors are 900 g of a 3.06 kg robot.

---

## Executive summary — top 5 replacements

Ranked by overall value, not by a single metric. "System saving" = 2 motors +
controller, against the current 1060 g (2 × 450 g + 160 g ODESC).

| # | Motor | Price ea | Mass ea | Peak τ | **System saving** | Why this one |
|---|---|---|---|---|---|---|
| **1** | **5010 360KV** (GoolRC / Readytosky / VGEBY) | **~$20** ⚠ | **80 g** | ~0.8 N·m @ 30 A | **740 g (24 %)** | Best mass *and* best price, by a wide margin. Covers every case except the extreme stand-up divergence. Cost of entry is a hub redesign, not money. |
| **2** | **T-Motor MN6007 II KV160** | ~$110 ⚠ | 159 g | 1.41 N·m @ 23.7 A | **582 g (19 %)** | The best *engineering* answer. Covers the full torque envelope including 57° stand-up, keeps ODrive + encoder, and its lower Kt actually improves ODESC current-sense resolution. Over budget. |
| **3** | **Maytech 5055-70KV** | ~$50 ⚠ | 330 g | 4.09 N·m @ 30 A | 240 g (8 %) | Lowest risk by far. Identical Kt (0.136) ⇒ **zero control retune**, same 50 mm family, same Hall-sensor option. A weekend change. Modest reward. |
| **4** | **T-Motor MN5008 KV170** | ~$90 ⚠ | 135 g | 0.82 N·m @ 14.6 A | 630 g (21 %) | Lightest motor with a real published rating. But 0.82 N·m sits exactly on the 34° stand-up case with no margin. |
| **5** | **CubeMars GL60 II KV28** | ~$150 ⚠ | 276 g | 1.0–1.5 N·m | 508 g (17 %) | The only option that **deletes the ODESC entirely** — integrated driver, native CAN + MIT mode, same protocol family as the AK45-10 hips. Heaviest motor here but a competitive system saving. |

⚠ **Prices are indicative and unverified except where noted below.** Only the
Flipsky figures were confirmed against a live listing. Re-check before ordering.

**Recommendation:** buy a pair of **5010 360KV for ~$40 total and bench them**.
At that price, testing costs less than deliberating — the risk was never the
money, it's the hub fabrication. Keep the **Maytech 5055** as the fallback if
the demand test (below) comes back above ~1 N·m.

---

## 1. The evidence

Five most recent runs containing real balancing, `robot_state == RUNNING` only.
Method and traps: `firmware/robot_teensy/AnalyzeLogClaude.md`.

| Run | RUNNING | active clamp | \|τ\| p50 | p99 | max | RMS | saturated |
|---|---|---|---|---|---|---|---|
| `20260726T060123` | 160.2 s | 0.200 | 0.0168 | 0.0675 | 0.2000 | 0.0283 | 0.24 % |
| `20260727T033345` | 145.0 s | 0.088 | 0.0182 | 0.0757 | 0.1000 | 0.0278 | 0.01 % |
| `20260727T044352` | 75.0 s | 0.100 | 0.0224 | 0.1000 | 0.1000 | 0.0344 | 0.72 % |
| `20260728T053232` | 161.8 s | 0.200 | 0.0184 | 0.0875 | 0.2418 | 0.0305 | 0.01 % |
| `20260802T174130` | 32.3 s | 0.092 | 0.0193 | 0.0716 | 0.1267 | 0.0285 | 0.06 % |

Pooled over 574.3 s, both wheels:

| | N·m | as current | % of 6.82 N·m peak |
|---|---|---|---|
| p50 | 0.0184 | 0.13 A | 0.27 % |
| p95 | 0.0561 | 0.41 A | 0.82 % |
| p99 | 0.0879 | 0.64 A | 1.29 % |
| worst sample | 0.2418 | 1.77 A | **3.55 %** |
| RMS | 0.0297 | 0.22 A | 0.44 % |

Peak mechanical power **4.2 W** per wheel; p99 is 0.38 W.

**The software clamp is not what's limiting this.** `lqr_torque_limit` is slewed
from the active CH9 profile (`profileN_torque_lim`, defaults 0.1 / 0.2 / 0.3
N·m), so round-number peaks were the first suspicion. But saturation is
0.01–0.72 % of samples, and two runs never reached their clamp at all (max
|τ_sym| 0.088 and 0.092 against a 0.1 limit). These are genuine demand figures.

**The speed side is equally lopsided.** No-load is 175.9 rad/s at 24 V = 13.2 m/s
at the wheel. The fastest clean sample was 2.0 turns/s ≈ 0.95 m/s. Even at a
2 m/s design target that is **15 % of no-load**.

---

## 2. Design cases — size from physics, not from these logs

The logs are gentle flat-floor tuning. The real ceiling is the hardest thing the
firmware will ask for. Per wheel, M = 3.1 kg, r = 0.075 m:

| Case | Identity | τ [N·m] |
|---|---|---|
| Balance p99 (measured) | — | 0.088 |
| 1 m/s² acceleration | `M·a·r/2` | 0.116 |
| 3 m/s² acceleration | | 0.349 |
| 10° slope, static hold | `M·g·sinθ·r/2` | 0.198 |
| 20° slope, static hold | | 0.390 |
| Catch a 20° lean | `M·g·tanθ·r/2` | 0.415 |
| **Catch 34° forward** — `standup_pitch_max` = +0.6 rad, the arm-time gate | | **0.78** |
| **Catch 57°** — `standup_div_fwd` = 1.0 rad, the abort limit | | **1.78** |

**The binding case is the stand-up catch at 1.78 N·m — 26 % of what you have.**
Everything else is under 0.4 N·m. Whether you need the 57° case or only the 34°
case is the single decision that separates a $20 motor from a $110 one.

---

## 3. Full candidate comparison

Kt = 9.55/KV. τ/g uses each vendor's own peak-current rating, so it is only
loosely comparable across categories (see the caveat below the table).

| Motor | $ ea | Mass | KV | Kt | I_max | Peak τ | τ/g [mN·m/g] | Saving |
|---|---|---|---|---|---|---|---|---|
| MTO5065-70 *(current)* | ~$90 | 450 g | 70 | 0.136 | 50 A | 6.82 | 15.2 | — |
| 5010 360KV | ~$20 | 80 g | 360 | 0.027 | 20–40 A | ~0.8 | 9.9 | 740 g |
| T-Motor MN5008 KV170 | ~$90 | 135 g | 170 | 0.056 | 14.6 A | 0.82 | 6.1 | 630 g |
| T-Motor MN6007 II KV160 | ~$110 | 159 g | 160 | 0.060 | 23.7 A (180 s) | 1.41 | 8.9 | 582 g |
| CubeMars GL60 KV25 | ~$130 | 230 g | 25 | 0.450 | 4 A | 1.75 | 7.6 | 440 g |
| CubeMars GL60 II KV28 | ~$150 | 276 g | 28 | 0.340 | 4.1 A | 1.0–1.5 | 5.4 | 508 g † |
| T-Motor U8 II KV85 | ~$180 | 277 g | 85 | 0.112 | 20 A | 2.25 | 8.1 | 346 g |
| Maytech 5055-70KV | ~$50 | 330 g | 70 | 0.136 | 30 A | 4.09 | 12.4 | 240 g |
| Flipsky 5065 200KV | **$56.80** ✓ | ~450 g | 200 | 0.048 | — | — | — | 0 — skip |
| Flipsky 6354 190KV | **$70.99** ✓ | 560 g | 190 | 0.050 | 65 A | 7.0 | 12.5 | *heavier* |
| GBM5208 gimbal | ~$40 | 185 g | — | — | — | ~0.25 | 1.4 | too weak |
| MyActuator RMD-L-4015 | ~$90 | 120 g | — | — | — | 0.49 | 4.1 | too weak |

† deletes the ODESC entirely. ✓ price verified against a live listing.

**Caveat on τ/g:** skateboard motors look best because their current ratings are
aggressive short bursts with no duty cycle; drone-motor ratings assume prop
airflow. Neither matches this application, where duty is essentially nil.

---

## 4. Why the cheap drone motor works

The counter-intuitive result is that the **$20, 80 g motor saves more mass than
the $110, 159 g one** (740 g vs 582 g). It is a smaller frame with a thinner
bell. What you give up is torque headroom and structure — not efficiency, not
control quality.

**Thermally it is a non-issue.** Continuous duty is 0.03 N·m RMS = 1.1 A ≈
**0.1 W** of copper loss. A 29 A stand-up burst is ~76 W for 0.3 s = 23 J ≈
**+2.4 °C** on the winding. Drone-motor current ratings assume prop airflow;
this application does not need it, so it can be sized on **saturation, not
heat** — which is what lets it go far smaller than any catalogue rating implies.

**5010 360KV against the design cases** (Kt = 0.027, ESC band 20–40 A):

| Case | τ | current | |
|---|---|---|---|
| Balance p99 | 0.088 | 3.3 A | ✓ trivial |
| 3 m/s² accel | 0.349 | 13 A | ✓ |
| 20° slope | 0.390 | 15 A | ✓ |
| **34° stand-up catch** | 0.78 | 29 A | ✓ top of band |
| **57° divergence** | 1.78 | 66 A | ✗ |

Two honest caveats: at 29 A a 10 mm stator will **saturate** and Kt will droop
perhaps 20–30 %, so budget 35–40 A for a real 0.78 N·m. And the low Kt raises
operating current to 1.1 A from today's 0.22 A — which **improves** ODESC shunt
resolution, since you are currently commanding 0.4 % of a 50 A full scale.

---

## 5. Risks and open questions

**Mechanical, and it is the crux.** A 5010 has a 4 mm shaft and prop-thrust
bearings. It cannot carry the robot. This only works if the wheel rides on its
own bearings and the motor supplies torque alone. The BOM may already support
this — `BRG_608_W` is specced at the wheel axle with F_peak 126 N and s₀ = 10.87
— but `COMPONENTS.md` also says the wheel is "D-shaft mount to 5065 motor".
**Check the CAD to establish which member actually carries the load.** If it is
the 608, this swap is far easier than it looks.

**Drive from the bell face, not the shaft.** 1.41 N·m through a 4 mm shaft is
~112 MPa shear, and it is a press-fit prop shaft. Bolt the wheel to the rotor
bell's mounting circle, which is how these motors are designed to take load.

**Dust ingestion.** An open drone bell 75 mm off the ground will eat grit. Your
skateboard motor is sealed for exactly that environment. This is the one thing
genuinely given up, and it is the reason the original MTO5065 choice was not
unreasonable.

**No Hall sensors** on any drone or gimbal motor here — commits you to the
magnetic encoder, which suits the current setup.

**Cogging.** Median torque command is 0.019 N·m; a gimbal motor's ~0.015 N·m
cogging is the same order. Likely still better than the MTO5065, but verify
rather than assume.

**Avoid KV > ~400.** Below that, back-EMF at the 255 rpm cruise falls under 1 V,
the ODrive PWMs a ~3 % duty into a low-resistance winding, and current ripple
swamps the setpoint. 360KV is about the practical ceiling.

**BOM inconsistency, unresolved.** `COMPONENTS.md` gives 450 g motor + 70 g wheel
= 520 g/side; `simulation/mujoco/master_sim/params.py` uses `m_wheel = 0.270` for
motor + hub + tyre. Both cannot be right, and the BOM is labelled best-estimate.
Every saving figure in this document moves with it.

---

## 6. Next steps

Two free steps decide the whole thing:

1. **Weigh the actual MTO5065.** Resolves §5's BOM inconsistency and rebases
   every number above.
2. **Measure real peak demand.** Raise `profile3_torque_lim` toward 1.0 N·m, set
   `standup_enable = 1`, and log an aggressive run plus a slope. This converts
   §2's physics table into evidence, and specifically answers whether the 57°
   divergence case is needed — the one number separating option 1 from option 2.

Then: buy a pair of 5010s (~$40) and bench them before committing to a hub design.
