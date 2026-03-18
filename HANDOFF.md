# HANDOFF.md — Project Status & Next Steps

Read this before starting work. Check `CLAUDE.md` for full design context and `docs/Control.MD` for controller architecture and full scenario specs.

---

## Current Status (2026-03-18)

### Phase 1 — Balance LQR ✅ COMPLETE & BASELINED

- **Scenario:** `1_LQR_pitch_step` — 5° pitch step + two ±4N kicks, VelocityPI OFF
- **Optimizer:** `optimize_lqr.py` — searches Q_PITCH, Q_PITCH_RATE, Q_VEL, R
- **Best run:** 4448 evals, 5-min run

| Param | Value |
|-------|-------|
| Q_PITCH | 0.138282 |
| Q_PITCH_RATE | 0.023379 |
| Q_VEL | 0.004591 |
| R | 9.998298 |
| fitness | 0.003267 |
| rms_pitch | 0.926° |

Gains in `sim_config.py` and `results_1_LQR_pitch_step.csv`.

**Character:** Low Q_PITCH relative to Q_PITCH_RATE → controller damps pitch rate, not pitch angle. Steady-state offset is intentional — handled by VelocityPI outer loop.

---

### Phase 2 — VelocityPI Outer Loop ✅ FULLY BASELINED

S2 and S3 jointly optimised via `combined_PI`. S3 extended to ±1 m/s with forced direction reversal. `theta_ref` rate-limited at 2.0 rad/s to suppress pitch-rate spikes on lean command steps.

**S2 — `2_VEL_PI_disturbance`** — position hold under disturbance kicks
- Equilibrium start, `v_desired = 0`, ±1N kicks at t=2/3s, 6s duration
- Metric: absolute wheel travel [m] / duration → units m/s

**S3 — `3_VEL_PI_staircase`** — velocity setpoint tracking
- Staircase: 0 → +0.3 → +0.6 → +1.0 → −0.5 → −1.0 → 0 m/s, 13s duration
- 1s settle window before first step (excluded from metric)
- Metric: `vel_track_rms_ms` [m/s] + 0.1 × rms_pitch_deg penalty

**Combined PI — `combined_PI`** — optimizer target
- fitness = 0.5 × S2_fitness + 0.5 × S3_fitness
- Results in `results_combined_PI.csv`

**Baselined gains (combined_PI, 30-min / 5488 evals, 2026-03-18):**
| Param | Value |
|-------|-------|
| KP_V | 0.502932 |
| KI_V | 0.012678 |
| fitness | 0.61 (on extended ±1 m/s S3) |
| THETA_REF_RATE_LIMIT | 2.0 rad/s |

---

## Scenario Reference

| # | Name | Controller | VelocityPI | v_desired | Disturbance | Metric | Status |
|---|------|------------|------------|-----------|-------------|--------|--------|
| 1 | `1_LQR_pitch_step` | LQR only | OFF | 0 | ±4N at t=2/3s | ISE pitch [rad²·s] | ✅ Optimised |
| 2 | `2_VEL_PI_disturbance` | VelocityPI + LQR | ON | 0 (hold) | ±1N at t=2/3s | wheel travel / duration [m/s] | ✅ Baselined |
| 3 | `3_VEL_PI_staircase` | VelocityPI + LQR | ON | 0→+0.3→+0.6→+1.0→−0.5→−1.0→0 | none | vel_track_rms_ms [m/s] | ✅ Baselined |
| — | `combined_PI` | VelocityPI + LQR | ON | S2+S3 | S2 kicks | 0.5×S2 + 0.5×S3 | ✅ Baselined |

Full details (timing, disturbance forces, fitness formulas) in `docs/Control.MD → Optimizer Scenarios`.

---

## Replay Usage

```bash
# View current gains in any scenario
python replay.py --baseline --scenario 1_LQR_pitch_step
python replay.py --baseline --scenario 2_VEL_PI_disturbance
python replay.py --baseline --scenario 3_VEL_PI_staircase

# Replay a specific run from CSV
python replay.py 42
python replay.py --top 3    # 3rd-best run by fitness
python replay.py --list     # list all runs
```

Telemetry panels: pitch + pitch cmd, wheel torque, pitch rate, robot velocity + commanded velocity (dashed orange). S3 staircase is auto-scheduled — no slider interaction needed.

---

## Next Steps

### Immediate — Phase 3: Yaw PI

Phase 2 is complete. Next step is Phase 3 — Yaw PI + Turn Mode. See `docs/Control.MD → Phase 3`.

To re-run the PI optimizer (e.g. after further S3 changes):
```bash
# Clear CSV first (scenario params changed)
rm results_combined_PI.csv
python optimize_vel_pi.py --hours 0.5
```

### Phase 3 — Yaw PI

See `docs/Control.MD → Phase 3`.

---

## Key Files

| File | Purpose |
|------|---------|
| `sim_config.py` | Single source of truth — all gains, durations, disturbance forces |
| `scenarios.py` | All scenario runners + `_run_sim_loop` shared physics backend |
| `optimize_lqr.py` | (1+8)-ES — searches Q_PITCH, Q_PITCH_RATE, Q_VEL, R |
| `optimize_vel_pi.py` | (1+8)-ES — searches KP_V, KI_V; supports `--scenario combined_PI` |
| `replay.py` | MuJoCo viewer + 4-panel matplotlib telemetry, staircase auto-schedule |
| `progress_window.py` | Floating tkinter progress window during optimizer runs |
| `lqr_design.py` | LQR solver + gain scheduling at 3 leg positions |
| `results_1_LQR_pitch_step.csv` | Scenario 1 run history |
| `results.csv` | Scenario 2/3/combined run history (clear before re-optimising after param changes) |
| `docs/Control.MD` | Full architecture, scenario specs, LQI rationale, phase plans |

---

## Hardware Tasks (Parallel, Not Blocking)

- [ ] Pick specific 5065 130KV motor SKU (Flipsky, T-Motor — needs Hall sensors, D-shaft, ≥40A, 24V)
- [ ] Finalise battery: 24V nominal, ≥4Ah, XT60, ≤750g (6S LiPo or 24V LiFePO4)
- [ ] Design wheel in CAD: PLA hub + TPU tread, 150mm OD, D-shaft mount, target 70g

---

## Future Phase Plans

See `docs/Control.MD` for Phases 3–5 (Yaw PI, Leg Suspension, Jump Recovery).
