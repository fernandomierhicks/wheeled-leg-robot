# Hardware Gain-Tuning: Phase 1 Operator Guide (Retracted Anchor)

## Goal

Tune the LQR pitch/rate, vel-PI, and yaw-PI gains on real hardware, starting from a single fully-retracted leg pose. Order: LQR pitch/rate first (balance is the foundation), then vel-PI, then yaw-PI. Fernando arms the robot (RC transmitter) and disarms if anything looks wrong; Claude sets parameters, drives velocity/yaw setpoints itself via the GUI motion channel, pulls the SD log, analyzes it, and proposes the next candidate — repeating with as little manual work from Fernando as possible beyond "arm now."

**Everything described below is built and hardware-verified.** No more implementation needed — this phase is about running trials.

## Constraints this phase

- **Hips are zip-tied fully retracted, hip motors disabled** (`hip_l_enable=0`, `hip_r_enable=0`) for all of Phase 1. There is no leg to hold — Fernando's only job during a run is the radio arm switch (CH10) and disarming if something looks wrong.
- **Real hip calibration can never complete** with hips zip-tied (it requires cycling them through their range), so `alpha` (the leg-height gain-schedule blend) can't be computed the normal way — see `alpha_force_ret_en` below.
- **Velocity/yaw setpoints for vel-PI/yaw-PI trials come from the GUI motion channel, not Fernando's sticks** — a human-held stick can't reproduce the same profile trial-to-trial. Arming stays exclusively radio-sourced throughout; the motion channel only ever gates *what* `v_cmd_ms`/`omega_cmd_rds` are set to, never *whether* the robot is armed. Fernando can always kill a run via CH10 regardless of what Claude is commanding.
- Mid/extended leg positions are out of scope for Phase 1 — see "Deferred to Phase 2" at the bottom.

## Start-of-session checklist

1. **Confirm both boards are connected and on `TELEM_VERSION 10`.** Launch the GUI (`python main.py` in `software/gui/`) if it isn't already running — the remote-control server auto-starts with it, no separate step needed. Then:
   ```
   python software/gui/tools/robot_ctl.py telem
   ```
   Check `telem.version == 10` and `telem.robot_state == 2` (STANDBY). If the ESP32 shows a version mismatch, it needs reflashing (`pio run -e esp32dev -t upload` in `firmware/robot_teensy/esp32`) — this shouldn't happen unless someone flashed one board and not the other.

2. **Set `alpha_force_ret_en=1`** — not persisted, must be set explicitly every session:
   ```
   python software/gui/tools/robot_ctl.py param_set alpha_force_ret_en 1
   ```
   This forces the leg gain-schedule blend to the fully-retracted anchor (`alpha=0.0`) without needing real calibration. **Its effect is only observable while armed** — the firmware code path that computes `alpha`/`gain_sched_alpha` doesn't run at all in STANDBY/ESTOP (only inside `RUNNING`/`JUMPING`), so don't bother checking it via `telem` until the first arm of the session.

3. **Load session state** to see where the last session left off:
   ```python
   import sys; sys.path.insert(0, "software/gui")
   from analysis.tuning_session import load_session
   state = load_session("software/gui/logs/retracted_tuning/session.json")
   print(state.stage, state.best_gains, state.best_fitness, state.step_size, state.trial_in_stage)
   ```
   No file yet → seeds fresh at stage `lqr` with firmware-default gains. This is resumable across conversations; always read it at session start rather than re-deriving progress from `trials.csv` by hand.

4. **First run of a brand-new campaign only**: validate the pipeline before searching anything — arm with the *current* gains completely unchanged (zero perturbation), run the Test Protocol below once, confirm log → download → analyze all work, *then* start proposing real candidates.

## Tool reference

### `robot_ctl.py` — CLI, one command per invocation

Talks to the GUI's local command server (`127.0.0.1:8765`, always running whenever the GUI is up — no separate launch step). Prints JSON to stdout, exits `0` on success / `1` on failure.

```
python robot_ctl.py telem
python robot_ctl.py param_get <name_or_id>              # e.g. lqr_k_pitch_ret, or 0x0424
python robot_ctl.py param_set <name_or_id> <value>
python robot_ctl.py set_mode <STANDBY|RUNNING|ESTOP|...>
python robot_ctl.py log_start [duration_ms]              # 0 = log until log_stop
python robot_ctl.py log_stop
python robot_ctl.py log_list
python robot_ctl.py log_download <file_index>             # saves to software/gui/logs/
python robot_ctl.py motion_set <v_m_s> <omega_rad_s>
python robot_ctl.py motion_release
```

**Important gotcha, confirmed live**: `motion_set` does **not** hold its value indefinitely. A firmware watchdog auto-reverts to radio control (zeroing `v_cmd_ms`/`omega_cmd_rds`) if no GUI command arrives within ~300 ms — this is intentional (a stalled Claude process shouldn't leave a stale setpoint applied forever), but it means a "hold at 0.3 m/s for 3 seconds" step needs `motion_set` **re-sent every ~150-200 ms throughout the hold**, not called once and left alone. A single one-shot `motion_set` will visibly revert to radio-sourced (zero, since the transmitter isn't driving anything during an unattended trial) within a third of a second.

### Analyzing a run

```
python software/gui/tools/analyze_hw_run.py <path.wlog> --stage {lqr,vel_pi,yaw_pi}
```
Prints a JSON metrics/fitness summary to stdout, verdict to stderr. Exit code `0`=safe, `2`=unsafe (safety-maxima violation — never adopt as new best regardless of fitness), `1`=couldn't decode.

Fernando can open the same file in the GUI's **Log Analyzer tab** at any point — it calls the exact same `analysis/wlog_metrics.py` functions this CLI script does, so the numbers on screen are exactly what drove the accept/reject decision, not a re-derivation. Six plot flavors in the combo box: LQR, Vel-PI, Yaw-PI, Torque/Current, Health/Saturation, Gain-schedule check.

### Session state / adaptive search (`analysis/tuning_session.py`)

Pure Python library (no Qt/hardware dependency) — Claude imports and calls it directly, there's no CLI wrapper for it. One hardware trial per call, serialized (physical trials can't be parallelized):

```python
import sys; sys.path.insert(0, "software/gui")
from analysis.tuning_session import (
    load_session, save_session, propose_next, record_trial,
    should_advance_stage, advance_stage, trial_row, append_trial,
)
from analysis.wlog_metrics import evaluate

SESSION = "software/gui/logs/retracted_tuning/session.json"
TRIALS  = "software/gui/logs/retracted_tuning/trials.csv"

state = load_session(SESSION)
candidate = propose_next(state)          # coupled ±step_size% perturbation of state.best_gains, clamped to firmware bounds

# ... set candidate's two params via param_set, run the Test Protocol below,
#     download the resulting .wlog ...

result = evaluate(wlog_path, state.stage)   # decode + metrics + safety + fitness, one call
accepted = record_trial(state, candidate, result["fitness"], result["safety_ok"])  # updates state in place
append_trial(TRIALS, trial_row(state.stage, candidate, result, accepted,
                                state.step_size, state.trial_in_stage,
                                state.trials_since_improvement, wlog_path=str(wlog_path)))
save_session(SESSION, state)

if should_advance_stage(state):
    advance_stage(state)   # -> next stage, reseeded from firmware defaults; None once yaw_pi is done
    save_session(SESSION, state)
```

Accept rule (1/5-success): `safety_ok and fitness < best_fitness` → adopt as new best, grow step ×1.2 (capped at 1.0). Otherwise discard, shrink step ×0.8 (floored at 0.01). A stage ends after 10 trials or 4 consecutive non-improving trials, whichever comes first.

**Note**: `record_trial`/`advance_stage` only update this module's own bookkeeping (`session.json`/`trials.csv`). Actually persisting an accepted candidate's gains to the firmware as the new default (`param_set` + relies on the param's own `PARAM_FLAG_PERSISTENT`) is a separate step — do it explicitly after each accepted trial, and again when advancing a stage (§5 of the original plan: "each stage's tuned result immediately becomes the new persisted default so the next stage starts from an already-improved base").

## Test protocol (one hardware trial)

1. Confirm STANDBY, `alpha_force_ret_en=1` (check every run — not persisted).
2. `param_set` the candidate gain pair for the stage under test.
3. `log_start <duration_ms>` — 15-20 s (10-30 s range: long enough for the arm-transient + several seconds of steady state).
4. Tell Fernando exactly what to do, then wait for him to confirm armed:
   - **LQR stage**: `PARAM_GUI_MOTION_CTRL_EN` stays 0 (don't touch it) — CH2/CH4 centered. *"Ready — arm now, hold ~18s, then disarm or let it auto-settle."*
   - **vel-PI / yaw-PI stage**: right after he arms, drive the step-and-hold sequence via repeated `motion_set` calls (see the re-send gotcha above) — e.g. 0 hold 2s → 0.3 hold 3s → 0 hold 2s → −0.4 hold 3s → 0 hold 2s → 0.5 hold 3s → `motion_release` (~17s total). Timing precision doesn't matter — analysis measures tracking error against the logged setpoint itself, not a nominal script.
5. Poll `telem` for `robot_state` to confirm arming happened, and that `gain_sched_alpha` reads `0.0` throughout (confirms `alpha_force_ret_en` actually took effect this run). Confirm it's back in STANDBY afterward.
6. `log_list` → `log_download` the newest file.
7. `evaluate()` it, report the summary + fitness + accept/reject, `record_trial`/`append_trial`/`save_session`, propose the next candidate.
8. Repeat from step 2, or advance to the next stage.

## Safety

- Existing hard clamps (`PARAM_LQR_TORQUE_LIMIT`, `PARAM_WHEEL_VEL_LIMIT_TURNS_S` + its 2× hard-fault multiplier, `PITCH_WATCHDOG_RAD` 50°, `MAX_HIP_DELTA_RAD`, and the vel-PI/yaw-PI clamp params) stay exactly as configured throughout — the search only ever varies the stage's own Kp/Ki (or K_pitch/K_rate) pair, never loosens a limit to "let a candidate through." They apply identically regardless of whether `v_cmd_ms`/`omega_cmd_rds` came from radio or the GUI motion channel.
- **Hip motors are physically disabled and legs zip-tied for all of Phase 1** — no hip-torque hazard this phase regardless of any parameter or stick.
- **Arming is unconditionally radio-sourced** — `gui_motion_ctrl_en` only ever gates `v_cmd_ms`/`omega_cmd_rds`, never CH10's arm/disarm authority. Fernando holds the arm switch every run and disarms immediately if the response looks wrong.
- The GUI motion channel has its own ~300 ms staleness watchdog (see the re-send gotcha above) — if Claude's process stalls mid-sequence, firmware zeroes the setpoints and reverts to radio control on its own, independent of Fernando noticing.
- Check every run's health flags, currents, and fault code *before* deciding whether to keep a candidate, independent of the fitness score — `evaluate()`'s `safety_ok` already does this; never adopt a candidate where it's `False`.
- State the exact gain values and, for vel-PI/yaw-PI, the full `motion_set` sequence about to be sent, before asking Fernando to arm. He's the safety supervisor throughout.

## Known gotchas (learned during build/verification)

- **`gain_sched_alpha` only reflects reality while armed.** In STANDBY/ESTOP the firmware code path that computes it never runs, so it just sits at `0.0` (the struct's zero-init default) — that's not proof `alpha_force_ret_en` is working, it reads that way regardless. Only trust it during/after an armed `RUNNING` (step 5 of the Test protocol).
- **`motion_set` needs periodic re-sends to hold** — see the CLI reference above. This is the single easiest mistake to make when scripting a vel-PI/yaw-PI hold sequence.
- **Two pre-existing sample logs in `software/gui/logs/`** (`LOG0001.WLOG`, `LOG0002.WLOG`) predate the `TELEM_VERSION` 9→10 bump and no longer decode with the current firmware. Don't use them as fixtures — capture something fresh (`log_start`/`log_download` work fine even unarmed, in STANDBY, if you just need a real V10 sample).
- If a board ever shows `FAULT_WHEEL_INIT_TIMEOUT` (`fault_code=12`) blocking STANDBY: at least one of `wheel_l_enable`/`wheel_r_enable` is on but that ODrive isn't answering CAN. `param_set` both to `0` (they're persistent — the write survives) and `set_mode`/reboot to clear it, if wheel power genuinely isn't available on the bench.

## Reference: param IDs added this phase

| Param name | ID | Persisted? | Purpose |
|---|---|---|---|
| `alpha_force_ret_en` | `0x042A` | No — set every session | Forces `gain_sched_alpha=0.0`, bypassing the calibration-validity check |
| `gui_motion_ctrl_en` | `0x042B` | No — auto-set by `motion_set`, auto-clears via watchdog or `motion_release` | Gates the source of `v_cmd_ms`/`omega_cmd_rds` (GUI vs. radio) |

## File reference

| Path | What |
|---|---|
| `software/gui/tabs/remote_control.py` | The command server (`QTcpServer`, `127.0.0.1:8765`) — starts automatically with the GUI |
| `software/gui/tools/robot_ctl.py` | CLI client — see Tool reference above |
| `software/gui/analysis/wlog_metrics.py` | Shared metrics + fitness module (`decode_wlog`, `compute_metrics`, `evaluate`) |
| `software/gui/tools/analyze_hw_run.py` | Thin CLI wrapper around `wlog_metrics.evaluate()` |
| `software/gui/tabs/log_analyzer_tab.py` | GUI "Log Analyzer" tab — same metrics module, visual cross-check |
| `software/gui/analysis/tuning_session.py` | `session.json`/`trials.csv` + adaptive-step search |
| `software/gui/logs/retracted_tuning/session.json` | Resumable session state |
| `software/gui/logs/retracted_tuning/trials.csv` | Append-only trial history |

## Deferred to Phase 2 (mid/extended leg positions)

Not built or scheduled in Phase 1 — revisit once the retracted anchor's LQR + vel-PI + yaw-PI gains are all converged and safety-clean.

**Add a 3rd LQR scheduling anchor** (retracted / nominal / extended) instead of just the current 2, so there's a real tuned point in the middle (`teensy/lib/ParamRegistry/param_ids.h`, `param_registry.cpp`, `teensy/src/control_loop.cpp`):
- New params `PARAM_LQR_K_PITCH_NOM`, `PARAM_LQR_K_RATE_NOM` (next free IDs: `0x042C`/`0x042D`), persistent, same min/max style as their RET/EXT siblings (`K_PITCH_NOM`: -20..0; `K_RATE_NOM`: -5..0).
- Default value = the current linear-interpolated midpoint of the (by-then Phase-1-tuned) RET/EXT gains — behavior at every `alpha` stays identical to today's 2-point blend until NOM is actually tuned away from that midpoint. Compute the actual numbers from the live RET/EXT values on the device, not guessed.
- `control_loop.cpp`'s gain interpolation becomes piecewise-linear: `alpha ∈ [0, 0.5]` blends RET→NOM, `alpha ∈ [0.5, 1.0]` blends NOM→EXT.

vel-PI and yaw-PI are not height-scheduled today, and this plan doesn't propose making them so. Phase 2's job for those two is *validating* the Phase-1-tuned gains still hold up at nominal/extended (re-tuning only if they clearly don't), not adding new scheduling machinery. Phase 2 also needs real hip calibration back (hips un-zip-tied) and `alpha_force_ret_en` cleared.
