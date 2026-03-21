# HANDOFF — Live Scenario Switching for master_sim Visualizer

**Status:** Planned, not yet implemented
**Date:** 2026-03-20

---

## Problem

Switching from Sandbox to S1 (or any scenario) requires closing the entire visualizer and relaunching with different arguments. This is slow and breaks workflow.

## Goal

The `launcher.py` tkinter GUI stays open as a persistent control panel. Clicking any button switches the running visualizer to that scenario live — no manual close/reopen.

---

## Architecture

```
launcher.py (tkinter)                    visualizer.py
┌──────────────────┐   switch_q          ┌─────────────────────────┐
│ [Sandbox]        │──("SWITCH","sandbox")──▶                      │
│ [S1 — LQR...]   │──("SWITCH","s01...")───▶  run_unified()        │
│ [S2 — Leg...]   │                      │    main loop drains     │
│ ...              │   mp.Queue           │    switch_q each frame  │
└──────────────────┘                      └─────────────────────────┘
     tkinter GUI                            MuJoCo + pyqtgraph
     (stays open)                           (single process pair)
```

- Launcher spawns `run_unified()` via `mp.Process` on first button click
- Passes a shared `mp.Queue` (`switch_q`) for commands
- Subsequent button clicks push `("SWITCH", scenario_key)` onto `switch_q`
- If the visualizer process dies, next button click spawns a fresh one

## World Groups

`WorldConfig` is a frozen dataclass — Python auto-generates `__eq__` from all fields. Scenarios with identical world geometry can be switched instantly without rebuilding the MuJoCo model.

| World | Scenarios |
|-------|-----------|
| Default flat floor (20×20m) | S1, S2, S3, S4, S6, S7 |
| Bumps | S5 |
| Terrain obstacles | S8 |
| Sandbox arena (28 obstacles + 6 props, 25×25m) | Sandbox |

**Same-world switch** (e.g. S1→S3): `mj_resetData` + `init_sim` + swap profiles + `ctrl.reset`. Viewer stays open. Instant.

**Different-world switch** (e.g. Sandbox→S1): Rebuild `model`/`data`, close viewer, reopen with new model. Pyqtgraph telemetry window stays alive (just clears buffers via `"RESET"` on `data_q`).

---

## Implementation Steps

### Step 1 — Add `run_unified()` to `viz/visualizer.py`

New function replacing both `replay()` and `sandbox()`:

```python
def run_unified(initial_scenario: str = "sandbox",
                switch_q: mp.Queue = None,
                rng_seed: int = 0):
```

**Outer/inner loop structure:**

```
build model for initial world
launch _plot_process ONCE (pyqtgraph telemetry — survives all switches)

OUTER LOOP (one iteration per world):
    with launch_passive(model, data) as viewer:
        INNER LOOP while viewer.is_running():
            drain switch_q → detect ("SWITCH", name)
            drain cmd_q → handle RESTART, sliders, toggles (same as today)

            if switch to SAME world:
                mj_resetData + init_sim + cfg.init_fn + ctrl.reset
                swap profile fns + controller flags
                send "RESET" + ("TITLE", ...) to data_q
                continue inner loop

            if switch to DIFFERENT world:
                rebuild model, data, ctrl for new world
                send "RESET" + ("TITLE", ...) to data_q
                break inner loop → viewer closes via `with` exit
                outer loop reopens viewer with new model

            run physics frame:
                sandbox (cfg is None): targets from sliders/gamepad, auto-reset on fall
                replay  (cfg is not None): targets from cfg profiles + dist_fn, duration tracked

            push telemetry, camera follow, frame pacing

    if viewer closed by user (not world switch): exit
```

The `with` statement in a loop handles the viewer lifecycle naturally — breaking out exits the context manager (closes viewer), the outer loop re-enters (opens new viewer).

`switch_q` is optional — if `None`, no external switching (backward compat for CLI usage).

### Step 2 — Modify `_plot_process` for title updates

In the `_update()` drain loop, handle new `data_q` message:
- `("TITLE", text)` → `main_win.setWindowTitle(text)`

This keeps the pyqtgraph window title in sync when the launcher triggers a switch.

### Step 3 — Rewrite `launcher.py` as persistent control panel

```python
import multiprocessing as mp

switch_q = mp.Queue(maxsize=8)
viz_proc = None

def on_click(scenario_key):
    global viz_proc
    if viz_proc is None or not viz_proc.is_alive():
        # First click or process died — spawn fresh
        viz_proc = mp.Process(target=run_unified,
                              args=(scenario_key,),
                              kwargs={"switch_q": switch_q})
        viz_proc.start()
    else:
        # Already running — send switch command
        switch_q.put_nowait(("SWITCH", scenario_key))
```

- Same buttons (Sandbox + S1–S8) but now call `on_click(key)` instead of spawning subprocesses
- Launcher imports `run_unified` from `master_sim.viz.visualizer`
- Tkinter stays responsive (it's the main thread, visualizer is a child process)

### Step 4 — Wire up `main()` CLI and backward compat

- `--mode sandbox` → `run_unified("sandbox")`
- `--mode replay --scenario X` → `run_unified(X)`
- Default (no args) → `run_unified("sandbox")`
- Keep `replay()` and `sandbox()` as thin wrappers for external callers

---

## Files to Modify

| File | Changes |
|------|---------|
| `simulation/mujoco/master_sim/viz/visualizer.py` | Add `run_unified()` with outer/inner loop, modify `_plot_process` for TITLE handling, update `main()` |
| `simulation/mujoco/master_sim/launcher.py` | Rewrite to use `mp.Queue` + `mp.Process` instead of `subprocess.Popen` |

No changes needed to `scenarios/base.py` or `sim_loop.py`.

---

## Verification

1. `python launcher.py` → click Sandbox → visualizer spawns, robot balances
2. Click S1 → viewer closes/reopens (different world), S1 runs with pitch step
3. Click S3 → instant reset (same default world), VelPI disturbance runs
4. Click S8 → viewer closes/reopens (terrain), bump scenario runs
5. Click Sandbox → viewer closes/reopens, back to free-drive arena
6. Close MuJoCo viewer → launcher stays open, next click spawns fresh visualizer
