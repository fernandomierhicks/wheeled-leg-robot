"""replay.py — Visual replay of any logged run in MuJoCo viewer.

Loads controller gains and scenario from the CSV, re-runs the exact same
simulation, and shows the MuJoCo viewer with a live matplotlib telemetry panel.

Usage:
    python replay.py                    # replay best run (by fitness)
    python replay.py 42                 # replay run_id = 42
    python replay.py --top 3            # replay 3rd-best run
    python replay.py --list             # list all runs
    python replay.py --resim 42         # re-simulate run 42 and overwrite CSV row
"""
import math
import multiprocessing as mp
import os
import sys
import time
import argparse
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sim_config import (
    ROBOT, Q_NOM, WHEEL_R,
    HIP_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT,
    LEG_K_S, LEG_B_S,
    LQR_Q_PITCH, LQR_Q_PITCH_RATE, LQR_Q_VEL, LQR_R,
    PITCH_NOISE_STD_RAD, PITCH_RATE_NOISE_STD_RAD_S,
    CTRL_STEPS,
)
from physics import build_xml, build_assets, get_equilibrium_pitch
from run_log import (
    CSV_PATH, load_run, load_all_runs, get_best_run,
    list_runs, overwrite_run,
)
import scenarios as _scenarios
from scenarios import (
    init_sim, get_pitch_and_rate, balance_torque, lqr_torque, evaluate,
    FALL_THRESHOLD, BALANCE_DURATION,
    DISTURBANCE_TIME, DISTURBANCE_FORCE, DISTURBANCE_DUR,
)
from lqr_design import compute_gain_table

RENDER_HZ    = 60
TELEMETRY_HZ = 100   # how often to push data to the plot process
WINDOW_S     = 8.0   # rolling time window shown in matplotlib


# ---------------------------------------------------------------------------
# CSV → gains dict
# ---------------------------------------------------------------------------
def _row_to_gains(row: dict) -> dict:
    # For LQR runs, KP/KD may be empty; use defaults
    def safe_float(val, default):
        if val is None or val == '' or val == 'nan':
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    return {
        "KP":     safe_float(row.get("KP"),     60.0),
        "KD":     safe_float(row.get("KD"),      5.0),
        "KP_pos": safe_float(row.get("KP_pos"),  0.30),
        "KP_vel": safe_float(row.get("KP_vel"),  0.30),
    }


def _get_best_run_id(rank: int = 1) -> int:
    rows = load_all_runs()
    pass_rows = []
    for r in rows:
        if r.get("status") == "PASS":
            try:
                pass_rows.append((float(r["fitness"]), int(r["run_id"])))
            except (ValueError, KeyError):
                pass
    if not pass_rows:
        raise ValueError("No PASS runs found in results.csv")
    pass_rows.sort(key=lambda x: x[0])   # ascending fitness (lower = better)
    if rank > len(pass_rows):
        raise ValueError(f"Only {len(pass_rows)} PASS runs exist, cannot get rank {rank}")
    return pass_rows[rank - 1][1]


# ---------------------------------------------------------------------------
# Matplotlib telemetry (separate process)
# ---------------------------------------------------------------------------
def _plot_process(data_q: mp.Queue, cmd_q: mp.Queue,
                  window_s: float, title: str) -> None:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    MAXLEN = int(window_s * TELEMETRY_HZ) + 500
    t_buf, pitch_buf, torque_buf, hip_buf = (deque(maxlen=MAXLEN) for _ in range(4))

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(6, 7))
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(hspace=0.50, top=0.91, bottom=0.13, left=0.18, right=0.96)

    specs = [
        ("Pitch",        "deg",  "#60d0ff"),
        ("Wheel Torque", "N·m",  "#f08040"),
        ("Hip Angle",    "rad",  "#b060ff"),
    ]
    lines = []
    for ax, (ttl, unit, col) in zip(axes, specs):
        ax.set_facecolor("#1e1e2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#555")
        ax.tick_params(colors="lightgray", labelsize=7)
        ax.grid(True, color="#333", linewidth=0.4, linestyle="--")
        ax.set_title(ttl, color="white", fontsize=9, pad=3)
        ax.set_ylabel(unit, color=col, fontsize=8)
        ax.set_xlabel("sim time [s]", color="lightgray", fontsize=7)
        ln, = ax.plot([], [], color=col, linewidth=1.5)
        lines.append(ln)

    # Restart button
    ax_btn = fig.add_axes([0.35, 0.03, 0.30, 0.05])
    btn    = Button(ax_btn, "Restart", color="#3a3a5e", hovercolor="#5a5a9e")
    btn.label.set_color("white"); btn.label.set_fontsize(9)
    btn.on_clicked(lambda _: cmd_q.put_nowait("RESTART"))

    fig.suptitle(title, color="white", fontsize=9)
    fig.show()

    while plt.fignum_exists(fig.number):
        items = []
        while True:
            try: items.append(data_q.get_nowait())
            except Exception: break
        if not items:
            plt.pause(1.0 / 30)
            continue

        for item in items:
            if item is None: return
            if item == "RESET":
                for buf in [t_buf, pitch_buf, torque_buf, hip_buf]:
                    buf.clear()
                for ln in lines: ln.set_data([], [])
                fig.canvas.flush_events()
                continue
            t, p, trq, hip = item
            t_buf.append(t); pitch_buf.append(p)
            torque_buf.append(trq); hip_buf.append(hip)

        if len(t_buf) < 2: continue
        tb     = list(t_buf)
        sim_t  = tb[-1]
        t0     = max(0.0, sim_t - window_s)
        idx    = next((i for i, t in enumerate(tb) if t >= t0), 0)
        tw     = tb[idx:]

        for ln, ax, buf in zip(lines, axes, [pitch_buf, torque_buf, hip_buf]):
            bw = list(buf)[idx:]
            ln.set_data(tw, bw)
            ax.set_xlim(t0, sim_t + 0.5)
            if len(bw) > 1:
                lo, hi = min(bw), max(bw)
                span = max(hi - lo, 0.1)
                ax.set_ylim(lo - span * 0.15, hi + span * 0.15)
        fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Main replay
# ---------------------------------------------------------------------------
def replay(run_id: int = None, baseline: bool = False, scenario_override: str = None,
           freefall: bool = False, slowmo: float = 1.0):
    """Launch MuJoCo viewer for the given run_id (best run if None).

    Pass baseline=True to skip CSV and use the baselined LQR gains from
    sim_config.py directly.  scenario_override sets the scenario regardless
    of what is stored in the CSV row.
    """
    if baseline:
        scenario_default = scenario_override or "balance_disturbance"
        row = dict(
            run_id=0, label="baseline_lqr",
            scenario=scenario_default,
            Q_PITCH=LQR_Q_PITCH, Q_PITCH_RATE=LQR_Q_PITCH_RATE,
            Q_VEL=LQR_Q_VEL, R=LQR_R,
            fitness="(baseline)",
        )
    else:
        if run_id is None:
            run_id = _get_best_run_id(rank=1)
        row = load_run(run_id)

    gains = _row_to_gains(row)
    label = row.get("label", "?")
    scenario = scenario_override or row.get("scenario", "balance")
    logged_fit = row.get("fitness", "?")

    # Load LQR gains from CSV row if present, otherwise fall back to sim_config defaults
    def _safe_float(v, default):
        try: return float(v) if v not in (None, '', 'nan') else default
        except (ValueError, TypeError): return default

    q_pitch      = _safe_float(row.get("Q_PITCH"),      LQR_Q_PITCH)
    q_pitch_rate = _safe_float(row.get("Q_PITCH_RATE"), LQR_Q_PITCH_RATE)
    q_vel        = _safe_float(row.get("Q_VEL"),        LQR_Q_VEL)
    r_val        = _safe_float(row.get("R"),             LQR_R)
    _scenarios.LQR_K_TABLE = compute_gain_table(
        ROBOT, Q_diag=[q_pitch, q_pitch_rate, q_vel], R_val=r_val
    )
    _scenarios.USE_PD_CONTROLLER = False
    use_lqr = True
    print(f"  LQR weights: Q=[{q_pitch:.4g}, {q_pitch_rate:.4g}, {q_vel:.4g}]  R={r_val:.4g}")

    print(f"\nReplaying run {run_id}  [{label}]  scenario={scenario}")
    print(f"  Logged fitness: {logged_fit}")
    print(f"  Gains: {gains}")

    # Build model
    xml    = build_xml()
    assets = build_assets()
    model  = mujoco.MjModel.from_xml_string(xml, assets)
    data   = mujoco.MjData(model)

    # Address lookups
    free_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")
    qpos_free = model.jnt_qposadr[free_id]
    dof_free  = model.jnt_dofadr[free_id]

    def _dof(name): return model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    def _qpos(name): return model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]

    dof_hip_L   = _dof("hip_L");          dof_hip_R   = _dof("hip_R")
    qpos_hip_L  = _qpos("hip_L");         qpos_hip_R  = _qpos("hip_R")
    dof_whl_L   = _dof("wheel_spin_L");   dof_whl_R   = _dof("wheel_spin_R")

    act_hip_L   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act_L")
    act_hip_R   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act_R")
    act_wheel_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_act_L")
    act_wheel_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_act_R")

    PHYSICS_HZ      = round(1.0 / model.opt.timestep)
    steps_per_frame = max(1, PHYSICS_HZ // RENDER_HZ)

    def _init():
        init_sim(model, data)

    _init()

    # Telemetry process
    plot_title = f"run {run_id} | {label} | fit={logged_fit}"
    data_q  = mp.Queue(maxsize=4000)
    cmd_q   = mp.Queue(maxsize=16)
    plot_proc = mp.Process(target=_plot_process,
                           args=(data_q, cmd_q, WINDOW_S, plot_title),
                           daemon=True)
    plot_proc.start()

    # Controller state
    step            = 0
    prev_sim_t      = 0.0
    last_push       = -1.0
    pitch_integral  = 0.0
    odo_x           = 0.0
    rng             = np.random.default_rng(0)
    box_bid         = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    d_pitch         = model.jnt_dofadr[mujoco.mj_name2id(
                          model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")] + 4
    disturbance_applied = False

    # 1-step sensor delay buffer — models ~2 ms BNO086 I2C + CAN latency.
    from physics import get_equilibrium_pitch as _gep
    _pitch_d      = _gep(ROBOT, Q_NOM)
    _pitch_rate_d = 0.0
    _wheel_vel_d  = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 35
        viewer.cam.elevation = -15
        viewer.cam.distance  =  2.5
        viewer.cam.lookat    = np.array([0.0, 0.0, 0.30])

        def _reset_state():
            nonlocal step, prev_sim_t, pitch_integral, odo_x
            nonlocal _pitch_d, _pitch_rate_d, _wheel_vel_d
            _init()
            step = 0; prev_sim_t = 0.0
            pitch_integral = 0.0; odo_x = 0.0
            _pitch_d = _gep(ROBOT, Q_NOM); _pitch_rate_d = 0.0; _wheel_vel_d = 0.0
            if not data_q.full(): data_q.put_nowait("RESET")

        while viewer.is_running():
            frame_start = time.perf_counter()
            sim_t = float(data.time)

            try:
                cmd = cmd_q.get_nowait()
                if cmd == "RESTART": _reset_state()
            except Exception:
                pass
            if sim_t < prev_sim_t - 0.01:
                _reset_state()
            prev_sim_t = sim_t

            # Physics: 2000 Hz.  Controller: 500 Hz = every CTRL_STEPS steps.
            dt = model.opt.timestep * CTRL_STEPS
            for _ in range(steps_per_frame):
                if step % CTRL_STEPS == 0:
                    pitch_true, pitch_rate_true = get_pitch_and_rate(data, box_bid, d_pitch)
                    pitch      = pitch_true      + rng.normal(0, PITCH_NOISE_STD_RAD)
                    pitch_rate = pitch_rate_true + rng.normal(0, PITCH_RATE_NOISE_STD_RAD_S)
                    wheel_vel  = (data.qvel[dof_whl_L] + data.qvel[dof_whl_R]) / 2.0

                    q_hip_avg = (data.qpos[qpos_hip_L] + data.qpos[qpos_hip_R]) / 2.0

                    # Rotate delay buffer (0ms: update before use)
                    _pitch_d, _pitch_rate_d, _wheel_vel_d = pitch, pitch_rate, wheel_vel

                    if freefall:
                        # No controller — all actuators zeroed, robot falls freely
                        data.ctrl[act_wheel_L] = 0.0
                        data.ctrl[act_wheel_R] = 0.0
                        data.ctrl[act_hip_L]   = 0.0
                        data.ctrl[act_hip_R]   = 0.0
                    elif use_lqr:
                        tau_wheel = lqr_torque(_pitch_d, _pitch_rate_d, _wheel_vel_d, q_hip_avg, v_ref=0.0)
                        data.ctrl[act_wheel_L] = tau_wheel
                        data.ctrl[act_wheel_R] = tau_wheel
                        for qpos_hip, dof_hip, act_hip in [
                            (qpos_hip_L, dof_hip_L, act_hip_L),
                            (qpos_hip_R, dof_hip_R, act_hip_R),
                        ]:
                            q_hip   = data.qpos[qpos_hip]
                            dq_hip  = data.qvel[dof_hip]
                            tau_hip = -(LEG_K_S * (q_hip - Q_NOM) + LEG_B_S * dq_hip)
                            data.ctrl[act_hip] = np.clip(tau_hip, -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)
                    else:
                        tau_wheel, pitch_integral, odo_x = balance_torque(
                            _pitch_d, _pitch_rate_d, pitch_integral, odo_x,
                            _wheel_vel_d, q_hip_avg, dt, gains)
                        data.ctrl[act_wheel_L] = tau_wheel
                        data.ctrl[act_wheel_R] = tau_wheel
                        for qpos_hip, dof_hip, act_hip in [
                            (qpos_hip_L, dof_hip_L, act_hip_L),
                            (qpos_hip_R, dof_hip_R, act_hip_R),
                        ]:
                            q_hip   = data.qpos[qpos_hip]
                            dq_hip  = data.qvel[dof_hip]
                            tau_hip = -(LEG_K_S * (q_hip - Q_NOM) + LEG_B_S * dq_hip)
                            data.ctrl[act_hip] = np.clip(tau_hip, -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)

                # Apply disturbance impulse if this is a disturbance scenario
                if scenario in ("balance_disturbance", "lqr_combined"):
                    if (DISTURBANCE_TIME <= data.time < DISTURBANCE_TIME + DISTURBANCE_DUR):
                        data.xfrc_applied[box_bid, 0] = DISTURBANCE_FORCE
                    else:
                        data.xfrc_applied[box_bid, 0] = 0.0

                mujoco.mj_step(model, data)
                step += 1

            viewer.sync()

            # Push telemetry
            wall_now = time.perf_counter()
            if wall_now - last_push >= 1.0 / TELEMETRY_HZ and not data_q.full():
                pitch_true, _ = get_pitch_and_rate(data, box_bid, d_pitch)
                q_hip_avg = (data.qpos[qpos_hip_L] + data.qpos[qpos_hip_R]) / 2.0
                data_q.put_nowait((sim_t,
                                   math.degrees(pitch_true),
                                   float(data.ctrl[act_wheel_L]),
                                   q_hip_avg))
                last_push = wall_now

            # HUD overlay: show run info
            viewer.user_scn.ngeom = 0
            g = viewer.user_scn.geoms[0]
            hud = f"run {run_id} | {label[:20]}"
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                [0.006, 0, 0], [-0.30, 0.15, 0.65],
                                np.eye(3).flatten(),
                                np.array([0.4, 1.0, 0.4, 1.0], dtype=np.float32))
            g.label = hud.encode()[:99]
            viewer.user_scn.ngeom = 1

            elapsed = time.perf_counter() - frame_start
            sleep_t = slowmo / RENDER_HZ - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    data_q.put(None)
    plot_proc.join(timeout=2)


# ---------------------------------------------------------------------------
# Re-simulate and overwrite
# ---------------------------------------------------------------------------
def resim(run_id: int):
    """Re-run the scenario for run_id and overwrite its CSV row."""
    row   = load_run(run_id)
    gains = _row_to_gains(row)
    label = row.get("label", f"resim_{run_id}")
    scenario = row.get("scenario", "balance")
    print(f"Re-simulating run {run_id} [{label}] ...")
    new_row = evaluate(gains, scenario=scenario, label=label,
                       run_id=run_id, csv_path=CSV_PATH)
    overwrite_run(run_id, new_row)
    print(f"Run {run_id} overwritten: fitness={new_row.get('fitness','?')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Replay or re-simulate logged runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python replay.py                 # replay best run
  python replay.py 42              # replay run_id=42
  python replay.py --top 3         # replay 3rd-best run
  python replay.py --list          # list all runs
  python replay.py --resim 42      # re-simulate run 42 and overwrite CSV
""")
    ap.add_argument("run_id", nargs="?", type=int, default=None,
                    help="run_id to replay")
    ap.add_argument("--top",   type=int, default=None,
                    help="Replay Nth-best run by fitness")
    ap.add_argument("--list",  action="store_true",
                    help="Print all runs and exit")
    ap.add_argument("--resim", type=int, default=None,
                    help="Re-simulate run_id and overwrite CSV row")
    ap.add_argument("--baseline", action="store_true",
                    help="Replay baselined LQR gains from sim_config (disturbance scenario)")
    ap.add_argument("--scenario", type=str, default=None,
                    help="Override scenario: balance | balance_disturbance | lqr_combined")
    ap.add_argument("--freefall", action="store_true",
                    help="Seed robot and run with zero torque — no controller active")
    ap.add_argument("--slowmo", type=float, default=1.0,
                    help="Slow-motion factor (e.g. 10 = 10x slower than real-time)")
    args = ap.parse_args()

    if args.list:
        list_runs()
        return

    if args.resim is not None:
        resim(args.resim)
        return

    if args.freefall:
        replay(baseline=True, scenario_override="balance", freefall=True,
               slowmo=args.slowmo)
        return

    if args.baseline:
        replay(baseline=True, scenario_override=args.scenario, slowmo=args.slowmo)
        return

    run_id = args.run_id
    if args.top is not None:
        run_id = _get_best_run_id(rank=args.top)
        print(f"Top-{args.top} run -> run_id={run_id}")

    replay(run_id, scenario_override=args.scenario, slowmo=args.slowmo)


if __name__ == "__main__":
    mp.freeze_support()
    main()
