"""
replay_run.py — replay any logged experiment in the MuJoCo GUI.

Usage:
    python simulation/mujoco/4bar_optimization/replay_run.py          # top 5 runs
    python simulation/mujoco/4bar_optimization/replay_run.py 2        # run_id = 2
    python simulation/mujoco/4bar_optimization/replay_run.py --list   # list all runs
    python simulation/mujoco/4bar_optimization/replay_run.py --top 1  # replay best run
"""

import csv
import math
import multiprocessing as mp
import os
import sys
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

# Import geometry / control helpers from eval_jump
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_jump import (
    build_xml, solve_ik,
    KP, KD, MAX_TORQUE, HOLD_TORQUE, OMEGA_MAX,
    JUMP_RAMP_S, JUMP_RAMPDOWN, STABLE_TIME,
    WHEEL_R, GROUNDED_TOL, WHEEL_CLEARANCE,
    CSV_PATH,
    FALLING, CROUCHING, JUMPING, LANDING,
)

RENDER_HZ  = 60
PUSH_HZ    = 200
WINDOW_S   = 12.0
CROUCH_TIME = 0.5  # seconds to complete crouch trajectory
JUMP_SLOWMO = 20   # slow-motion factor during JUMPING + LANDING


# ---------------------------------------------------------------------------
# Load run from CSV
# ---------------------------------------------------------------------------
def load_run(run_id: int | None) -> dict:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No results CSV found at {CSV_PATH}")

    rows = []
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(CSV_PATH, newline="", encoding="latin-1") as f:
            rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("CSV is empty")

    if run_id is None:
        row = rows[-1]
    else:
        matches = [r for r in rows if int(r["run_id"]) == run_id]
        if not matches:
            raise ValueError(f"run_id={run_id} not found in CSV")
        row = matches[0]

    if row["status"] != "PASS":
        raise ValueError(f"run_id={row['run_id']} has status={row['status']}: {row['fail_reason']}")

    # Reconstruct params dict (convert mm/g back to SI)
    p = dict(
        L_femur  = float(row["L_femur_mm"]) / 1000,
        L_stub   = float(row["L_stub_mm"])  / 1000,
        L_tibia  = float(row["L_tibia_mm"]) / 1000,
        Lc       = float(row["Lc_mm"])      / 1000,
        F_X      = float(row["F_X_mm"])     / 1000,
        F_Z      = float(row["F_Z_mm"])     / 1000,
        A_Z      = float(row["A_Z_mm"])     / 1000,
        m_box    = float(row["m_box_g"])    / 1000,
        m_femur  = float(row["m_femur_g"])  / 1000,
        m_tibia  = float(row["m_tibia_g"])  / 1000,
        m_coupler= float(row["m_coupler_g"])/ 1000,
        m_wheel  = float(row["m_wheel_g"])  / 1000,
    )
    Q_ret = float(row["Q_retracted_rad"])
    Q_ext = float(row["Q_extended_rad"])

    return row, p, Q_ret, Q_ext


def get_top_run_id(rank: int = 1) -> int:
    """Finds the run_id of the Nth best run in the CSV."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No results CSV found at {CSV_PATH}")

    rows = []
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(CSV_PATH, newline="", encoding="latin-1") as f:
            rows = list(csv.DictReader(f))

    pass_runs = []
    for r in rows:
        if r.get('status') == 'PASS':
            try:
                h = float(r['jump_height_mm'])
                pass_runs.append((h, int(r['run_id'])))
            except (ValueError, TypeError, KeyError):
                continue

    if not pass_runs:
        raise ValueError("No PASS runs with valid jump height found in CSV.")

    pass_runs.sort(key=lambda x: x[0], reverse=True)

    if len(pass_runs) < rank:
        raise ValueError(f"Cannot find rank {rank} run, only {len(pass_runs)} valid runs exist.")

    return pass_runs[rank - 1][1]


def list_runs():
    if not os.path.exists(CSV_PATH):
        print("No results.csv found.")
        return
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(CSV_PATH, newline="", encoding="latin-1") as f:
            rows = list(csv.DictReader(f))
    print(f"{'ID':>4}  {'Label':<22}  {'Status':<6}  {'Jump mm':>8}  {'Stub mm':>8}  {'Stroke deg':>10}  Timestamp")
    print("-" * 80)
    for r in rows:
        jh = r.get("jump_height_mm", "")
        print(f"{r['run_id']:>4}  {r['label']:<22}  {r['status']:<6}  "
              f"{jh:>8}  {r['L_stub_mm']:>8}  {r.get('stroke_deg',''):>10}  {r['timestamp']}")


# ---------------------------------------------------------------------------
# Matplotlib telemetry (side window)
# ---------------------------------------------------------------------------
def _plot_process(q: mp.Queue, window_s: float, title: str) -> None:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    MAXLEN = int(window_s * 1000) + 500
    t_buf = deque(maxlen=MAXLEN)
    hip_buf = deque(maxlen=MAXLEN)
    torque_buf = deque(maxlen=MAXLEN)
    height_buf = deque(maxlen=MAXLEN)

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(6, 7))
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(hspace=0.50, top=0.91, bottom=0.09, left=0.18, right=0.96)

    spec = [("Hip angle", "rad", "#f0c040"),
            ("Motor torque", "N*m", "#f08040"),
            ("Wheel height", "mm", "#60d0ff")]
    lines_list = []
    for ax, (ttl, unit, col) in zip(axes, spec):
        ax.set_facecolor("#1e1e2e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        ax.tick_params(colors="lightgray", labelsize=7)
        ax.grid(True, color="#333", linewidth=0.4, linestyle="--")
        ax.set_title(ttl, color="white", fontsize=9, pad=3)
        ax.set_ylabel(unit, color=col, fontsize=8)
        ax.set_xlabel("sim time [s]", color="lightgray", fontsize=7)
        ln, = ax.plot([], [], color=col, linewidth=1.5)
        lines_list.append(ln)

    fig.suptitle(title, color="white", fontsize=9)
    fig.show()

    while plt.fignum_exists(fig.number):
        while True:
            try:
                item = q.get_nowait()
            except Exception:
                break
            if item is None:
                return
            if item == "RESET":
                t_buf.clear(); hip_buf.clear()
                torque_buf.clear(); height_buf.clear()
                for ln in lines_list:
                    ln.set_data([], [])
                fig.canvas.flush_events()
                continue
            t_buf.append(item[0]); hip_buf.append(item[1])
            torque_buf.append(item[2]); height_buf.append(item[3])

        if len(t_buf) < 2:
            plt.pause(1.0 / 60)
            continue

        tb    = list(t_buf)
        sim_t = tb[-1]
        bufs  = [list(hip_buf), list(torque_buf), list(height_buf)]
        t0    = max(0.0, sim_t - window_s)
        idx   = next((i for i, t in enumerate(tb) if t >= t0), 0)
        tw    = tb[idx:]

        for ln, ax, buf in zip(lines_list, axes, bufs):
            bw = buf[idx:]
            ln.set_data(tw, bw)
            ax.set_xlim(t0, sim_t + 0.5)
            if len(bw) > 1:
                lo, hi = min(bw), max(bw)
                span = max(hi - lo, 0.01)
                ax.set_ylim(lo - span * 0.2, hi + span * 0.2)

        fig.canvas.flush_events()
        plt.pause(1.0 / 60)


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------
def replay(run_id: int | None = None):
    row, p, Q_ret, Q_ext = load_run(run_id)

    label   = row["label"]
    logged_h = row.get("jump_height_mm", "?")
    print(f"\nReplaying run {row['run_id']}  [{label}]")
    print(f"  L_femur={row['L_femur_mm']} mm  L_stub={row['L_stub_mm']} mm  "
          f"L_tibia={row['L_tibia_mm']} mm  Lc={row['Lc_mm']} mm")
    print(f"  F=({row['F_X_mm']}, {row['F_Z_mm']}) mm  A_Z={row['A_Z_mm']} mm")
    print(f"  Q_ret={Q_ret:.3f} rad  Q_ext={Q_ext:.3f} rad  "
          f"stroke={row['stroke_deg']} deg")
    print(f"  Logged jump height: {logged_h} mm\n")

    xml   = build_xml(p)
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)

    # Override equality anchors
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "4bar_close")
    model.eq_data[eq_id, 0:3] = [-p['Lc'],     0.0,          0.0      ]
    model.eq_data[eq_id, 3:6] = [  0.0,         0.0,    p['L_stub']   ]

    def _jqpos(n): return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def _jdof(n):  return model.jnt_dofadr [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def _bid(n):   return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

    s_slide   = _jqpos("body_slide")
    s_hF      = _jqpos("hinge_F")
    s_hip     = _jqpos("hip")
    s_knee    = _jqpos("knee_joint")
    d_hip     = _jdof("hip")
    wheel_bid = _bid("wheel_asm")

    def _init():
        ik = solve_ik(-0.8, p) or solve_ik(Q_ret, p)
        if ik is None:
            print("WARNING: IK failed at init angle")
            return
        data.qpos[s_hF]   = ik['q_coupler_F']
        data.qpos[s_hip]  = ik['q_hip']
        data.qpos[s_knee] = ik['q_knee']
        data.ctrl[0]      = 0.0
        mujoco.mj_forward(model, data)
        wz = float(data.xpos[wheel_bid][2])
        data.qpos[s_slide] += (WHEEL_R + WHEEL_CLEARANCE) - wz
        mujoco.mj_forward(model, data)
        wz = float(data.xpos[wheel_bid][2])
        print(f"Init: wheel_z={wz:.4f} m  (clearance={(wz-WHEEL_R)*1000:.1f} mm)")

    _init()

    PHYSICS_HZ      = round(1.0 / model.opt.timestep)
    CTRL_HZ         = RENDER_HZ
    steps_per_frame = max(1, PHYSICS_HZ // RENDER_HZ)
    steps_per_ctrl  = max(1, PHYSICS_HZ // CTRL_HZ)
    push_interval_wall = 1.0 / PUSH_HZ   # wall-clock gate for telemetry
    last_push_wall     = -1.0
    prev_sim_t         = 0.0
    ctrl_step_acc      = 0

    state          = FALLING
    crouch_start_t = jump_start_t = 0.0
    q_crouch_start = -0.8
    grounded_since = None
    max_height_m   = 0.0
    torque         = 0.0
    # Latest sensor readings — updated each control tick, read every frame for telemetry
    tel_sim_t = 0.0
    tel_hip_q = 0.0
    tel_wz    = 0.0

    plot_title = f"run {row['run_id']} | {label} | stub={row['L_stub_mm']}mm"
    plot_q     = mp.Queue(maxsize=4000)
    plot_proc  = mp.Process(target=_plot_process,
                             args=(plot_q, WINDOW_S, plot_title), daemon=True)
    plot_proc.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.sync()

        while viewer.is_running():
            frame_start = time.perf_counter()
            sim_t = float(data.time)

            if sim_t < prev_sim_t - 0.01:
                if not plot_q.full():
                    plot_q.put_nowait("RESET")
                last_push_wall = -1.0
                ctrl_step_acc  = 0
                state          = FALLING
                crouch_start_t = jump_start_t = 0.0
                grounded_since = None
                max_height_m   = 0.0
                _init()
            prev_sim_t = sim_t

            # ── Physics step — fewer steps during jump/flight for slow-mo ─
            n_steps = (max(1, steps_per_frame // JUMP_SLOWMO)
                       if state in (JUMPING, LANDING) else steps_per_frame)
            for _ in range(n_steps):
                mujoco.mj_step(model, data)

            ctrl_step_acc += n_steps

            # ── State machine at CTRL_HZ in sim-time ─────────────────────
            if ctrl_step_acc >= steps_per_ctrl:
                ctrl_step_acc -= steps_per_ctrl

                sim_t    = float(data.time)
                hip_q    = float(data.qpos[s_hip])
                omega    = float(data.qvel[d_hip])
                wheel_z  = float(data.xpos[wheel_bid][2])
                grounded = wheel_z < (WHEEL_R + GROUNDED_TOL)

                if state == FALLING:
                    torque = KP * (-0.8 - hip_q) + KD * (0.0 - omega)
                    torque = float(np.clip(torque, -HOLD_TORQUE, HOLD_TORQUE))
                    if grounded:
                        state = CROUCHING; crouch_start_t = sim_t; q_crouch_start = hip_q
                        print(f"[{sim_t:.3f}s] CROUCHING  hip={math.degrees(hip_q+math.pi/2):.1f}°")

                elif state == CROUCHING:
                    alpha_t = min(1.0, (sim_t - crouch_start_t) / CROUCH_TIME)
                    q_des   = q_crouch_start + (Q_ret - q_crouch_start) * alpha_t
                    torque  = KP * (q_des - hip_q) + KD * (0.0 - omega)
                    torque  = float(np.clip(torque, -HOLD_TORQUE, HOLD_TORQUE))
                    if alpha_t >= 1.0:
                        state = JUMPING; jump_start_t = sim_t
                        print(f"[{sim_t:.3f}s] JUMPING    hip={math.degrees(hip_q+math.pi/2):.1f}°")

                elif state == JUMPING:
                    ramp_in  = min(1.0, (sim_t - jump_start_t) / JUMP_RAMP_S)
                    ramp_out = min(1.0, max(0.0, (hip_q - Q_ext) / JUMP_RAMPDOWN))
                    torque   = -MAX_TORQUE * ramp_in * ramp_out * max(0.0, 1.0 - abs(omega) / OMEGA_MAX)
                    if not grounded or hip_q <= Q_ext + 0.02:
                        state = LANDING; torque = 0.0; grounded_since = None
                        print(f"[{sim_t:.3f}s] LANDING    hip={math.degrees(hip_q+math.pi/2):.1f}°")

                elif state == LANDING:
                    torque = KP * (Q_ext - hip_q) + KD * (0.0 - omega)
                    torque = float(np.clip(torque, -HOLD_TORQUE, HOLD_TORQUE))
                    if grounded:
                        if grounded_since is None:
                            grounded_since = sim_t
                        elif sim_t - grounded_since >= STABLE_TIME:
                            state = CROUCHING; crouch_start_t = sim_t; q_crouch_start = hip_q
                            grounded_since = None
                            print(f"[{sim_t:.3f}s] CROUCHING  (repeat)")
                    else:
                        grounded_since = None

                data.ctrl[0] = torque

                # Cache latest readings for per-frame telemetry push below
                tel_sim_t = float(data.time)
                tel_hip_q = hip_q
                tel_wz    = float(data.xpos[wheel_bid][2])

            viewer.sync()

            # ── Telemetry — push every render frame (wall-clock gated) ───
            # Using wall-clock rate so chart updates smoothly during slow-mo
            wall_now = time.perf_counter()
            if wall_now - last_push_wall >= push_interval_wall and not plot_q.full():
                plot_q.put_nowait((tel_sim_t, tel_hip_q, torque,
                                   max(0.0, (tel_wz - WHEEL_R) * 1000.0)))
                last_push_wall = wall_now

            # ── Time-factor label in MuJoCo scene ────────────────────────
            slow_now   = state in (JUMPING, LANDING)
            label_text = f"1/{JUMP_SLOWMO}x time" if slow_now else "1x time"
            label_rgba = ([1.0, 0.55, 0.0, 1.0] if slow_now else [0.4, 1.0, 0.4, 1.0])
            viewer.user_scn.ngeom = 0
            g = viewer.user_scn.geoms[0]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.008, 0.008, 0.008]),
                np.array([-0.25, 0.15, 0.60]),
                np.eye(3).flatten().astype(np.float32),
                np.array(label_rgba, dtype=np.float32),
            )
            g.label = label_text.encode()[:99]
            viewer.user_scn.ngeom = 1

            # Height tracking (every frame for accurate peak)
            wheel_z_now  = float(data.xpos[wheel_bid][2])
            max_height_m = max(max_height_m, wheel_z_now - WHEEL_R)

            elapsed = time.perf_counter() - frame_start
            sleep_t = 1.0 / RENDER_HZ - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    print(f"\nMax jump height (wheel contact patch): {max_height_m * 1000:.1f} mm  "
          f"(logged: {logged_h} mm)")
    plot_q.put(None)
    plot_proc.join(timeout=2)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            # No arguments, replay top 5
            print("--- Replaying top 5 runs by jump height ---")
            for rank in range(1, 6):
                try:
                    top_id = get_top_run_id(rank)
                    print(f"\n--- Replaying rank {rank} run (id={top_id}) ---")
                    replay(top_id)
                except ValueError as e:
                    print(f"Could not replay rank {rank}: {e}")
                    break # Stop if we run out of runs
        elif sys.argv[1] == "--list":
            list_runs()
        elif sys.argv[1] == "--top":
            rank = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            top_id = get_top_run_id(rank)
            print(f"--- Finding top run (rank {rank}) -> Found run_id {top_id} ---")
            replay(top_id)
        else:
            rid = int(sys.argv[1])
            replay(rid)
    except (ValueError, FileNotFoundError, IndexError) as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Usage: replay_run.py [run_id | --list | --top [rank]]", file=sys.stderr)
        sys.exit(1)
