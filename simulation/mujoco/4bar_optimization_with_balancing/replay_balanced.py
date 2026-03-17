"""
replay_balanced.py — replay any logged balanced experiment in the MuJoCo GUI.

Usage:
    python simulation/mujoco/4bar_optimization_with_balancing/replay_balanced.py          # top 5 runs
    python simulation/mujoco/4bar_optimization_with_balancing/replay_balanced.py 2        # run_id = 2
    python simulation/mujoco/4bar_optimization_with_balancing/replay_balanced.py --list   # list all runs
    python simulation/mujoco/4bar_optimization_with_balancing/replay_balanced.py --top 1  # replay best run
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

# Import from eval_jump_balanced
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_jump_balanced import (
    build_xml, solve_ik, get_equilibrium_pitch, DEFAULT, CSV_PATH,
    PITCH_KP, PITCH_KI, PITCH_KD, POSITION_KP, VELOCITY_KP, MAX_PITCH_CMD,
    WHEEL_TORQUE_LIMIT, HIP_KP, HIP_KD, HIP_TORQUE_LIMIT, OMEGA_MAX,
    JUMP_RAMP_S, JUMP_RAMPDOWN, CROUCH_DURATION_S,
    ACCEL_NOISE_STD, PITCH_NOISE_STD_RAD, PITCH_RATE_NOISE_STD_RAD_S,
    WHEEL_R
)

RENDER_HZ   = 60
PUSH_HZ     = 200
WINDOW_S    = 12.0
JUMP_SLOWMO = 10


# ---------------------------------------------------------------------------
# Load run from CSV
# ---------------------------------------------------------------------------
def load_run(run_id: int | None) -> dict:
    if not os.path.exists(CSV_PATH): raise FileNotFoundError(f"No results CSV found at {CSV_PATH}")
    rows = []
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(CSV_PATH, newline="", encoding="latin-1") as f: rows = list(csv.DictReader(f))
    if not rows: raise ValueError("CSV is empty")

    if run_id is None: row = rows[-1]
    else:
        matches = [r for r in rows if int(r["run_id"]) == run_id]
        if not matches: raise ValueError(f"run_id={run_id} not found in CSV")
        row = matches[0]

    if row["status"] != "PASS": raise ValueError(f"run_id={row['run_id']} has status={row['status']}: {row['fail_reason']}")

    p = dict(
        L_femur=float(row["L_femur_mm"])/1000, L_stub=float(row["L_stub_mm"])/1000,
        L_tibia=float(row["L_tibia_mm"])/1000, Lc=float(row["Lc_mm"])/1000,
        F_X=float(row["F_X_mm"])/1000, F_Z=float(row["F_Z_mm"])/1000,
        A_Z=float(row["A_Z_mm"])/1000, m_box=float(row["m_box_g"])/1000,
        m_femur=DEFAULT['m_femur'], m_tibia=DEFAULT['m_tibia'],
        m_coupler=DEFAULT['m_coupler'], m_wheel=DEFAULT['m_wheel'],
    )
    Q_ret, Q_ext = float(row["Q_retracted_rad"]), float(row["Q_extended_rad"])
    return row, p, Q_ret, Q_ext

def get_top_run_id(rank: int = 1) -> int:
    if not os.path.exists(CSV_PATH): raise FileNotFoundError(f"No results CSV found at {CSV_PATH}")
    rows = []
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(CSV_PATH, newline="", encoding="latin-1") as f: rows = list(csv.DictReader(f))
    pass_runs = []
    for r in rows:
        if r.get('status') == 'PASS':
            try: pass_runs.append((float(r['jump_height_mm']), int(r['run_id'])))
            except (ValueError, TypeError, KeyError): continue
    if not pass_runs: raise ValueError("No PASS runs with valid jump height found.")
    pass_runs.sort(key=lambda x: x[0], reverse=True)
    if len(pass_runs) < rank: raise ValueError(f"Cannot find rank {rank} run, only {len(pass_runs)} valid runs exist.")
    return pass_runs[rank - 1][1]

def list_runs():
    if not os.path.exists(CSV_PATH): print("No results_balanced.csv found."); return
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(CSV_PATH, newline="", encoding="latin-1") as f: rows = list(csv.DictReader(f))
    print(f"{'ID':>4}  {'Label':<22}  {'Status':<6}  {'Jump mm':>8}  {'Stub mm':>8}  {'Stroke deg':>10}  Timestamp")
    print("-" * 80)
    for r in rows:
        jh = r.get("jump_height_mm", "")
        print(f"{r['run_id']:>4}  {r['label']:<22}  {r['status']:<6}  "
              f"{jh:>8}  {r.get('L_stub_mm',''):>8}  {r.get('stroke_deg',''):>10}  {r['timestamp']}")


# ---------------------------------------------------------------------------
# Matplotlib telemetry
# ---------------------------------------------------------------------------
def _plot_process(q: mp.Queue, cmd_q: mp.Queue, window_s: float, title: str) -> None:
    import matplotlib; matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    MAXLEN = int(window_s * 1000) + 500
    t_buf, pitch_buf, torque_buf, height_buf = (deque(maxlen=MAXLEN) for _ in range(4))

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(6, 7))
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(hspace=0.50, top=0.91, bottom=0.13, left=0.18, right=0.96)

    spec = [("Pitch", "deg", "#60d0ff"), ("Wheel Torque", "N·m", "#f08040"), ("Jump Height", "mm", "#60ff60")]
    lines = []
    for ax, (ttl, unit, col) in zip(axes, spec):
        ax.set_facecolor("#1e1e2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#555")
        ax.tick_params(colors="lightgray", labelsize=7)
        ax.grid(True, color="#333", linewidth=0.4, linestyle="--")
        ax.set_title(ttl, color="white", fontsize=9, pad=3)
        ax.set_ylabel(unit, color=col, fontsize=8)
        ax.set_xlabel("sim time [s]", color="lightgray", fontsize=7)
        ln, = ax.plot([], [], color=col, linewidth=1.5); lines.append(ln)

    ax_btn = fig.add_axes([0.35, 0.03, 0.30, 0.05])
    btn = Button(ax_btn, "Restart", color="#3a3a5e", hovercolor="#5a5a9e")
    btn.label.set_color("white"); btn.label.set_fontsize(9)
    btn.on_clicked(lambda _: cmd_q.put_nowait("RESTART"))

    fig.suptitle(title, color="white", fontsize=9); fig.show()

    while plt.fignum_exists(fig.number):
        items = []
        while True:
            try: items.append(q.get_nowait())
            except Exception: break
        if not items: plt.pause(1.0 / 60); continue

        for item in items:
            if item is None: return
            if item == "RESET":
                for buf in [t_buf, pitch_buf, torque_buf, height_buf]: buf.clear()
                for ln in lines: ln.set_data([], [])
                fig.canvas.flush_events(); continue
            t, p, trq, h = item
            t_buf.append(t); pitch_buf.append(p); torque_buf.append(trq); height_buf.append(h)

        if len(t_buf) < 2: continue
        tb, sim_t = list(t_buf), t_buf[-1]
        t0 = max(0.0, sim_t - window_s)
        idx = next((i for i, t in enumerate(tb) if t >= t0), 0)
        tw = tb[idx:]

        for ln, ax, buf in zip(lines, axes, [pitch_buf, torque_buf, height_buf]):
            bw = list(buf)[idx:]
            ln.set_data(tw, bw)
            ax.set_xlim(t0, sim_t + 0.5)
            if len(bw) > 1:
                lo, hi = min(bw), max(bw)
                span = max(hi - lo, 0.05 * (1 if buf is not height_buf else 10))
                ax.set_ylim(lo - span * 0.2, hi + span * 0.2)
        fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------
def replay(run_id: int | None = None):
    row, p, Q_ret, Q_ext = load_run(run_id)
    label, logged_h = row["label"], row.get("jump_height_mm", "?")
    print(f"\nReplaying run {row['run_id']} [{label}]\n  Logged jump height: {logged_h} mm\n")

    xml = build_xml(p)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    Q_neutral = Q_ret + 0.30 * (Q_ext - Q_ret)

    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "4bar_close")
    model.eq_data[eq_id, 0:3] = [-p['Lc'], 0.0, 0.0]
    model.eq_data[eq_id, 3:6] = [0.0, 0.0, p['L_stub']]

    s_pitch, s_hip, s_root_z, s_hF, s_knee = (model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ["root_pitch", "hip", "root_z", "hinge_F", "knee_joint"])
    d_pitch, d_hip, d_wheel = (model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ["root_pitch", "hip", "wheel_spin"])
    wheel_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_asm")

    def _init():
        mujoco.mj_resetData(model, data)
        ik = solve_ik(Q_neutral, p)
        if not ik: raise RuntimeError("IK failed at initial neutral angle")
        data.qpos[s_hF], data.qpos[s_hip], data.qpos[s_knee] = ik['q_coupler_F'], ik['q_hip'], ik['q_knee']
        mujoco.mj_forward(model, data)
        data.qpos[s_root_z] += WHEEL_R - data.xpos[wheel_bid][2]
        # Seed balance
        data.qpos[s_pitch] = get_equilibrium_pitch(p, Q_neutral)
        mujoco.mj_forward(model, data)

    _init()

    PHYSICS_HZ = round(1.0 / model.opt.timestep)
    steps_per_frame = max(1, int(PHYSICS_HZ / RENDER_HZ))
    last_push_wall, prev_sim_t = -1.0, 0.0

    plot_title = f"run {row['run_id']} | {label} | stub={row['L_stub_mm']}mm"
    plot_q = mp.Queue(maxsize=4000)
    cmd_q  = mp.Queue(maxsize=16)
    plot_proc = mp.Process(target=_plot_process, args=(plot_q, cmd_q, WINDOW_S, plot_title), daemon=True)
    plot_proc.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 270, 0, 2.0
        viewer.cam.lookat = np.array([0.0, 0.0, 0.4])

        # Controller state
        pitch_integral, odo_x, grounded = 0.0, 0.0, True
        leg_state, current_hip_target = "NEUTRAL", Q_neutral
        jump_triggered, jump_start_t, crouch_start_t, max_height_m = False, 0.0, 0.0, 0.0
        was_airborne, land_t = False, -999.0
        slowmo_active = False

        def _reset_state():
            nonlocal pitch_integral, odo_x, grounded, leg_state, current_hip_target
            nonlocal jump_triggered, jump_start_t, crouch_start_t, max_height_m
            nonlocal was_airborne, land_t, slowmo_active
            _init()
            pitch_integral, odo_x, grounded = 0.0, 0.0, True
            leg_state, current_hip_target = "NEUTRAL", Q_neutral
            jump_triggered, jump_start_t, crouch_start_t, max_height_m = False, 0.0, 0.0, 0.0
            was_airborne, land_t = False, -999.0
            slowmo_active = False
            if not plot_q.full(): plot_q.put_nowait("RESET")

        while viewer.is_running():
            frame_start = time.perf_counter()
            sim_t = float(data.time)

            # MuJoCo viewer reset (ctrl+R) or matplotlib Restart button
            try:
                cmd = cmd_q.get_nowait()
                if cmd == "RESTART": _reset_state()
            except Exception: pass
            if sim_t < prev_sim_t - 0.01:
                _reset_state()
            prev_sim_t = sim_t

            # Slowmo: on from CROUCH start, off 0.5 s after landing
            if leg_state == "JUMP" and not slowmo_active:
                slowmo_active = True
            if slowmo_active and land_t > jump_start_t and sim_t > land_t + 0.5:
                slowmo_active = False
            slow_now = slowmo_active
            n_steps = max(1, steps_per_frame // JUMP_SLOWMO) if slow_now else steps_per_frame

            for _ in range(n_steps):
                sim_t = data.time
                pitch_true, pitch_rate_true = data.qpos[s_pitch], data.qvel[d_pitch]
                pitch = pitch_true + np.random.normal(0, PITCH_NOISE_STD_RAD)
                pitch_rate = pitch_rate_true + np.random.normal(0, PITCH_RATE_NOISE_STD_RAD_S)
                wheel_vel = data.qvel[d_wheel]

                accel_mag = np.linalg.norm(data.sensor("accel").data + np.random.normal(0, ACCEL_NOISE_STD, 3))
                if accel_mag < 3.0: grounded = False
                elif accel_mag > 7.0: grounded = True

                # Landing detection
                if jump_triggered and not grounded: was_airborne = True
                if jump_triggered and was_airborne and grounded and land_t < jump_start_t:
                    land_t = sim_t

                # Feedforward balance pitch
                pitch_ff = get_equilibrium_pitch(p, data.qpos[s_hip])

                target_pitch = 0.0
                if grounded:
                    vel_est = (wheel_vel + pitch_rate) * WHEEL_R
                    odo_x += vel_est * model.opt.timestep
                    pitch_fb = np.clip(-(POSITION_KP * odo_x + VELOCITY_KP * vel_est), -MAX_PITCH_CMD, MAX_PITCH_CMD)
                    target_pitch = pitch_ff + pitch_fb
                else:
                    pitch_integral = 0.0

                pitch_error = pitch - target_pitch
                pitch_integral = np.clip(pitch_integral + pitch_error * model.opt.timestep, -1.0, 1.0)
                u_bal = (PITCH_KP * pitch_error + PITCH_KI * pitch_integral + PITCH_KD * pitch_rate)
                data.ctrl[1] = np.clip(u_bal, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)

                # Sequence: Neutral(0-2) -> Crouch(2-3.5) -> Jump(3.5)
                if sim_t > 2.0 and leg_state == "NEUTRAL" and not jump_triggered:
                    leg_state, crouch_start_t = "CROUCH", sim_t
                
                if sim_t > 3.5 and leg_state == "CROUCH" and not jump_triggered:
                    leg_state, jump_start_t, jump_triggered = "JUMP", sim_t, True

                hip_q, hip_omega = data.qpos[s_hip], data.qvel[d_hip]
                if leg_state == "JUMP":
                    ramp_in = min(1.0, (sim_t - jump_start_t) / JUMP_RAMP_S)
                    ramp_out = min(1.0, max(0.0, (hip_q - Q_ext) / JUMP_RAMPDOWN))
                    speed_scale = max(0.0, 1.0 - abs(hip_omega) / OMEGA_MAX)
                    u_hip = -HIP_TORQUE_LIMIT * ramp_in * ramp_out * speed_scale
                    if (hip_q <= Q_ext + 0.05) or (not grounded and (sim_t - jump_start_t) > 0.05):
                        leg_state, current_hip_target = "NEUTRAL", Q_neutral
                else:
                    if leg_state == "CROUCH":
                        crouch_frac = min(1.0, (sim_t - crouch_start_t) / CROUCH_DURATION_S)
                        current_hip_target = Q_neutral + crouch_frac * (Q_ret - Q_neutral)
                    u_hip = HIP_KP * (current_hip_target - hip_q) - HIP_KD * hip_omega
                data.ctrl[0] = np.clip(u_hip, -HIP_TORQUE_LIMIT, HIP_TORQUE_LIMIT)

                mujoco.mj_step(model, data)

            viewer.sync()

            wall_now = time.perf_counter()
            if wall_now - last_push_wall >= (1.0/PUSH_HZ) and not plot_q.full():
                height_mm = max(0.0, (data.xpos[wheel_bid][2] - WHEEL_R) * 1000.0)
                plot_q.put_nowait((sim_t, math.degrees(data.qpos[s_pitch]), data.ctrl[1], height_mm))
                last_push_wall = wall_now

            viewer.user_scn.ngeom = 0
            g = viewer.user_scn.geoms[0]
            label_text = f"1/{JUMP_SLOWMO}x" if slow_now else "1x"
            label_rgba = [1.0, 0.55, 0.0, 1.0] if slow_now else [0.4, 1.0, 0.4, 1.0]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008,0,0], [-0.25, 0.15, 0.60], np.eye(3).flatten(), np.array(label_rgba, dtype=np.float32))
            g.label = label_text.encode()[:99]
            viewer.user_scn.ngeom = 1

            max_height_m = max(max_height_m, data.xpos[wheel_bid][2] - WHEEL_R)

            elapsed = time.perf_counter() - frame_start
            if (sleep_t := 1.0 / RENDER_HZ - elapsed) > 0: time.sleep(sleep_t)

    print(f"\nMax jump height (wheel contact patch): {max_height_m * 1000:.1f} mm (logged: {logged_h} mm)")
    plot_q.put(None); plot_proc.join(timeout=2)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            print("--- Replaying top 5 runs by jump height ---")
            for rank in range(1, 6):
                try:
                    top_id = get_top_run_id(rank)
                    print(f"\n--- Replaying rank {rank} run (id={top_id}) ---")
                    replay(top_id)
                except ValueError as e:
                    print(f"Could not replay rank {rank}: {e}"); break
        elif sys.argv[1] == "--list": list_runs()
        elif sys.argv[1] == "--top":
            rank = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            top_id = get_top_run_id(rank)
            print(f"--- Finding top run (rank {rank}) -> Found run_id {top_id} ---")
            replay(top_id)
        else:
            replay(int(sys.argv[1]))
    except (ValueError, FileNotFoundError, IndexError) as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Usage: replay_balanced.py [run_id | --list | --top [rank]]", file=sys.stderr)
        sys.exit(1)