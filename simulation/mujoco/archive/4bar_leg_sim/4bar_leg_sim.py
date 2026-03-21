"""
MuJoCo 4-bar leg viewer — interactive slider + matplotlib telemetry.

Use the MuJoCo viewer Controls panel slider to move the hip joint.
Matplotlib shows hip angle, knee angle, and wheel spin speed vs time.

Run:
    python simulation/mujoco/4bar_leg_sim/4bar_leg_sim.py
"""

import mujoco
import mujoco.viewer
import multiprocessing as mp
import numpy as np
import time
import os
from collections import deque

XML_PATH = os.path.join(os.path.dirname(__file__), "4bar_leg.xml")

RENDER_HZ = 60
WINDOW_S  = 8.0    # seconds of history shown in plots
PUSH_HZ   = 200    # telemetry push rate

# ── 4-bar geometry (MuJoCo box-frame XZ coordinates) ───────────────────────
L_femur = 0.100; L_stub = 0.015; L_tibia = 0.115; Lc = 0.110
A_Z = -0.0235
F_X = -0.015
F_Z =  0.0025

HIP_INIT_QHIP = -0.8   # rad — initial slider position


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def direct_ik(q_hip):
    C_x = -L_femur * np.cos(q_hip)
    C_z =  A_Z + L_femur * np.sin(q_hip)
    dx = C_x - F_X
    dz = C_z - F_Z
    K = (Lc**2 - dx**2 - dz**2 - L_stub**2) / (2.0 * L_stub)
    R = np.sqrt(dx**2 + dz**2)
    if abs(K) > R:
        return None
    phi      = np.arctan2(dz, dx)
    asin_val = np.arcsin(np.clip(K / R, -1.0, 1.0))
    alpha1 = _wrap(asin_val - phi)
    alpha2 = _wrap(np.pi - asin_val - phi)
    q_knee1 = alpha1 - q_hip
    q_knee2 = alpha2 - q_hip
    if abs(q_knee1) <= abs(q_knee2):
        alpha, q_knee = alpha1, q_knee1
    else:
        alpha, q_knee = alpha2, q_knee2
    E_x = C_x + L_stub * np.sin(alpha)
    E_z = C_z + L_stub * np.cos(alpha)
    q_coupler_F = np.arctan2(E_z - F_Z, F_X - E_x)
    return q_hip, q_knee, q_coupler_F


# ---------------------------------------------------------------------------
# Plot process
# ---------------------------------------------------------------------------
def _plot_process(q: mp.Queue, window_s: float) -> None:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    MAXLEN = int(window_s * 1000) + 500
    t_buf    = deque(maxlen=MAXLEN)
    hip_buf  = deque(maxlen=MAXLEN)
    knee_buf = deque(maxlen=MAXLEN)
    wspd_buf = deque(maxlen=MAXLEN)

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(6, 7))
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(hspace=0.50, top=0.91, bottom=0.09, left=0.18, right=0.96)

    spec = [
        ("Hip angle",         "rad",   "#f0c040"),
        ("Knee angle",        "rad",   "#c084f5"),
        ("Wheel spin speed",  "rad/s", "#60d0ff"),
    ]
    lines_list = []
    for ax, (title, unit, col) in zip(axes, spec):
        ax.set_facecolor("#1e1e2e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        ax.tick_params(colors="lightgray", labelsize=7)
        ax.grid(True, color="#333", linewidth=0.4, linestyle="--")
        ax.set_title(title, color="white", fontsize=9, pad=3)
        ax.set_ylabel(unit, color=col, fontsize=8)
        ax.set_xlabel("sim time [s]", color="lightgray", fontsize=7)
        ln, = ax.plot([], [], color=col, linewidth=1.5)
        lines_list.append(ln)

    fig.suptitle("t = 0.000 s", color="white", fontsize=10)
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
                t_buf.clear(); hip_buf.clear(); knee_buf.clear(); wspd_buf.clear()
                for ln in lines_list:
                    ln.set_data([], [])
                fig.canvas.flush_events()
                continue
            t_buf.append(item[0]); hip_buf.append(item[1])
            knee_buf.append(item[2]); wspd_buf.append(item[3])

        if len(t_buf) < 2:
            plt.pause(1.0 / 60)
            continue

        tb    = list(t_buf)
        sim_t = tb[-1]
        bufs  = [list(hip_buf), list(knee_buf), list(wspd_buf)]
        t0    = max(0.0, sim_t - window_s)
        idx   = next((i for i, t in enumerate(tb) if t >= t0), 0)
        tw    = tb[idx:]

        for ln, ax, buf in zip(lines_list, axes, bufs):
            bw = buf[idx:]
            ln.set_data(tw, bw)
            ax.set_xlim(t0, sim_t + 0.2)
            if len(bw) > 1:
                lo, hi = min(bw), max(bw)
                span = max(hi - lo, 0.01)
                ax.set_ylim(lo - span * 0.2, hi + span * 0.2)

        fig.suptitle(f"t = {sim_t:.3f} s", color="white", fontsize=10)
        fig.canvas.flush_events()
        plt.pause(1.0 / 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    print(f"Model: {model.nbody} bodies, {model.njnt} joints")
    for i in range(model.njnt):
        print(f"  Joint[{i}]: {model.joint(i).name:12s}  damping={model.dof_damping[i]:.4f}")

    PHYSICS_HZ      = round(1.0 / model.opt.timestep)
    steps_per_frame = max(1, PHYSICS_HZ // RENDER_HZ)
    push_interval   = 1.0 / PUSH_HZ
    last_push_t     = -1.0
    prev_sim_t      = 0.0

    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "4bar_close")
    model.eq_data[eq_id, 0:3] = [-Lc, 0.0,    0.0   ]
    model.eq_data[eq_id, 3:6] = [ 0.0, 0.0, L_stub]

    ik = direct_ik(HIP_INIT_QHIP)
    if ik is not None:
        q_hip, q_knee, q_coupler_F = ik
        data.qpos[0] = q_coupler_F
        data.qpos[1] = q_hip
        data.qpos[2] = q_knee
        data.ctrl[0] = q_hip
        print(f"Init: q_hip={q_hip:.4f}  q_knee={q_knee:.5f}  q_coupler_F={q_coupler_F:.4f}")
    else:
        print("WARNING: IK failed")

    mujoco.mj_forward(model, data)

    plot_q = mp.Queue(maxsize=4000)
    plot_proc = mp.Process(target=_plot_process, args=(plot_q, WINDOW_S), daemon=True)
    plot_proc.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)  # equivalent to clicking Align
        viewer.sync()
        while viewer.is_running():
            frame_start = time.perf_counter()

            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            viewer.sync()

            sim_t = float(data.time)

            if sim_t < prev_sim_t - 0.01:
                if not plot_q.full():
                    plot_q.put_nowait("RESET")
                last_push_t = -1.0
            prev_sim_t = sim_t

            if sim_t - last_push_t >= push_interval and not plot_q.full():
                plot_q.put_nowait((
                    sim_t,
                    float(data.qpos[1]),   # hip angle [rad]
                    float(data.qpos[2]),   # knee angle [rad]
                    float(data.qvel[3]),   # wheel spin [rad/s]
                ))
                last_push_t = sim_t

            elapsed = time.perf_counter() - frame_start
            sleep_t = 1.0 / RENDER_HZ - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    plot_q.put(None)
    plot_proc.join(timeout=2)
