"""
Femur-only jump simulation — same PD-to-torque state machine as 4bar_jump_sim.
Only the box, AK45-10 hip motor, and femur remain. The femur strikes the ground
to push off instead of a wheel.

Sequence:
  FALLING   — robot dropped from rest, PD holds hip at -0.8 rad
              after 5 s (settled on ground) → transitions to CROUCHING
  CROUCHING — PD tracks linear trajectory from current angle to Q_RETRACTED (-0.58) over 2 s
  JUMPING   — max torque toward Q_EXTENDED (-1.27), torque-speed limited
  LANDING   — PD holds extended, waits for stable femur-ground contact, then repeats

Motor model: AK45-10  7 N·m peak, 75 KV × 24V / 10:1 = 18.85 rad/s no-load
Torque-speed: T = MAX_TORQUE × max(0, 1 − |ω| / OMEGA_MAX)

Run:
    python simulation/mujoco/femur_jump/femur_jump_sim.py
"""

import mujoco
import mujoco.viewer
import multiprocessing as mp
import numpy as np
import time
import os
from collections import deque

XML_PATH = os.path.join(os.path.dirname(__file__), "femur_jump.xml")

RENDER_HZ = 60
WINDOW_S  = 12.0
PUSH_HZ   = 200

HIP_INIT_QHIP = -0.8    # rad — starting position

# ── Control parameters ──────────────────────────────────────────────────────
Q_RETRACTED  = -0.58    # fully crouched (start of jump stroke)
Q_EXTENDED   = -1.27    # fully extended (end of jump stroke)
CROUCH_TIME  =  2.0     # seconds to complete crouch trajectory

KP =  80.0              # PD proportional gain [N·m/rad]
KD =   3.0              # PD derivative gain   [N·m·s/rad]

MAX_TORQUE =  7.0       # AK45-10 peak torque [N·m]  — jump stroke only
HOLD_TORQUE = 7.0       # torque limit for PD position hold
OMEGA_MAX  = 18.85      # no-load speed at 24V, output shaft [rad/s]

FALL_WAIT    = 2.0      # seconds to wait for robot to fall and settle before crouching
STABLE_TIME  = 0.4      # seconds of continuous ground contact before re-crouching

# ── State machine ─────────────────────────────────────────────────────────────
FALLING   = 0
CROUCHING = 1
JUMPING   = 2
LANDING   = 3
DONE      = 4

STATE_NAMES = {FALLING: "FALLING", CROUCHING: "CROUCHING",
               JUMPING: "JUMPING", LANDING: "LANDING", DONE: "DONE"}


# ---------------------------------------------------------------------------
# Plot process
# ---------------------------------------------------------------------------
def _plot_process(q: mp.Queue, window_s: float) -> None:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    MAXLEN = int(window_s * 1000) + 500
    t_buf      = deque(maxlen=MAXLEN)
    hip_buf    = deque(maxlen=MAXLEN)
    torque_buf = deque(maxlen=MAXLEN)
    height_buf = deque(maxlen=MAXLEN)

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(6, 7))
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(hspace=0.50, top=0.91, bottom=0.09, left=0.18, right=0.96)

    spec = [
        ("Hip angle",    "rad",  "#f0c040"),
        ("Motor torque", "N·m",  "#f08040"),
        ("Box height",   "mm",   "#60d0ff"),
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

    title_obj = fig.suptitle("t = 0.000 s  |  FALLING", color="white", fontsize=10)
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

        title_obj.set_text(f"t = {sim_t:.3f} s")
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
        print(f"  Joint[{i}]: {model.joint(i).name:14s}  qposadr={model.jnt_qposadr[i]}")

    def jqpos(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid]

    def jdof(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_dofadr[jid]

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def gid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    s_slide  = jqpos("body_slide")
    s_hip    = jqpos("hip")
    d_hip    = jdof("hip")
    box_bid  = bid("box")

    geom_ground = gid("ground")
    geom_femur  = gid("femur_geom")

    # Simple init: set hip angle, drop from XML default height (box z=0.30)
    # Robot falls freely; state machine waits FALL_WAIT seconds before crouching
    data.qpos[s_hip] = HIP_INIT_QHIP
    data.ctrl[0]     = 0.0
    mujoco.mj_forward(model, data)
    print(f"Init: q_hip={HIP_INIT_QHIP:.4f}  box_z={float(data.xpos[box_bid][2]):.4f} m")
    print(f"Will start crouching at t={FALL_WAIT:.1f} s")

    PHYSICS_HZ      = round(1.0 / model.opt.timestep)
    steps_per_frame = max(1, PHYSICS_HZ // RENDER_HZ)
    push_interval   = 1.0 / PUSH_HZ
    last_push_t     = -1.0
    prev_sim_t      = 0.0

    # State machine
    state           = FALLING
    crouch_start_t  = 0.0
    q_crouch_start  = HIP_INIT_QHIP
    body_z_initial  = float(data.xpos[box_bid][2])
    max_height_m    = 0.0
    grounded_since  = None

    plot_q = mp.Queue(maxsize=4000)
    plot_proc = mp.Process(target=_plot_process, args=(plot_q, WINDOW_S), daemon=True)
    plot_proc.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.sync()

        while viewer.is_running():
            frame_start = time.perf_counter()

            sim_t = float(data.time)

            # Detect viewer reset
            if sim_t < prev_sim_t - 0.01:
                if not plot_q.full():
                    plot_q.put_nowait("RESET")
                last_push_t    = -1.0
                state          = FALLING
                crouch_start_t = 0.0
                max_height_m   = 0.0
                grounded_since = None
            prev_sim_t = sim_t

            # ── Grounded: femur_geom in contact with ground ───────────────
            grounded = any(
                (data.contact[i].geom1 == geom_ground and data.contact[i].geom2 == geom_femur) or
                (data.contact[i].geom1 == geom_femur  and data.contact[i].geom2 == geom_ground)
                for i in range(data.ncon)
            )

            # ── State machine + torque computation ────────────────────────
            hip_q = float(data.qpos[s_hip])
            omega = float(data.qvel[d_hip])

            if state == FALLING:
                torque = KP * (HIP_INIT_QHIP - hip_q) + KD * (0.0 - omega)
                torque = float(np.clip(torque, -HOLD_TORQUE, HOLD_TORQUE))
                if sim_t >= FALL_WAIT:
                    state          = CROUCHING
                    crouch_start_t = sim_t
                    q_crouch_start = hip_q
                    print(f"[{sim_t:.3f}s] CROUCHING  hip={np.degrees(hip_q+np.pi/2):.1f}°")

            elif state == CROUCHING:
                alpha  = min(1.0, (sim_t - crouch_start_t) / CROUCH_TIME)
                q_des  = q_crouch_start + (Q_RETRACTED - q_crouch_start) * alpha
                torque = KP * (q_des - hip_q) + KD * (0.0 - omega)
                torque = float(np.clip(torque, -HOLD_TORQUE, HOLD_TORQUE))
                if alpha >= 1.0:
                    state          = JUMPING
                    body_z_initial = float(data.xpos[box_bid][2])
                    max_height_m   = 0.0
                    print(f"[{sim_t:.3f}s] JUMPING    hip={np.degrees(hip_q+np.pi/2):.1f}°")

            elif state == JUMPING:
                torque = -MAX_TORQUE * max(0.0, 1.0 - abs(omega) / OMEGA_MAX)
                if not grounded or hip_q <= Q_EXTENDED + 0.02:
                    state          = LANDING
                    torque         = 0.0
                    grounded_since = None
                    print(f"[{sim_t:.3f}s] LANDING    hip={np.degrees(hip_q+np.pi/2):.1f}°"
                          f"  grounded={grounded}")

            elif state == LANDING:
                torque = KP * (Q_EXTENDED - hip_q) + KD * (0.0 - omega)
                torque = float(np.clip(torque, -HOLD_TORQUE, HOLD_TORQUE))
                if grounded:
                    if grounded_since is None:
                        grounded_since = sim_t
                    elif sim_t - grounded_since >= STABLE_TIME:
                        state          = CROUCHING
                        crouch_start_t = sim_t
                        q_crouch_start = hip_q
                        grounded_since = None
                        print(f"[{sim_t:.3f}s] CROUCHING  hip={np.degrees(hip_q+np.pi/2):.1f}°  (repeat)")
                else:
                    grounded_since = None

            else:  # DONE
                torque = 0.0

            data.ctrl[0] = torque

            # ── Physics step ──────────────────────────────────────────────
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            viewer.sync()

            sim_t = float(data.time)

            # Height = box CoM rise above its position at jump start
            box_z_now    = float(data.xpos[box_bid][2])
            height_mm    = max(0.0, (box_z_now - body_z_initial) * 1000.0)
            max_height_m = max(max_height_m, box_z_now - body_z_initial)

            if sim_t - last_push_t >= push_interval and not plot_q.full():
                plot_q.put_nowait((sim_t, hip_q, torque, height_mm))
                last_push_t = sim_t

            elapsed = time.perf_counter() - frame_start
            sleep_t = 1.0 / RENDER_HZ - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    print(f"\nMax jump height (box CoM above crouched position): {max_height_m * 1000:.1f} mm")
    plot_q.put(None)
    plot_proc.join(timeout=2)
