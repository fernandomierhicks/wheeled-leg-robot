"""
MuJoCo 4-bar jump simulation — PD-to-torque hip control.

Sequence:
  FALLING   — robot dropped from rest, PD holds hip at -0.8 rad
  CROUCHING — once wheel touches ground, PD tracks linear trajectory
              from current hip angle to Q_RETRACTED (-0.58) over 2 s
  JUMPING   — max torque toward Q_EXTENDED (-1.27), torque-speed limited
  DONE      — motor off after takeoff or full extension reached

Motor model: AK45-10  7 N·m peak, 75 KV × 24V / 10:1 = 18.85 rad/s no-load
Torque-speed: T = MAX_TORQUE × max(0, 1 − |ω| / OMEGA_MAX)

Run:
    python simulation/mujoco/4bar_optimization/4bar_jump_sim.py
"""

import mujoco
import mujoco.viewer
import multiprocessing as mp
import numpy as np
import time
import os
from collections import deque

XML_PATH = os.path.join(os.path.dirname(__file__), "4bar_jump.xml")

RENDER_HZ = 60
WINDOW_S  = 12.0   # enough to see full sequence
PUSH_HZ   = 200

# ── 4-bar geometry ──────────────────────────────────────────────────────────
L_femur = 0.100; L_stub = 0.025; L_tibia = 0.115; Lc = 0.110
A_Z = -0.0235; F_X = -0.020; F_Z = 0.01116

HIP_INIT_QHIP  = -0.8   # rad — starting position
WHEEL_CLEARANCE = 0.005 # m   — wheel starts this far above ground

# ── Control parameters ──────────────────────────────────────────────────────
Q_RETRACTED  = -0.58    # fully crouched (start of jump stroke)
Q_EXTENDED   = -1.27    # fully extended (end of jump stroke)
CROUCH_TIME  =  2.0     # seconds to complete crouch trajectory

KP =  20.0              # PD proportional gain [N·m/rad]  — softened, precise hold not needed
KD =   1.0              # PD derivative gain   [N·m·s/rad]

MAX_TORQUE =  7.0       # AK45-10 peak torque [N·m]  — jump stroke only
HOLD_TORQUE = 7.0       # torque limit for PD position hold (FALLING/CROUCHING/LANDING)
OMEGA_MAX  = 18.85      # no-load speed at 24V, output shaft [rad/s]  (75KV×24V/10:1=180RPM)
JUMP_RAMP_S = 0.020     # torque ramp duration at jump start [s] — protects 4-bar constraint

WHEEL_R      = 0.075    # wheel radius [m]
GROUNDED_TOL = 0.015    # ground contact threshold above wheel centre [m]

# ── State machine ────────────────────────────────────────────────────────────
FALLING   = 0   # falling, PD holds hip at HIP_INIT_QHIP
CROUCHING = 1   # PD trajectory from current angle to Q_RETRACTED
JUMPING   = 2   # torque-speed limited torque toward Q_EXTENDED
LANDING   = 3   # motor off, wait for stable ground contact, then repeat
DONE      = 4   # unused / final stop

STABLE_TIME  = 0.4   # seconds of continuous ground contact before re-crouching
CROUCH_TIME2 = 2.0   # crouch duration for repeat jumps (same as first)

STATE_NAMES = {FALLING: "FALLING", CROUCHING: "CROUCHING",
               JUMPING: "JUMPING", LANDING: "LANDING", DONE: "DONE"}


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def direct_ik(q_hip):
    C_x = -L_femur * np.cos(q_hip)
    C_z =  A_Z + L_femur * np.sin(q_hip)
    dx = C_x - F_X; dz = C_z - F_Z
    K = (Lc**2 - dx**2 - dz**2 - L_stub**2) / (2.0 * L_stub)
    R = np.sqrt(dx**2 + dz**2)
    if abs(K) > R:
        return None
    phi      = np.arctan2(dz, dx)
    asin_val = np.arcsin(np.clip(K / R, -1.0, 1.0))
    alpha1 = _wrap(asin_val - phi)
    alpha2 = _wrap(np.pi - asin_val - phi)
    q_knee1 = alpha1 - q_hip; q_knee2 = alpha2 - q_hip
    if abs(q_knee1) <= abs(q_knee2):
        alpha, q_knee = alpha1, q_knee1
    else:
        alpha, q_knee = alpha2, q_knee2
    E_x = C_x + L_stub * np.sin(alpha)
    E_z = C_z + L_stub * np.cos(alpha)
    q_coupler_F = np.arctan2(E_z - F_Z, F_X - E_x)
    return q_hip, q_knee, q_coupler_F


# ---------------------------------------------------------------------------
# Plot process — hip angle, applied torque, box height
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
        ("Hip angle",          "rad",  "#f0c040"),
        ("Motor torque",       "N·m",  "#f08040"),
        ("Box height",         "mm",   "#60d0ff"),
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

    s_slide  = jqpos("body_slide")
    s_hf     = jqpos("hinge_F")
    s_hip    = jqpos("hip")
    s_knee   = jqpos("knee_joint")
    d_hip    = jdof("hip")
    box_bid  = bid("box")
    wheel_bid = bid("wheel_asm")

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
        q_hip_init, q_knee_init, q_coupler_F = ik
        data.qpos[s_hf]    = q_coupler_F
        data.qpos[s_hip]   = q_hip_init
        data.qpos[s_knee]  = q_knee_init
        data.ctrl[0]       = 0.0
        # Forward pass to find actual wheel height, then adjust slide so
        # wheel starts WHEEL_CLEARANCE above ground — avoids violent drop impact
        mujoco.mj_forward(model, data)
        wheel_z = float(data.xpos[wheel_bid][2])
        data.qpos[s_slide] += (WHEEL_R + WHEEL_CLEARANCE) - wheel_z
        mujoco.mj_forward(model, data)
        wheel_z = float(data.xpos[wheel_bid][2])
        print(f"Init: q_hip={q_hip_init:.4f}  q_knee={q_knee_init:.5f}"
              f"  wheel_z={wheel_z:.4f} m  (clearance={1000*(wheel_z-WHEEL_R):.1f} mm)")
    else:
        print("WARNING: IK failed")

    mujoco.mj_forward(model, data)
    body_z_initial = float(data.xpos[box_bid][2])

    # State machine
    state           = FALLING
    crouch_start_t  = 0.0
    q_crouch_start  = HIP_INIT_QHIP
    jump_start_t    = 0.0
    max_height_m    = 0.0
    grounded_since  = None   # timestamp when continuous ground contact started

    plot_q = mp.Queue(maxsize=4000)
    plot_proc = mp.Process(target=_plot_process, args=(plot_q, WINDOW_S), daemon=True)
    plot_proc.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.sync()

        while viewer.is_running():
            frame_start = time.perf_counter()

            sim_t = float(data.time)

            # Detect viewer reset (time jumped backward)
            if sim_t < prev_sim_t - 0.01:
                if not plot_q.full():
                    plot_q.put_nowait("RESET")
                last_push_t    = -1.0
                state          = FALLING
                crouch_start_t = 0.0
                jump_start_t   = 0.0
                max_height_m   = 0.0
                grounded_since = None
            prev_sim_t = sim_t

            # ── State machine + torque computation ───────────────────────
            hip_q    = float(data.qpos[s_hip])
            omega    = float(data.qvel[d_hip])
            wheel_z  = float(data.xpos[wheel_bid][2])
            grounded = wheel_z < (WHEEL_R + GROUNDED_TOL)

            if state == FALLING:
                # PD hold at initial angle while falling
                q_des  = HIP_INIT_QHIP
                torque = KP * (q_des - hip_q) + KD * (0.0 - omega)
                torque = float(np.clip(torque, -HOLD_TORQUE, HOLD_TORQUE))
                if grounded:
                    state          = CROUCHING
                    crouch_start_t = sim_t
                    q_crouch_start = hip_q
                    print(f"[{sim_t:.3f}s] CROUCHING  hip={np.degrees(hip_q+np.pi/2):.1f}°")

            elif state == CROUCHING:
                # Linear trajectory from q_crouch_start → Q_RETRACTED
                alpha  = min(1.0, (sim_t - crouch_start_t) / CROUCH_TIME)
                q_des  = q_crouch_start + (Q_RETRACTED - q_crouch_start) * alpha
                torque = KP * (q_des - hip_q) + KD * (0.0 - omega)
                torque = float(np.clip(torque, -HOLD_TORQUE, HOLD_TORQUE))
                if alpha >= 1.0:
                    state          = JUMPING
                    jump_start_t   = sim_t
                    body_z_initial = float(data.xpos[box_bid][2])  # height ref at jump start
                    max_height_m   = 0.0
                    print(f"[{sim_t:.3f}s] JUMPING    hip={np.degrees(hip_q+np.pi/2):.1f}°")

            elif state == JUMPING:
                # Torque-speed curve with linear ramp-up to protect 4-bar constraint
                ramp   = min(1.0, (sim_t - jump_start_t) / JUMP_RAMP_S)
                torque = -MAX_TORQUE * ramp * max(0.0, 1.0 - abs(omega) / OMEGA_MAX)
                if not grounded or hip_q <= Q_EXTENDED + 0.02:
                    state          = LANDING
                    torque         = 0.0
                    grounded_since = None
                    print(f"[{sim_t:.3f}s] LANDING    hip={np.degrees(hip_q+np.pi/2):.1f}°"
                          f"  grounded={grounded}")

            elif state == LANDING:
                # Hold extended position while airborne/bouncing — keeps 4-bar intact
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
                    grounded_since = None  # reset if briefly airborne

            else:  # DONE
                torque = 0.0

            data.ctrl[0] = torque

            # ── Physics step ─────────────────────────────────────────────
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            viewer.sync()

            sim_t = float(data.time)

            # Height = wheel centre above ground (= 0 when resting, > 0 when airborne)
            wheel_z_now  = float(data.xpos[wheel_bid][2])
            height_mm    = max(0.0, (wheel_z_now - WHEEL_R) * 1000.0)
            max_height_m = max(max_height_m, wheel_z_now - WHEEL_R)

            # Push telemetry
            if sim_t - last_push_t >= push_interval and not plot_q.full():
                plot_q.put_nowait((sim_t, hip_q, torque, height_mm))
                last_push_t = sim_t

            elapsed = time.perf_counter() - frame_start
            sleep_t = 1.0 / RENDER_HZ - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    print(f"\nMax jump height (wheel centre above ground): {max_height_m * 1000:.1f} mm")
    plot_q.put(None)
    plot_proc.join(timeout=2)
