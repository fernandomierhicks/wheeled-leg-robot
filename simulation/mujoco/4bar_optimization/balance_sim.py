"""
balance_sim.py — 1-DOF leg balancing simulation.

A single leg with a free-floating body (planar X-Z translation + Pitch rotation).
A PD controller on the wheel motor attempts to balance the body upright.
The hip motor holds a fixed crouch angle.

Run:
    python simulation/mujoco/balancing/balance_sim.py
"""

import math
import multiprocessing as mp
import os
import sys
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

# ── Add parent directory to path to import from 4bar_optimization ──────────
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
sys.path.insert(0, _PARENT_DIR)

from mujoco.four_bar_optimization.eval_jump import DEFAULT, MOTOR_MASS, LEG_Y

# ── Simulation settings ─────────────────────────────────────────────────────
RENDER_HZ = 60
PUSH_HZ   = 200
WINDOW_S  = 15.0

# ── Robot parameters ────────────────────────────────────────────────────────
PARAMS = DEFAULT
WHEEL_R = 0.075

# ── Control ─────────────────────────────────────────────────────────────────
# Balance controller (on body pitch)
BALANCE_KP = 6.0    # Proportional gain [N·m/rad]
BALANCE_KD = 0.4    # Derivative gain [N·m·s/rad]
WHEEL_TORQUE_LIMIT = 3.0  # [N·m]

# Hip position controller
HIP_KP = 20.0
HIP_KD = 1.0
HIP_Q_TARGET = -0.8  # Target hip angle [rad]
HIP_TORQUE_LIMIT = 7.0

# ── Initial state ───────────────────────────────────────────────────────────
INITIAL_PITCH_RAD = 0.15  # Small initial tilt to test controller

# --- Sensor Noise Simulation ───────────────────────────────────────────────
# Simulates noise on the IMU readings. Based on BNO085 datasheet values.
PITCH_NOISE_STD_RAD = math.radians(0.1)     # Std dev of pitch angle noise [rad]
PITCH_RATE_NOISE_STD_RAD_S = math.radians(0.5) # Std dev of pitch rate noise [rad/s]


# ---------------------------------------------------------------------------
# Kinematics (copied from eval_jump.py)
# ---------------------------------------------------------------------------
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def solve_ik(q_hip: float, p: dict) -> dict | None:
    L_f = p['L_femur']; L_s = p['L_stub']; L_t = p['L_tibia']
    Lc  = p['Lc'];      F_X = p['F_X'];    F_Z = p['F_Z']; A_Z = p['A_Z']
    C_x = -L_f * math.cos(q_hip)
    C_z =  A_Z + L_f * math.sin(q_hip)
    dx = C_x - F_X; dz = C_z - F_Z
    R  = math.sqrt(dx * dx + dz * dz)
    if R < 1e-9: return None
    K = (Lc**2 - dx**2 - dz**2 - L_s**2) / (2.0 * L_s)
    if abs(K) / R >= 1.0: return None
    phi   = math.atan2(dz, dx)
    asinv = math.asin(max(-1.0, min(1.0, K / R)))
    a1    = _wrap(asinv - phi)
    a2    = _wrap(math.pi - asinv - phi)
    qk1   = a1 - q_hip; qk2 = a2 - q_hip
    alpha = a1 if abs(qk1) <= abs(qk2) else a2
    q_knee = alpha - q_hip
    E_x = C_x + L_s * math.sin(alpha)
    E_z = C_z + L_s * math.cos(alpha)
    q_coupler_F = math.atan2(E_z - F_Z, F_X - E_x)
    return dict(q_hip=q_hip, q_knee=q_knee, q_coupler_F=q_coupler_F)


# ---------------------------------------------------------------------------
# Dynamic XML builder (adapted from eval_jump.py)
# ---------------------------------------------------------------------------
def build_xml(p: dict) -> str:
    L_f = p['L_femur']; L_s = p['L_stub']; L_t = p['L_tibia']
    Lc  = p['Lc'];      F_X = p['F_X'];    F_Z = p['F_Z']; A_Z = p['A_Z']
    tib_cz  = (L_s - L_t) / 2.0
    tib_hsz = (L_s + L_t) / 2.0
    motor_y = LEG_Y - 0.0215
    maytech_mass = min(MOTOR_MASS, p['m_wheel'])
    tyre_mass    = max(0.005, p['m_wheel'] - maytech_mass)

    return f"""<mujoco model="balance_leg">
  <option gravity="0 0 -9.81" timestep="0.0005" solver="Newton" iterations="200" tolerance="1e-10"/>
  <visual><global offwidth="1280" offheight="720" azimuth="270" elevation="-5"/></visual>
  <worldbody>
    <light name="main" pos="0 -1 1" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" pos="0 0 0" size="5 5 0.01" rgba="0.3 0.3 0.3 1" contype="1" conaffinity="1" friction="0.8 0.005 0.0001"/>
    <body name="box" pos="0 0 0.40">
      <joint name="root_x" type="slide" axis="1 0 0" pos="0 0 0" damping="0.5"/>
      <joint name="root_z" type="slide" axis="0 0 1" pos="0 0 0" damping="0.5"/>
      <joint name="root_pitch" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.01"/>
      <geom name="box_geom" type="box" size="0.05 0.10 0.05" rgba="0.29 0.56 0.85 0.55" contype="0" conaffinity="0" mass="{p['m_box']:.4f}"/>
      <geom name="motor_geom" type="cylinder" pos="0 {motor_y:.4f} {A_Z:.5f}" euler="90 0 0" size="0.0265 0.0215" rgba="0.87 0.36 0.36 0.55" contype="0" conaffinity="0" mass="{MOTOR_MASS:.3f}"/>
      <body name="coupler" pos="{F_X:.5f} {LEG_Y:.4f} {F_Z:.5f}">
        <joint name="hinge_F" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.001"/>
        <geom type="cylinder" pos="0 0 0" euler="90 0 0" size="0.011 0.010" rgba="0 1 1 0.9" contype="0" conaffinity="0" mass="0"/>
        <geom name="coupler_geom" type="box" pos="{-Lc/2:.5f} 0 0" size="{Lc/2:.5f} 0.01 0.01" rgba="0.30 0.90 0.90 0.55" contype="0" conaffinity="0" mass="{p['m_coupler']:.4f}"/>
      </body>
      <body name="femur" pos="0 {LEG_Y:.4f} {A_Z:.5f}">
        <joint name="hip" type="hinge" axis="0 1 0" pos="0 0 0" range="-180 180" armature="0.01" damping="0.05"/>
        <geom type="cylinder" pos="0 0 0" euler="90 0 0" size="0.005 0.015" rgba="1 1 0 0.9" contype="0" conaffinity="0" mass="0"/>
        <geom name="femur_geom" type="box" pos="{-L_f/2:.5f} 0 0" size="{L_f/2:.5f} 0.01 0.01" rgba="0.94 0.75 0.25 0.55" contype="0" conaffinity="0" mass="{p['m_femur']:.4f}"/>
        <geom name="knee_geom" type="cylinder" pos="{-L_f:.5f} 0 0" euler="90 0 0" size="0.011 0.010" rgba="1 1 1 0.9" contype="0" conaffinity="0" mass="0"/>
        <body name="tibia" pos="{-L_f:.5f} 0 0">
          <joint name="knee_joint" type="hinge" axis="0 1 0" pos="0 0 0" range="-60 60" damping="0.001"/>
          <geom name="tibia_geom" type="box" pos="0 0 {tib_cz:.5f}" size="0.010 0.010 {tib_hsz:.5f}" rgba="0.75 0.45 0.90 0.55" contype="0" conaffinity="0" mass="{p['m_tibia']:.4f}"/>
          <geom name="stub_geom" type="cylinder" pos="0 0 {L_s:.5f}" euler="90 0 0" size="0.011 0.010" rgba="1 0.3 0.3 0.9" contype="0" conaffinity="0" mass="0"/>
          <body name="wheel_asm" pos="0 0 {-L_t:.5f}">
            <joint name="wheel_spin" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.001"/>
            <geom name="motor_body_geom" type="cylinder" pos="0 0.036 0" euler="90 0 0" size="0.035 0.026" rgba="0.15 0.15 0.15 0.55" contype="0" conaffinity="0" mass="{maytech_mass:.4f}"/>
            <geom name="wheel_tire_geom" type="cylinder" pos="0 0.036 0" euler="90 0 0" size="{WHEEL_R} 0.015" rgba="0.25 0.70 0.35 0.80" contype="1" conaffinity="1" friction="0.8 0.01 0.001" solref="0.01 1.0" solimp="0.9 0.99 0.001" mass="{tyre_mass:.4f}"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <connect name="4bar_close" body1="coupler" body2="tibia" anchor="0 0 0" solref="0.00005 1" solimp="0.9999 0.9999 0.0001"/>
  </equality>
  <actuator>
    <motor name="hip_motor" joint="hip" ctrlrange="-{HIP_TORQUE_LIMIT} {HIP_TORQUE_LIMIT}" gear="1"/>
    <motor name="wheel_motor" joint="wheel_spin" ctrlrange="-{WHEEL_TORQUE_LIMIT} {WHEEL_TORQUE_LIMIT}" gear="1"/>
  </actuator>
  <statistic center="0 0 0.30" extent="0.75"/>
</mujoco>"""


# ---------------------------------------------------------------------------
# Matplotlib telemetry (side window)
# ---------------------------------------------------------------------------
def _plot_process(q: mp.Queue, window_s: float) -> None:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    MAXLEN = int(window_s * 1000) + 500
    t_buf = deque(maxlen=MAXLEN)
    pitch_buf = deque(maxlen=MAXLEN)
    pitch_rate_buf = deque(maxlen=MAXLEN)
    torque_buf = deque(maxlen=MAXLEN)

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(6, 7))
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(hspace=0.50, top=0.91, bottom=0.09, left=0.18, right=0.96)

    spec = [("Body Pitch", "rad", "#60d0ff"),
            ("Pitch Rate", "rad/s", "#f0c040"),
            ("Wheel Torque", "N·m", "#f08040")]
    lines_list = []
    for ax, (ttl, unit, col) in zip(axes, spec):
        ax.set_facecolor("#1e1e2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#555")
        ax.tick_params(colors="lightgray", labelsize=7)
        ax.grid(True, color="#333", linewidth=0.4, linestyle="--")
        ax.set_title(ttl, color="white", fontsize=9, pad=3)
        ax.set_ylabel(unit, color=col, fontsize=8)
        ax.set_xlabel("sim time [s]", color="lightgray", fontsize=7)
        ln, = ax.plot([], [], color=col, linewidth=1.5)
        lines_list.append(ln)

    fig.suptitle("Balance Controller Telemetry", color="white", fontsize=10)
    fig.show()

    while plt.fignum_exists(fig.number):
        items = []
        while True:
            try: items.append(q.get_nowait())
            except Exception: break
        if not items:
            plt.pause(1.0 / 60); continue

        for item in items:
            if item is None: return
            if item == "RESET":
                t_buf.clear(); pitch_buf.clear(); pitch_rate_buf.clear(); torque_buf.clear()
                for ln in lines_list: ln.set_data([], [])
                fig.canvas.flush_events()
                continue
            t_buf.append(item[0]); pitch_buf.append(item[1])
            pitch_rate_buf.append(item[2]); torque_buf.append(item[3])

        if len(t_buf) < 2: continue

        tb = list(t_buf); sim_t = tb[-1]
        bufs = [list(pitch_buf), list(pitch_rate_buf), list(torque_buf)]
        t0 = max(0.0, sim_t - window_s)
        idx = next((i for i, t in enumerate(tb) if t >= t0), 0)
        tw = tb[idx:]

        for ln, ax, buf in zip(lines_list, axes, bufs):
            bw = buf[idx:]
            ln.set_data(tw, bw)
            ax.set_xlim(t0, sim_t + 0.5)
            if len(bw) > 1:
                lo, hi = min(bw), max(bw)
                span = max(hi - lo, 0.05)
                ax.set_ylim(lo - span * 0.2, hi + span * 0.2)

        fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    xml = build_xml(PARAMS)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Override equality anchors
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "4bar_close")
    model.eq_data[eq_id, 0:3] = [-PARAMS['Lc'], 0.0, 0.0]
    model.eq_data[eq_id, 3:6] = [0.0, 0.0, PARAMS['L_stub']]

    # Joint/body name->ID lookups
    def _jqpos(n): return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def _jdof(n): return model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def _bid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

    s_root_x = _jqpos("root_x"); s_root_z = _jqpos("root_z"); s_pitch = _jqpos("root_pitch")
    s_hF = _jqpos("hinge_F"); s_hip = _jqpos("hip"); s_knee = _jqpos("knee_joint")
    d_pitch = _jdof("root_pitch"); d_hip = _jdof("hip")
    wheel_bid = _bid("wheel_asm")

    # --- Initial pose ---
    def _init_pose():
        mujoco.mj_resetData(model, data)
        ik = solve_ik(HIP_Q_TARGET, PARAMS)
        if ik is None:
            raise RuntimeError("IK failed at initial hip angle")

        data.qpos[s_hF] = ik['q_coupler_F']
        data.qpos[s_hip] = ik['q_hip']
        data.qpos[s_knee] = ik['q_knee']

        # Place on ground
        mujoco.mj_forward(model, data)
        wheel_z = data.xpos[wheel_bid][2]
        data.qpos[s_root_z] += WHEEL_R - wheel_z

        # Set initial tilt
        data.qpos[s_pitch] = INITIAL_PITCH_RAD
        mujoco.mj_forward(model, data)
        print(f"Initialized at pitch {INITIAL_PITCH_RAD:.2f} rad")

    _init_pose()

    # --- Simulation setup ---
    PHYSICS_HZ = round(1.0 / model.opt.timestep)
    steps_per_frame = max(1, PHYSICS_HZ // RENDER_HZ)
    last_push_wall = -1.0
    prev_sim_t = 0.0

    plot_q = mp.Queue(maxsize=4000)
    plot_proc = mp.Process(target=_plot_process, args=(plot_q, WINDOW_S), daemon=True)
    plot_proc.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.cam.elevation = -10
        viewer.cam.distance = 2.5
        viewer.sync()

        while viewer.is_running():
            frame_start = time.perf_counter()
            sim_t = float(data.time)

            # Detect viewer reset
            if sim_t < prev_sim_t - 0.01:
                _init_pose()
                if not plot_q.full(): plot_q.put_nowait("RESET")
                last_push_wall = -1.0
            prev_sim_t = sim_t

            # --- Controller ---
            pitch_true = float(data.qpos[s_pitch])
            pitch_rate_true = float(data.qvel[d_pitch])

            # Add simulated sensor noise
            pitch = pitch_true + np.random.normal(scale=PITCH_NOISE_STD_RAD)
            pitch_rate = pitch_rate_true + np.random.normal(scale=PITCH_RATE_NOISE_STD_RAD_S)

            hip_q = float(data.qpos[s_hip])
            hip_omega = float(data.qvel[d_hip])

            # Balance PD controller -> wheel torque
            # DISABLED to observe free-fall with noisy sensors
            data.ctrl[1] = 0.0

            # Hip PD controller -> hold position
            hip_torque = HIP_KP * (HIP_Q_TARGET - hip_q) + HIP_KD * (0.0 - hip_omega)
            data.ctrl[0] = np.clip(hip_torque, -HIP_TORQUE_LIMIT, HIP_TORQUE_LIMIT)

            # --- Physics step ---
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            viewer.sync()

            # --- Telemetry ---
            wall_now = time.perf_counter()
            if wall_now - last_push_wall >= (1.0/PUSH_HZ) and not plot_q.full():
                plot_q.put_nowait((sim_t, pitch, pitch_rate, data.ctrl[1]))
                last_push_wall = wall_now

            # --- Info text in viewer ---
            viewer.user_scn.ngeom = 0
            g = viewer.user_scn.geoms[0]
            info_text = (f"Pitch (true): {math.degrees(pitch_true):>5.1f} deg\n"
                         f"Pitch (noisy): {math.degrees(pitch):>5.1f} deg\n"
                         f"Wheel Torque: {data.ctrl[1]:>5.2f} Nm")
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.008, 0.008, 0.008]),
                np.array([-0.25, 0.15, 0.60]), np.eye(3).flatten(),
                np.array([0.4, 1.0, 0.4, 1.0], dtype=np.float32))
            g.label = info_text.encode()[:99]
            viewer.user_scn.ngeom = 1

            # --- Sync to render framerate ---
            elapsed = time.perf_counter() - frame_start
            sleep_t = 1.0 / RENDER_HZ - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    plot_q.put(None)
    plot_proc.join(timeout=2)