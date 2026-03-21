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
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

# ── Standalone Robot Constants ──────────────────────────────────────────────
LEG_Y            = 0.1430    # Y-offset of leg plane from box centre [m] (motor mount)
MOTOR_MASS       = 0.260     # AK45-10 [kg]

DEFAULT = dict(
    L_femur  = 0.100,    # A→C  [m]
    L_stub   = 0.025,    # C→E tibia stub (above knee)  [m]
    L_tibia  = 0.115,    # C→W to wheel centre  [m]
    Lc       = 0.110,    # F→E coupler length  [m]
    F_X      = -0.020,   # F x-position in box frame  [m]  (A is at x=0)
    F_Z      =  0.01116, # F z-position in box frame  [m]
    A_Z      = -0.0235,  # A z-position in box frame  [m]  (hip motor centre)
    m_box    = 0.477,    # body electronics + structure  [kg]
    m_femur  = 0.025,
    m_tibia  = 0.035,
    m_coupler= 0.015,
    m_wheel  = 0.410,    # Maytech motor + TPU tyre  [kg]
)

# ── Simulation settings ─────────────────────────────────────────────────────
RENDER_HZ = 60
PUSH_HZ   = 200
WINDOW_S  = 15.0
SLOW_MO   = 1.0     # Slow motion factor (1.0 = real-time)

# ── Robot parameters ────────────────────────────────────────────────────────
PARAMS = DEFAULT
WHEEL_R = 0.075

# ── Control ─────────────────────────────────────────────────────────────────
# Balance controller (PID on Pitch + PD on Translation)
PITCH_KP = 60.0       # [N·m/rad] High stiffness to minimize wheel travel
PITCH_KI = 0.0        # [N·m/(rad·s)] Zeroed to prevent windup
PITCH_KD = 5.0        # [N·m·s/rad] High damping to suppress oscillation

# Outer Loop: Position -> Pitch
# "Higher level controller that tweaks desired pitch"
POSITION_KP = 0.30    # [rad/m] Increased for tighter position holding
VELOCITY_KP = 0.30    # [rad/(m/s)] Increased for tighter velocity damping
MAX_PITCH_CMD = 0.25  # [rad] Max lean angle commanded by position loop

WHEEL_TORQUE_LIMIT = 7.0  # [N·m]

# Hip position controller
HIP_KP = 30.0
HIP_KD = 1.0
HIP_Q_TARGET = -0.8  # Target hip angle [rad]
HIP_TORQUE_LIMIT = 7.0
HIP_Q_CROUCH = -0.58
HIP_Q_EXTENDED = -1.27
JUMP_RAMP_S = 0.010
JUMP_RAMPDOWN = 0.15
OMEGA_MAX = 18.85

# ── Initial state ───────────────────────────────────────────────────────────
INITIAL_PITCH_RAD = 0.1  # Small initial tilt to test controller

# --- Sensor Noise Simulation ───────────────────────────────────────────────
# Simulates noise on the BNO085 IMU pitch solution
PITCH_NOISE_STD_RAD = math.radians(0.1)     # Std dev of pitch angle noise [rad]
PITCH_RATE_NOISE_STD_RAD_S = math.radians(0.5) # Std dev of pitch rate noise [rad/s]
ACCEL_NOISE_STD = 0.2                       # [m/s^2] Noise on accelerometer


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
  <visual><global offwidth="1280" offheight="720" azimuth="270" elevation="0"/></visual>
  <worldbody>
    <light name="main" pos="0 -1 1" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
    <light name="fill" pos="0  1 1" dir="0 -1 -1" diffuse="0.3 0.3 0.3"/>
    <geom name="ground" type="plane" pos="0 0 0" size="5 5 0.01" rgba="0.3 0.3 0.3 1" contype="1" conaffinity="1" friction="0.8 0.005 0.0001" solref="0.02 1"/>
    <!-- XYZ axis markers -->
    <geom name="axis_x" type="cylinder" fromto="0 0 0.001  0.06 0 0.001" size="0.003" rgba="1 0 0 1" contype="0" conaffinity="0" mass="0"/>
    <geom name="axis_y" type="cylinder" fromto="0 0 0.001  0 0.06 0.001" size="0.003" rgba="0 1 0 1" contype="0" conaffinity="0" mass="0"/>
    <geom name="axis_z" type="cylinder" fromto="0 0 0.001  0 0 0.061" size="0.003" rgba="0 0 1 1" contype="0" conaffinity="0" mass="0"/>

    <body name="box" pos="0 0 0.40">
      <joint name="root_x" type="slide" axis="1 0 0" pos="0 0 0" damping="0.5"/>
      <joint name="root_z" type="slide" axis="0 0 1" pos="0 0 0" damping="0.5"/>
      <joint name="root_pitch" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.01"/>
      <geom name="box_geom" type="box" size="0.05 0.10 0.05" rgba="0.29 0.56 0.85 0.55" contype="1" conaffinity="1" mass="{p['m_box']:.4f}" solref="0.02 1"/>
      <site name="imu" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
      <!-- CG Marker -->
      <site name="box_cg" type="sphere" size="0.01" rgba="1 0 1 0.8" pos="0 0 0"/>
      <geom name="motor_geom" type="cylinder" pos="0 {motor_y:.4f} {A_Z:.5f}" euler="90 0 0" size="0.0265 0.0215" rgba="0.87 0.36 0.36 0.55" contype="1" conaffinity="1" mass="{MOTOR_MASS:.3f}" solref="0.02 1"/>
      <body name="coupler" pos="{F_X:.5f} {LEG_Y:.4f} {F_Z:.5f}">
        <joint name="hinge_F" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.001"/>
        <geom type="cylinder" pos="0 0 0" euler="90 0 0" size="0.011 0.010" rgba="0 1 1 0.9" contype="0" conaffinity="0" mass="0"/>
        <geom name="coupler_geom" type="box" pos="{-Lc/2:.5f} 0 0" size="{Lc/2:.5f} 0.01 0.01" rgba="0.30 0.90 0.90 0.55" contype="1" conaffinity="1" mass="{p['m_coupler']:.4f}" solref="0.02 1"/>
      </body>
      <body name="femur" pos="0 {LEG_Y:.4f} {A_Z:.5f}">
        <joint name="hip" type="hinge" axis="0 1 0" pos="0 0 0" range="-180 180" armature="0.01" damping="0.05"/>
        <geom type="cylinder" pos="0 0 0" euler="90 0 0" size="0.005 0.015" rgba="1 1 0 0.9" contype="0" conaffinity="0" mass="0"/>
        <geom name="femur_geom" type="box" pos="{-L_f/2:.5f} 0 0" size="{L_f/2:.5f} 0.01 0.01" rgba="0.94 0.75 0.25 0.55" contype="1" conaffinity="1" mass="{p['m_femur']:.4f}" solref="0.02 1"/>
        <geom name="knee_geom" type="cylinder" pos="{-L_f:.5f} 0 0" euler="90 0 0" size="0.011 0.010" rgba="1 1 1 0.9" contype="0" conaffinity="0" mass="0" solref="0.02 1"/>
        <body name="tibia" pos="{-L_f:.5f} 0 0">
          <joint name="knee_joint" type="hinge" axis="0 1 0" pos="0 0 0" range="-60 60" damping="0.001"/>
          <geom name="tibia_geom" type="box" pos="0 0 {tib_cz:.5f}" size="0.010 0.010 {tib_hsz:.5f}" rgba="0.75 0.45 0.90 0.55" contype="1" conaffinity="1" mass="{p['m_tibia']:.4f}" solref="0.02 1"/>
          <geom name="stub_geom" type="cylinder" pos="0 0 {L_s:.5f}" euler="90 0 0" size="0.011 0.010" rgba="1 0.3 0.3 0.9" contype="0" conaffinity="0" mass="0" solref="0.02 1"/>
          <body name="wheel_asm" pos="0 0 {-L_t:.5f}">
            <joint name="wheel_spin" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.001"/>
            <geom name="motor_body_geom" type="cylinder" pos="0 0.036 0" euler="90 0 0" size="0.035 0.026" rgba="0.15 0.15 0.15 0.55" contype="1" conaffinity="1" mass="{maytech_mass:.4f}" solref="0.02 1"/>
            <geom name="wheel_tire_geom" type="cylinder" pos="0 0.036 0" euler="90 0 0" size="{WHEEL_R} 0.015" rgba="0.25 0.70 0.35 0.80" contype="1" conaffinity="1" friction="0.8 0.01 0.001" solref="0.02 1" solimp="0.9 0.99 0.001" mass="{tyre_mass:.4f}"/>
            <geom name="wheel_spoke" type="box" pos="0 0.052 0" size="{WHEEL_R*0.8:.4f} 0.005 0.005" rgba="1 1 1 0.9" contype="0" conaffinity="0" mass="0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <connect name="4bar_close" body1="coupler" body2="tibia" anchor="0 0 0" solref="0.00005 1" solimp="0.9999 0.9999 0.0001"/>
  </equality>
  <sensor>
    <accelerometer name="accel" site="imu"/>
  </sensor>
  <actuator>
    <motor name="hip_motor" joint="hip" ctrlrange="-{HIP_TORQUE_LIMIT} {HIP_TORQUE_LIMIT}" gear="1"/>
    <motor name="wheel_motor" joint="wheel_spin" ctrlrange="-{WHEEL_TORQUE_LIMIT} {WHEEL_TORQUE_LIMIT}" gear="1"/>
  </actuator>
  <statistic center="0 0 0.30" extent="0.75"/>
</mujoco>"""


# ---------------------------------------------------------------------------
# Matplotlib telemetry (side window)
# ---------------------------------------------------------------------------
def _plot_process(q: mp.Queue, cmd_q: mp.Queue, window_s: float) -> None:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.widgets as widgets

    MAXLEN = int(window_s * 1000) + 500
    t_buf = deque(maxlen=MAXLEN)
    pitch_buf = deque(maxlen=MAXLEN)
    pitch_rate_buf = deque(maxlen=MAXLEN)
    torque_buf = deque(maxlen=MAXLEN)

    plt.ion() # Interactive mode
    # Single chart with dual axes
    fig, ax1 = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#1e1e2e")
    plt.subplots_adjust(top=0.92, bottom=0.15, left=0.15, right=0.85)

    # Axis 1: Pitch
    ax1.set_facecolor("#1e1e2e")
    ax1.set_xlabel("Time [s]", color="lightgray")
    ax1.set_ylabel("Pitch [rad]", color="#60d0ff", fontweight="bold")
    ax1.tick_params(axis='x', colors="lightgray")
    ax1.tick_params(axis='y', colors="#60d0ff")
    ax1.grid(True, color="#333", linewidth=0.4, linestyle="--")
    ln1, = ax1.plot([], [], color="#60d0ff", linewidth=1.5, label="Pitch")

    # Axis 2: Torque (Twin)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Torque [N·m]", color="#f08040", fontweight="bold")
    ax2.tick_params(axis='y', colors="#f08040")
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ln2, = ax2.plot([], [], color="#f08040", linewidth=1.0, alpha=0.8, label="Torque")

    ax_start = plt.axes([0.60, 0.025, 0.15, 0.04])
    btn_start = widgets.Button(ax_start, 'Start', color='#60d060', hovercolor='#80e080')

    def on_start(event):
        cmd_q.put("START")
    btn_start.on_clicked(on_start)

    ax_reset = plt.axes([0.8, 0.025, 0.15, 0.04])
    btn_reset = widgets.Button(ax_reset, 'Restart', color='#f0c040', hovercolor='#ffd060')
    
    def on_reset(event):
        cmd_q.put("RESET")
    btn_reset.on_clicked(on_reset)

    ax_crouch = plt.axes([0.05, 0.025, 0.15, 0.04])
    btn_crouch = widgets.Button(ax_crouch, 'Crouch', color='#6060d0', hovercolor='#8080e0')
    def on_crouch(event):
        cmd_q.put("CROUCH")
    btn_crouch.on_clicked(on_crouch)

    ax_jump = plt.axes([0.22, 0.025, 0.15, 0.04])
    btn_jump = widgets.Button(ax_jump, 'Jump', color='#d060d0', hovercolor='#e080e0')
    def on_jump(event):
        cmd_q.put("JUMP")
    btn_jump.on_clicked(on_jump)

    ax_kick = plt.axes([0.40, 0.025, 0.12, 0.04])
    btn_kick = widgets.Button(ax_kick, 'Kick (-X)', color='#ff6060', hovercolor='#ff8080')
    
    def on_kick(event):
        cmd_q.put("KICK")
    btn_kick.on_clicked(on_kick)

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
                ln1.set_data([], []); ln2.set_data([], [])
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

        # Update Pitch
        pitch_data = list(pitch_buf)[idx:]
        ln1.set_data(tw, pitch_data)
        ax1.set_xlim(t0, sim_t + 0.5)
        if len(pitch_data) > 1:
            lo, hi = min(pitch_data), max(pitch_data)
            span = max(hi - lo, 0.1)
            ax1.set_ylim(lo - span * 0.2, hi + span * 0.2)

        # Update Torque
        trq_data = list(torque_buf)[idx:]
        ln2.set_data(tw, trq_data)
        if len(trq_data) > 1:
            mx = max(abs(min(trq_data)), abs(max(trq_data)), 0.1)
            ax2.set_ylim(-mx*1.2, mx*1.2)

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
    d_wheel_spin = _jdof("wheel_spin")
    d_root_x = _jdof("root_x")
    wheel_bid = _bid("wheel_asm")
    box_bid = _bid("box")

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

        # Reset time and velocities. The model is now in its initial qpos.
        # The first mj_step will happen in the main loop, allowing the solver
        # to resolve any constraint violations.
        data.time = 0.0
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data) # Update sensors
        print(f"Initialized at pitch {INITIAL_PITCH_RAD:.2f} rad")

    _init_pose()

    # Controller state
    pitch_integral = 0.0
    odo_x = 0.0
    target_pitch = 0.0
    kick_end_t = 0.0
    leg_state = "HOLD"  # HOLD or JUMP
    current_hip_target = HIP_Q_TARGET
    jump_start_t = 0.0
    grounded = True

    # --- Simulation setup ---
    PHYSICS_HZ = round(1.0 / model.opt.timestep)
    steps_per_frame = max(1, int(PHYSICS_HZ / (RENDER_HZ * SLOW_MO)))
    last_push_wall = -1.0
    prev_sim_t = 0.0

    plot_q = mp.Queue(maxsize=4000)
    cmd_q = mp.Queue()
    plot_proc = mp.Process(target=_plot_process, args=(plot_q, cmd_q, WINDOW_S), daemon=True)
    plot_proc.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 270
        viewer.cam.elevation = 0
        viewer.cam.distance = 2.0
        viewer.cam.lookat = np.array([0.0, 0.0, 0.4])

        # Pause at t=0 to inspect the initial state. Press SPACE to unpause.
        with viewer.lock():
            viewer.is_paused = True
        viewer.sync()

        while viewer.is_running():
            frame_start = time.perf_counter()
            sim_t = float(data.time)

            # Detect viewer reset
            if sim_t < prev_sim_t - 0.01:
                _init_pose()
                pitch_integral = 0.0
                odo_x = 0.0
                kick_end_t = 0.0
                leg_state = "HOLD"
                current_hip_target = HIP_Q_TARGET
                grounded = True
                if not plot_q.full(): plot_q.put_nowait("RESET")
                last_push_wall = -1.0
                with viewer.lock():
                    viewer.is_paused = True
            prev_sim_t = sim_t

            while not cmd_q.empty():
                cmd = cmd_q.get()
                if cmd == "RESET":
                    _init_pose()
                    pitch_integral = 0.0
                    odo_x = 0.0
                    kick_end_t = 0.0
                    leg_state = "HOLD"
                    current_hip_target = HIP_Q_TARGET
                    grounded = True
                    if not plot_q.full(): plot_q.put_nowait("RESET")
                    last_push_wall = -1.0
                    with viewer.lock():
                        viewer.is_paused = True
                    prev_sim_t = data.time
                elif cmd == "START":
                    with viewer.lock():
                        viewer.is_paused = False
                elif cmd == "KICK":
                    kick_end_t = data.time + 0.20  # Apply force for 200ms
                elif cmd == "CROUCH":
                    current_hip_target = HIP_Q_CROUCH
                elif cmd == "JUMP":
                    leg_state = "JUMP"
                    jump_start_t = data.time

            # --- Physics step ---
            if not viewer.is_paused:
                for _ in range(steps_per_frame):
                    # --- Controller (runs at PHYSICS_HZ) ---
                    # Sensor data (IMU + Encoders) only - No cheating with true world x/vel
                    pitch_true = float(data.qpos[s_pitch])
                    pitch_rate_true = float(data.qvel[d_pitch])
                    
                    # Add BNO085 simulated noise
                    pitch = pitch_true + np.random.normal(0, PITCH_NOISE_STD_RAD)
                    pitch_rate = pitch_rate_true + np.random.normal(0, PITCH_RATE_NOISE_STD_RAD_S)

                    wheel_vel = data.qvel[d_wheel_spin]

                    # Detect ground contact via IMU Acceleration
                    # Free fall < 2.0 m/s^2. Standing/Moving > 5.0 m/s^2.
                    # Hysteresis prevents flickering.
                    accel_raw = data.sensor("accel").data
                    accel_noisy = accel_raw + np.random.normal(0, ACCEL_NOISE_STD, 3)
                    accel_mag = np.linalg.norm(accel_noisy)
                    
                    if accel_mag < 3.0:    # ~0.3g (Airborne)
                        grounded = False
                    elif accel_mag > 7.0:  # ~0.7g (Grounded/Landing)
                        grounded = True
                        
                    if grounded:
                        # Estimate ground speed/pos via odometry
                        vel_est = (wheel_vel + pitch_rate) * WHEEL_R
                        odo_x += vel_est * model.opt.timestep

                        # Outer loop: Position Control -> Desired Pitch
                        # If we drift forward (+x), we want to lean back (-pitch) to accelerate backward
                        target_pitch = -(POSITION_KP * odo_x + VELOCITY_KP * vel_est)
                        target_pitch = np.clip(target_pitch, -MAX_PITCH_CMD, MAX_PITCH_CMD)
                    else:
                        # Air Control / Reaction Wheel
                        # Disable position/velocity loops, maintain 0 pitch relative to world
                        target_pitch = 0.0
                        pitch_integral = 0.0  # Prevent windup in air

                    # Inner loop: Balance Control (Regulate pitch to target_pitch)
                    pitch_error = pitch - target_pitch

                    # Integral term (limited anti-windup)
                    pitch_integral += pitch_error * model.opt.timestep
                    pitch_integral = np.clip(pitch_integral, -1.0, 1.0)

                    u_bal = (PITCH_KP * pitch_error +
                             PITCH_KI * pitch_integral +
                             PITCH_KD * pitch_rate)
                    data.ctrl[1] = np.clip(u_bal, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)

                    # Hip Control
                    hip_q = data.qpos[s_hip]
                    hip_omega = data.qvel[d_hip]

                    if leg_state == "JUMP":
                        ramp_in = min(1.0, (data.time - jump_start_t) / JUMP_RAMP_S)
                        ramp_out = min(1.0, max(0.0, (hip_q - HIP_Q_EXTENDED) / JUMP_RAMPDOWN))
                        u_hip = -HIP_TORQUE_LIMIT * ramp_in * ramp_out * max(0.0, 1.0 - abs(hip_omega) / OMEGA_MAX)
                        # Retract if fully extended OR if we've left the ground (airborne)
                        if (hip_q <= HIP_Q_EXTENDED + 0.05) or (not grounded and (data.time - jump_start_t) > 0.05):
                            leg_state = "HOLD"
                            current_hip_target = HIP_Q_TARGET
                    else:
                        u_hip = HIP_KP * (current_hip_target - hip_q) - HIP_KD * hip_omega

                    data.ctrl[0] = np.clip(u_hip, -HIP_TORQUE_LIMIT, HIP_TORQUE_LIMIT)

                    # Apply disturbance force if active
                    if data.time < kick_end_t:
                        data.xfrc_applied[box_bid, 0] = -0.7  # [N] Push backward (reduced 30x)
                    else:
                        data.xfrc_applied[box_bid, 0] = 0.0

                    mujoco.mj_step(model, data)

            viewer.sync()

            # --- Telemetry ---
            wall_now = time.perf_counter()
            if wall_now - last_push_wall >= (1.0/PUSH_HZ) and not plot_q.full():
                # Re-sample noisy state for telemetry to match sensor performance
                # (We regenerate noise here to visualize what the sensor output looks like)
                pitch = float(data.qpos[s_pitch]) + np.random.normal(0, PITCH_NOISE_STD_RAD)
                pitch_rate = float(data.qvel[d_pitch]) + np.random.normal(0, PITCH_RATE_NOISE_STD_RAD_S)
                plot_q.put_nowait((sim_t, pitch, pitch_rate, data.ctrl[1]))
                last_push_wall = wall_now

            # --- Info text in viewer ---
            pitch = float(data.qpos[s_pitch])
            com = data.subtree_com[box_bid]
            viewer.user_scn.ngeom = 0
            g = viewer.user_scn.geoms[0]
            info_text = (f"Pitch: {math.degrees(pitch):>5.1f} deg\n"
                         f"Tgt Pitch: {math.degrees(target_pitch):>5.1f} deg\n"
                         f"Pos X: {data.qpos[s_root_x]:>5.3f} m\n"
                         f"Wheel Torque: {data.ctrl[1]:>5.2f} Nm\n"
                         f"Speed: 1/{SLOW_MO:.0f}x")
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.008, 0.008, 0.008]),
                np.array([-0.25, 0.15, 0.60]), np.eye(3).flatten(),
                np.array([0.4, 1.0, 0.4, 1.0], dtype=np.float32))
            g.label = info_text.encode()[:99]

            # Vertical line from CG to ground (Magenta)
            g_line = viewer.user_scn.geoms[1]
            mujoco.mjv_initGeom(
                g_line, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.002, com[2]/2.0, 0.0]),
                np.array([com[0], com[1], com[2]/2.0]), np.eye(3).flatten(),
                np.array([1.0, 0.0, 1.0, 0.5]))

            # Sphere at the CG (also magenta)
            g_cg_sphere = viewer.user_scn.geoms[2]
            mujoco.mjv_initGeom(
                g_cg_sphere, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.01, 0, 0]),
                com, np.eye(3).flatten(),
                np.array([1.0, 0.0, 1.0, 0.8]))
            viewer.user_scn.ngeom = 3

            # --- Sync to render framerate ---
            elapsed = time.perf_counter() - frame_start
            sleep_t = 1.0 / RENDER_HZ - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    plot_q.put(None)
    plot_proc.join(timeout=2)