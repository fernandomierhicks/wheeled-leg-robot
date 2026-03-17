"""
Step 08 — Jump physics simulation (pymunk rigid body).

Single-leg, body constrained to vertical-only motion.
Simulates one leg + half robot mass — mirrors the symmetric 2-leg jump.

Topology (from step07):
    A  hip motor pivot  (body-fixed)
    F  passive pivot    (body-fixed, offset from A)
    Femur  A → C       (driven by motor at A)
    Tibia  E – C – W   (E = 1.5 cm stub above knee, W = wheel)
    Coupler F → E      (passive, closes the 4-bar)

Constraints:
    - Body box moves on vertical-only groove (no balance controller needed)
    - Body angle locked at 0 (no tipping)
    - Hip motor: 7 N·m peak, 14.1 rad/s max output (AK45-10, 10:1, 18 V)

Run:
    python simulation/2d/step08_jump_sim.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pymunk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# Step07 geometry
L_femur = 0.100    # A → C  [m]
L_stub  = 0.015    # C → E  stub above knee  [m]
L_tibia = 0.115    # C → W  below knee  [m]
Lc      = 0.110    # F → E  coupler  [m]
F_dx    = +0.015   # F relative to A  [m]
F_dy    = +0.026
WHEEL_R = 0.075    # wheel radius  [m]

# Body box
BODY_W, BODY_H = 0.100, 0.100
MOTOR_R = 0.053 / 2   # AK45-10

# A and F in body-local coords (body CoM = box centre)
A_LOCAL = np.array([0.0,   -BODY_H/2 + MOTOR_R])   # (0, -0.0235)
F_LOCAL = A_LOCAL + np.array([F_dx, F_dy])           # (0.015, 0.0025)

# Mass budget  (one leg, total = robot/2 ≈ 1.745 kg)
M_BOX     = 1.260   # body incl. hip motor (bolted)
M_FEMUR   = 0.025
M_TIBIA   = 0.035   # full rod E→W
M_COUPLER = 0.015
M_WHEEL   = 0.410   # Maytech 380 g + TPU wheel 30 g

# Motor
MAX_TORQUE = 7.0    # N·m  AK45-10 peak
MAX_RATE   = 14.1   # rad/s  output shaft (75 KV × 18 V / 10:1 gear × 2π/60)

# Sim
HIP_START    = 50.0   # deg  crouched
HIP_STOP     = 5.0    # deg  cut motor at full extension
DT           = 1/1000 # physics step [s]
SIM_STEPS    = 10     # physics steps per animation frame
FPS          = 60
SIM_DURATION = 1.5    # seconds total

# ---------------------------------------------------------------------------
# Step-07 FK solver (inline — no file dependency)
# ---------------------------------------------------------------------------
def mechanism_fk_world(hip_rad, A_w, F_w):
    """
    Solve 4-bar FK given hip angle and world positions of A, F.
    Returns (C, E, W, phi) in world metres, or (None,None,None,None).
    """
    C = A_w + L_femur * np.array([np.sin(hip_rad), -np.cos(hip_rad)])

    def residual(phi):
        E = C + L_stub * np.array([-np.sin(phi), np.cos(phi)])
        d = F_w - E
        return float(np.dot(d, d)) - Lc**2

    phi_ideal = -hip_rad
    phis = np.linspace(-np.pi, np.pi, 720)
    rs   = [residual(p) for p in phis]

    brackets = [(phis[i], phis[i+1])
                for i in range(len(phis)-1)
                if np.sign(rs[i]) != np.sign(rs[i+1])]
    if not brackets:
        return None, None, None, None

    roots = []
    for lo, hi in brackets:
        try:
            roots.append(brentq(residual, lo, hi, xtol=1e-9))
        except ValueError:
            pass
    if not roots:
        return None, None, None, None

    phi = min(roots, key=lambda r: abs(r - phi_ideal))
    E = C + L_stub  * np.array([-np.sin(phi),  np.cos(phi)])
    W = C + L_tibia * np.array([ np.sin(phi), -np.cos(phi)])
    return C, E, W, phi

# ---------------------------------------------------------------------------
# Compute initial (crouched) world positions
# ---------------------------------------------------------------------------
hip0_rad = np.radians(HIP_START)

# Compute with A temporarily at origin, then shift so wheel touches ground
A_tmp = np.array([0.0, 0.0])
F_tmp = A_tmp + np.array([F_dx, F_dy])
C0, E0, W0, phi0 = mechanism_fk_world(hip0_rad, A_tmp, F_tmp)
if C0 is None:
    raise RuntimeError(f"FK failed at hip={HIP_START} deg — check parameters")

# Shift body up so wheel centre is at y = WHEEL_R
shift_y = WHEEL_R - W0[1]
A_w0 = A_tmp + np.array([0.0, shift_y])
F_w0 = F_tmp + np.array([0.0, shift_y])
C_w0 = C0    + np.array([0.0, shift_y])
E_w0 = E0    + np.array([0.0, shift_y])
W_w0 = W0    + np.array([0.0, shift_y])

box_com0 = A_w0 - A_LOCAL   # body CoM in world

print("Initial positions:")
print(f"  box CoM  ({box_com0[0]*100:.1f}, {box_com0[1]*100:.1f}) cm")
print(f"  A        ({A_w0[0]*100:.1f}, {A_w0[1]*100:.1f}) cm")
print(f"  F        ({F_w0[0]*100:.1f}, {F_w0[1]*100:.1f}) cm")
print(f"  W (wheel)({W_w0[0]*100:.1f}, {W_w0[1]*100:.1f}) cm")
print(f"  phi0     {np.degrees(phi0):.1f} deg")
print(f"  Total mass: {M_BOX+M_FEMUR+M_TIBIA+M_COUPLER+M_WHEEL:.3f} kg")

# ---------------------------------------------------------------------------
# Pymunk space
# ---------------------------------------------------------------------------
space = pymunk.Space()
space.gravity = (0.0, -9.81)
space.damping = 0.999   # negligible air damping

# Ground
ground = pymunk.Segment(space.static_body, (-3.0, 0.0), (3.0, 0.0), 0.003)
ground.friction   = 1.0
ground.elasticity = 0.0
space.add(ground)

# ---------------------------------------------------------------------------
# Helper: rigid link body
# ---------------------------------------------------------------------------
def make_link(mass, p1_w, p2_w, vis_radius=0.006):
    """
    Rigid link from p1_w to p2_w (world coords).
    CoM at midpoint; body +x axis points p1→p2.
    Returns (pymunk.Body, pymunk.Segment).
    """
    com   = (p1_w + p2_w) / 2.0
    half  = np.linalg.norm(p2_w - p1_w) / 2.0
    angle = float(np.arctan2(p2_w[1] - p1_w[1], p2_w[0] - p1_w[0]))
    mom   = pymunk.moment_for_segment(mass, (-half, 0), (half, 0), vis_radius)
    body  = pymunk.Body(mass, mom)
    body.position = tuple(com)
    body.angle    = angle
    seg   = pymunk.Segment(body, (-half, 0), (half, 0), vis_radius)
    seg.filter    = pymunk.ShapeFilter(group=1)   # no robot self-collision
    return body, seg

# ---------------------------------------------------------------------------
# Create bodies
# ---------------------------------------------------------------------------

# 1. Body box
box_body  = pymunk.Body(M_BOX, pymunk.moment_for_box(M_BOX, (BODY_W, BODY_H)))
box_body.position = tuple(box_com0)
box_body.angle    = 0.0
box_shape = pymunk.Poly.create_box(box_body, (BODY_W, BODY_H))
box_shape.filter  = pymunk.ShapeFilter(group=1)
space.add(box_body, box_shape)

# 2. Femur  A → C
femur_body, femur_seg = make_link(M_FEMUR, A_w0, C_w0)
space.add(femur_body, femur_seg)

# 3. Tibia  E → C → W  (full rod E to W; C is NOT the midpoint)
tibia_body, tibia_seg = make_link(M_TIBIA, E_w0, W_w0)
space.add(tibia_body, tibia_seg)

# 4. Coupler  F → E
coupler_body, coupler_seg = make_link(M_COUPLER, F_w0, E_w0)
space.add(coupler_body, coupler_seg)

# 5. Wheel assembly  (circle)
wheel_body  = pymunk.Body(M_WHEEL, pymunk.moment_for_circle(M_WHEEL, 0, WHEEL_R))
wheel_body.position = tuple(W_w0)
wheel_shape = pymunk.Circle(wheel_body, WHEEL_R)
wheel_shape.friction   = 1.0
wheel_shape.elasticity = 0.0
wheel_shape.filter     = pymunk.ShapeFilter(group=1)
space.add(wheel_body, wheel_shape)

# ---------------------------------------------------------------------------
# Joints
# ---------------------------------------------------------------------------

# --- Vertical constraint on body box ---
# GrooveJoint: CoM of box_body slides along x=0 vertical line in world
groove = pymunk.GrooveJoint(
    space.static_body, box_body,
    (0.0, -6.0), (0.0,  6.0),  # groove endpoints in static body (world) coords
    (0.0,  0.0)                 # anchor on box_body local coords = CoM
)
# Lock box angle to 0
angle_lock = pymunk.RotaryLimitJoint(space.static_body, box_body, 0.0, 0.0)
space.add(groove, angle_lock)

# --- 4-bar mechanism joints ---
j_hip     = pymunk.PivotJoint(box_body,    femur_body,   tuple(A_w0))  # A
j_knee    = pymunk.PivotJoint(femur_body,  tibia_body,   tuple(C_w0))  # C
j_F_body  = pymunk.PivotJoint(box_body,    coupler_body, tuple(F_w0))  # F
j_E_coup  = pymunk.PivotJoint(coupler_body, tibia_body,  tuple(E_w0))  # E
j_axle    = pymunk.PivotJoint(tibia_body,  wheel_body,   tuple(W_w0))  # W
space.add(j_hip, j_knee, j_F_body, j_E_coup, j_axle)

# --- Hip motor ---
# rate < 0: femur rotates CW relative to box → leg extends (hip angle decreases)
motor = pymunk.SimpleMotor(box_body, femur_body, -MAX_RATE)
motor.max_force = MAX_TORQUE
space.add(motor)

# ---------------------------------------------------------------------------
# Helper: extract joint world positions from current body states
# ---------------------------------------------------------------------------
half_f = L_femur / 2.0
half_t = (L_stub + L_tibia) / 2.0
half_c = Lc / 2.0

def get_joint_positions():
    """
    Return world-coord positions of A, C, E, W, F from body states.
    Convention from make_link(p1, p2): p1 is at local (-half, 0), p2 at (+half, 0).
    """
    bpos = np.array(box_body.position)   # box CoM (angle locked to 0)
    A_w = bpos + A_LOCAL
    F_w = bpos + F_LOCAL

    fa = femur_body.angle
    fd = np.array([np.cos(fa), np.sin(fa)])
    fc = np.array(femur_body.position)
    C_w = fc + half_f * fd          # p2 end of femur = C

    ta = tibia_body.angle
    td = np.array([np.cos(ta), np.sin(ta)])
    tc = np.array(tibia_body.position)
    E_w = tc - half_t * td          # p1 end of tibia = E
    W_w = tc + half_t * td          # p2 end of tibia = W

    return A_w, C_w, E_w, W_w, F_w

def get_hip_deg():
    """Hip angle in degrees from femur body angle."""
    # femur body angle = hip_rad - pi/2  →  hip_rad = femur.angle + pi/2
    return float(np.degrees(femur_body.angle + np.pi / 2))

# ---------------------------------------------------------------------------
# Telemetry storage
# ---------------------------------------------------------------------------
t_hist      = [0.0]
height_hist = [0.0]   # body CoM above initial, mm
hip_hist    = [HIP_START]
torque_hist = [MAX_TORQUE]

initial_y = float(box_body.position.y)
max_height_mm = 0.0
sim_time      = 0.0
phase         = "crouched"   # crouched → thrusting → airborne → done

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor("#1e1e2e")
plt.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.91, wspace=0.30)

ax_sim  = fig.add_subplot(1, 2, 1)
ax_plot = fig.add_subplot(1, 2, 2)

# --- Sim panel ---
ax_sim.set_facecolor("#1e1e2e")
ax_sim.set_aspect("equal")
ax_sim.set_xlim(-0.28, 0.28)
ax_sim.set_ylim(-0.05, 0.75)

def m_to_cm(v, _): return f"{v*100:.0f}"
for ax in [ax_sim]:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
    ax.set_xlabel("x [cm]", color="lightgray")
    ax.set_ylabel("y [cm]", color="lightgray")
    ax.tick_params(colors="lightgray")
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")
    ax.grid(True, color="#333", linewidth=0.4, linestyle="--")

ax_sim.axhline(0.0, color="#888", linewidth=2.0)
ax_sim.axvline(0.0, color="#40ff40", linewidth=0.7, linestyle="--", alpha=0.45,
               label="x = 0")
ax_sim.set_title("Step 08 — Jump Simulation (pymunk)", color="white", fontsize=11)
ax_sim.text(0.22, 0.003, "ground", color="#777", fontsize=8)

# --- Plot panel ---
ax_plot.set_facecolor("#1e1e2e")
ax_plot.tick_params(colors="lightgray")
for sp in ax_plot.spines.values():
    sp.set_edgecolor("#555")
ax_plot.grid(True, color="#333", linewidth=0.4, linestyle="--")
ax_plot.set_xlabel("time [s]", color="lightgray")
ax_plot.set_ylabel("body CoM height above start [mm]", color="#60d0ff")
ax_plot.set_title("Jump height trace", color="white", fontsize=11)
ax_plot.set_xlim(0.0, SIM_DURATION)
ax_plot.set_ylim(-10, 350)

# --- Dynamic artists ---
# Body rectangle
box_patch = mpatches.Rectangle(
    (box_com0[0] - BODY_W/2, box_com0[1] - BODY_H/2),
    BODY_W, BODY_H,
    linewidth=1.5, edgecolor="#aaa", facecolor="#4a90d9", alpha=0.88, zorder=3
)
ax_sim.add_patch(box_patch)

# Motor circle (drawn on body, moves with box)
motor_circ = mpatches.Circle(
    tuple(A_w0), MOTOR_R,
    linewidth=1.2, edgecolor="#aaa", facecolor="#e05c5c", alpha=0.9, zorder=4
)
ax_sim.add_patch(motor_circ)

# Links
femur_art,   = ax_sim.plot([], [], color="#f0c040", linewidth=5,
                            solid_capstyle="round", zorder=4)
stub_art,    = ax_sim.plot([], [], color="#c084f5", linewidth=4,
                            solid_capstyle="round", zorder=4)
tibia_art,   = ax_sim.plot([], [], color="#c084f5", linewidth=5,
                            solid_capstyle="round", zorder=4)
coupler_art, = ax_sim.plot([], [], color="#aafafa", linewidth=2.5,
                            linestyle="--", zorder=4)

# Wheel
wheel_art = mpatches.Circle(
    tuple(W_w0), WHEEL_R,
    linewidth=1.5, edgecolor="limegreen", facecolor="#1e1e2e", alpha=0.45, zorder=2
)
ax_sim.add_patch(wheel_art)

# Pivots
piv_colors = ["yellow", "white", "#aafafa", "#aafafa", "limegreen"]
piv_arts   = [ax_sim.plot([], [], "o", color=c, markersize=6, zorder=7)[0]
              for c in piv_colors]   # A, C, E, W, F

# Height trace
height_line, = ax_plot.plot([], [], color="#60d0ff", linewidth=2.0)
ax_plot.axhline(0, color="#555", linewidth=0.8, linestyle="--")

# Info text
max_text  = ax_sim.text(0.02, 0.97, "", transform=ax_sim.transAxes,
                         color="white", fontsize=11, va="top",
                         family="monospace", fontweight="bold")
info_text = ax_sim.text(0.02, 0.87, "", transform=ax_sim.transAxes,
                         color="lightgray", fontsize=8.5, va="top",
                         family="monospace")

# ---------------------------------------------------------------------------
# Animation update
# ---------------------------------------------------------------------------
TOTAL_FRAMES = int(SIM_DURATION * FPS)

def animate(frame):
    global sim_time, max_height_mm, phase

    motor_on = False
    actual_torque = 0.0

    for _ in range(SIM_STEPS):
        hip_deg  = get_hip_deg()
        wheel_y  = float(wheel_body.position.y)
        on_ground = wheel_y <= WHEEL_R + 0.008   # 8 mm tolerance

        if on_ground and HIP_STOP < hip_deg < 90.0:
            motor.max_force = MAX_TORQUE
            motor.rate      = -MAX_RATE
            motor_on        = True
            phase           = "thrusting"
        else:
            motor.max_force  = 0.0
            if not on_ground:
                phase = "airborne"

        space.step(DT)
        sim_time += DT

    # Telemetry
    current_y  = float(box_body.position.y)
    height_mm  = (current_y - initial_y) * 1000.0
    max_height_mm = max(max_height_mm, height_mm)
    t_hist.append(sim_time)
    height_hist.append(height_mm)
    hip_hist.append(get_hip_deg())

    # Joint world positions
    A_w, C_w, E_w, W_w, F_w = get_joint_positions()
    bpos = np.array(box_body.position)

    # Update visual artists
    box_patch.set_xy((bpos[0] - BODY_W/2, bpos[1] - BODY_H/2))
    motor_circ.center = tuple(A_w)

    femur_art.set_data([A_w[0], C_w[0]], [A_w[1], C_w[1]])
    stub_art.set_data( [E_w[0], C_w[0]], [E_w[1], C_w[1]])
    tibia_art.set_data([C_w[0], W_w[0]], [C_w[1], W_w[1]])
    coupler_art.set_data([F_w[0], E_w[0]], [F_w[1], E_w[1]])
    wheel_art.center = tuple(W_w)

    for pt, art in zip([A_w, C_w, E_w, W_w, F_w], piv_arts):
        art.set_data([pt[0]], [pt[1]])

    # Expand sim panel y-limit if needed
    top_y = bpos[1] + BODY_H
    if top_y + 0.05 > ax_sim.get_ylim()[1]:
        ax_sim.set_ylim(-0.05, top_y + 0.10)

    # Update plot
    height_line.set_data(t_hist, height_hist)
    if max_height_mm > 50:
        ax_plot.set_ylim(-10, max_height_mm * 1.25)

    max_text.set_text(f"Max: {max_height_mm:.1f} mm")
    info_text.set_text(
        f"t       = {sim_time:.3f} s\n"
        f"phase   = {phase}\n"
        f"hip     = {get_hip_deg():.1f} deg\n"
        f"motor   = {'ON  ' if motor_on else 'OFF '}"
        f"  ({MAX_TORQUE if motor_on else 0.0:.1f} N·m)\n"
        f"body y  = {current_y*100:.1f} cm\n"
        f"wheel y = {float(wheel_body.position.y)*100:.1f} cm"
    )

    return (box_patch, motor_circ, femur_art, stub_art, tibia_art,
            coupler_art, wheel_art, height_line, max_text, info_text,
            *piv_arts)

ani = animation.FuncAnimation(
    fig, animate, frames=TOTAL_FRAMES,
    interval=int(1000 / FPS), blit=False, repeat=False
)

fig.suptitle(
    f"Jump sim — half robot  |  mass={M_BOX+M_FEMUR+M_TIBIA+M_COUPLER+M_WHEEL:.2f} kg"
    f"  |  motor {MAX_TORQUE:.0f} N·m  {np.degrees(MAX_RATE):.0f} °/s",
    color="white", fontsize=11
)
plt.show()

# Print summary after window closes
print(f"\n--- Jump result ---")
print(f"  Max body CoM height above start : {max_height_mm:.1f} mm")
print(f"  Sim duration                    : {sim_time:.2f} s")
if height_hist:
    liftoff_times = [t_hist[i] for i, h in enumerate(height_hist) if h > 5]
    if liftoff_times:
        print(f"  Approx liftoff time             : {liftoff_times[0]:.3f} s")
