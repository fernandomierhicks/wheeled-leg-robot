"""
Step 08b — Motor + femur isolated test.

Body box is FIXED to world.
Hip motor applies constant 0.5 N·m torque until the AK45-10 reaches max speed.

Controls: Play | Pause | Restart
Right panel: motor torque, motor speed, hip angle vs simulation time.

Run:
    python simulation/2d/step08b_motor_femur.py
"""

import numpy as np
import pymunk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
BODY_W, BODY_H = 0.100, 0.100
MOTOR_R = 0.053 / 2                                    # AK45-10 radius
A_LOCAL = np.array([0.0, -BODY_H/2 + MOTOR_R])        # hip pivot in body-local

L_FEMUR      = 0.100     # 10 cm
M_FEMUR      = 0.025     # 25 g PLA link

MAX_OMEGA    = 14.14     # rad/s — AK45-10 output shaft (75 KV × 18 V / 10:1 × 2π/60)

# PD position controller
MAX_TORQUE_CTRL = 0.5    # N·m — torque limit (low enough to see accel/decel)
KP              = 2.0    # position gain
KD              = 0.05   # damping / velocity gain

# Square-wave target
TARGETS_DEG = [0.0, 180.0]   # alternating positions
HOLD_TIME   = 1.0            # sim-seconds to hold at each target before switching

DT         = 1 / 1000   # physics timestep [s]
FPS        = 60
BASE_STEPS = round(1.0 / (DT * FPS))   # steps/frame for 1× real-time (= 17)

# ---------------------------------------------------------------------------
# Build a fresh pymunk space
# ---------------------------------------------------------------------------
def build():
    space = pymunk.Space()
    space.gravity = (0.0, -9.81)
    space.damping = 0.999

    # Body box — STATIC (fixed to world)
    box = pymunk.Body(body_type=pymunk.Body.STATIC)
    box.position = (0.0, 0.0)
    box_shape = pymunk.Poly.create_box(box, (BODY_W, BODY_H))
    box_shape.filter = pymunk.ShapeFilter(group=1)
    space.add(box, box_shape)

    # Hip pivot A in world coords
    A_w = np.array(box.position) + A_LOCAL

    # Femur — starts hanging straight down (pymunk angle = -π/2)
    init_angle = -np.pi / 2
    half_f     = L_FEMUR / 2
    femur_com  = A_w + half_f * np.array([np.cos(init_angle), np.sin(init_angle)])

    moment_f = pymunk.moment_for_segment(M_FEMUR, (-half_f, 0), (half_f, 0), 0.006)
    femur = pymunk.Body(M_FEMUR, moment_f)
    femur.position = tuple(femur_com)
    femur.angle    = init_angle

    seg = pymunk.Segment(femur, (-half_f, 0), (half_f, 0), 0.006)
    seg.filter = pymunk.ShapeFilter(group=1)
    space.add(femur, seg)

    # Pin femur to box at A
    pivot = pymunk.PivotJoint(box, femur, tuple(A_w))
    space.add(pivot)

    return space, box, femur, A_w

# ---------------------------------------------------------------------------
# Simulation state  (mutable container so callbacks can update it cleanly)
# ---------------------------------------------------------------------------
state = {
    "space":   None,
    "box":     None,
    "femur":   None,
    "A_world": None,
    "running": False,
    "time":    0.0,
    "t":       [0.0],
    "torque":  [0.0],
    "omega":   [0.0],
    "hip":     [0.0],
}

def reset_state():
    sp, bx, fm, Aw = build()
    sq["idx"] = 0
    sq["last_switch"] = 0.0
    state.update(
        space=sp, box=bx, femur=fm, A_world=Aw,
        running=False, time=0.0,
        t=[0.0], torque=[0.0], omega=[0.0], hip=[0.0], hip_target=[0.0],
    )

sq = {"idx": 0, "last_switch": 0.0}   # square-wave target state

reset_state()

def get_hip_deg():
    return float(np.degrees(state["femur"].angle + np.pi / 2))

def get_target_deg():
    return float(TARGETS_DEG[sq["idx"] % len(TARGETS_DEG)])

def step_target(sim_time):
    if sim_time - sq["last_switch"] >= HOLD_TIME:
        sq["idx"] += 1
        sq["last_switch"] = sim_time

def compute_torque(hip_rad, omega, sim_time):
    """PD position controller, torque-limited."""
    step_target(sim_time)
    target_rad = np.radians(get_target_deg())
    error = target_rad - hip_rad
    error = (error + np.pi) % (2 * np.pi) - np.pi   # wrap to [-π, π]
    tau = KP * error - KD * omega
    return float(np.clip(tau, -MAX_TORQUE_CTRL, MAX_TORQUE_CTRL))

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor("#1e1e2e")
plt.subplots_adjust(left=0.06, right=0.97, bottom=0.12, top=0.92,
                    wspace=0.35, hspace=0.55)

ax_sim    = fig.add_subplot(1, 2, 1)
ax_torque = fig.add_subplot(3, 2, 2)
ax_omega  = fig.add_subplot(3, 2, 4)
ax_hip    = fig.add_subplot(3, 2, 6)

# --- Sim panel style ---
ax_sim.set_facecolor("#1e1e2e")
ax_sim.set_aspect("equal")
ax_sim.set_xlim(-0.20, 0.20)
ax_sim.set_ylim(-0.22, 0.20)

def m_to_cm(v, _): return f"{v*100:.0f}"
ax_sim.xaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
ax_sim.yaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
ax_sim.set_xlabel("x [cm]", color="lightgray")
ax_sim.set_ylabel("y [cm]", color="lightgray")
ax_sim.tick_params(colors="lightgray")
for sp in ax_sim.spines.values(): sp.set_edgecolor("#555")
ax_sim.grid(True, color="#333", linewidth=0.4, linestyle="--")
ax_sim.axhline(0, color="#888", linewidth=1.0)
ax_sim.set_title("Body (fixed) + hip motor + femur", color="white", fontsize=11)

# --- Telemetry panel styles ---
plot_specs = [
    (ax_torque, "Motor torque",  "torque [N·m]",  "#f0c040"),
    (ax_omega,  "Motor speed",   "speed [rad/s]", "#c084f5"),
    (ax_hip,    "Hip angle",     "angle [deg]",   "#60d0ff"),
]
for ax, title, ylabel, col in plot_specs:
    ax.set_facecolor("#1e1e2e")
    ax.tick_params(colors="lightgray", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#555")
    ax.grid(True, color="#333", linewidth=0.4, linestyle="--")
    ax.set_title(title, color="white", fontsize=9)
    ax.set_ylabel(ylabel, color=col, fontsize=8)
    ax.set_xlabel("time [s]", color="lightgray", fontsize=8)
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.1, 1)

ax_torque.set_ylim(-MAX_TORQUE_CTRL * 1.5, MAX_TORQUE_CTRL * 1.5)
ax_omega.set_ylim(-0.5, MAX_OMEGA * 1.3)
ax_omega.axhline(MAX_OMEGA, color="#ff6060", linewidth=0.8, linestyle="--",
                 alpha=0.7, label=f"max {MAX_OMEGA:.1f} rad/s")
ax_omega.legend(fontsize=7, facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")
ax_hip.set_ylim(-200, 200)

# ---------------------------------------------------------------------------
# Static visual elements
# ---------------------------------------------------------------------------
# Body box (stays fixed)
ax_sim.add_patch(mpatches.FancyBboxPatch(
    (-BODY_W/2, -BODY_H/2), BODY_W, BODY_H,
    boxstyle="square,pad=0", linewidth=1.5,
    edgecolor="#aaa", facecolor="#4a90d9", alpha=0.88, zorder=3))
ax_sim.text(0, 0.005, "body", color="white", fontsize=8,
            ha="center", va="center", zorder=5)

# Hip motor circle
A_w = state["A_world"]
ax_sim.add_patch(mpatches.Circle(tuple(A_w), MOTOR_R,
    linewidth=1.2, edgecolor="#aaa", facecolor="#e05c5c", alpha=0.9, zorder=4))
ax_sim.text(*A_w, "AK45-10", color="white", fontsize=6,
            ha="center", va="center", zorder=6)
ax_sim.plot(*A_w, "o", color="yellow", markersize=5, zorder=7)
ax_sim.text(A_w[0] - 0.013, A_w[1] + 0.007, "A",
            color="yellow", fontsize=9, fontweight="bold", zorder=8)

# ---------------------------------------------------------------------------
# Dynamic visual elements (femur)
# ---------------------------------------------------------------------------
C_init = A_w + L_FEMUR * np.array([np.cos(-np.pi/2), np.sin(-np.pi/2)])

femur_art, = ax_sim.plot([A_w[0], C_init[0]], [A_w[1], C_init[1]],
                          color="#f0c040", linewidth=5,
                          solid_capstyle="round", zorder=4, label="femur")
knee_dot,  = ax_sim.plot([C_init[0]], [C_init[1]],
                          "o", color="white", markersize=7, zorder=6)
knee_lbl   = ax_sim.text(C_init[0]+0.008, C_init[1], "C",
                          color="white", fontsize=8, fontweight="bold", zorder=9)

# Telemetry lines
torque_line,      = ax_torque.plot([], [], color="#f0c040", linewidth=1.5)
omega_line,       = ax_omega.plot([], [],  color="#c084f5", linewidth=1.5)
hip_line,         = ax_hip.plot([], [],    color="#60d0ff", linewidth=1.5, label="actual")
hip_target_line,  = ax_hip.plot([], [],    color="#ff6060", linewidth=1.0,
                                 linestyle="--", label="target")
ax_hip.legend(fontsize=7, facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")

# Status / info text
status_txt = ax_sim.text(0.02, 0.98, "PAUSED — press Play",
                          transform=ax_sim.transAxes, color="#aaaaaa",
                          fontsize=9, va="top", family="monospace")
info_txt   = ax_sim.text(0.02, 0.88, "", transform=ax_sim.transAxes,
                          color="lightgray", fontsize=8.5, va="top",
                          family="monospace")

# ---------------------------------------------------------------------------
# Buttons
# ---------------------------------------------------------------------------
bstyle = dict(color="#2a2a3e", hovercolor="#4a4a6e")
ax_bp = plt.axes([0.22, 0.02, 0.07, 0.045]); btn_play  = Button(ax_bp, "Play",    **bstyle)
ax_bu = plt.axes([0.31, 0.02, 0.07, 0.045]); btn_pause = Button(ax_bu, "Pause",   **bstyle)
ax_br = plt.axes([0.40, 0.02, 0.07, 0.045]); btn_reset = Button(ax_br, "Restart", **bstyle)

# Speed multiplier slider  (0.1× … 5×, default 1×)
ax_spd  = plt.axes([0.54, 0.025, 0.22, 0.025], facecolor="#2a2a3e")
spd_slider = Slider(ax_spd, "speed", 0.1, 5.0, valinit=1.0, valstep=0.1, color="#60d0ff")
spd_slider.label.set_color("lightgray")
spd_slider.valtext.set_color("lightgray")

for b in [btn_play, btn_pause, btn_reset]:
    b.label.set_color("white")
    b.label.set_fontsize(9)

def on_play(event):
    state["running"] = True
    status_txt.set_text("RUNNING")
    status_txt.set_color("#60d060")
    fig.canvas.draw_idle()

def on_pause(event):
    state["running"] = False
    status_txt.set_text("PAUSED")
    status_txt.set_color("#aaaaaa")
    fig.canvas.draw_idle()

def on_restart(event):
    reset_state()
    # Reset visual
    C_r = state["A_world"] + L_FEMUR * np.array([0.0, -1.0])
    femur_art.set_data([state["A_world"][0], C_r[0]],
                       [state["A_world"][1], C_r[1]])
    knee_dot.set_data([C_r[0]], [C_r[1]])
    knee_lbl.set_position((C_r[0]+0.008, C_r[1]))
    for ln in [torque_line, omega_line, hip_line, hip_target_line]:
        ln.set_data([], [])
    for ax in [ax_torque, ax_omega, ax_hip]:
        ax.set_xlim(0, 5)
    ax_hip.set_ylim(-200, 200)
    status_txt.set_text("PAUSED — press Play")
    status_txt.set_color("#aaaaaa")
    info_txt.set_text("")
    fig.canvas.draw_idle()

btn_play.on_clicked(on_play)
btn_pause.on_clicked(on_pause)
btn_reset.on_clicked(on_restart)

# ---------------------------------------------------------------------------
# Animation loop
# ---------------------------------------------------------------------------
def animate(_frame):
    if not state["running"]:
        return (femur_art, knee_dot, torque_line, omega_line, hip_line)

    femur = state["femur"]
    sp    = state["space"]

    steps = max(1, round(spd_slider.val * BASE_STEPS))
    for _ in range(steps):
        hip_rad = femur.angle + np.pi / 2
        omega   = float(femur.angular_velocity)
        tau     = compute_torque(hip_rad, omega, state["time"])
        femur.torque = tau
        sp.step(DT)
        state["time"] += DT

    # Current state
    hip_rad = femur.angle + np.pi / 2
    omega   = float(femur.angular_velocity)
    hip_deg = get_hip_deg()
    tau     = compute_torque(hip_rad, omega, state["time"])

    # Femur endpoints from body state
    fa  = femur.angle
    fd  = np.array([np.cos(fa), np.sin(fa)])
    fc  = np.array(femur.position)
    A_e = fc - (L_FEMUR/2) * fd    # A end (pivot)
    C_e = fc + (L_FEMUR/2) * fd    # C end (knee tip)

    # Record history
    t = state["time"]
    state["t"].append(t)
    state["torque"].append(tau)
    state["omega"].append(omega)
    state["hip"].append(hip_deg)
    state["hip_target"].append(get_target_deg())

    # Update simulation visuals
    femur_art.set_data([A_e[0], C_e[0]], [A_e[1], C_e[1]])
    knee_dot.set_data([C_e[0]], [C_e[1]])
    knee_lbl.set_position((C_e[0]+0.008, C_e[1]))

    # Update telemetry plots
    th = state["t"]
    torque_line.set_data(th, state["torque"])
    omega_line.set_data(th,  state["omega"])
    hip_line.set_data(th,    state["hip"])
    hip_target_line.set_data(th, state["hip_target"])

    # Auto-scale: explicit set_ylim from data
    x_max = max(5.0, t + 0.5)
    for ax in [ax_torque, ax_omega, ax_hip]:
        ax.set_xlim(0, x_max)
    for ax, data in [(ax_torque, state["torque"]),
                     (ax_omega,  state["omega"]),
                     (ax_hip,    state["hip"] + state["hip_target"])]:
        if len(data) > 1:
            lo, hi = min(data), max(data)
            span = max(hi - lo, 0.1)
            ax.set_ylim(lo - span * 0.15, hi + span * 0.15)

    info_txt.set_text(
        f"t      = {t:.3f} s\n"
        f"omega  = {omega:+.2f} rad/s\n"
        f"torque = {tau:.2f} N·m\n"
        f"hip    = {hip_deg:+.1f} deg"
    )

    return (femur_art, knee_dot, torque_line, omega_line, hip_line)

ani = animation.FuncAnimation(
    fig, animate,
    interval=int(1000 / FPS),
    blit=False,
    cache_frame_data=False,
)

fig.suptitle(
    f"Motor model  |  τ_max = {MAX_TORQUE_CTRL} N·m  ω_max = {MAX_OMEGA:.1f} rad/s  "
    f"(AK45-10, 10:1, 18 V)",
    color="white", fontsize=11,
)
plt.show()
