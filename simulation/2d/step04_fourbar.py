"""
Step 04 — Full 4-bar parallelogram leg.

Topology (side view):

    Body
  A ———————— B     A = hip motor pivot (driven, left)
  |          |     B = passive bearing pivot (right, same height)
  | femur    | parallel link
  |          |
  C ———————— D  ← wheel axle — CD stays horizontal at ALL hip angles
       ◎

Parallelogram constraint:
  |AC| = |BD| = FEMUR_L        (link lengths equal)
  |AB| = |CD| = PIVOT_SEP      (top and bottom spans equal)
  AB ∥ CD  →  CD is always horizontal

Wheel centre is at midpoint of CD.
The wheel centre traces a CIRCULAR ARC (not a vertical line) as hip angle
changes — this is physically correct. The parallelogram guarantees the axle
stays level, which is what matters for wheel-balancing.

Slider: hip angle −80° … +80° from vertical downward (0° = leg straight down)

Run:
    python simulation/2d/step04_fourbar.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Parameters (metres)
# ---------------------------------------------------------------------------
BODY_W, BODY_H   = 0.10, 0.10
BODY_CX, BODY_CY = 0.00, 0.20

MOTOR_OD = 0.053
MOTOR_R  = MOTOR_OD / 2

BODY_BOTTOM = BODY_CY - BODY_H / 2       # 0.15 m
MOTOR_CX    = BODY_CX                    # 0.00 m
MOTOR_CY    = BODY_BOTTOM + MOTOR_R      # 0.1765 m

FEMUR_L    = 0.140     # both AC and BD  (140 mm baseline)
PIVOT_SEP  = 0.060     # AB = CD span    (60 mm — distance between the two top pivots)

# Pivot positions (fixed to body)
AX, AY = MOTOR_CX,              MOTOR_CY   # hip motor pivot
BX, BY = MOTOR_CX + PIVOT_SEP,  MOTOR_CY   # passive bearing pivot

HIP_INIT_DEG = 30.0

# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def fourbar_fk(hip_deg: float):
    """
    Forward kinematics.
    Returns: C, D, wheel_centre  (each as (x,y) in metres)
    """
    theta = np.radians(hip_deg)
    cx = AX + FEMUR_L * np.sin(theta)
    cy = AY - FEMUR_L * np.cos(theta)
    dx = BX + FEMUR_L * np.sin(theta)
    dy = BY - FEMUR_L * np.cos(theta)
    wx = (cx + dx) / 2
    wy = (cy + dy) / 2
    return (cx, cy), (dx, dy), (wx, wy)

# Precompute locus for all angles
locus_angles = np.linspace(-80, 80, 300)
locus_pts = [fourbar_fk(a)[2] for a in locus_angles]
locus_x = [p[0] for p in locus_pts]
locus_y = [p[1] for p in locus_pts]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 8))
plt.subplots_adjust(bottom=0.20)
fig.patch.set_facecolor("#1e1e2e")
ax.set_facecolor("#1e1e2e")
ax.set_aspect("equal")
ax.set_xlim(-0.15, 0.30)
ax.set_ylim(-0.05, 0.38)

def m_to_cm(x, _): return f"{x*100:.0f}"
ax.xaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
ax.set_xlabel("x  [cm]", color="lightgray")
ax.set_ylabel("y  [cm]", color="lightgray")
ax.tick_params(colors="lightgray")
for spine in ax.spines.values():
    spine.set_edgecolor("#555")
ax.grid(True, color="#333", linewidth=0.5, linestyle="--")
ax.axhline(0, color="#888", linewidth=1.5)
ax.set_title("Step 04 — 4-bar parallelogram leg", color="white", fontsize=13)

# ---------------------------------------------------------------------------
# Static: body + motor + passive bearing + locus
# ---------------------------------------------------------------------------

# Body box
body_patch = mpatches.FancyBboxPatch(
    (BODY_CX - BODY_W/2, BODY_CY - BODY_H/2), BODY_W, BODY_H,
    boxstyle="square,pad=0",
    linewidth=1.5, edgecolor="#aaa", facecolor="#4a90d9", alpha=0.85, zorder=2
)
ax.add_patch(body_patch)
ax.text(BODY_CX, BODY_CY + 0.005, "body", color="white",
        fontsize=8, ha="center", va="center", zorder=5)

# Hip motor circle
motor_patch = mpatches.Circle(
    (AX, AY), MOTOR_R,
    linewidth=1.5, edgecolor="#aaa", facecolor="#e05c5c", alpha=0.9, zorder=3
)
ax.add_patch(motor_patch)
ax.text(AX, AY, "AK45-10", color="white", fontsize=6, ha="center", va="center", zorder=5)

# Passive bearing B (smaller circle)
bearing_patch = mpatches.Circle(
    (BX, BY), 0.008,
    linewidth=1.5, edgecolor="#aaa", facecolor="#aaa", alpha=0.9, zorder=3
)
ax.add_patch(bearing_patch)

# Pivot labels A, B
ax.text(AX - 0.010, AY + 0.008, "A", color="yellow", fontsize=9, fontweight="bold", zorder=7)
ax.text(BX + 0.004, BY + 0.008, "B", color="#aaa",   fontsize=9, fontweight="bold", zorder=7)

# Wheel centre locus (dashed arc)
ax.plot(locus_x, locus_y, "--", color="#555", linewidth=1, zorder=1, label="wheel centre locus")

# ---------------------------------------------------------------------------
# Dynamic elements (updated by slider)
# ---------------------------------------------------------------------------
C0, D0, W0 = fourbar_fk(HIP_INIT_DEG)

# Links
femur_line, = ax.plot([AX, C0[0]], [AY, C0[1]],
                      color="#f0c040", linewidth=4, solid_capstyle="round",
                      zorder=4, label="femur (AC)")

parallel_line, = ax.plot([BX, D0[0]], [BY, D0[1]],
                         color="#c084f5", linewidth=4, solid_capstyle="round",
                         zorder=4, label="parallel link (BD)")

axle_line, = ax.plot([C0[0], D0[0]], [C0[1], D0[1]],
                     color="#60d060", linewidth=3, solid_capstyle="round",
                     zorder=4, label="wheel axle (CD)")

top_link_line, = ax.plot([AX, BX], [AY, BY],
                         color="#aaa", linewidth=2, linestyle=":",
                         zorder=3, label="body span (AB)")

# Pivot dots
pivot_c, = ax.plot(*C0, "o", color="#f0c040", markersize=7, zorder=6)
pivot_d, = ax.plot(*D0, "o", color="#c084f5",  markersize=7, zorder=6)
wheel_dot, = ax.plot(*W0, "o", color="white",  markersize=5, zorder=7)

# Pivot labels C, D
label_c = ax.text(C0[0] - 0.012, C0[1], "C", color="#f0c040",
                  fontsize=9, fontweight="bold", zorder=8)
label_d = ax.text(D0[0] + 0.004, D0[1], "D", color="#c084f5",
                  fontsize=9, fontweight="bold", zorder=8)

# Wheel circle
WHEEL_R = 0.075  # 150 mm diameter
wheel_patch = mpatches.Circle(W0, WHEEL_R,
                               linewidth=1.5, edgecolor="#60d060",
                               facecolor="#1e1e2e", alpha=0.6, zorder=2)
ax.add_patch(wheel_patch)

# Info text
info_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                    color="lightgray", fontsize=8.5, va="top", family="monospace")

def update_info(hip_deg, C, D, W):
    cd_angle = np.degrees(np.arctan2(D[1]-C[1], D[0]-C[0]))
    info_text.set_text(
        f"hip angle  : {hip_deg:+.1f}°\n"
        f"femur len  : {FEMUR_L*1000:.0f} mm\n"
        f"pivot sep  : {PIVOT_SEP*1000:.0f} mm\n"
        f"C pos      : ({C[0]*100:.1f}, {C[1]*100:.1f}) cm\n"
        f"wheel ctr  : ({W[0]*100:.1f}, {W[1]*100:.1f}) cm\n"
        f"CD angle   : {cd_angle:.2f}°  ({'✓ horizontal' if abs(cd_angle) < 0.01 else '✗'})"
    )

update_info(HIP_INIT_DEG, C0, D0, W0)

ax.legend(loc="upper right", fontsize=7.5,
          facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")

# ---------------------------------------------------------------------------
# Slider
# ---------------------------------------------------------------------------
ax_slider = plt.axes([0.15, 0.09, 0.70, 0.03], facecolor="#2a2a3e")
slider = Slider(ax_slider, "hip angle  [°]", -80, 80,
                valinit=HIP_INIT_DEG, color="#f0c040")
slider.label.set_color("lightgray")
slider.valtext.set_color("lightgray")

def on_slider(val):
    hip_deg = slider.val
    C, D, W = fourbar_fk(hip_deg)

    femur_line.set_data([AX, C[0]], [AY, C[1]])
    parallel_line.set_data([BX, D[0]], [BY, D[1]])
    axle_line.set_data([C[0], D[0]], [C[1], D[1]])
    pivot_c.set_data([C[0]], [C[1]])
    pivot_d.set_data([D[0]], [D[1]])
    wheel_dot.set_data([W[0]], [W[1]])
    wheel_patch.center = W
    label_c.set_position((C[0] - 0.012, C[1]))
    label_d.set_position((D[0] + 0.004, D[1]))
    update_info(hip_deg, C, D, W)
    fig.canvas.draw_idle()

slider.on_changed(on_slider)
update_info(HIP_INIT_DEG, C0, D0, W0)

plt.show()
