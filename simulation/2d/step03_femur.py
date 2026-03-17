"""
Step 03 — Body + AK45-10 hip motor + femur link.

Femur:
  Length = 140 mm (baseline from CLAUDE.md — user queried 150 mm, kept 140 mm)
  Pivot A = centre of hip motor (bottom of body box)
  Pivot C = knee end (free, shown as dot)

Hip angle convention:
  0°  = femur pointing straight down (vertical)
  +θ  = rotated forward (counterclockwise when viewed from the right)
  Range shown: -80° to +80° via slider

Run:
    python simulation/2d/step03_femur.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Parameters (all in metres)
# ---------------------------------------------------------------------------
BODY_W, BODY_H   = 0.10, 0.10
BODY_CX, BODY_CY = 0.00, 0.20

MOTOR_OD = 0.053
MOTOR_R  = MOTOR_OD / 2

BODY_BOTTOM = BODY_CY - BODY_H / 2          # 0.15 m
MOTOR_CX    = BODY_CX
MOTOR_CY    = BODY_BOTTOM + MOTOR_R         # 0.1765 m  (inside box, tangent to bottom)

FEMUR_L = 0.140                              # 140 mm

HIP_ANGLE_INIT_DEG = 30.0                   # starting angle from vertical downward

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def femur_endpoint(hip_deg: float):
    """Return (Cx, Cy) — knee pivot — given hip angle in degrees from vertical."""
    theta = np.radians(hip_deg)
    cx = MOTOR_CX + FEMUR_L * np.sin(theta)
    cy = MOTOR_CY - FEMUR_L * np.cos(theta)
    return cx, cy


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 8))
plt.subplots_adjust(bottom=0.18)
fig.patch.set_facecolor("#1e1e2e")
ax.set_facecolor("#1e1e2e")

ax.set_aspect("equal")
ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-0.05, 0.38)

import matplotlib.ticker as ticker
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
ax.set_title("Step 03 — Body + Hip motor + Femur (140 mm)", color="white", fontsize=13)

# ---------------------------------------------------------------------------
# Static patches: body box
# ---------------------------------------------------------------------------
body_patch = mpatches.FancyBboxPatch(
    (BODY_CX - BODY_W/2, BODY_CY - BODY_H/2), BODY_W, BODY_H,
    boxstyle="square,pad=0",
    linewidth=1.5, edgecolor="#aaa", facecolor="#4a90d9", alpha=0.85, zorder=2
)
ax.add_patch(body_patch)
ax.text(BODY_CX, BODY_CY + 0.005, "body", color="white",
        fontsize=8, ha="center", va="center", zorder=5)

# Motor circle
motor_patch = mpatches.Circle(
    (MOTOR_CX, MOTOR_CY), MOTOR_R,
    linewidth=1.5, edgecolor="#aaa", facecolor="#e05c5c", alpha=0.9, zorder=3
)
ax.add_patch(motor_patch)
ax.text(MOTOR_CX, MOTOR_CY, "AK45-10", color="white",
        fontsize=6.5, ha="center", va="center", zorder=5)

# Pivot A dot (motor centre = hip pivot)
pivot_a, = ax.plot(MOTOR_CX, MOTOR_CY, "o", color="yellow",
                   markersize=5, zorder=6, label="A (hip pivot)")

# ---------------------------------------------------------------------------
# Dynamic: femur link + knee pivot C
# ---------------------------------------------------------------------------
cx0, cy0 = femur_endpoint(HIP_ANGLE_INIT_DEG)

femur_line, = ax.plot(
    [MOTOR_CX, cx0], [MOTOR_CY, cy0],
    color="#f0c040", linewidth=4, solid_capstyle="round", zorder=4, label="femur"
)

pivot_c, = ax.plot(cx0, cy0, "o", color="limegreen",
                   markersize=7, zorder=6, label="C (knee pivot)")

# Labels A / C
label_a = ax.text(MOTOR_CX + 0.008, MOTOR_CY + 0.008, "A", color="yellow",
                  fontsize=9, fontweight="bold", zorder=7)
label_c = ax.text(cx0 + 0.008, cy0, "C", color="limegreen",
                  fontsize=9, fontweight="bold", zorder=7)

# Femur length readout
info_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                    color="lightgray", fontsize=9, va="top", family="monospace")

def update_info(hip_deg, cx, cy):
    length_mm = np.hypot(cx - MOTOR_CX, cy - MOTOR_CY) * 1000
    info_text.set_text(
        f"hip angle : {hip_deg:+.1f}°\n"
        f"femur len : {length_mm:.1f} mm\n"
        f"C pos     : ({cx*100:.1f}, {cy*100:.1f}) cm"
    )

update_info(HIP_ANGLE_INIT_DEG, cx0, cy0)

ax.legend(loc="upper right", fontsize=8,
          facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")

# ---------------------------------------------------------------------------
# Slider
# ---------------------------------------------------------------------------
ax_slider = plt.axes([0.15, 0.07, 0.70, 0.03], facecolor="#2a2a3e")
slider = Slider(ax_slider, "hip angle  [°]", -80, 80,
                valinit=HIP_ANGLE_INIT_DEG, color="#f0c040")
slider.label.set_color("lightgray")
slider.valtext.set_color("lightgray")

def on_slider(val):
    hip_deg = slider.val
    cx, cy = femur_endpoint(hip_deg)
    femur_line.set_data([MOTOR_CX, cx], [MOTOR_CY, cy])
    pivot_c.set_data([cx], [cy])
    label_c.set_position((cx + 0.008, cy))
    update_info(hip_deg, cx, cy)
    fig.canvas.draw_idle()

slider.on_changed(on_slider)
update_info(HIP_ANGLE_INIT_DEG, cx0, cy0)

plt.show()
