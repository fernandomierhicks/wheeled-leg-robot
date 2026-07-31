"""
Step 05 — Simple femur + tibia leg, single motor, vertical wheel motion.

Topology:
    A  ← hip motor pivot (bottom-centre of body)
    |
    | femur (140 mm)
    |
    C  ← knee
    |
    | tibia (140 mm)
    |
    W  ← wheel centre

Key insight:
  With femur_len == tibia_len == L and knee_angle = -2 * hip_angle,
  the wheel centre W stays on the vertical line x = 0 (directly below body centre).

  Proof:
    knee  = A + L*(sin θ,        -cos θ)
    wheel = A + L*(sin θ,        -cos θ)
              + L*(sin(θ + θ_k), -cos(θ + θ_k))
    with θ_k = -2θ:
    wheel_x = L*sin(θ) + L*sin(-θ) = 0  ✓
    wheel_y = A_y - 2L*cos(θ)            (pure vertical)

  This constraint (θ_knee = -2·θ_hip) is what a 4-bar or belt/gear mechanism
  must enforce mechanically.

Slider: hip angle 0° … 70° (0° = leg fully extended, 70° = fully retracted)

Run:
    python simulation/2d/step05_femur_tibia.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider

# ---------------------------------------------------------------------------
# Parameters (metres)
# ---------------------------------------------------------------------------
BODY_W, BODY_H   = 0.10, 0.10
BODY_CX, BODY_CY = 0.00, 0.20

MOTOR_OD  = 0.053
MOTOR_R   = MOTOR_OD / 2
BODY_BOTTOM = BODY_CY - BODY_H / 2
AX = BODY_CX
AY = BODY_BOTTOM + MOTOR_R        # hip pivot = motor centre

FEMUR_L  = 0.140                  # 140 mm
TIBIA_L  = 0.140                  # 140 mm (equal → vertical constraint works)
WHEEL_R  = 0.075                  # 150 mm wheel

HIP_INIT_DEG = 20.0

# ---------------------------------------------------------------------------
# Forward kinematics  (θ_knee = -2·θ_hip enforced)
# ---------------------------------------------------------------------------
def leg_fk(hip_deg: float):
    th_hip  = np.radians(hip_deg)
    th_knee = -2.0 * th_hip          # the magic constraint

    kx = AX + FEMUR_L * np.sin(th_hip)
    ky = AY - FEMUR_L * np.cos(th_hip)

    wx = kx + TIBIA_L * np.sin(th_hip + th_knee)
    wy = ky - TIBIA_L * np.cos(th_hip + th_knee)

    return (kx, ky), (wx, wy)

# Precompute locus to verify it's a vertical line
locus_angles = np.linspace(0, 70, 200)
locus_pts    = [leg_fk(a)[1] for a in locus_angles]
locus_x      = [p[0] for p in locus_pts]
locus_y      = [p[1] for p in locus_pts]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 8))
plt.subplots_adjust(bottom=0.20)
fig.patch.set_facecolor("#1e1e2e")
ax.set_facecolor("#1e1e2e")
ax.set_aspect("equal")
ax.set_xlim(-0.20, 0.20)
ax.set_ylim(-0.10, 0.38)

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
ax.set_title("Step 05 — Femur + Tibia, single motor\n(θ_knee = −2·θ_hip → vertical wheel)", color="white", fontsize=12)

# Vertical reference line
ax.axvline(AX, color="#444", linewidth=1, linestyle="--", zorder=1)

# ---------------------------------------------------------------------------
# Static: body, motor
# ---------------------------------------------------------------------------
ax.add_patch(mpatches.FancyBboxPatch(
    (BODY_CX - BODY_W/2, BODY_CY - BODY_H/2), BODY_W, BODY_H,
    boxstyle="square,pad=0", linewidth=1.5,
    edgecolor="#aaa", facecolor="#4a90d9", alpha=0.85, zorder=2))
ax.text(BODY_CX, BODY_CY + 0.005, "body", color="white",
        fontsize=8, ha="center", va="center", zorder=5)

ax.add_patch(mpatches.Circle((AX, AY), MOTOR_R,
    linewidth=1.5, edgecolor="#aaa", facecolor="#e05c5c", alpha=0.9, zorder=3))
ax.text(AX, AY, "AK45-10", color="white", fontsize=6, ha="center", va="center", zorder=5)
ax.text(AX - 0.010, AY + 0.008, "A", color="yellow", fontsize=9, fontweight="bold", zorder=7)

# Wheel locus (should be a vertical line at x=0)
ax.plot(locus_x, locus_y, color="#ff6060", linewidth=2,
        linestyle="-", alpha=0.4, zorder=1, label="wheel locus")

# ---------------------------------------------------------------------------
# Dynamic elements
# ---------------------------------------------------------------------------
K0, W0 = leg_fk(HIP_INIT_DEG)

femur_line, = ax.plot([AX, K0[0]], [AY, K0[1]],
                      color="#f0c040", linewidth=5,
                      solid_capstyle="round", zorder=4, label="femur")

tibia_line, = ax.plot([K0[0], W0[0]], [K0[1], W0[1]],
                      color="#c084f5", linewidth=5,
                      solid_capstyle="round", zorder=4, label="tibia")

pivot_k, = ax.plot(*K0, "o", color="white",    markersize=8, zorder=6)
pivot_w, = ax.plot(*W0, "o", color="limegreen", markersize=6, zorder=7)

label_k = ax.text(K0[0] + 0.008, K0[1], "C (knee)", color="white",
                  fontsize=8, zorder=8)
label_w = ax.text(W0[0] + 0.008, W0[1] + 0.005, "W (wheel)", color="limegreen",
                  fontsize=8, zorder=8)

wheel_patch = mpatches.Circle(W0, WHEEL_R,
    linewidth=2, edgecolor="limegreen", facecolor="#1e1e2e", alpha=0.5, zorder=2)
ax.add_patch(wheel_patch)

info_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                    color="lightgray", fontsize=8.5, va="top", family="monospace")

def update_info(hip_deg, K, W):
    th_knee = -2.0 * hip_deg
    leg_ext = AY - W[1]          # how far wheel is below hip pivot
    info_text.set_text(
        f"hip  angle : {hip_deg:+.1f}°\n"
        f"knee angle : {th_knee:+.1f}°  (= -2 × hip)\n"
        f"knee pos   : ({K[0]*100:.1f}, {K[1]*100:.1f}) cm\n"
        f"wheel x    : {W[0]*100:.2f} cm  ({'✓ on centreline' if abs(W[0]) < 1e-9 else '✗'})\n"
        f"wheel y    : {W[1]*100:.1f} cm\n"
        f"leg ext    : {leg_ext*100:.1f} cm"
    )

update_info(HIP_INIT_DEG, K0, W0)

ax.legend(loc="upper right", fontsize=8,
          facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")

# ---------------------------------------------------------------------------
# Slider
# ---------------------------------------------------------------------------
ax_slider = plt.axes([0.15, 0.09, 0.70, 0.03], facecolor="#2a2a3e")
slider = Slider(ax_slider, "hip angle  [°]", 0, 70,
                valinit=HIP_INIT_DEG, color="#f0c040")
slider.label.set_color("lightgray")
slider.valtext.set_color("lightgray")

def on_slider(val):
    hip_deg = slider.val
    K, W = leg_fk(hip_deg)
    femur_line.set_data([AX, K[0]], [AY, K[1]])
    tibia_line.set_data([K[0], W[0]], [K[1], W[1]])
    pivot_k.set_data([K[0]], [K[1]])
    pivot_w.set_data([W[0]], [W[1]])
    wheel_patch.center = W
    label_k.set_position((K[0] + 0.008, K[1]))
    label_w.set_position((W[0] + 0.008, W[1] + 0.005))
    update_info(hip_deg, K, W)
    fig.canvas.draw_idle()

slider.on_changed(on_slider)

plt.show()
