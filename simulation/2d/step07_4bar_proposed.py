"""
Step 07 — User-proposed 4-bar mechanism.

Topology:
    A  = hip motor pivot (body-fixed, active)
    F  = body-fixed passive pivot (~5 cm up and left of A, inside body)

    Femur:  A → C   (rigid, driven by motor at A)
    Tibia:  E – C – W  (rigid rod; C is knee pivot 2 cm from one end)
              E = stub end (2 cm above C, toward body)
              W = wheel end
    Coupler: F → E  (passive link, free to rotate at both ends)

    4-bar: A(fixed) — femur — C — [tibia body] — E — coupler — F(fixed)

    As motor rotates femur, C moves → E is constrained by |F–E| = Lc
    → tibia angle is forced → W traces a coupler curve (hopefully near-vertical)

Parameters (tweak at top to explore):
    L_femur  — femur length A→C
    L_stub   — tibia stub C→E (above knee, toward body)
    L_tibia  — tibia length C→W (knee to wheel, below knee)
    F_offset — (dx, dy) of passive pivot F relative to A

Run:
    python simulation/2d/step07_4bar_proposed.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Parameters — edit these to explore variants
# ---------------------------------------------------------------------------
BODY_W, BODY_H   = 0.10, 0.10
BODY_CX, BODY_CY = 0.00, 0.20
MOTOR_R  = 0.053 / 2
BODY_BOTTOM = BODY_CY - BODY_H / 2

AX = 0.0
AY = BODY_BOTTOM + MOTOR_R       # hip pivot = motor centre = 0.1765 m

L_femur   = 0.100                # femur A→C  [m]  (10 cm)
L_stub    = 0.015                # tibia stub C→E  (1.5 cm above knee)  [m]
L_tibia   = 0.115                # tibia C→W  (11.5 cm knee to wheel)  [m]
Lc        = 0.110                # coupler F→E  (11 cm)  [m]

# Passive body pivot F: relative to A
F_dx = +0.015                    # +1.5 cm in x from motor centre
F_dy = +0.026                    # +2.6 cm in y from motor centre
FX   = AX + F_dx
FY   = AY + F_dy

WHEEL_R = 0.075                  # 150 mm wheel

HIP_RANGE = (0.0, 65.0)          # operating range [deg]
HIP_INIT  = 25.0

F  = np.array([FX, FY])

print(f"Geometry summary:")
print(f"  A        = ({AX*100:.1f}, {AY*100:.1f}) cm  [hip motor]")
print(f"  F        = ({FX*100:.1f}, {FY*100:.1f}) cm  [body passive pivot]")
print(f"  L_femur  = {L_femur*1000:.0f} mm")
print(f"  L_stub   = {L_stub*1000:.0f} mm")
print(f"  L_tibia  = {L_tibia*1000:.0f} mm")
print(f"  Lc       = {Lc*1000:.0f} mm  [coupler F to E]")

# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------
def mechanism_fk(hip_deg):
    """
    Given hip angle, solve for tibia angle φ via coupler constraint.
    Returns: C, E, W, phi  (all in metres)
    """
    th = np.radians(hip_deg)

    # Knee position
    Cx = AX + L_femur * np.sin(th)
    Cy = AY - L_femur * np.cos(th)
    C  = np.array([Cx, Cy])

    # Tibia geometry:
    #   stub direction: (-sin φ, +cos φ)   [opposite to wheel direction]
    #   E = C + L_stub * (-sin φ, cos φ)
    #   W = C + L_tibia * (+sin φ, -cos φ)
    #
    # Constraint: |F - E(φ)|² = Lc²
    def residual(phi):
        E = C + L_stub * np.array([-np.sin(phi), np.cos(phi)])
        return np.dot(F - E, F - E) - Lc**2

    # Sweep phi to find all sign changes, pick solution closest to ideal -th
    phi_ideal = -th
    phi_sweep = np.linspace(-np.pi, np.pi, 720)
    r_sweep   = [residual(p) for p in phi_sweep]

    brackets = []
    for i in range(len(phi_sweep) - 1):
        if np.sign(r_sweep[i]) != np.sign(r_sweep[i+1]):
            brackets.append((phi_sweep[i], phi_sweep[i+1]))

    if not brackets:
        return None, None, None, None

    # Solve each bracket, take root closest to ideal
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
# Precompute locus
# ---------------------------------------------------------------------------
angles  = np.linspace(HIP_RANGE[0], HIP_RANGE[1], 400)
locus_W = []
locus_ok = []
for a in angles:
    result = mechanism_fk(a)
    if result[0] is not None:
        locus_W.append(result[2])
        locus_ok.append(a)
    else:
        locus_W.append(None)

locus_x = [p[0] if p is not None else np.nan for p in locus_W]
locus_y = [p[1] if p is not None else np.nan for p in locus_W]

wheel_x_err_mm = [abs(x)*1000 if not np.isnan(x) else np.nan for x in locus_x]
valid_err = [e for e in wheel_x_err_mm if not np.isnan(e)]
max_err = max(valid_err) if valid_err else float('nan')
print(f"\nWheel x deviation from centreline:")
print(f"  Feasible solutions: {len(valid_err)}/{len(angles)}")
print(f"  Max over full range (0-65°): {max_err:.2f} mm")

# Jump stroke error (10°-50°)
jump_errs = [abs(locus_x[i])*1000
             for i, a in enumerate(angles)
             if 10.0 <= a <= 50.0 and not np.isnan(locus_x[i])]
print(f"  Max over jump stroke (10-50°): {max(jump_errs):.2f} mm" if jump_errs else "  No solutions in jump stroke range")

# ---------------------------------------------------------------------------
# Figure: sim panel + error plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 7))
fig.patch.set_facecolor("#1e1e2e")
plt.subplots_adjust(left=0.06, right=0.97, bottom=0.15, top=0.92,
                    wspace=0.30)

ax_sim = fig.add_subplot(1, 2, 1)
ax_err = fig.add_subplot(1, 2, 2)

# --- Sim panel style ---
ax_sim.set_facecolor("#1e1e2e")
ax_sim.set_aspect("equal")
ax_sim.set_xlim(-0.22, 0.22)
ax_sim.set_ylim(-0.18, 0.38)

def m_to_cm(x, _): return f"{x*100:.0f}"
for ax in [ax_sim]:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
    ax.set_xlabel("x  [cm]", color="lightgray")
    ax.set_ylabel("y  [cm]", color="lightgray")
    ax.tick_params(colors="lightgray")
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")
    ax.grid(True, color="#333", linewidth=0.4, linestyle="--")

ax_sim.axhline(0, color="#888", linewidth=1.5)
ax_sim.axvline(0, color="#40ff40", linewidth=0.8, linestyle="--", alpha=0.6, label="ideal x=0")
ax_sim.set_title("Step 07 — User-proposed 4-bar mechanism", color="white", fontsize=11)

# --- Error panel style ---
ax_err.set_facecolor("#1e1e2e")
ax_err.tick_params(colors="lightgray")
for sp in ax_err.spines.values():
    sp.set_edgecolor("#555")
ax_err.grid(True, color="#333", linewidth=0.4, linestyle="--")
ax_err.set_xlabel("hip angle [°]", color="lightgray")
ax_err.set_ylabel("|wheel x| deviation from centreline [mm]", color="lightgray")
ax_err.set_title("Wheel x error vs hip angle", color="white", fontsize=11)
ax_err.set_xlim(HIP_RANGE)

# --- Static shapes ---
# Body box
ax_sim.add_patch(mpatches.FancyBboxPatch(
    (BODY_CX - BODY_W/2, BODY_CY - BODY_H/2), BODY_W, BODY_H,
    boxstyle="square,pad=0", linewidth=1.5,
    edgecolor="#aaa", facecolor="#4a90d9", alpha=0.85, zorder=2))
ax_sim.text(BODY_CX, BODY_CY + 0.005, "body",
            color="white", fontsize=8, ha="center", va="center", zorder=5)

# Motor circle
ax_sim.add_patch(mpatches.Circle((AX, AY), MOTOR_R,
    linewidth=1.5, edgecolor="#aaa", facecolor="#e05c5c", alpha=0.9, zorder=3))
ax_sim.text(AX, AY, "AK45-10", color="white", fontsize=6, ha="center", va="center")

# Pivot A label
ax_sim.plot(AX, AY, "o", color="yellow", markersize=6, zorder=7)
ax_sim.text(AX - 0.012, AY + 0.006, "A", color="yellow",
            fontsize=9, fontweight="bold", zorder=8)

# Fixed body pivot F
ax_sim.plot(FX, FY, "s", color="#aafafa", markersize=8, zorder=7,
            label=f"F (body pivot)")
ax_sim.text(FX - 0.014, FY + 0.006, "F", color="#aafafa",
            fontsize=9, fontweight="bold", zorder=8)

# Locus of W
ax_sim.plot(locus_x, locus_y, color="#ff6060", linewidth=2,
            linestyle="-", alpha=0.45, zorder=1, label="W locus")

# --- Error curve ---
ax_err.fill_between(locus_ok if locus_ok else angles,
                    wheel_x_err_mm[:len(locus_ok)] if locus_ok else wheel_x_err_mm,
                    color="#c084f5", alpha=0.35)
ax_err.plot(locus_ok if locus_ok else angles,
            wheel_x_err_mm[:len(locus_ok)] if locus_ok else wheel_x_err_mm,
            color="#c084f5", linewidth=2)
ax_err.axhline(5, color="#ff6060", linewidth=1, linestyle="--", alpha=0.6,
               label="5 mm threshold")
# Jump stroke band (10°–50°)
ax_err.axvspan(10, 50, color="#ffff40", alpha=0.06, label="jump stroke 10°–50°")
ax_err.axvline(10, color="#ffff40", linewidth=0.8, linestyle=":", alpha=0.5)
ax_err.axvline(50, color="#ffff40", linewidth=0.8, linestyle=":", alpha=0.5)
ax_err.legend(fontsize=8, facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")
ax_err.set_ylim(bottom=0)

# Vertical cursor on error plot
vline_err = ax_err.axvline(HIP_INIT, color="#f0c040", linewidth=1, linestyle="--")

# --- Dynamic mechanism ---
C0r, E0r, W0r, phi0 = mechanism_fk(HIP_INIT)

femur_line, = ax_sim.plot([AX, C0r[0]], [AY, C0r[1]],
                           color="#f0c040", linewidth=5,
                           solid_capstyle="round", zorder=4, label="femur A→C")
stub_line,  = ax_sim.plot([C0r[0], E0r[0]], [C0r[1], E0r[1]],
                           color="#c084f5", linewidth=4, linestyle="-",
                           solid_capstyle="round", zorder=4, label="tibia stub C→E")
tibia_line, = ax_sim.plot([C0r[0], W0r[0]], [C0r[1], W0r[1]],
                           color="#c084f5", linewidth=5,
                           solid_capstyle="round", zorder=4, label="tibia C→W")
coupler_line, = ax_sim.plot([FX, E0r[0]], [FY, E0r[1]],
                             color="#aafafa", linewidth=2.5, linestyle="--",
                             zorder=4, label="coupler F→E")

knee_dot,  = ax_sim.plot(*C0r, "o", color="white",     markersize=8, zorder=6)
stub_dot,  = ax_sim.plot(*E0r, "o", color="#aafafa",   markersize=7, zorder=6)
wheel_dot, = ax_sim.plot(*W0r, "o", color="limegreen", markersize=6, zorder=7)

label_c = ax_sim.text(C0r[0]+0.008, C0r[1], "C", color="white",     fontsize=8, fontweight="bold", zorder=9)
label_e = ax_sim.text(E0r[0]+0.008, E0r[1], "E", color="#aafafa",   fontsize=8, fontweight="bold", zorder=9)
label_w = ax_sim.text(W0r[0]+0.008, W0r[1], "W", color="limegreen", fontsize=8, fontweight="bold", zorder=9)

wheel_patch = mpatches.Circle(W0r, WHEEL_R,
    linewidth=1.5, edgecolor="limegreen", facecolor="#1e1e2e", alpha=0.4, zorder=2)
ax_sim.add_patch(wheel_patch)

info_text = ax_sim.text(0.02, 0.97, "", transform=ax_sim.transAxes,
                         color="lightgray", fontsize=8, va="top", family="monospace")

def update_info(hip_deg, W, phi):
    if W is None:
        info_text.set_text("No solution")
        return
    info_text.set_text(
        f"hip angle  : {hip_deg:+.1f}°\n"
        f"tibia angle: {np.degrees(phi):+.1f}°\n"
        f"wheel x    : {W[0]*1000:+.2f} mm  ({'✓' if abs(W[0])<0.005 else '~'} vertical)\n"
        f"wheel y    : {W[1]*100:.1f} cm\n"
        f"coupler Lc : {Lc*1000:.1f} mm\n"
        f"F offset   : ({F_dx*100:.1f}, {F_dy*100:.1f}) cm from A"
    )

update_info(HIP_INIT, W0r, phi0)
ax_sim.legend(loc="lower right", fontsize=7.5,
              facecolor="#2a2a3e", edgecolor="#555", labelcolor="white")
fig.suptitle(f"4-bar leg: femur={L_femur*1000:.0f}mm  tibia={L_tibia*1000:.0f}mm  "
             f"stub={L_stub*1000:.0f}mm  Lc={Lc*1000:.0f}mm",
             color="white", fontsize=11)

# ---------------------------------------------------------------------------
# Slider
# ---------------------------------------------------------------------------
ax_sl = plt.axes([0.15, 0.05, 0.70, 0.025], facecolor="#2a2a3e")
slider = Slider(ax_sl, "hip angle [°]", HIP_RANGE[0], HIP_RANGE[1],
                valinit=HIP_INIT, color="#f0c040")
slider.label.set_color("lightgray")
slider.valtext.set_color("lightgray")

def on_slider(val):
    ang  = slider.val
    result = mechanism_fk(ang)
    C, E, W, phi = result
    if C is None:
        return
    femur_line.set_data([AX, C[0]],  [AY, C[1]])
    stub_line.set_data( [C[0], E[0]], [C[1], E[1]])
    tibia_line.set_data([C[0], W[0]], [C[1], W[1]])
    coupler_line.set_data([FX, E[0]], [FY, E[1]])
    knee_dot.set_data( [C[0]], [C[1]])
    stub_dot.set_data( [E[0]], [E[1]])
    wheel_dot.set_data([W[0]], [W[1]])
    wheel_patch.center = W
    label_c.set_position((C[0]+0.008, C[1]))
    label_e.set_position((E[0]+0.008, E[1]))
    label_w.set_position((W[0]+0.008, W[1]))
    vline_err.set_xdata([ang, ang])
    update_info(ang, W, phi)
    fig.canvas.draw_idle()

slider.on_changed(on_slider)

plt.show()
