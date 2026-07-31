"""
Step 06 — Three mechanism options compared side by side.

Left:   Parallelogram 4-bar   (θ_knee = 0,      wheel traces arc)
Centre: Synthesised 4-bar     (θ_knee ≈ -2·θ_hip, near-vertical)
Right:  Ideal 2:1 constraint  (θ_knee = -2·θ_hip, exact vertical)

Bottom row: error plots (wheel x-deviation from centreline vs hip angle)

Run:
    python simulation/2d/step06_mechanism_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
L     = 0.140    # femur = tibia = 140 mm
AX    = 0.0
AY    = 0.0      # hip pivot at origin for clarity in each subplot

WHEEL_R = 0.075

HIP_INIT_DEG = 25.0
HIP_ANGLES   = np.linspace(0, 65, 400)   # operating range

# ---------------------------------------------------------------------------
# Mechanism 1 — Parallelogram (θ_knee_rel = 0)
# ---------------------------------------------------------------------------
def parallelogram_fk(hip_deg):
    th = np.radians(hip_deg)
    kx = AX + L * np.sin(th)
    ky = AY - L * np.cos(th)
    # tibia parallel to femur (θ_knee_rel = 0)
    wx = kx + L * np.sin(th)
    wy = ky - L * np.cos(th)
    return (kx, ky), (wx, wy)

# ---------------------------------------------------------------------------
# Mechanism 3 — Ideal 2:1 (exact vertical)
# ---------------------------------------------------------------------------
def ideal_fk(hip_deg):
    th = np.radians(hip_deg)
    kx = AX + L * np.sin(th)
    ky = AY - L * np.cos(th)
    # tibia at -θ absolute (θ_knee_rel = -2θ_hip)
    th_knee_rel = -2.0 * th
    wx = kx + L * np.sin(th + th_knee_rel)
    wy = ky - L * np.cos(th + th_knee_rel)
    return (kx, ky), (wx, wy)

# ---------------------------------------------------------------------------
# Mechanism 2 — Synthesised 4-bar
#
# Arrangement:
#   Body-fixed pivots: A=(0,0) and B=(Bx, By)
#   Femur AC (driven, length L)
#   Coupler link B→D (length Ld), where D is on the tibia at fraction t from C
#   Tibia CW (length L), D at distance t*L from C along tibia
#
# For a given (hip_deg), we solve for tibia angle that satisfies |BD| = Ld.
# We optimise [Bx, By, Ld, t] to minimise deviation of wheel from x=0
# over the operating range.
# ---------------------------------------------------------------------------

def synthesised_fk_given_params(hip_deg, Bx, By, Ld, t):
    """
    Solve for tibia angle given 4-bar params.
    Returns (knee, wheel) or None if no solution.
    """
    th_hip = np.radians(hip_deg)
    kx = AX + L * np.sin(th_hip)
    ky = AY - L * np.cos(th_hip)

    # Solve: |BD|² = Ld²  where D = C + t*L*(sin(th_t), -cos(th_t))
    # D = (kx + t*L*sin(th_t), ky - t*L*cos(th_t))
    # |B - D|² = Ld²
    # Search over th_t in [-pi, pi]
    def residual(th_t):
        dx = kx + t * L * np.sin(th_t) - Bx
        dy = ky - t * L * np.cos(th_t) - By
        return (dx**2 + dy**2 - Ld**2)**2

    # Try candidate near ideal solution
    th_ideal = th_hip + (-2.0 * th_hip)   # = -th_hip
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(residual,
                          bounds=(th_ideal - np.pi/2, th_ideal + np.pi/2),
                          method='bounded')
    th_t = res.x
    wx = kx + L * np.sin(th_t)
    wy = ky - L * np.cos(th_t)
    return (kx, ky), (wx, wy), th_t

def optimise_fourbar():
    """Find B, Ld, t that minimise max |wheel_x| over operating range."""
    def cost(params):
        Bx, By, Ld, t = params
        if Ld <= 0 or t <= 0.1 or t >= 1.0:
            return 1e6
        errors = []
        for ang in np.linspace(5, 60, 20):
            try:
                _, (wx, wy), _ = synthesised_fk_given_params(ang, Bx, By, Ld, t)
                errors.append(wx**2)
            except Exception:
                errors.append(1.0)
        return np.max(errors) + 0.1 * np.mean(errors)

    # Initial guess: B above A, moderate link length
    x0 = [0.02, 0.08, 0.10, 0.5]
    bounds = [(-0.05, 0.05), (0.02, 0.20), (0.05, 0.25), (0.2, 0.9)]
    res = minimize(cost, x0, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-8})
    return res.x

print("Optimising 4-bar linkage geometry …", flush=True)
Bx_opt, By_opt, Ld_opt, t_opt = optimise_fourbar()
print(f"  B = ({Bx_opt*1000:.1f}, {By_opt*1000:.1f}) mm  "
      f"Ld = {Ld_opt*1000:.1f} mm  t = {t_opt:.3f}")

def synth_fk(hip_deg):
    result = synthesised_fk_given_params(hip_deg, Bx_opt, By_opt, Ld_opt, t_opt)
    return result[0], result[1]   # (knee, wheel)

# ---------------------------------------------------------------------------
# Precompute loci
# ---------------------------------------------------------------------------
para_locus  = [parallelogram_fk(a)[1] for a in HIP_ANGLES]
ideal_locus = [ideal_fk(a)[1]         for a in HIP_ANGLES]
try:
    synth_locus = [synth_fk(a)[1] for a in HIP_ANGLES]
except Exception:
    synth_locus = ideal_locus   # fallback

# Error: wheel x-deviation from centreline
para_err  = [abs(p[0]) * 1000 for p in para_locus]
ideal_err = [0.0 for _ in HIP_ANGLES]
synth_err = [abs(p[0]) * 1000 for p in synth_locus]

# ---------------------------------------------------------------------------
# Figure layout: 2 rows × 3 columns
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor("#1e1e2e")
plt.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.16,
                    hspace=0.35, wspace=0.3)

TITLES = [
    "Option A — Parallelogram\n(θ_knee = 0, wheel traces arc)",
    "Option B — Synthesised 4-bar\n(θ_knee ≈ −2·θ, near-vertical)",
    "Option C — Ideal 2:1 constraint\n(θ_knee = −2·θ, exact vertical)",
]
COLORS = ["#f0c040", "#c084f5", "#60d060"]

ax_sim  = [fig.add_subplot(2, 3, i+1) for i in range(3)]
ax_err  = [fig.add_subplot(2, 3, i+4) for i in range(3)]

def style_sim_ax(ax, title, col):
    ax.set_facecolor("#1e1e2e")
    ax.set_aspect("equal")
    ax.set_xlim(-0.22, 0.22)
    ax.set_ylim(-0.38, 0.12)
    def m_to_cm(x, _): return f"{x*100:.0f}"
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
    ax.tick_params(colors="lightgray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    ax.grid(True, color="#333", linewidth=0.4, linestyle="--")
    ax.axhline(0, color="#888", linewidth=1)
    ax.axvline(0, color=col, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.set_xlabel("x [cm]", color="lightgray", fontsize=8)
    ax.set_ylabel("y [cm]", color="lightgray", fontsize=8)

def style_err_ax(ax, col):
    ax.set_facecolor("#1e1e2e")
    ax.tick_params(colors="lightgray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    ax.grid(True, color="#333", linewidth=0.4, linestyle="--")
    ax.set_xlabel("hip angle [°]", color="lightgray", fontsize=8)
    ax.set_ylabel("|wheel x| [mm]", color="lightgray", fontsize=8)
    ax.set_xlim(0, 65)

for i in range(3):
    style_sim_ax(ax_sim[i], TITLES[i], COLORS[i])
    style_err_ax(ax_err[i], COLORS[i])

fig.suptitle("Leg Mechanism Comparison — Wheel Centre Trajectory", color="white", fontsize=13)

# --- Draw loci ---
ax_sim[0].plot([p[0] for p in para_locus],  [p[1] for p in para_locus],
               color=COLORS[0], alpha=0.35, linewidth=2)
ax_sim[1].plot([p[0] for p in synth_locus], [p[1] for p in synth_locus],
               color=COLORS[1], alpha=0.35, linewidth=2)
ax_sim[2].plot([p[0] for p in ideal_locus], [p[1] for p in ideal_locus],
               color=COLORS[2], alpha=0.35, linewidth=2)

# --- Error plots ---
ax_err[0].fill_between(HIP_ANGLES, para_err,  color=COLORS[0], alpha=0.4)
ax_err[0].plot(HIP_ANGLES, para_err,  color=COLORS[0], linewidth=1.5)
ax_err[1].fill_between(HIP_ANGLES, synth_err, color=COLORS[1], alpha=0.4)
ax_err[1].plot(HIP_ANGLES, synth_err, color=COLORS[1], linewidth=1.5)
ax_err[2].fill_between(HIP_ANGLES, ideal_err, color=COLORS[2], alpha=0.4)
ax_err[2].plot(HIP_ANGLES, ideal_err, color=COLORS[2], linewidth=1.5)
for ax in ax_err:
    ax.set_ylim(bottom=0)

# --- Dynamic elements (updated by slider) ---
line_arts = []  # list of dicts per mechanism
fk_fns    = [parallelogram_fk, synth_fk, ideal_fk]
B_pts     = [None, (Bx_opt, By_opt), None]   # body pivot B for synth

for i, (ax, fk, col) in enumerate(zip(ax_sim, fk_fns, COLORS)):
    # Hip pivot dot
    ax.plot(AX, AY, "o", color="yellow", markersize=6, zorder=6)
    ax.text(AX - 0.015, AY + 0.005, "A", color="yellow", fontsize=8, fontweight="bold")

    if B_pts[i]:
        bx, by = B_pts[i]
        ax.plot(bx, by, "s", color="#aaa", markersize=5, zorder=6)
        ax.text(bx + 0.005, by + 0.005, "B", color="#aaa", fontsize=7)

    K0, W0 = fk(HIP_INIT_DEG)

    femur, = ax.plot([AX, K0[0]], [AY, K0[1]],
                     color=col, linewidth=4, solid_capstyle="round", zorder=4)
    tibia, = ax.plot([K0[0], W0[0]], [K0[1], W0[1]],
                     color=col, linewidth=4, alpha=0.6, solid_capstyle="round", zorder=4)
    knee_dot,  = ax.plot(*K0, "o", color="white", markersize=7, zorder=6)
    wheel_dot, = ax.plot(*W0, "o", color=col,     markersize=5, zorder=7)
    wcirc = mpatches.Circle(W0, WHEEL_R,
                             linewidth=1.5, edgecolor=col,
                             facecolor="#1e1e2e", alpha=0.4, zorder=2)
    ax.add_patch(wcirc)

    # Cursor line on error plot
    vline = ax_err[i].axvline(HIP_INIT_DEG, color=col, linewidth=1, linestyle="--")

    line_arts.append(dict(femur=femur, tibia=tibia,
                          knee=knee_dot, wheel=wheel_dot,
                          wcirc=wcirc, vline=vline, fk=fk))

# ---------------------------------------------------------------------------
# Slider
# ---------------------------------------------------------------------------
ax_sl = plt.axes([0.20, 0.06, 0.60, 0.025], facecolor="#2a2a3e")
slider = Slider(ax_sl, "hip angle [°]", 0, 65,
                valinit=HIP_INIT_DEG, color="#f0c040")
slider.label.set_color("lightgray")
slider.valtext.set_color("lightgray")

def on_slider(val):
    ang = slider.val
    for art in line_arts:
        K, W = art["fk"](ang)
        art["femur"].set_data([AX, K[0]], [AY, K[1]])
        art["tibia"].set_data([K[0], W[0]], [K[1], W[1]])
        art["knee"].set_data([K[0]], [K[1]])
        art["wheel"].set_data([W[0]], [W[1]])
        art["wcirc"].center = W
        art["vline"].set_xdata([ang, ang])
    fig.canvas.draw_idle()

slider.on_changed(on_slider)

plt.show()
