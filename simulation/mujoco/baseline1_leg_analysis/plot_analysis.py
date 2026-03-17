"""plot_analysis.py — 6-panel deep-dive analysis of baseline-1 winning geometry.

Usage:
    python simulation/mujoco/baseline1_leg_analysis/plot_analysis.py
Input:  telemetry.npz   (created by run_sim.py)
Output: analysis.png
"""
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_config import (ROBOT, Q_RET, Q_EXT, Q_NEUTRAL,
                        HIP_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT, WHEEL_R)
from physics import solve_ik

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Style ─────────────────────────────────────────────────────────────────────
BG, FG, GR = "#1e1e2e", "white", "#444455"
PHASE_COLORS = {
    "NEUTRAL": "#1e3040",
    "CROUCH":  "#2a3a10",
    "JUMP":    "#4a1818",
    "FLIGHT":  "#103a28",
    "LAND":    "#2a1a3a",
}


# ---------------------------------------------------------------------------
def _phases(t, leg_state, grounded):
    """Return list of (t0, t1, label) for background shading."""
    def lbl(s, g):
        if not g: return "FLIGHT"
        s = int(round(s))
        return {0: "NEUTRAL", 1: "CROUCH", 2: "JUMP"}.get(s, "NEUTRAL")

    spans, cur, t0 = [], lbl(leg_state[0], grounded[0] > 0.5), t[0]
    for i in range(1, len(t)):
        l = lbl(leg_state[i], grounded[i] > 0.5)
        if l != cur:
            spans.append((t0, t[i], cur)); cur = l; t0 = t[i]
    spans.append((t0, t[-1], cur))
    return spans


def _shade(ax, spans):
    for t0, t1, lbl in spans:
        ax.axvspan(t0, t1, alpha=0.18, color=PHASE_COLORS.get(lbl, GR), linewidth=0)


def _style(ax, title, ylabel, col):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_edgecolor(GR)
    ax.tick_params(colors=FG, labelsize=7)
    ax.set_title(title, color=FG, fontsize=9, pad=4)
    ax.set_ylabel(ylabel, color=col, fontsize=8)
    ax.set_xlabel("time [s]", color=FG, fontsize=7)
    ax.grid(True, color=GR, linewidth=0.4, linestyle="--")


# ---------------------------------------------------------------------------
def draw_linkage(ax, p, q_vals, colors, labels):
    """Plot 4-bar linkage at multiple hip angles in the body-frame X-Z plane."""
    A_Z = p['A_Z']
    for q, col, lbl in zip(q_vals, colors, labels):
        ik = solve_ik(q, p)
        if ik is None: continue
        A = (0.0, A_Z)
        C, E, F, W = ik['C'], ik['E'], ik['F'], ik['W']

        # Links: femur, tibia, stub, coupler
        for (x0, z0), (x1, z1), lw in [
            (A, C, 2.5), (C, W, 1.5), (C, E, 1.5), (F, E, 2.0)
        ]:
            ax.plot([x0, x1], [z0, z1], color=col, lw=lw, alpha=0.85, solid_capstyle='round')

        # Pivots
        for (px, pz), ms in [(A, 9), (C, 7), (E, 7), (F, 7), (W, 11)]:
            ax.plot(px, pz, 'o', color=col, ms=ms, alpha=0.9, zorder=5)

        # Wheel
        ax.add_patch(plt.Circle((W[0], W[1]), WHEEL_R,
                                color=col, fill=False, lw=1.5, alpha=0.5, ls='--'))
        # Body box
        bx = bz = 0.05
        ax.add_patch(mpatches.FancyBboxPatch(
            (-bx, -bz), 2*bx, 2*bz,
            boxstyle="round,pad=0.004",
            edgecolor=col, facecolor="none", lw=1.5, alpha=0.4))

    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_edgecolor(GR)
    ax.tick_params(colors=FG, labelsize=7)
    ax.set_xlabel("X [m]", color=FG, fontsize=8)
    ax.set_ylabel("Z [m]", color=FG, fontsize=8)
    ax.set_title("4-bar Poses  (ret / neutral / ext)", color=FG, fontsize=9, pad=4)
    ax.grid(True, color=GR, lw=0.4, ls='--')
    handles = [Line2D([0], [0], color=c, lw=2, label=l)
               for c, l in zip(colors, labels)]
    ax.legend(handles=handles, fontsize=7,
              facecolor='#2a2a3e', labelcolor=FG, loc='lower right')


# ---------------------------------------------------------------------------
def main():
    path = os.path.join(_DIR, "telemetry.npz")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found — run run_sim.py first.")
        sys.exit(1)

    d          = np.load(path)
    t          = d['t']
    pitch_deg  = np.degrees(d['pitch'])
    hip_deg    = np.degrees(d['hip_q'])
    wheel_z    = d['wheel_z']
    u_hip      = d['u_hip']
    u_wheel    = d['u_wheel']
    pitch_ff   = np.degrees(d['pitch_ff'])
    leg_state  = d['leg_state']
    grounded   = d['grounded']
    height_mm  = np.maximum(0.0, (wheel_z - WHEEL_R) * 1000.0)
    max_h      = height_mm.max()

    # Use wheel_z for physical grounding (cleaner than noisy accel flag)
    phys_grounded = (wheel_z <= WHEEL_R + 0.003).astype(float)
    spans = _phases(t, leg_state, phys_grounded)

    # Key event times from wheel_z
    liftoff_mask = (phys_grounded < 0.5) & (t > 3.0)
    liftoff_t    = t[liftoff_mask][0] if liftoff_mask.any() else None
    if liftoff_t is not None:
        land_mask = (t > liftoff_t) & (phys_grounded > 0.5)
        land_t    = t[land_mask][0] if land_mask.any() else None
    else:
        land_t = None

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    fig.patch.set_facecolor(BG)
    plt.subplots_adjust(hspace=0.50, wspace=0.32,
                        top=0.90, bottom=0.07, left=0.09, right=0.97)

    p = ROBOT
    fig.suptitle(
        f"Baseline-1  |  Winning Geometry Analysis\n"
        f"L_femur={p['L_femur']*1000:.0f}mm  "
        f"L_tibia={p['L_tibia']*1000:.0f}mm  "
        f"Lc={p['Lc']*1000:.0f}mm  "
        f"F_X={p['F_X']*1000:.0f}mm  "
        f"F_Z={p['F_Z']*1000:.1f}mm  "
        f"L_stub={p['L_stub']*1000:.0f}mm  "
        f"|  Peak jump: {max_h:.0f} mm",
        color=FG, fontsize=10)

    # Phase legend (shared)
    legend_patches = [mpatches.Patch(color=v, alpha=0.5, label=k)
                      for k, v in PHASE_COLORS.items()]
    fig.legend(handles=legend_patches, ncol=5, loc='lower center',
               facecolor='#2a2a3e', labelcolor=FG, fontsize=7,
               bbox_to_anchor=(0.5, 0.0))

    # ── Panel 1: Body Pitch ──────────────────────────────────────────────────
    ax = axes[0, 0]
    _style(ax, "Body Pitch", "deg", "#60d0ff")
    _shade(ax, spans); ax.set_xlim(t[0], t[-1])
    ax.plot(t, pitch_deg, color="#60d0ff", lw=1.3, label="pitch")
    ax.plot(t, pitch_ff,  color="#60d0ff", lw=0.8, ls='--', alpha=0.55, label="FF pitch")
    ax.axhline(0, color=GR, lw=0.8, ls=':')
    ax.legend(fontsize=6, facecolor='#2a2a3e', labelcolor=FG, loc='upper right')

    # ── Panel 2: Wheel Height ────────────────────────────────────────────────
    ax = axes[0, 1]
    _style(ax, "Wheel Height  (jump profile)", "mm", "#60ff60")
    _shade(ax, spans); ax.set_xlim(t[0], t[-1])
    ax.plot(t, height_mm, color="#60ff60", lw=1.5)
    ax.axhline(max_h, color="#60ff60", lw=0.8, ls='--', alpha=0.5,
               label=f"peak {max_h:.0f} mm")
    if liftoff_t:
        ax.axvline(liftoff_t, color="yellow", lw=1.0, ls=':', label=f"liftoff {liftoff_t:.2f}s")
    if land_t:
        ax.axvline(land_t, color="orange", lw=1.0, ls=':', label=f"land {land_t:.2f}s")
    ax.legend(fontsize=6, facecolor='#2a2a3e', labelcolor=FG)

    # ── Panel 3: Hip Angle ───────────────────────────────────────────────────
    ax = axes[1, 0]
    _style(ax, "Hip Angle", "deg", "#f0d040")
    _shade(ax, spans); ax.set_xlim(t[0], t[-1])
    ax.plot(t, hip_deg, color="#f0d040", lw=1.3)
    ax.axhline(math.degrees(Q_RET), color="#f0d040", lw=0.8, ls='--', alpha=0.6,
               label=f"Q_ret {math.degrees(Q_RET):.1f}°")
    ax.axhline(math.degrees(Q_EXT), color="#f07040", lw=0.8, ls='--', alpha=0.6,
               label=f"Q_ext {math.degrees(Q_EXT):.1f}°")
    ax.axhline(math.degrees(Q_NEUTRAL), color="#80c080", lw=0.8, ls=':', alpha=0.6,
               label=f"Q_neutral {math.degrees(Q_NEUTRAL):.1f}°")
    ax.legend(fontsize=6, facecolor='#2a2a3e', labelcolor=FG)

    # ── Panel 4: Hip Torque ──────────────────────────────────────────────────
    ax = axes[1, 1]
    _style(ax, "Hip Torque  (commanded)", "N·m", "#f08040")
    _shade(ax, spans); ax.set_xlim(t[0], t[-1])
    ax.plot(t, u_hip, color="#f08040", lw=1.3)
    ax.axhline( HIP_TORQUE_LIMIT, color="red", lw=0.9, ls='--', alpha=0.7,
                label=f"±{HIP_TORQUE_LIMIT} N·m limit")
    ax.axhline(-HIP_TORQUE_LIMIT, color="red", lw=0.9, ls='--', alpha=0.7)
    ax.axhline(0, color=GR, lw=0.5, ls=':')
    ax.legend(fontsize=6, facecolor='#2a2a3e', labelcolor=FG)

    # ── Panel 5: Wheel Torque ────────────────────────────────────────────────
    ax = axes[2, 0]
    _style(ax, "Wheel Torque  (balance effort)", "N·m", "#c070ff")
    _shade(ax, spans); ax.set_xlim(t[0], t[-1])
    ax.plot(t, u_wheel, color="#c070ff", lw=1.3)
    ax.axhline( WHEEL_TORQUE_LIMIT, color="red", lw=0.9, ls='--', alpha=0.7,
                label=f"±{WHEEL_TORQUE_LIMIT} N·m limit")
    ax.axhline(-WHEEL_TORQUE_LIMIT, color="red", lw=0.9, ls='--', alpha=0.7)
    ax.axhline(0, color=GR, lw=0.5, ls=':')
    ax.legend(fontsize=6, facecolor='#2a2a3e', labelcolor=FG)

    # ── Panel 6: 4-bar Linkage Diagram ───────────────────────────────────────
    ax = axes[2, 1]
    draw_linkage(
        ax, ROBOT,
        q_vals=[Q_RET, Q_NEUTRAL, Q_EXT],
        colors=["#f08040", "#60d0ff", "#60ff60"],
        labels=[
            f"Retracted  ({math.degrees(Q_RET):.0f}°)",
            f"Neutral     ({math.degrees(Q_NEUTRAL):.0f}°)",
            f"Extended  ({math.degrees(Q_EXT):.0f}°)",
        ],
    )

    out = os.path.join(_DIR, "analysis.png")
    plt.savefig(out, dpi=150, facecolor=BG)
    print(f"Saved -> {out}")
    print(f"Peak jump: {max_h:.0f} mm"
          + (f"  |  liftoff {liftoff_t:.2f}s" if liftoff_t else "")
          + (f"  |  land {land_t:.2f}s" if land_t else ""))


if __name__ == "__main__":
    main()
