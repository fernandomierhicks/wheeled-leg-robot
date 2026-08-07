"""gui.py — PyQt6 app for designing and inspecting the 4-bar.

Launch:
    python simulation/2d/fourbar_optimizer_gui/gui.py
    python simulation/2d/fourbar_optimizer_gui/api.py gui

This module is a *consumer* of the core; the core never imports it.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from model import (LinkageSpec, baseline1, TracePoint, ALL_BODIES, MOVING_LINKS,
                   FEMUR, TIBIA, COUPLER, FEMUR_SHAFT, MOTOR, WHEEL,
                   WHEEL_MOTOR, pair_key)
from kinematics import solve_pose, trace_world
from geometry import ShapeSet, collisions, body_shapes, RES_FAST, RES_DRAW
from evaluate import find_range, evaluate, hip_torque

BG = "#1e1e2e"
FG = "lightgray"
LINK_COLOR = {FEMUR: "#f0c040", TIBIA: "#c084f5", COUPLER: "#5ad7d7"}
NODE_COLOR = {"A": "#ffe066", "C": "#ffffff", "E": "#aafafa",
              "F": "#5ad7d7", "W": "#40ff80"}
SHAFT_COLOR = "#9aa8c0"

# Drawn once in the right dock instead of under every canvas: an in-plot legend
# costs a strip of height on BOTH compare panels for a key that never changes.
# Colours must track the literals used in LinkageCanvas.draw_state.
LEGEND_ENTRIES = (
    ("femur", LINK_COLOR[FEMUR]),
    ("tibia", LINK_COLOR[TIBIA]),
    ("coupler", LINK_COLOR[COUPLER]),
    ("knee shaft", SHAFT_COLOR),
    ("traced path", "#ff6060"),
    ("wheel travel box", "#40ff40"),
    ("ride-height limit", "#ff9040"),
    ("ground", "#8a5a2a"),
    ("range-end ghosts", "#3a3a4a"),
    ("dimensions", "#ffb340"),
    ("collision", "#ff4040"),
)

WORKING_EDITED = "— working (edited) —"
# The optimizer pushes specs into both panels that came from no preset at all.
# Without a name of their own the combo keeps showing whatever was last picked
# while the canvas draws something else entirely.
OPT_SNAPSHOT = "— optimizer start snapshot —"
OPT_BEST = "— optimizer best —"
PLACEHOLDERS = (WORKING_EDITED, OPT_SNAPSHOT, OPT_BEST)

# Spec fields the panel edits in millimetres (stored in metres).
MM_FIELDS = ("L_femur", "L_stub", "L_tibia", "Lc", "F_X", "F_Z", "w_perp",
             "femur_r_A", "femur_r_C", "tibia_r_E", "tibia_r_C", "tibia_r_W",
             "coupler_r_F", "coupler_r_E", "femur_shaft_r", "motor_r",
             "wheel_r", "wheel_motor_r")


# ---------------------------------------------------------------------------
def _spin(value, lo, hi, step=1.0, dec=3, suffix=" mm"):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(dec)
    s.setValue(value)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    s.setMinimumWidth(110)
    return s


class ParamPanel(QtWidgets.QWidget):
    """All editable dimensions.  Lengths in mm, angles in degrees."""

    changed = QtCore.pyqtSignal()

    def __init__(self, spec: LinkageSpec):
        super().__init__()
        self.w = {}
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        def group(title, rows):
            box = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(box)
            form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            for key, label, widget in rows:
                self.w[key] = widget
                form.addRow(label, widget)
                if isinstance(widget, QtWidgets.QDoubleSpinBox):
                    widget.valueChanged.connect(self.changed)
                elif isinstance(widget, QtWidgets.QCheckBox):
                    widget.toggled.connect(self.changed)
            lay.addWidget(box)
            return box

        group("Bar lengths", [
            ("L_femur", "femur  A→C", _spin(spec.L_femur * 1e3, 10, 400)),
            ("L_stub", "stub  C→E", _spin(spec.L_stub * 1e3, 1, 200)),
            ("L_tibia", "tibia  C→W", _spin(spec.L_tibia * 1e3, 10, 400)),
            ("Lc", "coupler  F→E", _spin(spec.Lc * 1e3, 10, 400)),
        ])
        group("Pivot F  (relative to motor A at origin)", [
            ("F_X", "F_X", _spin(spec.F_X * 1e3, -300, 300)),
            ("F_Z", "F_Z", _spin(spec.F_Z * 1e3, -300, 300)),
        ])
        group("Tibia dogleg", [
            ("w_perp", "W ⊥ offset", _spin(spec.w_perp * 1e3, -200, 200)),
        ])
        group("Link end radii  (per link — two links at one pin may differ)", [
            ("femur_r_A", "femur @ A", _spin(spec.femur_r_A * 1e3, 0.5, 100, 0.5)),
            ("femur_r_C", "femur @ C", _spin(spec.femur_r_C * 1e3, 0.5, 100, 0.5)),
            ("tibia_r_E", "tibia @ E", _spin(spec.tibia_r_E * 1e3, 0.5, 100, 0.5)),
            ("tibia_r_C", "tibia @ C", _spin(spec.tibia_r_C * 1e3, 0.5, 100, 0.5)),
            ("tibia_r_W", "tibia @ W", _spin(spec.tibia_r_W * 1e3, 0.5, 100, 0.5)),
            ("coupler_r_F", "coupler @ F", _spin(spec.coupler_r_F * 1e3, 0.5, 100, 0.5)),
            ("coupler_r_E", "coupler @ E", _spin(spec.coupler_r_E * 1e3, 0.5, 100, 0.5)),
        ])
        group("Circular bodies", [
            ("femur_shaft_r", "knee shaft radius",
             _spin(spec.femur_shaft_r * 1e3, 0.5, 100, 0.5)),
            ("motor_r", "motor radius", _spin(spec.motor_r * 1e3, 1, 200, 0.5)),
            ("wheel_enabled", "wheel at traced pt",
             QtWidgets.QCheckBox("enabled")),
            ("wheel_r", "wheel radius", _spin(spec.wheel_r * 1e3, 1, 400)),
            ("wheel_motor_r", "hub motor radius",
             _spin(spec.wheel_motor_r * 1e3, 1, 400, 0.5)),
        ])
        self.w["wheel_enabled"].setChecked(spec.wheel_enabled)
        self.w["femur_shaft_r"].setToolTip(
            "The femur's knee shaft at C, Ø23 mm as built.  Its own body, not\n"
            "part of the femur plate: the coupler passes the plate on another\n"
            "plane, but the shaft protrudes into the coupler's plane.  Only the\n"
            "coupler pair is on by default — see the collision-pair matrix.")
        self.w["wheel_motor_r"].setToolTip(
            "Hub motor, concentric with the wheel.  Always tested (it is on the\n"
            "leg whether or not the tyre outline is drawn); which bodies it is\n"
            "tested against is set in the collision-pair matrix.")
        group("Pose limits  (bind even with no collision)", [
            ("enforce_wheel_below_hip", "W below hip axis",
             QtWidgets.QCheckBox("enforce")),
            ("wheel_below_hip_margin", "limit (z)",
             _spin(spec.wheel_below_hip_margin * 1e3, -300, 300, 1.0)),
            ("enforce_W_below_E", "W below E",
             QtWidgets.QCheckBox("enforce")),
            ("enforce_ground", "ground plane",
             QtWidgets.QCheckBox("enforce")),
        ])
        self.w["enforce_ground"].setChecked(spec.enforce_ground)
        self.w["enforce_ground"].setToolTip(
            "Horizontal ground tangent to the bottom of the wheel, so it\n"
            "tracks the wheel as the leg moves.  No link and not the motor\n"
            "may reach it.")
        self.w["enforce_wheel_below_hip"].setChecked(spec.enforce_wheel_below_hip)
        self.w["enforce_wheel_below_hip"].setToolTip(
            "The wheel's UPPER tangent (W_z + wheel_r) may not rise above the\n"
            "hip motor axis.  By the time the top of the tyre reaches the axis\n"
            "the chassis is already on the ground and there is no jump left.")
        self.w["enforce_W_below_E"].setChecked(spec.enforce_W_below_E)
        self.w["enforce_W_below_E"].setToolTip(
            "The wheel pivot W may not rise above the tibia stub end E.\n"
            "W above E means the tibia has leaned back past horizontal and\n"
            "its far end swings toward the ground.")

        group("Hip torque limit  (static hold, wheel on the ground)", [
            ("enforce_torque_limit", "torque limit",
             QtWidgets.QCheckBox("enforce")),
            ("leg_load_kg", "load on this leg",
             _spin(spec.leg_load_kg, 0.0, 100.0, 0.1, 2, " kg")),
            ("torque_limit_nm", "max hip torque",
             _spin(spec.torque_limit_nm, 0.01, 100.0, 0.1, 2, " N·m")),
        ])
        self.w["enforce_torque_limit"].setChecked(spec.enforce_torque_limit)
        self.w["enforce_torque_limit"].setToolTip(
            "Trim the stroke to the poses the hip motor can actually hold.\n"
            "  tau(q) = load * g * |dz_W/dq|\n"
            "The wheel rolls, so only the VERTICAL gear ratio does work; a\n"
            "pose can be perfectly reachable and still need more torque than\n"
            "the motor has.  Blocks like a collision does.")
        self.w["leg_load_kg"].setToolTip(
            "Mass carried by THIS leg — half the body for a two-hip robot\n"
            "(2 kg body, motors included -> 1.0 kg per leg).")
        self.w["torque_limit_nm"].setToolTip(
            "AK45-10 continuous rating is 2.5 N·m.  Peak is higher, but a\n"
            "standing robot holds this torque indefinitely, so continuous is\n"
            "the number that binds.")

        group("Hip sweep", [
            ("q_seed", "seed angle", _spin(math.degrees(spec.q_seed), -180, 180,
                                           1.0, 2, " °")),
            ("q_min", "limit min", _spin(math.degrees(spec.q_min), -360, 360,
                                         5.0, 2, " °")),
            ("q_max", "limit max", _spin(math.degrees(spec.q_max), -360, 360,
                                         5.0, 2, " °")),
        ])
        lay.addStretch(1)

    def apply_to(self, spec: LinkageSpec):
        for k in MM_FIELDS:
            setattr(spec, k, self.w[k].value() / 1000.0)
        for k in ("q_seed", "q_min", "q_max"):
            setattr(spec, k, math.radians(self.w[k].value()))
        spec.wheel_enabled = self.w["wheel_enabled"].isChecked()
        spec.enforce_wheel_below_hip = \
            self.w["enforce_wheel_below_hip"].isChecked()
        spec.wheel_below_hip_margin = \
            self.w["wheel_below_hip_margin"].value() / 1000.0
        spec.enforce_W_below_E = self.w["enforce_W_below_E"].isChecked()
        spec.enforce_ground = self.w["enforce_ground"].isChecked()
        spec.enforce_torque_limit = self.w["enforce_torque_limit"].isChecked()
        spec.leg_load_kg = self.w["leg_load_kg"].value()
        spec.torque_limit_nm = self.w["torque_limit_nm"].value()
        spec.sync_primary_to_W()
        return spec

    def load_from(self, spec: LinkageSpec):
        blocked = [w for w in self.w.values()]
        for w in blocked:
            w.blockSignals(True)
        for k in MM_FIELDS:
            self.w[k].setValue(getattr(spec, k) * 1000.0)
        for k in ("q_seed", "q_min", "q_max"):
            self.w[k].setValue(math.degrees(getattr(spec, k)))
        self.w["wheel_enabled"].setChecked(spec.wheel_enabled)
        self.w["enforce_wheel_below_hip"].setChecked(spec.enforce_wheel_below_hip)
        self.w["wheel_below_hip_margin"].setValue(
            spec.wheel_below_hip_margin * 1000.0)
        self.w["enforce_W_below_E"].setChecked(spec.enforce_W_below_E)
        self.w["enforce_ground"].setChecked(spec.enforce_ground)
        self.w["enforce_torque_limit"].setChecked(spec.enforce_torque_limit)
        self.w["leg_load_kg"].setValue(spec.leg_load_kg)
        self.w["torque_limit_nm"].setValue(spec.torque_limit_nm)
        for w in blocked:
            w.blockSignals(False)


class CollisionMatrix(QtWidgets.QGroupBox):
    """Checkbox grid: which body pairs are tested."""

    changed = QtCore.pyqtSignal()

    def __init__(self, spec: LinkageSpec):
        super().__init__("Collision pairs  (ticked = tested)")
        self.boxes = {}
        grid = QtWidgets.QGridLayout(self)
        grid.setSpacing(2)
        for i, a in enumerate(ALL_BODIES):
            for b in ALL_BODIES[i + 1:]:
                k = pair_key(a, b)
                cb = QtWidgets.QCheckBox(f"{a} ↔ {b}")
                cb.setChecked(spec.collides(a, b))
                cb.toggled.connect(self.changed)
                self.boxes[k] = cb
        for row, (k, cb) in enumerate(sorted(self.boxes.items())):
            grid.addWidget(cb, row // 2, row % 2)

        note = QtWidgets.QLabel(
            "Pairs sharing a pin (femur/tibia, tibia/coupler), the femur on the\n"
            "motor shaft, and the wheel on the tibia are off by default — those\n"
            "bodies sit on different plates and cannot interfere.  wheel/wheel_motor\n"
            "is off because the two are concentric and so always overlap.\n"
            "femur_shaft is the Ø23 mm knee shaft: it is the femur's own part and\n"
            "the tibia turns on it, so only coupler ↔ femur_shaft is on.")
        note.setStyleSheet("color: #888; font-size: 10px;")
        grid.addWidget(note, (len(self.boxes) + 1) // 2, 0, 1, 2)

    def apply_to(self, spec: LinkageSpec):
        for k, cb in self.boxes.items():
            spec.collide[k] = cb.isChecked()
        return spec

    def load_from(self, spec: LinkageSpec):
        for k, cb in self.boxes.items():
            cb.blockSignals(True)
            cb.setChecked(spec.collides(*k))
            cb.blockSignals(False)


class LinkageCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(7, 7), facecolor=BG)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.15)

    def _draw_F_dimensions(self, ax, spec: LinkageSpec, S: float):
        """Engineering-style dimension callouts for the coupler pivot F
        relative to the hip motor A at the origin."""
        fx, fz = spec.F_X * S, spec.F_Z * S
        col = "#ffb340"
        mr = spec.motor_r * S

        # Dimension lines placed in clear space: the X band above everything,
        # the Z band to the right of the motor (the mechanism runs down-left).
        z_dim = max(0.0, fz) + mr + 30.0
        x_dim = mr + 34.0

        thin = dict(color=col, linewidth=0.7, linestyle=":", alpha=0.6, zorder=9)
        ax.plot([0, 0], [0, z_dim], **thin)          # extension from A
        ax.plot([fx, fx], [fz, z_dim], **thin)       # extension from F
        ax.plot([0, x_dim], [0, 0], **thin)
        ax.plot([fx, x_dim], [fz, fz], **thin)

        arrow = dict(arrowstyle="<->", color=col, lw=1.4, shrinkA=0, shrinkB=0)

        # horizontal offset  A -> F  (X)
        ax.annotate("", xy=(fx, z_dim), xytext=(0.0, z_dim),
                    arrowprops=arrow, zorder=10)
        ax.text(fx / 2.0, z_dim + 5.0, f"X  {spec.F_X * 1000:+.2f} mm",
                color=col, fontsize=10, fontweight="bold",
                ha="center", va="bottom", zorder=11)

        # vertical offset  A -> F  (Z).  Usually only a few mm; below ~15 mm an
        # arrowhead pair just smudges, so fall back to CAD-style end ticks.
        if abs(fz) >= 15.0:
            ax.annotate("", xy=(x_dim, fz), xytext=(x_dim, 0.0),
                        arrowprops=arrow, zorder=10)
        else:
            ax.plot([x_dim, x_dim], [0.0, fz], color=col, lw=1.4, zorder=10)
            for zt in (0.0, fz):
                ax.plot([x_dim - 4.5, x_dim + 4.5], [zt, zt],
                        color=col, lw=1.4, zorder=10)
        ax.text(x_dim + 7.0, fz / 2.0, f"Z  {spec.F_Z * 1000:+.2f} mm",
                color=col, fontsize=10, fontweight="bold",
                ha="left", va="center", zorder=11)

        # centre-to-centre distance = the AF clearance the archive checked but
        # never enforced.  Text stacked above the X label, clear of the links.
        af = math.hypot(spec.F_X, spec.F_Z) * 1000.0
        ax.plot([0, fx], [0, fz], color=col, linewidth=1.0, linestyle="--",
                alpha=0.5, zorder=9)
        ax.text(fx / 2.0, z_dim + 20.0,
                f"AF {af:.2f} mm   (motor clearance {af - mr:+.2f} mm)",
                color=col, fontsize=8.5, ha="center", va="bottom", zorder=11)

    def _seg_label(self, ax, p0, p1, text, color, offset_mm, fontsize=8.5):
        """Label a bar along its own axis, pushed clear on the normal."""
        (x0, z0), (x1, z1) = p0, p1
        ang = math.degrees(math.atan2(z1 - z0, x1 - x0))
        rot = ang
        while rot > 90:
            rot -= 180
        while rot < -90:
            rot += 180
        nx, nz = -math.sin(math.radians(ang)), math.cos(math.radians(ang))
        ax.text((x0 + x1) / 2 + nx * offset_mm, (z0 + z1) / 2 + nz * offset_mm,
                text, color=color, fontsize=fontsize, fontweight="bold",
                ha="center", va="center", rotation=rot,
                rotation_mode="anchor", zorder=11,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#1e1e2e",
                          edgecolor="none", alpha=0.72))

    def _dim_between(self, ax, p0, p1, text, color, offset_mm, fontsize=8.0):
        """Proper dimension: an offset line with extension ticks and arrows,
        so it reads as a measurement between two points rather than as a label
        stuck on a bar."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = float(np.hypot(*d))
        if L < 1e-9:
            return
        nx, nz = -d[1] / L, d[0] / L
        off = np.array([nx, nz]) * offset_mm
        a, b = p0 + off, p1 + off

        thin = dict(color=color, linewidth=0.6, linestyle=":", alpha=0.55, zorder=9)
        ax.plot([p0[0], a[0]], [p0[1], a[1]], **thin)
        ax.plot([p1[0], b[0]], [p1[1], b[1]], **thin)
        ax.annotate("", xy=tuple(b), xytext=tuple(a), zorder=10,
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.2,
                                    shrinkA=0, shrinkB=0))
        self._seg_label(ax, a, b, text, color, 0.0, fontsize)

    def _dim_path_box(self, ax, box, mean_x: float, x_clear: float):
        """Dimension the traced point's bounding box against the hip axis at
        the origin.

        `box` is (x0, x1, z0, z1) in mm and `mean_x` the best-fit vertical the
        deviation metric is measured from.  The traced point never leaves the
        box over the valid stroke, so its height IS the stroke; what the hip
        needs to know on top of that is how far down the box starts and how far
        the band sits fore or aft of the motor axis.
        """
        col = "#40ff40"
        x0, x1, z0, z1 = box
        arrow = dict(arrowstyle="<->", color=col, lw=1.1, shrinkA=0, shrinkB=0)
        thin = dict(color=col, linewidth=0.6, linestyle=":", alpha=0.5, zorder=9)
        bbox = dict(boxstyle="round,pad=0.18", facecolor="#1e1e2e",
                    edgecolor="none", alpha=0.75)

        # Z: origin down to the top edge, then the box height itself.  Chained
        # rather than two measurements from the origin, so the second number is
        # the stroke — the thing being designed for — and the bottom edge is
        # just their sum.  The first hangs off the z = 0 line the axes already
        # draw, and both sit clear of the F band, which owns the strip right of A.
        xd = max(x1 + 34.0, x_clear)
        ax.plot([x1, xd], [z1, z1], **thin)
        ax.annotate("", xy=(xd, z1), xytext=(xd, 0.0), arrowprops=arrow,
                    zorder=10)
        ax.text(xd + 3.0, z1 / 2.0, f"Z top {z1:+.2f} mm", color=col,
                fontsize=7.5, fontweight="bold", ha="center", va="bottom",
                rotation=90, rotation_mode="anchor", zorder=11, bbox=bbox)

        xs = xd + 30.0
        for zz in (z1, z0):
            ax.plot([x1, xs], [zz, zz], **thin)
        ax.annotate("", xy=(xs, z0), xytext=(xs, z1), arrowprops=arrow,
                    zorder=10)
        ax.text(xs + 3.0, (z0 + z1) / 2.0, f"stroke {z1 - z0:.2f} mm", color=col,
                fontsize=7.5, fontweight="bold", ha="center", va="bottom",
                rotation=90, rotation_mode="anchor", zorder=11, bbox=bbox)

        # X: how wide the wander is, then where that band sits relative to the
        # hip.  Stacked below the box; the second hangs off the x = 0 line the
        # axes already draw.
        zw, zm = z0 - 20.0, z0 - 40.0

        ax.plot([x0, x0], [z0, zw], **thin)
        ax.plot([x1, x1], [z0, zw], **thin)
        ax.annotate("", xy=(x1, zw), xytext=(x0, zw), arrowprops=arrow, zorder=10)
        ax.text((x0 + x1) / 2.0, zw - 3.0, f"width {x1 - x0:.2f} mm", color=col,
                fontsize=7.5, fontweight="bold", ha="center", va="top",
                zorder=11, bbox=bbox)

        ax.plot([mean_x, mean_x], [z0, zm], **thin)
        ax.annotate("", xy=(mean_x, zm), xytext=(0.0, zm), arrowprops=arrow,
                    zorder=10)
        ax.text(mean_x / 2.0, zm - 3.0, f"best-fit X {mean_x:+.2f} mm", color=col,
                fontsize=7.5, fontweight="bold", ha="center", va="top",
                zorder=11, bbox=bbox)
        # No W × H label on the box itself: the top edge is exactly where the
        # femur sweeps, and both spans already follow from the dimensions —
        # the height is the stroke.

    def _draw_link_dimensions(self, ax, spec: LinkageSpec, pose, S: float):
        """Length of every bar, drawn on the bar itself."""
        n = pose.nodes
        A, C, E, F, W = (np.array(n[k]) * S for k in "ACEFW")
        col = "#e8e8f4"

        self._seg_label(ax, A, C, f"femur  {spec.L_femur*1000:.2f} mm", col,
                        spec.femur_r_A * S + 11)
        self._seg_label(ax, F, E, f"coupler  {spec.Lc*1000:.2f} mm", col,
                        -(spec.coupler_r_F * S + 11))
        # The tibia carries three nodes, so name every span explicitly — with
        # w_perp = 0 they are collinear and there is otherwise no way to see
        # whether a length is measured to C or along E–W.
        # Every label below is the TRUE distance between the two points the
        # arrow spans.  L_tibia is only the axial component of C->W, so it is
        # NOT |CW| once w_perp bends the bar — label it as such or CAD built
        # from these numbers comes out short.
        tib = "#d9b3ff"
        r = spec.tibia_r_W * S
        ec_len = math.hypot(*(C - E))
        cw_len = math.hypot(*(W - C))
        ew_len = math.hypot(*(W - E))

        self._dim_between(ax, E, C, f"EC  {ec_len:.2f} mm",
                          tib, -(r + 16), fontsize=7.5)
        self._dim_between(ax, C, W, f"CW  {cw_len:.2f} mm",
                          tib, -(r + 16), fontsize=7.5)
        self._dim_between(ax, E, W, f"EW  {ew_len:.2f} mm",
                          tib, -(r + 42), fontsize=7.5)

        # Dogleg: the straight E–W run, and how far the knee sits off it.
        if abs(spec.w_perp) > 1e-6:
            dog = "#ff9de2"
            ax.plot([E[0], W[0]], [E[1], W[1]], color=dog, lw=1.0,
                    linestyle="--", alpha=0.85, zorder=10)
            d = W - E
            nx, nz = -d[1] / ew_len, d[0] / ew_len
            co = spec.c_offset() * S
            sign = 1.0 if ((C[0] - E[0]) * nx + (C[1] - E[1]) * nz) > 0 else -1.0
            foot = C - np.array([nx, nz]) * co * sign
            ax.plot([C[0], foot[0]], [C[1], foot[1]], color=dog, lw=1.4,
                    zorder=10)
            ax.text(C[0] - nx * sign * co * 0.5 + 6,
                    C[1] - nz * sign * co * 0.5,
                    f"C_offset {spec.c_offset()*1000:.2f} mm",
                    color=dog, fontsize=7.5, fontweight="bold",
                    ha="left", va="center", zorder=11)
        else:
            # Collinear: say so, and show that the spans must add up.
            ax.text(*(E + W) / 2, "", color=tib)
            self._seg_label(
                ax, E, W,
                f"collinear:  {spec.L_stub*1000:.2f} + {spec.L_tibia*1000:.2f}"
                f" = {ew_len:.2f}",
                "#9a86b8", -(r + 58), fontsize=6.5)

    def draw_state(self, spec: LinkageSpec, q: float, metrics, shapes: ShapeSet,
                   show_ghosts=True, show_path=True, show_dims=True,
                   xlim=None, ylim=None, clean=True):
        """`clean` drops the in-plot stats box and legend — and the empty band
        reserved for the stats box — so the mechanism gets the whole panel.
        The same numbers and key live in the right dock either way."""
        ax = self.ax
        ax.clear()
        # The legend sits under the axes, so its margin is only worth paying
        # for when it is actually drawn.
        self.fig.subplots_adjust(bottom=0.15 if not clean else 0.09)
        ax.set_facecolor(BG)
        ax.set_aspect("equal")
        ax.grid(True, color="#333", linewidth=0.4, linestyle="--")
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.axvline(0, color="#555", linewidth=0.8)

        if spec.enforce_wheel_below_hip:
            zl = spec.wheel_below_hip_margin * 1000.0
            ax.axhline(zl, color="#ff9040", linewidth=1.2, linestyle="-.",
                       alpha=0.75, zorder=2,
                       label="ride-height limit")
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        ax.set_xlabel("x  [mm]  (forward →)", color=FG, fontsize=9)
        ax.set_ylabel("z  [mm]  (up ↑)", color=FG, fontsize=9)

        S = 1000.0  # metres -> mm for display

        # ── ghost outlines at the range ends ────────────────────────────────
        if show_ghosts and metrics is not None and metrics.valid:
            # Range-end silhouettes: barely-there reference outlines.
            for qg in (metrics.q_lo, metrics.q_hi):
                pg = solve_pose(spec, qg, metrics.alpha_at(qg))
                if pg is None:
                    continue
                for link in MOVING_LINKS:
                    poly = shapes.world(pg, link) * S
                    ax.add_patch(mpatches.Polygon(
                        poly, closed=True, facecolor="none", edgecolor="#3a3a4a",
                        linewidth=0.6, alpha=0.45, zorder=1))


        # ── traced path ─────────────────────────────────────────────────────
        if show_path and metrics is not None and metrics.valid and metrics.path is not None:
            p = metrics.path * S
            ax.plot(p[:, 0], p[:, 1], color="#ff6060", linewidth=2.0,
                    alpha=0.55, zorder=2, label="traced path")
            # Bounding box of the path over the valid stroke: the traced point
            # stays inside it for every reachable pose, so its height is the
            # travel and its width the fore/aft wander.
            box = (float(p[:, 0].min()), float(p[:, 0].max()),
                   float(p[:, 1].min()), float(p[:, 1].max()))
            ax.add_patch(mpatches.Rectangle(
                (box[0], box[2]), box[1] - box[0], box[3] - box[2],
                facecolor="#40ff40", alpha=0.06, linewidth=0.0, zorder=1))
            ax.add_patch(mpatches.Rectangle(
                (box[0], box[2]), box[1] - box[0], box[3] - box[2],
                facecolor="none", edgecolor="#40ff40", linewidth=1.1,
                linestyle="--", alpha=0.85, zorder=3, label="wheel travel box"))
            # Best-fit vertical, clipped to the box rather than run across the
            # whole plot: it is the line max_dev is measured from, so it only
            # means anything over the stroke.
            ax.plot([metrics.mean_x_mm] * 2, [box[2], box[3]], color="#40ff40",
                    linewidth=0.9, linestyle=":", alpha=0.8, zorder=3)
            if show_dims:
                self._dim_path_box(ax, box, metrics.mean_x_mm,
                                   spec.motor_r * S + 52.0)

        # ── current pose ────────────────────────────────────────────────────
        # Stay on the same assembly branch the sweep used.
        alpha_prev = metrics.alpha_at(q) if (metrics and metrics.valid) else None
        pose = solve_pose(spec, q, alpha_prev)
        if pose is None:
            ax.set_title("SINGULAR at this hip angle", color="#ff6060", fontsize=11)
            self.draw_idle()
            return

        hits = collisions(spec, pose, shapes)
        hit_bodies = {b for pr in hits for b in pr}

        for link in MOVING_LINKS:
            poly = shapes.world(pose, link) * S
            bad = link in hit_bodies
            ax.add_patch(mpatches.Polygon(
                poly, closed=True,
                facecolor=("#ff4040" if bad else LINK_COLOR[link]),
                edgecolor=("#ff8080" if bad else "#ddd"),
                alpha=0.55 if not bad else 0.75, linewidth=1.4, zorder=4,
                label=link))

        # motor
        ax.add_patch(mpatches.Circle(
            (0, 0), spec.motor_r * S,
            facecolor=("#ff4040" if MOTOR in hit_bodies else "#e05c5c"),
            edgecolor="#ddd", alpha=0.5, linewidth=1.4, zorder=3))
        ax.text(0, 0, "motor", color="white", fontsize=6.5,
                ha="center", va="center", zorder=9)

        # Knee shaft on the femur at C.  Drawn on top of the plates it sits
        # between, because what it fouls is the coupler passing over it.
        ax.add_patch(mpatches.Circle(
            np.array(pose.nodes["C"]) * S, spec.femur_shaft_r * S,
            facecolor=("#ff4040" if FEMUR_SHAFT in hit_bodies else "none"),
            edgecolor=("#ff8080" if FEMUR_SHAFT in hit_bodies else SHAFT_COLOR),
            alpha=0.9, linewidth=1.4, zorder=7, label="knee shaft"))

        # wheel + hub motor (concentric)
        wc = np.array(trace_world(pose, spec.primary_trace())) * S
        if spec.wheel_enabled:
            ax.add_patch(mpatches.Circle(
                wc, spec.wheel_r * S,
                facecolor="none",
                edgecolor=("#ff4040" if WHEEL in hit_bodies else "#40ff80"),
                linewidth=1.6, alpha=0.8, zorder=3))
        bad_hub = WHEEL_MOTOR in hit_bodies
        ax.add_patch(mpatches.Circle(
            wc, spec.wheel_motor_r * S,
            facecolor=("#ff4040" if bad_hub else "#e05c5c"),
            edgecolor="#ddd", alpha=0.5, linewidth=1.4, zorder=3))
        ax.text(wc[0], wc[1], "hub", color="white", fontsize=6.5,
                ha="center", va="center", zorder=9)

        # nodes
        for name, (x, z) in pose.nodes.items():
            ax.plot(x * S, z * S, "o", color=NODE_COLOR.get(name, "white"),
                    markersize=6, zorder=8)
            ax.annotate(name, (x * S, z * S), textcoords="offset points",
                        xytext=(7, 5), color=NODE_COLOR.get(name, "white"),
                        fontsize=9, fontweight="bold", zorder=9)

        # trace points other than the primary
        for tp in spec.trace_points:
            if tp.primary:
                continue
            x, z = trace_world(pose, tp)
            ax.plot(x * S, z * S, "x", color=tp.color, markersize=8, zorder=8)

        # Upper tangent of the wheel — the surface the ride-height limit tests.
        if spec.enforce_wheel_below_hip:
            wc = trace_world(pose, spec.primary_trace())
            top = (wc[1] + spec.wheel_r) * S
            lim = spec.wheel_below_hip_margin * S
            near = top > lim - 6.0
            ax.plot([(wc[0] - spec.wheel_r) * S, (wc[0] + spec.wheel_r) * S],
                    [top, top],
                    color=("#ff5050" if near else "#ff9040"),
                    linewidth=1.6, linestyle="--", alpha=0.9, zorder=6)
            ax.annotate("", xy=((wc[0]) * S, lim), xytext=((wc[0]) * S, top),
                        arrowprops=dict(arrowstyle="<->", color="#ff9040",
                                        lw=1.0, shrinkA=0, shrinkB=0), zorder=6)
            ax.text((wc[0]) * S + 5, (top + lim) / 2,
                    f"{(lim - top):.1f} mm", color="#ff9040", fontsize=7.5,
                    fontweight="bold", ha="left", va="center", zorder=11)

        # Ground plane, tangent to the wheel bottom at THIS pose.
        if spec.enforce_ground:
            from evaluate import ground_z
            gz = ground_z(spec, pose) * S
            ax.axhline(gz, color="#8a5a2a", linewidth=2.0, alpha=0.9, zorder=2)
            ax.axhspan(gz - 400, gz, facecolor="#5a3a1a", alpha=0.22, zorder=0)
            ax.text(0.985, gz, " ground", color="#d09050", fontsize=8,
                    ha="right", va="bottom", transform=ax.get_yaxis_transform(),
                    zorder=11)

        if show_dims:
            self._draw_F_dimensions(ax, spec, S)
            self._draw_link_dimensions(ax, spec, pose, S)

        # ── on-plot statistics ──────────────────────────────────────────────
        # Whether a config is any good comes down to stroke, deviation and how
        # much torque headroom is left, so those stay ON the diagram in both
        # modes.  Clean mode shrinks them into the empty top-left corner rather
        # than holding a band open below the mechanism for a full-size box.
        if metrics is not None and metrics.valid:
            tau = hip_torque(spec, pose)
            if clean:
                stats = "\n".join((
                    f"stroke  {metrics.travel_mm:7.2f} mm  ({metrics.stroke_deg:.2f}°)",
                    f"max dev {metrics.max_dev_mm:7.2f} mm",
                    f"τ peak  {metrics.max_torque_nm:7.2f} / {spec.torque_limit_nm:.2f} N·m",
                ))
            else:
                stats = "\n".join((
                    f"max vertical deviation {metrics.max_dev_mm:7.2f} mm",
                    f"vertical travel ret→ext{metrics.travel_mm:7.2f} mm",
                    f"stroke {metrics.stroke_deg:6.2f}°  rms dev {metrics.rms_dev_mm:5.2f} mm",
                    f"τ here {tau:5.2f}   peak {metrics.max_torque_nm:5.2f} N·m",
                    f"τ limit {spec.torque_limit_nm:.2f} @ {spec.leg_load_kg:.2f} kg/leg",
                ))
        else:
            stats = "INVALID\n" + (metrics.reason if metrics is not None else "")

        # Clean: top-left, the one corner nothing else claims — the F dimension
        # band starts at the origin and runs right, and the mechanism hangs
        # below it.  Full: bottom-left, inside the band reserved for it, where
        # the far left of the ground strip is reliably empty.
        if clean:
            ax.text(0.012, 0.988, stats, transform=ax.transAxes,
                    color="#e8e8f4", fontsize=7.5, family="monospace",
                    fontweight="bold", va="top", ha="left", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#2a2a3e",
                              edgecolor="#6a6a85", alpha=0.85))
        else:
            ax.text(0.015, 0.022, stats, transform=ax.transAxes,
                    color="#ffffff", fontsize=9.0, family="monospace",
                    fontweight="bold", va="bottom", ha="left", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2a3e",
                              edgecolor="#6a6a85", alpha=0.92))

        title = f"hip {math.degrees(q):+.2f}°"
        if hits:
            title += "   COLLISION: " + ", ".join(f"{a}/{b}" for a, b in hits)
            ax.set_title(title, color="#ff6060", fontsize=11)
        else:
            ax.set_title(title + "   clear", color="#8fe08f", fontsize=11)

        if xlim is not None and ylim is not None:
            # Fixed axes: computed once over the whole sweep so the view never
            # shifts while scrubbing, and both compare panels share one scale.
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        else:
            ax.relim()
            ax.autoscale_view()
            lo_x, hi_x = ax.get_xlim()
            lo_z, hi_z = ax.get_ylim()
            pad = 20.0
            top_pad = 58.0 if show_dims else pad
            right_pad = 95.0 if show_dims else pad
            band = 0.0 if clean else STATS_BAND * (hi_z - lo_z)
            ax.set_xlim(lo_x - pad, hi_x + right_pad)
            ax.set_ylim(lo_z - band - pad, hi_z + top_pad)
        # Outside the axes, under the x-label.  Anywhere inside the data area
        # eventually collides: the mechanism sweeps the lower-left, the F
        # dimension band owns the top, and the stats box the lower-right.
        if not clean:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.085), ncol=6,
                      fontsize=6.5, facecolor="#2a2a3e", edgecolor="#555",
                      labelcolor="white", framealpha=0.9, borderpad=0.4,
                      labelspacing=0.3, columnspacing=1.1, handlelength=1.6)
        self.draw_idle()


# Let the slider run this far past each collision/singularity limit, so the
# interfering pose can actually be seen (drawn with solid red fills).
OVERSHOOT = math.radians(1.0)

# Fraction of the mechanism's own height reserved as empty band below it, so
# the stats box always has somewhere to sit.  At full extension the leg spans
# the whole width of the plot, so no in-axes corner is free by luck — the room
# has to be made deliberately.
STATS_BAND = 0.22

REF_COLOR = "#5ad7d7"
WORK_COLOR = "#f0c040"


class TorqueCanvas(FigureCanvasQTAgg):
    """Static hold torque across the stroke, with a cursor at the slider pose.

    x is % of each config's OWN reachable range — the same convention the
    slider uses — so two mechanisms with different strokes overlay directly.
    The cursor can sit outside 0-100%: the slider is allowed to run OVERSHOOT
    past each end so the blocking pose is visible, and torque out there is
    exactly what the motor cannot deliver.
    """

    def __init__(self):
        self.fig = Figure(figsize=(7, 2.2), facecolor=BG)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.085, right=0.985, top=0.97, bottom=0.30)
        self.setMinimumHeight(200)
        self.setMaximumHeight(270)

    def _cursor(self, panel, frac: float):
        """(stroke_pct, q, tau) for this panel at the slider position."""
        m = panel.metrics
        span = m.q_hi - m.q_lo
        if span < 1e-9:
            return None
        q = (m.q_lo - OVERSHOOT) + (span + 2.0 * OVERSHOOT) * frac
        pose = solve_pose(panel.spec, q, m.alpha_at(q))
        if pose is None:
            return None
        return (q - m.q_lo) / span * 100.0, q, hip_torque(panel.spec, pose)

    def draw_curves(self, entries, frac: float):
        """entries = [(label, color, ComparePanel), ...]"""
        ax = self.ax
        ax.clear()
        ax.set_facecolor(BG)
        ax.grid(True, color="#333", linewidth=0.4, linestyle="--")
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        ax.set_ylabel("hip torque\n[N·m]", color=FG, fontsize=8)

        live = [(lab, col, p) for lab, col, p in entries
                if p.spec is not None and p.metrics is not None
                and p.metrics.valid and p.metrics.torques is not None]
        if not live:
            ax.set_xlabel("% of reachable stroke", color=FG, fontsize=8)
            ax.text(0.5, 0.5, "no valid configuration", color="#ff6060",
                    fontsize=10, ha="center", va="center",
                    transform=ax.transAxes)
            self.draw_idle()
            return

        # Which end is which is a property of the mechanism, not a convention:
        # read it off the traced path rather than assuming q_lo is extended.
        z = live[0][2].metrics.path[:, 1]
        lo_lab, hi_lab = (("extended", "retracted") if z[0] < z[-1]
                          else ("retracted", "extended"))
        ax.set_xlabel(f"% of reachable stroke   (0 = {lo_lab}  →  "
                      f"100 = {hi_lab})", color=FG, fontsize=8)

        # One dashed line per distinct limit — the two sides may differ.
        for lim in sorted({p.spec.torque_limit_nm for _, _, p in live}):
            ax.axhline(lim, color="#ff5050", linewidth=1.3, linestyle="--",
                       alpha=0.9, zorder=3)
            # Mid-span: the note box owns the top-left and the cursor label the
            # end the slider is nearest, so both edges are contested.
            ax.text(0.42, lim, f"limit {lim:.2f}", color="#ff5050",
                    fontsize=7.5, fontweight="bold", ha="left", va="bottom",
                    transform=ax.get_yaxis_transform(), zorder=6)

        note = []
        for label, color, p in live:
            m = p.metrics
            span = m.q_hi - m.q_lo
            x = (m.q_samples - m.q_lo) / span * 100.0
            ax.plot(x, m.torques, color=color, linewidth=1.8, zorder=4,
                    label=f"{label}  peak {m.max_torque_nm:.2f} N·m "
                          f"@ {math.degrees(m.q_max_torque):+.1f}°")
            xp = (m.q_max_torque - m.q_lo) / span * 100.0
            ax.plot([xp], [m.max_torque_nm], "v", color=color, markersize=5,
                    alpha=0.8, zorder=5)

            cur = self._cursor(p, frac)
            if cur is None:
                note.append(f"{label:9s}  — singular at this pose")
                continue
            cx, cq, ctau = cur
            ax.axvline(cx, color=color, linewidth=1.0, linestyle=":",
                       alpha=0.65, zorder=3)
            ax.plot([cx], [ctau], "o", color=color, markersize=7,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=7)
            # Flip the callout inboard near the right end, or it is clipped.
            side = 1 if cx <= 55.0 else -1
            ax.annotate(f"{ctau:.2f} N·m", (cx, ctau),
                        textcoords="offset points", xytext=(8 * side, 6),
                        ha="left" if side > 0 else "right",
                        color=color, fontsize=9, fontweight="bold", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="#1e1e2e",
                                  edgecolor=color, alpha=0.85))
            pct = ctau / p.spec.torque_limit_nm * 100.0
            over = "  OVER LIMIT" if ctau > p.spec.torque_limit_nm else ""
            note.append(f"{label:9s} {ctau:6.2f} N·m  @ hip {math.degrees(cq):+7.2f}°"
                        f"   {pct:5.1f}% of limit{over}")

        ax.text(0.008, 0.97, "\n".join(note), transform=ax.transAxes,
                color="#e0e0f0", fontsize=8.5, family="monospace",
                va="top", ha="left", zorder=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#2a2a3e",
                          edgecolor="#6a6a85", alpha=0.92))

        ax.set_xlim(-6, 106)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(0.0, lo), hi * 1.28)      # headroom for the labels
        ax.legend(loc="lower right", fontsize=7, facecolor="#2a2a3e",
                  edgecolor="#555", labelcolor="white", framealpha=0.85,
                  borderpad=0.35, labelspacing=0.25)
        self.draw_idle()


def config_bounds(spec: LinkageSpec, metrics, shapes: ShapeSet,
                  n: int = 48) -> tuple[float, float, float, float]:
    """Bounding box in mm over the whole (overshot) sweep, so the axes can be
    fixed once and never move while scrubbing."""
    xs, zs = [], []
    if metrics is not None and metrics.valid:
        q0, q1 = metrics.q_lo - OVERSHOOT, metrics.q_hi + OVERSHOOT
    else:
        q0, q1 = spec.q_min, spec.q_max

    for q in np.linspace(q0, q1, n):
        ap = metrics.alpha_at(q) if (metrics is not None and metrics.valid) else None
        p = solve_pose(spec, float(q), ap)
        if p is None:
            continue
        for link in MOVING_LINKS:
            poly = shapes.world(p, link)
            xs += [poly[:, 0].min(), poly[:, 0].max()]
            zs += [poly[:, 1].min(), poly[:, 1].max()]
        wx, wz = trace_world(p, spec.primary_trace())
        r = max(spec.wheel_motor_r, spec.wheel_r if spec.wheel_enabled else 0.0)
        xs += [wx - r, wx + r]
        zs += [wz - r, wz + r]

    xs += [-spec.motor_r, spec.motor_r]
    zs += [-spec.motor_r, spec.motor_r]
    if not xs:
        return -300.0, 150.0, -350.0, 100.0
    return (min(xs) * 1000.0, max(xs) * 1000.0,
            min(zs) * 1000.0, max(zs) * 1000.0)


def spec_report(spec: LinkageSpec, metrics=None) -> str:
    """Full human-readable dump of a configuration."""
    L = []
    mm = lambda v: f"{v * 1000:9.2f} mm"

    cw = math.hypot(spec.L_tibia, spec.w_perp)
    ew = math.hypot(spec.L_tibia + spec.L_stub, spec.w_perp)

    L.append("MODEL PARAMETERS  (what the optimizer varies)")
    L.append(f"  L_femur        {mm(spec.L_femur)}   = |AC|")
    L.append(f"  L_stub         {mm(spec.L_stub)}   = |EC|")
    L.append(f"  Lc             {mm(spec.Lc)}   = |FE|")
    L.append(f"  L_tibia        {mm(spec.L_tibia)}   AXIAL component of C→W")
    L.append(f"  w_perp         {mm(spec.w_perp)}   PERPENDICULAR component")
    L.append("")
    L.append("*** BUILD FROM THESE — true point-to-point distances ***")
    L.append(f"  |AC|  femur    {mm(spec.L_femur)}")
    L.append(f"  |FE|  coupler  {mm(spec.Lc)}")
    L.append(f"  |EC|  stub     {mm(spec.L_stub)}")
    L.append(f"  |CW|           {mm(cw)}   = hypot(L_tibia, w_perp)")
    L.append(f"  |EW|           {mm(ew)}")
    if abs(spec.w_perp) > 1e-9:
        L.append(f"  NOTE: |CW| != L_tibia ({cw*1000:.3f} vs "
                 f"{spec.L_tibia*1000:.3f} mm).  Using L_tibia as the C→W")
        L.append(f"        distance builds the bar {(cw-spec.L_tibia)*1000:.3f}"
                 f" mm short.")
    L.append("")
    L.append("LINK SPANS  (radius-centre to radius-centre)")
    for k, v in spec.link_spans().items():
        L.append(f"  {k:8s}       {mm(v)}")
    L.append(f"  largest        {mm(spec.max_link_span())}")
    L.append("")
    L.append("PIVOT F  (hip motor A is the origin)")
    L.append(f"  F_X            {mm(spec.F_X)}")
    L.append(f"  F_Z            {mm(spec.F_Z)}")
    af = math.hypot(spec.F_X, spec.F_Z)
    L.append(f"  |AF|           {mm(af)}")
    L.append(f"  motor clear    {mm(af - spec.motor_r)}")
    L.append(f"  F_Z in body frame (A_Z = -23.50 mm):"
             f" {spec.to_body_frame_fz()*1000:+.2f} mm")
    L.append("")
    L.append("TIBIA DOGLEG  (three readings of the same bend — all different)")
    L.append(f"  w_perp         {mm(spec.w_perp)}   W off the C→E axis"
             f"  [free variable]")
    L.append(f"  C_offset       {mm(spec.c_offset())}   C off the E–W line"
             f"  [derived]")
    ang = math.degrees(math.atan2(spec.w_perp, spec.L_tibia))
    L.append(f"  dogleg angle   {ang:+9.3f} °   of C→W off the tibia axis")
    L.append("    c_offset = L_stub * |w_perp| / |EW|, so it is always smallest")
    L.append("")
    L.append("LINK END RADII  (per link, per end)")
    for link, nodes in spec.node_radii().items():
        for node, r in nodes.items():
            L.append(f"  {link:8s} @ {node}  {mm(r)}")
    L.append("")
    L.append("CIRCULAR BODIES")
    L.append(f"  knee shaft     r = {spec.femur_shaft_r*1000:.2f} mm"
             f"  (Ø{spec.femur_shaft_r*2000:.0f} mm) on the femur at C")
    L.append(f"  motor radius   {mm(spec.motor_r)}")
    L.append(f"  wheel          {'enabled' if spec.wheel_enabled else 'disabled'}"
             f"   r = {spec.wheel_r*1000:.2f} mm"
             f"  (Ø{spec.wheel_r*2000:.0f} mm)")
    L.append(f"  hub motor      r = {spec.wheel_motor_r*1000:.2f} mm"
             f"  (Ø{spec.wheel_motor_r*2000:.0f} mm), concentric with the wheel")
    L.append("")
    L.append("POSE LIMITS")
    L.append(f"  wheel top      {'ON ' if spec.enforce_wheel_below_hip else 'off'}"
             f"   W_z + wheel_r ≤ {spec.wheel_below_hip_margin*1000:+.2f} mm"
             f"   i.e. W_z ≤ {(spec.wheel_below_hip_margin - spec.wheel_r)*1000:+.2f} mm")
    L.append("                     upper tangent of the tyre, not its centre")
    L.append(f"  W below E      {'ON ' if spec.enforce_W_below_E else 'off'}"
             f"   (above it the tibia has leaned back past horizontal)")
    L.append(f"  ground plane   {'ON ' if spec.enforce_ground else 'off'}"
             f"   tangent to wheel bottom, {spec.wheel_r*1000:.1f} mm below W")
    L.append("                     no link and not the motor may reach it")
    L.append(f"  hip torque     {'ON ' if spec.enforce_torque_limit else 'off'}"
             f"   tau ≤ {spec.torque_limit_nm:.2f} N·m"
             f"   at {spec.leg_load_kg:.2f} kg on this leg")
    L.append("                     tau = load * g * |dz_W/dq|, static hold with")
    L.append("                     the wheel on the ground.  Only the vertical")
    L.append("                     gear ratio does work — the wheel rolls.")
    L.append("")
    L.append("HIP SWEEP")
    L.append(f"  seed           {math.degrees(spec.q_seed):+9.2f} °")
    L.append(f"  limits         {math.degrees(spec.q_min):+9.2f} °"
             f"  to {math.degrees(spec.q_max):+.2f} °")
    L.append("")
    L.append("COLLISION PAIRS TESTED")
    on = sorted(f"{a}/{b}" for (a, b), v in spec.collide.items() if v)
    off = sorted(f"{a}/{b}" for (a, b), v in spec.collide.items() if not v)
    L.append("  on : " + (", ".join(on) or "none"))
    L.append("  off: " + (", ".join(off) or "none"))
    L.append("")
    tp = spec.primary_trace()
    L.append("TRACED POINT (primary)")
    L.append(f"  {tp.name}  on {tp.body}  local"
             f" ({tp.local_x*1000:+.2f}, {tp.local_z*1000:+.2f}) mm")

    if metrics is not None:
        L.append("")
        L.append("RESULTS")
        if not metrics.valid:
            L.append(f"  INVALID — {metrics.reason}")
        else:
            L.append(f"  max vertical deviation   {metrics.max_dev_mm:9.2f} mm")
            L.append(f"  rms deviation            {metrics.rms_dev_mm:9.2f} mm")
            L.append(f"  vertical travel ret→ext  {metrics.travel_mm:9.2f} mm")
            L.append(f"  hip stroke               {metrics.stroke_deg:9.2f} °")
            L.append(f"  range   [{math.degrees(metrics.q_lo):+.2f},"
                     f" {math.degrees(metrics.q_hi):+.2f}] °")
            L.append(f"  best-fit vertical at x   {metrics.mean_x_mm:9.2f} mm")
            L.append(f"  peak hip torque          {metrics.max_torque_nm:9.2f} N·m"
                     f"  at {math.degrees(metrics.q_max_torque):+.2f} °")
            L.append(f"  torque headroom          "
                     f"{spec.torque_limit_nm - metrics.max_torque_nm:9.2f} N·m")
            L.append(f"  blocked lo   {metrics.stop_lo}")
            L.append(f"  blocked hi   {metrics.stop_hi}")
    return "\n".join(L)


class DetailsDialog(QtWidgets.QDialog):
    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 760)
        lay = QtWidgets.QVBoxLayout(self)
        box = QtWidgets.QPlainTextEdit(text)
        box.setReadOnly(True)
        box.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        box.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        lay.addWidget(box)
        row = QtWidgets.QHBoxLayout()
        b_copy = QtWidgets.QPushButton("Copy")
        b_copy.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(text))
        b_close = QtWidgets.QPushButton("Close")
        b_close.clicked.connect(self.accept)
        row.addStretch(1)
        row.addWidget(b_copy)
        row.addWidget(b_close)
        lay.addLayout(row)


class ComparePanel(QtWidgets.QWidget):
    """One side of the A/B view: a canvas plus its own metrics line.

    The reference side gets a preset dropdown; the working side is driven by
    the parameter panel.
    """

    preset_changed = QtCore.pyqtSignal(str)

    def __init__(self, title: str, selectable: bool):
        super().__init__()
        self.spec: LinkageSpec | None = None
        self.metrics = None
        self.shapes: ShapeSet | None = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(3)

        head = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(title)
        lbl.setStyleSheet("font-weight: bold;")
        head.addWidget(lbl)
        self.combo = None
        if selectable:
            self.combo = QtWidgets.QComboBox()
            self.combo.setMinimumWidth(180)
            self.combo.currentTextChanged.connect(self.preset_changed)
            head.addWidget(self.combo, 1)
        else:
            head.addStretch(1)

        self.b_export = QtWidgets.QPushButton("Export…")
        self.b_export.setToolTip(
            "Save this diagram exactly as drawn — PDF or SVG for print,\n"
            "PNG for a quick paste")
        self.b_export.setMaximumWidth(80)
        self.b_export.clicked.connect(self.export_diagram)
        head.addWidget(self.b_export)
        lay.addLayout(head)

        self.canvas = LinkageCanvas()
        lay.addWidget(self.canvas, 1)

        # Metrics text and the details button are built here — this panel owns
        # them — but MainWindow parents them into the right dock.  Every row
        # under the canvas costs height on both panels at once, and the diagram
        # is what needs the room.
        self.info = QtWidgets.QWidget()
        ilay = QtWidgets.QVBoxLayout(self.info)
        ilay.setContentsMargins(0, 0, 0, 0)
        ilay.setSpacing(3)

        self.metrics_lbl = QtWidgets.QLabel("—")
        self.metrics_lbl.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px; color: #c0c0d0;")
        self.metrics_lbl.setWordWrap(True)
        ilay.addWidget(self.metrics_lbl)

        self.b_details = QtWidgets.QPushButton("Show details…")
        self.b_details.setToolTip(
            "Every dimension of this configuration, plus its results")
        self.b_details.clicked.connect(self.show_details)
        ilay.addWidget(self.b_details)

        self._title = title

    def export_diagram(self):
        """Write the canvas as it stands, so what you see is what prints."""
        if self.spec is None:
            return
        name = self.combo.currentText() if self.combo else self._title
        stem = "".join(c if (c.isalnum() or c in "-_.") else "_"
                       for c in name).strip("_") or "diagram"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export diagram", f"{stem}.pdf",
            "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)")
        if not path:
            return
        fig = self.canvas.fig
        try:
            fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
        except Exception as e:                       # bad path, locked file…
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))

    def show_details(self):
        if self.spec is None:
            return
        name = self.combo.currentText() if self.combo else self._title
        DetailsDialog(f"{self._title} — {name}",
                      spec_report(self.spec, self.metrics), self).exec()

    def _set_combo_label(self, label: str):
        """Point the dropdown at `label`, inserting it if it is a placeholder.

        Signals stay blocked: this reports what is already being shown, and
        must never be mistaken for the user picking an entry.
        """
        if self.combo is None:
            return
        self.combo.blockSignals(True)
        if self.combo.findText(label) < 0:
            self.combo.insertItem(0, label)
        self.combo.setCurrentText(label)
        self.combo.blockSignals(False)

    def set_spec(self, spec: LinkageSpec, use_collisions: bool,
                 label: str | None = None):
        """`label` names what is being shown.  Pass it whenever the spec did
        NOT come from the dropdown, or the two silently drift apart."""
        if label is not None:
            self._set_combo_label(label)
        self.spec = spec
        self.shapes = ShapeSet(spec, RES_DRAW)
        self.metrics = evaluate(spec, ShapeSet(spec, RES_FAST), use_collisions)
        m = self.metrics
        if m.valid:
            self.metrics_lbl.setText(
                f"stroke {m.stroke_deg:6.2f}°   travel {m.travel_mm:7.2f} mm   "
                f"dev {m.max_dev_mm:6.2f} mm\n"
                f"peak τ {m.max_torque_nm:5.2f} / {spec.torque_limit_nm:.2f} N·m"
                f"   @ {spec.leg_load_kg:.2f} kg/leg\n"
                f"lo: {m.stop_lo}\nhi: {m.stop_hi}")
        else:
            self.metrics_lbl.setText(f"INVALID — {m.reason}")

    def draw_at_fraction(self, frac: float, ghosts: bool, path: bool, dims: bool,
                         xlim=None, ylim=None, clean: bool = True):
        """frac in [0,1] across this config's OWN reachable range, so two
        mechanisms with different strokes stay comparable.  The range is
        extended by OVERSHOOT at each end so the blocking pose is reachable and
        renders with solid red fills."""
        if self.spec is None:
            return None
        m = self.metrics
        if m is not None and m.valid:
            q0, q1 = m.q_lo - OVERSHOOT, m.q_hi + OVERSHOOT
        else:
            q0, q1 = self.spec.q_min, self.spec.q_max
        q = q0 + (q1 - q0) * frac
        self.canvas.draw_state(self.spec, q, m, self.shapes, ghosts, path, dims,
                               xlim, ylim, clean)
        return q


class OptWorker(QtCore.QThread):
    """Runs the search off the UI thread so the window stays responsive."""

    progress = QtCore.pyqtSignal(int, float, object, object)
    restarted = QtCore.pyqtSignal(int, object)
    done = QtCore.pyqtSignal(object)

    def __init__(self, spec: LinkageSpec, cfg, restarts: int = 1,
                 time_limit_s: float | None = None,
                 converge_pct: float | None = None):
        super().__init__()
        self.spec, self.cfg, self.restarts = spec, cfg, restarts
        self.time_limit_s, self.converge_pct = time_limit_s, converge_pct
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from optimize import optimize_multi
        mr = optimize_multi(
            self.spec, self.cfg, restarts=self.restarts,
            callback=lambda n, f, s, m: self.progress.emit(n, f, s, m),
            should_stop=lambda: self._stop,
            time_limit_s=self.time_limit_s, converge_pct=self.converge_pct,
            on_restart=lambda i, r, partial: self.restarted.emit(i, partial))
        self.done.emit(mr)


class OptimizerPanel(QtWidgets.QWidget):
    """Free-variable bounds, objective choice, and run controls."""

    apply_best = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        from optimize import VARS, DEFAULT_BOUNDS, MAX_LINK_M
        self.VARS, self.DEFAULT_BOUNDS = VARS, DEFAULT_BOUNDS
        self.MAX_LINK_M = MAX_LINK_M
        self.worker = None
        self.best_spec = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # ── free variables + bounds ─────────────────────────────────────────
        box = QtWidgets.QGroupBox("Free variables and bounds  [mm]")
        grid = QtWidgets.QGridLayout(box)
        grid.addWidget(QtWidgets.QLabel("<b>free</b>"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("<b>min</b>"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("<b>max</b>"), 0, 2)
        self.var_rows = {}
        for r, k in enumerate(VARS, start=1):
            cb = QtWidgets.QCheckBox(k)
            cb.setChecked(True)
            lo, hi = DEFAULT_BOUNDS[k]
            s_lo = _spin(lo * 1e3, -500, 500, 5.0, 1)
            s_hi = _spin(hi * 1e3, -500, 500, 5.0, 1)
            grid.addWidget(cb, r, 0)
            grid.addWidget(s_lo, r, 1)
            grid.addWidget(s_hi, r, 2)
            self.var_rows[k] = (cb, s_lo, s_hi)

        r = len(VARS) + 1
        grid.addWidget(QtWidgets.QLabel("max link span"), r, 0)
        self.max_link = _spin(MAX_LINK_M * 1e3, 20, 1000, 5.0, 1)
        self.max_link.setToolTip(
            "Hard cap on any link, radius-centre to radius-centre.\n"
            "For the tibia this is the full E→W span (L_stub + L_tibia),\n"
            "which the per-variable bounds cannot express on their own.")
        grid.addWidget(self.max_link, r, 1, 1, 2)
        lay.addWidget(box)

        # ── torque constraint ───────────────────────────────────────────────
        tb = QtWidgets.QGroupBox("Hip torque constraint")
        tform = QtWidgets.QFormLayout(tb)
        self.tq_override = QtWidgets.QCheckBox("override for this run")
        self.tq_override.setToolTip(
            "Off: the search uses whatever the working config carries, so a\n"
            "preset's own load and limit are never silently replaced.\n"
            "On: these values are stamped onto every candidate instead.")
        self.tq_enforce = QtWidgets.QCheckBox("enforce")
        self.tq_enforce.setChecked(True)
        self.tq_enforce.setToolTip(
            "Untick to search blind to torque.  Useful only as a control —\n"
            "a torque-blind winner can need well over twice what the motor\n"
            "can hold, and loses most of its travel once trimmed.")
        self.tq_load = _spin(1.0, 0.0, 100.0, 0.1, 2, " kg")
        self.tq_load.setToolTip("Mass carried by ONE leg.")
        self.tq_limit = _spin(2.5, 0.01, 100.0, 0.1, 2, " N·m")
        self.tq_limit.setToolTip("AK45-10 continuous is 2.5 N·m.")
        tform.addRow(self.tq_override)
        tform.addRow("enforce", self.tq_enforce)
        tform.addRow("load / leg", self.tq_load)
        tform.addRow("max torque", self.tq_limit)
        lay.addWidget(tb)

        self.tq_override.toggled.connect(self._sync_torque_enabled)
        self._sync_torque_enabled(False)

        # ── objective ───────────────────────────────────────────────────────
        ob = QtWidgets.QGroupBox("Objective — maximise vertical travel")
        form = QtWidgets.QFormLayout(ob)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["constrained", "weighted"])
        self.mode.setToolTip(
            "constrained: maximise travel, reject |deviation| over tolerance\n"
            "weighted:    maximise w_travel*travel - w_vert*deviation")
        self.tol = _spin(5.0, 0.1, 200, 0.5, 2)
        self.w_travel = _spin(1.0, 0, 100, 0.1, 2, "")
        self.w_vert = _spin(2.0, 0, 100, 0.1, 2, "")
        form.addRow("mode", self.mode)
        form.addRow("deviation tol", self.tol)
        form.addRow("w_travel", self.w_travel)
        form.addRow("w_vert", self.w_vert)
        lay.addWidget(ob)

        # ── run ─────────────────────────────────────────────────────────────
        rb = QtWidgets.QGroupBox("Run")
        rform = QtWidgets.QFormLayout(rb)
        # The item TEXT is prose; the item DATA is the value optimize.py wants.
        self.algo = QtWidgets.QComboBox()
        self.algo.addItem("de — differential evolution (global search)", "de")
        self.algo.addItem("es — (1+λ) evolution strategy (refines current)", "es")
        self.algo.setToolTip(
            "de — differential evolution (scipy, best1bin, Sobol init).\n"
            "     A POPULATION searching the whole bounds box.  It ignores your\n"
            "     working geometry, so it can land on a different family of\n"
            "     mechanism entirely.  Needs the larger budget: measured, DE\n"
            "     scatters 24% across restarts at 2000 evals and agrees within\n"
            "     2.7% at 6000.\n"
            "\n"
            "es — (1+λ) evolution strategy, λ=8, 1/5-success step adaptation.\n"
            "     A LOCAL search that starts FROM the working geometry and\n"
            "     mutates it, shrinking the step as it stops winning.  This is\n"
            "     the scheme that produced baseline-1.  Cheaper per unit of\n"
            "     progress, but it only refines the basin it starts in — and\n"
            "     every restart starts from that same parent, so ES restarts\n"
            "     agreeing proves less than DE restarts agreeing.")
        self.budget = QtWidgets.QSpinBox()
        self.budget.setRange(100, 2_000_000)
        self.budget.setSingleStep(1000)
        self.budget.setValue(6000)
        # budget/restarts tooltips are mode-dependent — set in _sync_run_mode.
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 10_000)
        self.seed.setToolTip(
            "Seed of restart 0; restart i uses seed+i.  With 'randomise' on\n"
            "this is overwritten at every start and left showing the seed the\n"
            "last run actually used, so any result can be reproduced by\n"
            "unticking and typing it back.")
        self.run_mode = QtWidgets.QComboBox()
        self.run_mode.addItem("budget × restarts", "evals")
        self.run_mode.addItem("time limit", "time")
        self.run_mode.setToolTip(
            "budget × restarts — run exactly N restarts of N evals each.\n"
            "time limit      — you give minutes; the session keeps restarting\n"
            "                  until the clock runs out, dividing the time by\n"
            "                  'restarts' to size each one.  The eval budget\n"
            "                  stays on as a ceiling, so a run that settles\n"
            "                  early hands its remaining time to the next.")
        self.minutes = _spin(10.0, 0.1, 600.0, 1.0, 1, " min")
        self.minutes.setToolTip(
            "Total wall clock for the whole session, all restarts included.")
        self.auto_stop = QtWidgets.QCheckBox("stop early when restarts agree")
        self.auto_stop.setChecked(True)
        self.auto_stop.setToolTip(
            "End the session as soon as the spread across restarts drops under\n"
            "2% — the same threshold verdict() calls CONVERGED.  Needs at least\n"
            "3 completed restarts before it will trigger, since two runs can\n"
            "agree by luck.  Applies in both modes.")
        self.seed_random = QtWidgets.QCheckBox("randomise each run")
        self.seed_random.setChecked(True)
        self.seed_random.setToolTip(
            "Draw a fresh seed on every Start.  Repeating a run with a fixed\n"
            "seed re-walks the identical search, which hides scatter that is\n"
            "really there — the restart spread only means something when the\n"
            "runs are independent.")
        self.restarts = QtWidgets.QSpinBox()
        self.restarts.setRange(1, 50)
        self.restarts.setValue(5)
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(self.seed, 1)
        srow.addWidget(self.seed_random)
        rform.addRow("algorithm", self.algo)
        rform.addRow("stop after", self.run_mode)
        rform.addRow("time limit", self.minutes)
        rform.addRow("budget (evals)", self.budget)
        self.lbl_restarts = QtWidgets.QLabel("restarts")
        rform.addRow(self.lbl_restarts, self.restarts)
        rform.addRow("", self.auto_stop)
        rform.addRow("seed", srow)
        lay.addWidget(rb)

        self.run_mode.currentIndexChanged.connect(self._sync_run_mode)
        self._sync_run_mode()

        brow = QtWidgets.QHBoxLayout()
        self.b_start = QtWidgets.QPushButton("▶ Start")
        self.b_stop = QtWidgets.QPushButton("■ Stop")
        self.b_stop.setEnabled(False)
        self.b_apply = QtWidgets.QPushButton("Apply best → working")
        self.b_apply.setEnabled(False)
        for b in (self.b_start, self.b_stop, self.b_apply):
            brow.addWidget(b)
        lay.addLayout(brow)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        lay.addWidget(self.bar)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        lay.addWidget(self.log, 1)

        self.b_apply.clicked.connect(
            lambda: self.best_spec and self.apply_best.emit(self.best_spec))

        # Persist settings: restore whatever was last used, then autosave on
        # every change (debounced) and again on exit.
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self.save_settings)
        self.load_settings()
        self._connect_autosave()

    def is_timed(self) -> bool:
        return self.run_mode.currentData() == "time"

    def roll_seed(self):
        """Draw the seed for a run about to start.  Deliberately NOT part of
        config(), which is also called after a run to read back the free-
        variable list — a getter that moved the seed would rewrite history.
        The spinner keeps the value, so the run stays reproducible."""
        if self.seed_random.isChecked():
            self.seed.setValue(random.randint(0, self.seed.maximum()))

    def _sync_run_mode(self):
        """Grey out what the chosen mode does not use, and say what 'restarts'
        means in each — it is a count in one mode and a divisor in the other."""
        timed = self.is_timed()
        self.minutes.setEnabled(timed)
        self.lbl_restarts.setText("restarts ÷ time" if timed else "restarts")
        self.restarts.setToolTip(
            ("The time limit is divided by this to size each run, so it sets\n"
             "how many restarts the session aims for rather than how many it\n"
             "will do — a run that hits its eval ceiling early leaves time for\n"
             "more."
             if timed else
             "Independent runs from different seeds.  Agreement between them is\n"
             "the only convergence evidence a stochastic search can give.\n"
             "Measured: DE at 2000 evals scatters 24% across restarts;\n"
             "at 6000 it agrees within 2.7%."))
        self.budget.setToolTip(
            ("Per-run eval ceiling.  In timed mode the clock usually binds\n"
             "first; this stops one run from spending the whole slice if it\n"
             "somehow evaluates very fast."
             if timed else
             "Evaluations per restart, so total work is budget × restarts.\n"
             "The run reports whether it was still improving when it hit this\n"
             "cutoff — if it says STILL IMPROVING, the number is too low."))

    def _sync_torque_enabled(self, on: bool):
        for w in (self.tq_enforce, self.tq_load, self.tq_limit):
            w.setEnabled(on)

    # ── settings persistence ────────────────────────────────────────────────
    @staticmethod
    def settings_path():
        from pathlib import Path
        return Path(os.path.dirname(os.path.abspath(__file__))) / "opt_settings.json"

    def _widgets(self):
        w = {"max_link": self.max_link, "mode": self.mode, "tol": self.tol,
             "w_travel": self.w_travel, "w_vert": self.w_vert,
             "algo": self.algo, "budget": self.budget,
             "restarts": self.restarts, "seed": self.seed,
             "seed_random": self.seed_random, "run_mode": self.run_mode,
             "minutes": self.minutes, "auto_stop": self.auto_stop,
             "tq_override": self.tq_override, "tq_enforce": self.tq_enforce,
             "tq_load": self.tq_load, "tq_limit": self.tq_limit}
        for k, (cb, lo, hi) in self.var_rows.items():
            w[f"{k}.free"] = cb
            w[f"{k}.min"] = lo
            w[f"{k}.max"] = hi
        return w

    def _connect_autosave(self):
        for widget in self._widgets().values():
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.toggled.connect(self._save_timer.start)
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(self._save_timer.start)
            else:
                widget.valueChanged.connect(self._save_timer.start)

    def save_settings(self):
        data = {}
        for key, widget in self._widgets().items():
            if isinstance(widget, QtWidgets.QCheckBox):
                data[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QComboBox):
                # Store the value, never the label — labels are prose and may
                # be reworded, and the value is what optimize.py reads.
                d = widget.currentData()
                data[key] = widget.currentText() if d is None else d
            else:
                data[key] = widget.value()
        try:
            self.settings_path().write_text(json.dumps(data, indent=2),
                                            encoding="utf-8")
        except OSError:
            pass          # never let a settings write break the session

    def load_settings(self):
        p = self.settings_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for key, widget in self._widgets().items():
            if key not in data:
                continue
            widget.blockSignals(True)
            try:
                if isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(bool(data[key]))
                elif isinstance(widget, QtWidgets.QComboBox):
                    # By value first, by label second: settings written before
                    # the labels became prose still resolve.
                    v = str(data[key])
                    i = widget.findData(v)
                    if i < 0:
                        i = widget.findText(v)
                    if i >= 0:
                        widget.setCurrentIndex(i)
                else:
                    widget.setValue(type(widget.value())(data[key]))
            except (TypeError, ValueError):
                pass
            widget.blockSignals(False)
        self._sync_torque_enabled(self.tq_override.isChecked())
        self._sync_run_mode()

    def config(self):
        from optimize import OptConfig
        free, bounds = [], {}
        for k, (cb, lo, hi) in self.var_rows.items():
            if cb.isChecked():
                free.append(k)
            bounds[k] = (lo.value() / 1000.0, hi.value() / 1000.0)
        ovr = self.tq_override.isChecked()
        return OptConfig(
            free=tuple(free) or self.VARS, bounds=bounds,
            mode=self.mode.currentText(), tol_mm=self.tol.value(),
            w_travel=self.w_travel.value(), w_vert=self.w_vert.value(),
            algo=self.algo.currentData(), budget=self.budget.value(),
            seed=self.seed.value(), max_link_mm=self.max_link.value(),
            enforce_torque_limit=self.tq_enforce.isChecked() if ovr else None,
            leg_load_kg=self.tq_load.value() if ovr else None,
            torque_limit_nm=self.tq_limit.value() if ovr else None,
        )


# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, preset: str | None = None):
        super().__init__()
        self.setWindowTitle("4-bar Linkage Lab")
        self.resize(1500, 950)

        self.spec = LinkageSpec.load(preset) if preset else baseline1()
        self.metrics = None
        self.metrics_free = None
        self.shapes = ShapeSet(self.spec, RES_DRAW)
        self._xlim = self._ylim = None
        self._opt_best_f = -math.inf
        self._opt_last_n = 0
        self._opt_base_evals = 0
        self._opt_restart = 0
        self._opt_timed = False
        self._opt_limit_s = None
        self._opt_t0 = 0.0
        # What the working panel is actually showing, so its combo can be
        # re-asserted after the optimizer has borrowed the panel.
        self._work_label = WORKING_EDITED

        self.params = ParamPanel(self.spec)
        self.matrix = CollisionMatrix(self.spec)
        self.ref_panel = ComparePanel("Reference", selectable=True)
        self.work_panel = ComparePanel("Working / optimizer result", selectable=True)
        self.work_panel.combo.setToolTip(
            "Load a saved favorite straight into the working config.\n"
            "Reverts to '(edited)' as soon as you change a parameter.")
        self.canvas = self.work_panel.canvas

        # ── left dock ────────────────────────────────────────────────────────
        left = QtWidgets.QScrollArea()
        left.setWidget(self.params)
        left.setWidgetResizable(True)
        left.setMinimumWidth(300)
        left.setMaximumWidth(340)

        # ── right dock ───────────────────────────────────────────────────────
        right = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(right)
        rlay.addWidget(self.matrix)

        self.readout = QtWidgets.QPlainTextEdit()
        self.readout.setReadOnly(True)
        self.readout.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px;")
        self.readout.setMinimumHeight(260)
        rlay.addWidget(QtWidgets.QLabel("Results"))
        rlay.addWidget(self.readout)

        self.chk_collide = QtWidgets.QCheckBox("Limit range by collisions")
        self.chk_collide.setChecked(True)
        self.chk_collide.toggled.connect(self.recompute)
        self.chk_ghost = QtWidgets.QCheckBox("Show range-end ghosts")
        self.chk_ghost.setChecked(True)
        self.chk_ghost.toggled.connect(self.redraw)
        self.chk_path = QtWidgets.QCheckBox("Show traced path")
        self.chk_path.setChecked(True)
        self.chk_path.toggled.connect(self.redraw)
        self.chk_dims = QtWidgets.QCheckBox("Show F offset dimensions")
        self.chk_dims.setChecked(True)
        self.chk_dims.toggled.connect(self.redraw)
        self.chk_clean = QtWidgets.QCheckBox("Clean diagram (print-ready)")
        self.chk_clean.setChecked(True)
        self.chk_clean.setToolTip(
            "Drops the legend strip, shrinks the stats to stroke / deviation /\n"
            "torque in the top-left corner, and reclaims the empty band held\n"
            "below the mechanism for the full-size box — the diagram gets the\n"
            "whole panel.  The rest of the numbers are in this dock either way.")
        self.chk_clean.toggled.connect(self.redraw)
        for c in (self.chk_collide, self.chk_ghost, self.chk_path,
                  self.chk_dims, self.chk_clean):
            rlay.addWidget(c)

        # Per-panel metrics, moved off the canvas so no row of text competes
        # with the mechanism for height.
        for name, panel in (("Reference", self.ref_panel),
                            ("Working", self.work_panel)):
            gb = QtWidgets.QGroupBox(name)
            gl = QtWidgets.QVBoxLayout(gb)
            gl.setContentsMargins(6, 4, 6, 4)
            gl.addWidget(panel.info)
            rlay.addWidget(gb)

        key = QtWidgets.QGroupBox("Diagram key")
        klay = QtWidgets.QVBoxLayout(key)
        klay.setContentsMargins(6, 4, 6, 4)
        key_lbl = QtWidgets.QLabel(
            "&nbsp;&nbsp;".join(f'<span style="color:{c}">■</span>&nbsp;{n}'
                                for n, c in LEGEND_ENTRIES))
        key_lbl.setWordWrap(True)
        key_lbl.setStyleSheet("font-size: 10px; color: #c0c0d0;")
        klay.addWidget(key_lbl)
        rlay.addWidget(key)

        rlay.addWidget(QtWidgets.QLabel("Reference vs working"))
        self.delta = QtWidgets.QPlainTextEdit()
        self.delta.setReadOnly(True)
        self.delta.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px;")
        self.delta.setMaximumHeight(120)
        rlay.addWidget(self.delta)

        btns = QtWidgets.QHBoxLayout()
        b_reset = QtWidgets.QPushButton("Reset")
        b_reset.clicked.connect(self.reset_baseline)
        b_load = QtWidgets.QPushButton("Load…")
        b_load.clicked.connect(self.load_preset)
        b_save = QtWidgets.QPushButton("Save…")
        b_save.clicked.connect(self.save_preset)
        for b in (b_reset, b_load, b_save):
            btns.addWidget(b)
        rlay.addLayout(btns)

        btns2 = QtWidgets.QHBoxLayout()
        b_fav = QtWidgets.QPushButton("★ Save as favorite")
        b_fav.setToolTip("Snapshot the working config into the reference dropdown")
        b_fav.clicked.connect(self.save_favorite)
        b_pull = QtWidgets.QPushButton("← Load reference into working")
        b_pull.clicked.connect(self.pull_reference)
        for b in (b_fav, b_pull):
            btns2.addWidget(b)
        rlay.addLayout(btns2)

        b_del = QtWidgets.QPushButton("🗑 Delete selected reference preset")
        b_del.setToolTip("Deletes the preset currently chosen in the LEFT "
                         "(reference) dropdown.  Asks first.")
        b_del.clicked.connect(self.delete_preset)
        rlay.addWidget(b_del)
        rlay.addStretch(1)

        self.opt_panel = OptimizerPanel()
        self.opt_panel.b_start.clicked.connect(self.start_optimization)
        self.opt_panel.b_stop.clicked.connect(self.stop_optimization)
        self.opt_panel.apply_best.connect(self.apply_optimized)

        tabs = QtWidgets.QTabWidget()
        rscroll = QtWidgets.QScrollArea()
        rscroll.setWidget(right)
        rscroll.setWidgetResizable(True)
        tabs.addTab(rscroll, "Setup / Results")
        oscroll = QtWidgets.QScrollArea()
        oscroll.setWidget(self.opt_panel)
        oscroll.setWidgetResizable(True)
        tabs.addTab(oscroll, "Optimizer")
        tabs.setMinimumWidth(390)
        tabs.setMaximumWidth(460)
        right = tabs

        # ── centre: side-by-side comparison ──────────────────────────────────
        centre = QtWidgets.QWidget()
        clay = QtWidgets.QVBoxLayout(centre)

        ab = QtWidgets.QSplitter()
        ab.addWidget(self.ref_panel)
        ab.addWidget(self.work_panel)
        ab.setSizes([700, 700])
        clay.addWidget(ab, 1)

        srow = QtWidgets.QHBoxLayout()
        self.lbl_q = QtWidgets.QLabel("hip")
        self.lbl_q.setMinimumWidth(210)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(self.on_slider)
        srow.addWidget(QtWidgets.QLabel("stroke position"))
        srow.addWidget(self.slider, 1)
        srow.addWidget(self.lbl_q)
        clay.addLayout(srow)

        hint = QtWidgets.QLabel(
            "The slider is % of each config's OWN reachable range, so two "
            "mechanisms with different strokes stay comparable.")
        hint.setStyleSheet("color: #888; font-size: 10px;")
        clay.addWidget(hint)

        self.torque_canvas = TorqueCanvas()
        self.chk_torque = QtWidgets.QCheckBox(
            "Show hip torque across the stroke")
        self.chk_torque.setChecked(True)
        self.chk_torque.setToolTip(
            "Static hold torque with the wheel on the ground:\n"
            "  tau = load * g * |dz_W/dq|\n"
            "The marker is the torque required at the current slider pose.")
        self.chk_torque.toggled.connect(self.on_torque_toggled)
        clay.addWidget(self.chk_torque)
        clay.addWidget(self.torque_canvas)

        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(centre)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        # debounce so dragging a spinbox doesn't recompute on every keystroke
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.recompute)
        self.params.changed.connect(self._timer.start)
        self.params.changed.connect(self.mark_working_edited)
        self.matrix.changed.connect(self._timer.start)
        self.ref_panel.preset_changed.connect(self.on_reference_changed)
        self.work_panel.preset_changed.connect(self.on_working_preset_changed)

        self.refresh_presets()
        self.recompute()

    # -----------------------------------------------------------------------
    def refresh_presets(self, select: str | None = None):
        """Populate both dropdowns from the presets folder."""
        names = sorted(p.stem for p in _presets_dir().glob("*.json"))

        ref = self.ref_panel.combo
        ref.blockSignals(True)
        current = select or ref.currentText()
        ref.clear()
        ref.addItems(names)
        # A rebuild must not silently rename what the panel is showing: if the
        # reference is an optimizer snapshot, keep that entry alive.
        if current in PLACEHOLDERS and self.ref_panel.spec is not None:
            ref.insertItem(0, current)
        if current in names or current in PLACEHOLDERS:
            ref.setCurrentText(current)
        ref.blockSignals(False)

        # The working combo loads a favorite into the editable config, so it
        # carries a placeholder for "not one of these any more".
        work = self.work_panel.combo
        work.blockSignals(True)
        keep = work.currentText()
        work.clear()
        work.addItem(WORKING_EDITED)
        work.addItems(names)
        if keep in PLACEHOLDERS and keep != WORKING_EDITED:
            work.insertItem(1, keep)
        work.setCurrentText(
            keep if (keep in names or keep in PLACEHOLDERS) else WORKING_EDITED)
        work.blockSignals(False)

        if ref.count():
            self.on_reference_changed(ref.currentText())

    def mark_working_edited(self):
        """Parameters diverged from whatever favorite was loaded."""
        self._work_label = WORKING_EDITED
        work = self.work_panel.combo
        if work.currentText() != WORKING_EDITED:
            work.blockSignals(True)
            work.setCurrentText(WORKING_EDITED)
            work.blockSignals(False)

    def on_working_preset_changed(self, name: str):
        if not name or name == WORKING_EDITED:
            return
        path = _presets_dir() / f"{name}.json"
        if not path.exists():
            return
        self.spec = LinkageSpec.load(path)
        self._work_label = name
        self.params.load_from(self.spec)
        self.matrix.load_from(self.spec)
        self.recompute()
        self.statusBar().showMessage(f"loaded '{name}' into working config", 4000)

    def on_reference_changed(self, name: str):
        if not name:
            return
        path = _presets_dir() / f"{name}.json"
        if not path.exists():
            return
        self.ref_panel.set_spec(LinkageSpec.load(path),
                                self.chk_collide.isChecked())
        self.update_delta()
        self.redraw()

    # -----------------------------------------------------------------------
    def sync_spec(self):
        self.params.apply_to(self.spec)
        self.matrix.apply_to(self.spec)
        self.shapes = ShapeSet(self.spec, RES_DRAW)

    def recompute(self):
        self.sync_spec()
        use_coll = self.chk_collide.isChecked()
        fast = ShapeSet(self.spec, RES_FAST)
        self.metrics = evaluate(self.spec, fast, use_coll)
        self.metrics_free = evaluate(self.spec, fast, False)

        # Re-label every time: the optimizer may have left the combo reading
        # "optimizer best" while this panel is about to show the working spec.
        self.work_panel.set_spec(self.spec, use_coll, label=self._work_label)
        if self.ref_panel.spec is not None:
            self.ref_panel.set_spec(self.ref_panel.spec, use_coll)

        self.update_bounds()
        self.update_readout()
        self.update_delta()
        self.on_slider(self.slider.value())

    def update_bounds(self):
        """One fixed viewport shared by both panels — same scale and origin, so
        the two mechanisms are directly comparable and nothing jumps."""
        boxes = []
        for p in (self.ref_panel, self.work_panel):
            if p.spec is not None and p.shapes is not None:
                boxes.append(config_bounds(p.spec, p.metrics, p.shapes))
        if not boxes:
            self._xlim = self._ylim = None
            return
        x0 = min(b[0] for b in boxes)
        x1 = max(b[1] for b in boxes)
        z0 = min(b[2] for b in boxes)
        z1 = max(b[3] for b in boxes)
        pad = 22.0
        top = 60.0 if self.chk_dims.isChecked() else pad
        right = 100.0 if self.chk_dims.isChecked() else pad
        # No stats box in clean mode, so nothing needs the band held below it.
        band = 0.0 if self.chk_clean.isChecked() else STATS_BAND * (z1 - z0)
        self._xlim = (x0 - pad, x1 + right)
        self._ylim = (z0 - band - pad, z1 + top)

    def on_slider(self, v: int):
        frac = v / 1000.0
        g, p, d = (self.chk_ghost.isChecked(), self.chk_path.isChecked(),
                   self.chk_dims.isChecked())
        c = self.chk_clean.isChecked()
        qr = self.ref_panel.draw_at_fraction(frac, g, p, d, self._xlim,
                                             self._ylim, c)
        qw = self.work_panel.draw_at_fraction(frac, g, p, d, self._xlim,
                                              self._ylim, c)
        txt = f"{frac*100:5.1f}%   "
        if qr is not None:
            txt += f"ref {math.degrees(qr):+7.2f}°   "
        if qw is not None:
            txt += f"work {math.degrees(qw):+7.2f}°"
        self.lbl_q.setText(txt)

        if self.chk_torque.isChecked():
            self.torque_canvas.draw_curves(
                [("reference", REF_COLOR, self.ref_panel),
                 ("working", WORK_COLOR, self.work_panel)], frac)

    def on_torque_toggled(self, on: bool):
        self.torque_canvas.setVisible(on)
        if on:
            self.on_slider(self.slider.value())

    def redraw(self):
        self.update_bounds()
        self.on_slider(self.slider.value())

    def update_delta(self):
        r = self.ref_panel.metrics
        w = self.work_panel.metrics
        if r is None or w is None or not r.valid or not w.valid:
            self.delta.setPlainText("(one side invalid — no comparison)")
            return
        rows = [("stroke °", r.stroke_deg, w.stroke_deg, "+"),
                ("travel mm", r.travel_mm, w.travel_mm, "+"),
                ("max dev mm", r.max_dev_mm, w.max_dev_mm, "-"),
                ("peak τ N·m", r.max_torque_nm, w.max_torque_nm, "-")]
        L = [f"{'':11s}{'ref':>9s}{'work':>9s}{'delta':>10s}"]
        for name, a, b, better in rows:
            d = b - a
            mark = "" if abs(d) < 1e-9 else (
                " ✓" if ((d > 0) == (better == "+")) else " ✗")
            L.append(f"{name:11s}{a:9.2f}{b:9.2f}{d:+10.2f}{mark}")
        self.delta.setPlainText("\n".join(L))

    def update_readout(self):
        m, f = self.metrics, self.metrics_free
        L = []
        if m is None or not m.valid:
            L.append(f"INVALID: {m.reason if m else '?'}")
        else:
            L.append("── with collisions ─────────────────────")
            L.append(f"  vertical travel   {m.travel_mm:9.2f} mm")
            L.append(f"  max |deviation|   {m.max_dev_mm:9.2f} mm")
            L.append(f"  rms deviation     {m.rms_dev_mm:9.2f} mm")
            L.append(f"  hip stroke        {m.stroke_deg:9.2f} °")
            L.append(f"  range   [{math.degrees(m.q_lo):+7.2f}, "
                     f"{math.degrees(m.q_hi):+7.2f}] °")
            for tag, stop, qb in (("lo", m.stop_lo, m.q_block_lo),
                                  ("hi", m.stop_hi, m.q_block_hi)):
                extra = (f"  (contact at {math.degrees(qb):+.2f}°)"
                         if qb is not None and stop.startswith("collision") else "")
                L.append(f"  blocked {tag}  {stop}{extra}")
            L.append(f"  best-fit x        {m.mean_x_mm:9.2f} mm")
            L.append(f"  peak hip torque   {m.max_torque_nm:9.2f} N·m"
                     f"  at {math.degrees(m.q_max_torque):+7.2f} °")
            L.append(f"  torque headroom   "
                     f"{self.spec.torque_limit_nm - m.max_torque_nm:9.2f} N·m"
                     f"   (limit {self.spec.torque_limit_nm:.2f} @ "
                     f"{self.spec.leg_load_kg:.2f} kg/leg)")
        if f is not None and f.valid and m is not None and m.valid:
            L.append("")
            L.append("── without collisions ──────────────────")
            L.append(f"  travel {f.travel_mm:8.2f} mm   stroke {f.stroke_deg:7.2f} °")
            L.append("")
            L.append("── cost of real link widths ────────────")
            L.append(f"  stroke lost       {f.stroke_deg - m.stroke_deg:9.2f} °")
            L.append(f"  travel lost       {f.travel_mm - m.travel_mm:9.2f} mm")
        L.append("")
        L.append("── frame conversion ────────────────────")
        L.append(f"  F_Z (body frame)  {self.spec.to_body_frame_fz()*1000:9.2f} mm")
        self.readout.setPlainText("\n".join(L))

    # -----------------------------------------------------------------------
    def reset_baseline(self):
        self.spec = baseline1()
        self._work_label = WORKING_EDITED
        self.params.load_from(self.spec)
        self.matrix.load_from(self.spec)
        self.recompute()

    def load_preset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load preset", str(_presets_dir()), "JSON (*.json)")
        if not path:
            return
        self.spec = LinkageSpec.load(path)
        self._work_label = WORKING_EDITED
        self.params.load_from(self.spec)
        self.matrix.load_from(self.spec)
        self.recompute()

    def save_preset(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save preset", str(_presets_dir() / "custom.json"), "JSON (*.json)")
        if not path:
            return
        self.sync_spec()
        self.spec.save(path)
        self.refresh_presets()
        self.statusBar().showMessage(f"saved {path}", 4000)

    def save_favorite(self):
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save as favorite", "Name:", text="favorite")
        if not ok or not name.strip():
            return
        self.sync_spec()
        safe = "".join(c for c in name.strip() if c.isalnum() or c in "-_ ").strip()
        self.spec.save(_presets_dir() / f"{safe}.json")
        self.refresh_presets(select=safe)
        self.statusBar().showMessage(f"favorite '{safe}' saved", 4000)

    # ── optimization ────────────────────────────────────────────────────────
    def start_optimization(self):
        if self.opt_panel.worker is not None:
            return
        self.sync_spec()
        p = self.opt_panel
        p.roll_seed()
        cfg = p.config()
        timed = p.is_timed()
        limit_s = p.minutes.value() * 60.0 if timed else None
        converge = 2.0 if p.auto_stop.isChecked() else None
        self._opt_timed = timed
        self._opt_limit_s = limit_s
        self._opt_t0 = time.perf_counter()
        self.opt_panel.log.clear()
        self.opt_panel.log.appendPlainText(
            f"{cfg.algo.upper()}  budget {cfg.budget}  mode {cfg.mode}"
            + (f"  tol {cfg.tol_mm} mm" if cfg.mode == "constrained" else
               f"  w={cfg.w_travel}/{cfg.w_vert}")
            + (f"\ntime limit {p.minutes.value():.1f} min, sliced "
               f"{limit_s / max(1, p.restarts.value()):.0f} s per restart"
               if timed else "")
            + ("\nauto-stop when restarts agree within 2%"
               if converge is not None else "")
            # Logged every run: with a random seed this line is the only record
            # of what to type back to reproduce the result.
            + f"\nseed {cfg.seed}"
            + ("  (random)" if self.opt_panel.seed_random.isChecked() else "")
            + f", restarts use {cfg.seed}..{cfg.seed + self.opt_panel.restarts.value() - 1}"
            + f"\n{cfg.torque_note()}"
            + f"\nmax link span: {cfg.max_link_mm:.0f} mm"
            + f"\nfree: {', '.join(cfg.free)}\n")
        self.opt_panel.bar.setValue(0)
        self.opt_panel.b_start.setEnabled(False)
        self.opt_panel.b_stop.setEnabled(True)
        self._opt_best_f = -math.inf
        self._opt_last_n = 0
        self._opt_base_evals = 0
        self._opt_restart = 0

        # Snapshot the reference side so you can watch the search pull away
        # from where you started.
        self.ref_panel.set_spec(self.spec.copy(), self.chk_collide.isChecked(),
                                label=OPT_SNAPSHOT)
        self.update_delta()

        w = OptWorker(self.spec.copy(), cfg, self.opt_panel.restarts.value(),
                      time_limit_s=limit_s, converge_pct=converge)
        w.progress.connect(self.on_opt_progress)
        w.restarted.connect(self.on_opt_restarted)
        w.done.connect(self.on_opt_done)
        self.opt_panel.worker = w
        w.start()

    def stop_optimization(self):
        if self.opt_panel.worker:
            self.opt_panel.worker.stop()
            self.opt_panel.log.appendPlainText("stopping…")

    def on_opt_progress(self, n, f, spec, m):
        p = self.opt_panel

        # Each restart's callback reports improvements relative to ITS OWN
        # best, which starts from nothing — so track a global best here and
        # never let the display step backwards when a new restart begins.
        if n < self._opt_last_n:                     # eval counter reset => new restart
            self._opt_base_evals += self._opt_last_n
            self._opt_restart += 1
            # In timed mode the restart count is not known ahead of time — the
            # session runs as many as fit — so do not pretend to a total.
            of = "" if self._opt_timed else f"/{p.restarts.value()}"
            p.log.appendPlainText(
                f"--- restart {self._opt_restart + 1}{of} "
                f"(holding best {self._opt_best_f:.2f}) ---")
        self._opt_last_n = n

        total = self._opt_base_evals + n
        if self._opt_timed and self._opt_limit_s:
            frac = (time.perf_counter() - self._opt_t0) / self._opt_limit_s
        else:
            frac = total / max(1, p.budget.value() * p.restarts.value())
        p.bar.setValue(min(100, int(100 * frac)))

        if m is None or not m.valid or f <= self._opt_best_f:
            return                                   # not a new winner: leave the view alone

        self._opt_best_f = f
        p.log.appendPlainText(
            f"[{total:7d}] NEW BEST {f:9.2f}   travel {m.travel_mm:7.2f}  "
            f"dev {m.max_dev_mm:6.2f}  stroke {m.stroke_deg:6.2f}")
        p.best_spec = spec
        p.b_apply.setEnabled(True)
        self.work_panel.set_spec(spec, self.chk_collide.isChecked(),
                                 label=OPT_BEST)
        self.update_bounds()
        self.update_delta()
        self.on_slider(self.slider.value())

    def on_opt_restarted(self, i, partial):
        """One restart finished — report the spread so far.  This is the number
        the auto-stop is watching, so it should be visible while it decides."""
        p = self.opt_panel
        s = partial.spread_pct()
        left = ""
        if self._opt_timed and self._opt_limit_s:
            left = f"   {max(0.0, self._opt_limit_s - (time.perf_counter() - self._opt_t0)):.0f}s left"
        p.log.appendPlainText(
            f"    restart {i} done   best {partial.best.best_fitness:.2f}"
            + (f"   spread {s:.1f}%" if not math.isnan(s) else "   spread n/a")
            + left)

    def on_opt_done(self, mr):
        p = self.opt_panel
        p.worker = None
        p.b_start.setEnabled(True)
        p.b_stop.setEnabled(False)
        p.bar.setValue(100)
        res = mr.best
        m = res.best_metrics

        if len(mr.runs) > 1:
            p.log.appendPlainText(
                "\n── convergence across restarts ──\n"
                + "  " + "  ".join(f"{f:.1f}" for f in mr.fitnesses())
                + f"\n  spread {mr.spread_pct():.1f}%\n  {mr.verdict()}"
                + f"\n  {mr.session_note()}")
        else:
            p.log.appendPlainText(f"\n  {res.convergence_note()}"
                                  f"\n  {mr.session_note()}")

        p.log.appendPlainText(
            f"\n{'stopped' if res.stopped_early else 'finished'}: "
            f"{res.n_evals} evals in {res.elapsed_s:.1f}s\n"
            f"best fitness {res.best_fitness:.2f}\n"
            + (f"travel {m.travel_mm:.2f} mm   dev {m.max_dev_mm:.2f} mm   "
               f"stroke {m.stroke_deg:.2f}°\n" if m and m.valid else "")
            + "\n".join(f"  {k:9s} {getattr(res.best_spec, k)*1000:+8.2f} mm"
                        for k in p.config().free)
            + "\n\nPress 'Apply best → working' to load it.")
        p.best_spec = res.best_spec
        p.b_apply.setEnabled(True)
        # Make certain the view ends on the overall winner, whichever restart
        # produced it.
        if res.best_metrics is not None and res.best_metrics.valid:
            self.work_panel.set_spec(res.best_spec, self.chk_collide.isChecked(),
                                     label=OPT_BEST)
            self.update_bounds()
            self.update_delta()
            self.on_slider(self.slider.value())

    def apply_optimized(self, spec: LinkageSpec):
        self.spec = spec.copy()
        self._work_label = WORKING_EDITED
        self.params.load_from(self.spec)
        self.matrix.load_from(self.spec)
        self.recompute()
        self.statusBar().showMessage("optimizer result applied to working config", 5000)

    def closeEvent(self, ev):
        if self.opt_panel.worker is not None:
            self.opt_panel.worker.stop()
            self.opt_panel.worker.wait(3000)
        self.opt_panel.save_settings()
        super().closeEvent(ev)

    def delete_preset(self):
        name = self.ref_panel.combo.currentText()
        if not name:
            return
        path = _presets_dir() / f"{name}.json"
        if not path.exists():
            self.statusBar().showMessage(f"'{name}' no longer on disk", 4000)
            self.refresh_presets()
            return

        # Deleting is not undoable, so show what is about to be lost.
        m = self.ref_panel.metrics
        detail = ""
        if m is not None and m.valid:
            detail = (f"\n\ntravel {m.travel_mm:.2f} mm"
                      f"\ndeviation {m.max_dev_mm:.2f} mm"
                      f"\nstroke {m.stroke_deg:.2f}°")
        if QtWidgets.QMessageBox.question(
                self, "Delete preset",
                f"Permanently delete '{name}'?{detail}",
                QtWidgets.QMessageBox.StandardButton.Yes |
                QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
        ) != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            path.unlink()
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Delete failed", str(e))
            return
        self.refresh_presets()
        self.statusBar().showMessage(f"deleted '{name}'", 4000)

    def pull_reference(self):
        if self.ref_panel.spec is None:
            return
        self.spec = self.ref_panel.spec.copy()
        self.params.load_from(self.spec)
        self.matrix.load_from(self.spec)
        self.recompute()


def _presets_dir():
    from pathlib import Path
    d = Path(os.path.dirname(os.path.abspath(__file__))) / "presets"
    d.mkdir(exist_ok=True)
    return d


def main(preset: str | None = None) -> int:
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    win = MainWindow(preset)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
