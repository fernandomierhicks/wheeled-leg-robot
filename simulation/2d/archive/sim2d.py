"""
sim2d.py — 2D rigid-body simulation framework
Uses pymunk for physics, matplotlib for rendering.

Units: metres, kg, seconds, radians (converted to cm for display).

Usage pattern:
    world = World(gravity=(0, -9.81))
    box   = world.add_box(pos=(0, 0), size=(0.1, 0.1), mass=None)  # None = static
    world.run()   # opens interactive matplotlib window
"""

import numpy as np
import pymunk
import pymunk.matplotlib_util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

class World:
    """Top-level simulation container."""

    def __init__(self, gravity: Tuple[float, float] = (0, -9.81), dt: float = 1/500):
        self.space = pymunk.Space()
        self.space.gravity = gravity
        self.dt = dt
        self.time = 0.0
        self._bodies: List["Body2D"] = []
        self._joints: List = []

    # -----------------------------------------------------------------------
    # Factory helpers
    # -----------------------------------------------------------------------

    def add_box(
        self,
        pos: Tuple[float, float],
        size: Tuple[float, float],
        mass: Optional[float] = None,
        angle: float = 0.0,
        color: str = "steelblue",
        label: str = "",
        fixed: bool = False,
    ) -> "Body2D":
        """
        Add a rectangular rigid body.
        mass=None or fixed=True → static (infinite mass, doesn't move).
        pos is the centre of the box, in metres.
        size is (width, height) in metres.
        """
        w, h = size
        if mass is None or fixed:
            pymunk_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        else:
            moment = pymunk.moment_for_box(mass, size)
            pymunk_body = pymunk.Body(mass, moment)

        pymunk_body.position = pos
        pymunk_body.angle = angle

        shape = pymunk.Poly.create_box(pymunk_body, size)
        shape.friction = 0.8
        shape.elasticity = 0.2

        self.space.add(pymunk_body, shape)

        body2d = Body2D(pymunk_body=pymunk_body, shape=shape,
                        size=size, color=color, label=label)
        self._bodies.append(body2d)
        return body2d

    def add_circle(
        self,
        pos: Tuple[float, float],
        radius: float,
        mass: Optional[float] = None,
        color: str = "tomato",
        label: str = "",
        fixed: bool = False,
    ) -> "Body2D":
        """
        Add a circular rigid body.
        mass=None or fixed=True → static.
        pos is the centre, in metres.
        """
        if mass is None or fixed:
            pymunk_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        else:
            moment = pymunk.moment_for_circle(mass, 0, radius)
            pymunk_body = pymunk.Body(mass, moment)

        pymunk_body.position = pos

        shape = pymunk.Circle(pymunk_body, radius)
        shape.friction = 0.8
        shape.elasticity = 0.2

        self.space.add(pymunk_body, shape)

        body2d = Body2D(pymunk_body=pymunk_body, shape=shape,
                        size=(radius*2, radius*2), color=color, label=label)
        body2d.set_color(color)
        self._bodies.append(body2d)
        return body2d

    def add_segment(
        self,
        body: "Body2D",
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float = 0.005,
        color: str = "dimgray",
    ) -> pymunk.Segment:
        """Add a line segment shape to an existing body (for links/rods)."""
        seg = pymunk.Segment(body.pymunk_body, a, b, radius)
        seg.friction = 0.8
        self.space.add(seg)
        return seg

    def add_pivot_joint(
        self,
        body_a: "Body2D",
        body_b: "Body2D",
        anchor_world: Tuple[float, float],
    ) -> pymunk.PivotJoint:
        """Pin body_a to body_b at a world-space point."""
        joint = pymunk.PivotJoint(
            body_a.pymunk_body, body_b.pymunk_body, anchor_world
        )
        self.space.add(joint)
        self._joints.append(joint)
        return joint

    def add_pin_to_world(
        self,
        body: "Body2D",
        anchor_world: Tuple[float, float],
    ) -> pymunk.PivotJoint:
        """Pin a body to the static world at a world-space anchor."""
        joint = pymunk.PivotJoint(
            self.space.static_body, body.pymunk_body, anchor_world
        )
        self.space.add(joint)
        self._joints.append(joint)
        return joint

    # -----------------------------------------------------------------------
    # Simulation step
    # -----------------------------------------------------------------------

    def step(self, steps: int = 1):
        for _ in range(steps):
            self.space.step(self.dt)
            self.time += self.dt

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def run(
        self,
        title: str = "2D Robot Sim",
        xlim: Tuple[float, float] = (-0.4, 0.4),
        ylim: Tuple[float, float] = (-0.1, 0.6),
        fps: int = 60,
        sim_steps_per_frame: int = 8,
        show_grid: bool = True,
    ):
        """Open a matplotlib window and animate the simulation."""
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#1e1e2e")
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Axis labels in cm
        def m_to_cm(x, _): return f"{x*100:.0f}"
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
        ax.set_xlabel("x  [cm]", color="lightgray")
        ax.set_ylabel("y  [cm]", color="lightgray")
        ax.tick_params(colors="lightgray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

        if show_grid:
            ax.grid(True, color="#333", linewidth=0.5, linestyle="--")

        # Ground line
        ax.axhline(0, color="#888", linewidth=1.5, linestyle="-")

        title_obj = ax.set_title(title, color="white", fontsize=13)

        # Draw options for pymunk
        draw_options = pymunk.matplotlib_util.DrawOptions(ax)
        draw_options.flags = (
            pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
            | pymunk.SpaceDebugDrawOptions.DRAW_CONSTRAINTS
        )

        # Collect artist handles for body labels
        label_texts = []
        for b in self._bodies:
            if b.label:
                t = ax.text(0, 0, b.label, color="white", fontsize=8,
                            ha="center", va="center", zorder=10)
                label_texts.append((b, t))

        time_text = ax.text(
            0.02, 0.97, "", transform=ax.transAxes,
            color="lightgray", fontsize=9, va="top"
        )

        def _update(frame):
            ax.cla()
            ax.set_facecolor("#1e1e2e")
            ax.set_aspect("equal")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(m_to_cm))
            ax.set_xlabel("x  [cm]", color="lightgray")
            ax.set_ylabel("y  [cm]", color="lightgray")
            ax.tick_params(colors="lightgray")
            for spine in ax.spines.values():
                spine.set_edgecolor("#555")
            if show_grid:
                ax.grid(True, color="#333", linewidth=0.5, linestyle="--")
            ax.axhline(0, color="#888", linewidth=1.5)
            ax.set_title(title, color="white", fontsize=13)

            # Advance physics
            self.step(sim_steps_per_frame)

            # Draw pymunk shapes
            draw_options = pymunk.matplotlib_util.DrawOptions(ax)
            self.space.debug_draw(draw_options)

            # Labels on each body
            for b in self._bodies:
                if b.label:
                    x, y = b.pymunk_body.position
                    ax.text(x, y, b.label, color="white", fontsize=8,
                            ha="center", va="center", zorder=10)

            ax.text(0.02, 0.97, f"t = {self.time:.3f} s",
                    transform=ax.transAxes, color="lightgray",
                    fontsize=9, va="top")

        interval_ms = int(1000 / fps)
        ani = animation.FuncAnimation(
            fig, _update, interval=interval_ms, cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()
        return ani


# ---------------------------------------------------------------------------
# Body2D — thin wrapper that keeps metadata alongside pymunk body
# ---------------------------------------------------------------------------

@dataclass
class Body2D:
    pymunk_body: pymunk.Body
    shape: pymunk.Shape
    size: Tuple[float, float]
    color: str = "steelblue"
    label: str = ""

    @property
    def pos(self) -> Tuple[float, float]:
        return tuple(self.pymunk_body.position)

    @property
    def angle(self) -> float:
        return self.pymunk_body.angle

    @property
    def vel(self) -> Tuple[float, float]:
        return tuple(self.pymunk_body.velocity)

    def set_color(self, color: str):
        """Set shape color (used by pymunk debug draw via color attr)."""
        # pymunk 7.x uses shape.color as RGBA tuple 0-255
        from matplotlib.colors import to_rgba
        r, g, b, a = to_rgba(color)
        self.shape.color = (int(r*255), int(g*255), int(b*255), int(a*255))
        self.color = color
