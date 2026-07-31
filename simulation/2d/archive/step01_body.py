"""
Step 01 — Static robot body (10 cm × 10 cm box).

Run:
    python simulation/2d/step01_body.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from sim2d import World

# --- world setup -----------------------------------------------------------
world = World(gravity=(0, -9.81))

# Robot body: 10 cm × 10 cm, centred 20 cm above ground, fixed (static)
body = world.add_box(
    pos=(0.0, 0.20),          # centre of box, metres
    size=(0.10, 0.10),        # 10 cm × 10 cm
    mass=None,                # None = static / fixed in space
    color="steelblue",
    label="body",
)
body.set_color("steelblue")

# --- run -------------------------------------------------------------------
world.run(
    title="Step 01 — Robot body (static)",
    xlim=(-0.30, 0.30),
    ylim=(-0.05, 0.45),
)
