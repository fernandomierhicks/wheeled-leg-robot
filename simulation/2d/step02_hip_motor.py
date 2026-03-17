"""
Step 02 — Robot body + AK45-10 hip motor.

Motor (CubeMars AK45-10):
  OD = 53 mm  (confirmed from datasheet, Φ53×43 mm)
  Mounted fixed to the body; output shaft centre = hip pivot A.
  Placed so the motor circle is tangent to the bottom face of the body,
  centred horizontally.

Run:
    python simulation/2d/step02_hip_motor.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from sim2d import World

# --- Parameters (metres) ---------------------------------------------------
BODY_W, BODY_H   = 0.10, 0.10          # body box
BODY_CX, BODY_CY = 0.00, 0.20          # body centre

MOTOR_OD   = 0.053                      # AK45-10 OD = 53 mm
MOTOR_R    = MOTOR_OD / 2               # 26.5 mm

# Motor centre: tangent to bottom of body, centred horizontally
BODY_BOTTOM = BODY_CY - BODY_H / 2     # 0.15 m
MOTOR_CX    = BODY_CX                  # centred
MOTOR_CY    = BODY_BOTTOM + MOTOR_R    # 0.1765 m — inside the box, tangent to bottom face

# --- World -----------------------------------------------------------------
world = World(gravity=(0, -9.81))

# Body box
body = world.add_box(
    pos=(BODY_CX, BODY_CY),
    size=(BODY_W, BODY_H),
    mass=None,
    color="steelblue",
    label="body",
)
body.set_color("steelblue")

# Hip motor circle (fixed to body, static for now)
motor = world.add_circle(
    pos=(MOTOR_CX, MOTOR_CY),
    radius=MOTOR_R,
    mass=None,
    color="tomato",
    label="AK45-10",
)

# --- Run -------------------------------------------------------------------
world.run(
    title="Step 02 — Body + AK45-10 hip motor (Φ53 mm)",
    xlim=(-0.25, 0.25),
    ylim=(-0.05, 0.38),
)
