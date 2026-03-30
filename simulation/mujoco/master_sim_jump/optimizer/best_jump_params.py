"""best_jump_params.py — best geometry found by optimize_geometry.py

Generated: 2026-03-29 23:29:26
Fitness:   -1.567000  (lower is better; negative = peak_z > 300 mm)
Peak body Z: 770.1 mm  (approx, assumes fell=False)
"""

# Copy these values into params.py when satisfied.
# RobotGeometry fields:
L_femur    = 0.247378   # [m]
L_tibia    = 0.220000   # [m]
Lc         = 0.221670   # [m]
L_stub     = 0.025000   # [m]
Q_RET      = -0.565531   # [rad]  auto-computed by auto_stroke_angles()
Q_EXT      = -1.282365   # [rad]  auto-computed by auto_stroke_angles()

# JumpGains fields:
crouch_time = 0.124588   # [s]
max_torque  = 7.0              # [N·m]  fixed at motor limit
