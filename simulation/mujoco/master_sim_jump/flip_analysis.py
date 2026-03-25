"""flip_analysis.py -- Backflip feasibility via drive-fast-then-jump strategy.

Strategy:
  1. Drive forward at high speed (wheels spinning, body has forward momentum)
  2. Brake hard / lean back -> imparts backward pitch angular momentum
  3. Jump (leg extension) while pitching back -> airborne with rotation
  4. Retract legs mid-air -> reduce MOI -> spin faster (figure-skater effect)

Angular momentum is conserved in flight: L = I * omega = const
  - Extended legs: I_ext, omega_ext
  - Retracted legs: I_ret, omega_ret = omega_ext * I_ext / I_ret
"""
import math
import sys

# ── Robot parameters ─────────────────────────────────────────────────────
# From params.py (SimParams defaults)

# Masses
m_box = 0.477        # [kg] body + electronics
m_motor_hip = 0.260  # [kg] AK45-10 each, x2
m_femur = 0.0192     # [kg] each, x2
m_tibia = 0.0183     # [kg] each, x2
m_coupler = 0.0094   # [kg] each, x2
m_bearing = 0.012    # [kg] each, x2
m_wheel = 0.270      # [kg] wheel assembly each, x2

# Geometry
L_femur = 0.17378    # [m]
L_tibia = 0.12939    # [m]
L_stub = 0.03513     # [m]
wheel_r = 0.075      # [m]
A_Z = -0.0235        # [m] hip motor Z offset
leg_y = 0.1430       # [m] leg Y offset

# Motor specs
wheel_KV = 70.0      # [RPM/V]
wheel_I_max = 50.0   # [A]
wheel_Kt = 9.55 / wheel_KV   # [N.m/A]
wheel_tau_max = wheel_Kt * wheel_I_max  # [N.m] per wheel
V_batt = 24.0        # [V]
omega_noload = wheel_KV * V_batt * 2 * math.pi / 60  # [rad/s]

hip_tau_max = 7.0     # [N.m] AK45-10 peak

# Derived
m_leg = m_femur + m_tibia + m_coupler + m_bearing  # one side
m_total = m_box + 2 * m_motor_hip + 2 * m_leg + 2 * m_wheel

g = 9.81
print("=" * 70)
print("BACKFLIP FEASIBILITY -- Drive + Jump + Retract Strategy")
print("=" * 70)
print(f"Total robot mass:     {m_total:.3f} kg")
print(f"Weight:               {m_total * g:.1f} N")
print()

# ── Phase 1: Driving angular momentum from braking ───────────────────────
# When driving forward at speed v, each wheel spins at omega_w = v / wheel_r
# Total wheel angular momentum: L_wheels = 2 * I_wheel * omega_w
# When we brake hard, that angular momentum transfers to the body as pitch.
#
# Wheel MOI (solid cylinder approx): I = 0.5 * m * r^2
# More accurately for a motor: I ~ 0.4 * m * r^2 (hollow rotor)

I_wheel = 0.5 * m_wheel * wheel_r**2  # per wheel

print("--- Phase 1: Wheel braking angular momentum ---")
for v_drive in [2.0, 4.0, 6.0, 8.0, 10.0]:
    omega_w = v_drive / wheel_r
    L_wheels = 2 * I_wheel * omega_w
    print(f"  v={v_drive:4.1f} m/s  omega_w={omega_w:6.1f} rad/s  "
          f"L_wheels={L_wheels:.4f} N.m.s")

print()

# ── MOI in extended vs retracted pose ────────────────────────────────────
# Extended: legs fully out (Q_EXT), wheels far from body
# Retracted: legs tucked (Q_RET), wheels close to body
#
# Approximate: body rotates about its CoG.
# Extended leg length from hip to wheel: L_femur + L_tibia = 0.303 m
# Retracted: much shorter effective radius.
#
# For extended pose, wheel center is ~L_femur + L_tibia below hip
# For retracted pose, wheel is much closer.
# From Q_EXT = -1.43 rad and Q_RET = -0.787 rad:

Q_EXT = -1.43161
Q_RET = -0.78705

# Approximate wheel distance from body CoG for each pose
# Extended: femur points down-ish, wheel at ~full leg length
d_wheel_ext = L_femur + L_tibia  # ~0.303 m (approximate, ignoring angles)
# Retracted: leg folded up, wheel much closer
# At Q_RET the 4-bar is compact; rough estimate ~60% of extended
d_wheel_ret = 0.10  # ~100 mm from body CoG when fully crouched (conservative)

# Body box MOI (about its own CoG, pitch axis)
# Approximate body as 150mm tall x 80mm deep box
I_box = (1.0/12) * m_box * (0.15**2 + 0.08**2)

# Hip motors (on body, near CoG)
I_hip_motors = 2 * m_motor_hip * A_Z**2  # small, close to CoG

# Legs as point masses at midpoint of their length
d_leg_ext = (L_femur + L_tibia) / 2
d_leg_ret = d_wheel_ret / 2

# EXTENDED MOI
I_legs_ext = 2 * m_leg * d_leg_ext**2
I_wheels_ext = 2 * m_wheel * d_wheel_ext**2
I_extended = I_box + I_hip_motors + I_legs_ext + I_wheels_ext

# RETRACTED MOI
I_legs_ret = 2 * m_leg * d_leg_ret**2
I_wheels_ret = 2 * m_wheel * d_wheel_ret**2
I_retracted = I_box + I_hip_motors + I_legs_ret + I_wheels_ret

ratio = I_extended / I_retracted

print("--- MOI comparison (pitch axis, about CoG) ---")
print(f"  I_box:           {I_box:.5f} kg.m2")
print(f"  I_hip_motors:    {I_hip_motors:.5f} kg.m2")
print(f"  Extended pose:")
print(f"    I_legs:        {I_legs_ext:.5f} kg.m2  (d={d_leg_ext*1000:.0f} mm)")
print(f"    I_wheels:      {I_wheels_ext:.5f} kg.m2  (d={d_wheel_ext*1000:.0f} mm)")
print(f"    I_total:       {I_extended:.5f} kg.m2")
print(f"  Retracted pose:")
print(f"    I_legs:        {I_legs_ret:.5f} kg.m2  (d={d_leg_ret*1000:.0f} mm)")
print(f"    I_wheels:      {I_wheels_ret:.5f} kg.m2  (d={d_wheel_ret*1000:.0f} mm)")
print(f"    I_total:       {I_retracted:.5f} kg.m2")
print(f"  MOI ratio:       {ratio:.2f}x  (retract speeds up spin by {ratio:.2f}x)")
print()

# ── Phase 2: Jump flight time ───────────────────────────────────────────
# From S10 sim: flight_time ~ 0.238 s, peak height ~ 70 mm
# The jump design targets 283 mm -> longer flight
flight_time_current = 0.238
h_current = g * flight_time_current**2 / 8

# Target jump height from project overview: 283 mm
h_target = 0.283
t_flight_target = math.sqrt(8 * h_target / g)

print("--- Phase 2: Jump flight time ---")
print(f"  Current jump:   h={h_current*1000:.0f} mm, t_flight={flight_time_current:.3f} s")
print(f"  Target jump:    h={h_target*1000:.0f} mm, t_flight={t_flight_target:.3f} s")
print()

# ── Phase 3: Full backflip budget ────────────────────────────────────────
# Total angular momentum sources:
#   1. Wheel braking: L_brake = 2 * I_wheel * omega_wheel
#   2. Hip extension torque during ground contact (small pitch-back impulse)
#   3. Body lean-back before jump (initial pitch rate)
#
# In flight, L is conserved. We need theta = 2*pi total rotation.
#
# Strategy: start with omega_0 from braking, then retract to speed up.
# Split flight into two phases:
#   Phase A: extended (brief, during retraction ~0.05s)
#   Phase B: retracted (rest of flight)
#
# Simpler: assume instant retraction at liftoff (upper bound)
# omega_ret = L / I_ret, need omega_ret * t_flight >= 2*pi

theta_target = 2 * math.pi

print("--- Phase 3: Backflip angular budget ---")
print()
print("  Required rotation: 360 deg")
print()

# For each driving speed, compute if flip is achievable
print(f"  {'v_drive':>7s}  {'L_brake':>8s}  {'omega_ext':>10s}  {'omega_ret':>10s}  "
      f"{'theta_cur':>10s}  {'theta_tgt':>10s}  {'result_cur':>10s}  {'result_tgt':>10s}")
print(f"  {'[m/s]':>7s}  {'[N.m.s]':>8s}  {'[rad/s]':>10s}  {'[rad/s]':>10s}  "
      f"{'[deg]':>10s}  {'[deg]':>10s}  {'':>10s}  {'':>10s}")
print("  " + "-" * 90)

for v_drive in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
    omega_w = v_drive / wheel_r
    L_brake = 2 * I_wheel * omega_w

    # This is the angular momentum transferred to body when wheels lock
    # omega_body_ext = L_brake / I_extended (if all L goes to body pitch)
    omega_ext = L_brake / I_extended

    # After retraction (conservation of L about CoG):
    omega_ret = L_brake / I_retracted

    # Rotation during flight (using retracted omega, assume instant retract)
    theta_current = math.degrees(omega_ret * flight_time_current)
    theta_target_jump = math.degrees(omega_ret * t_flight_target)

    res_cur = "FLIP!" if theta_current >= 360 else f"need {360-theta_current:.0f} more"
    res_tgt = "FLIP!" if theta_target_jump >= 360 else f"need {360-theta_target_jump:.0f} more"

    print(f"  {v_drive:7.1f}  {L_brake:8.4f}  {omega_ext:10.2f}  {omega_ret:10.2f}  "
          f"{theta_current:10.1f}  {theta_target_jump:10.1f}  {res_cur:>10s}  {res_tgt:>10s}")

print()

# ── What driving speed do we need? ───────────────────────────────────────
# omega_ret * t_flight = 2*pi
# (L / I_ret) * t_flight = 2*pi
# L = 2*pi * I_ret / t_flight
# 2 * I_wheel * v / (wheel_r * I_ret) * t_flight  -- wait, let me be precise
# L = 2 * I_wheel * (v/wheel_r) = 2*pi * I_ret / t_flight
# v = 2*pi * I_ret * wheel_r / (t_flight * 2 * I_wheel)

for label, t_fl in [("current jump", flight_time_current),
                     ("target 283mm", t_flight_target)]:
    v_needed = theta_target * I_retracted * wheel_r / (t_fl * 2 * I_wheel)
    omega_w_needed = v_needed / wheel_r
    print(f"  Min driving speed for flip ({label}, t={t_fl:.3f}s):")
    print(f"    v = {v_needed:.2f} m/s ({v_needed*3.6:.1f} km/h)")
    print(f"    wheel omega = {omega_w_needed:.1f} rad/s "
          f"({'OK' if omega_w_needed < omega_noload else 'EXCEEDS no-load!'})")
    print()

# ── Additional: hip torque kick ──────────────────────────────────────────
# During the extension phase (~0.14s on ground), the hip torque also
# creates a pitch-back moment. This adds to angular momentum.
# tau_hip * t_extend = additional L
t_extend = 0.14  # from S10 sim
L_hip_kick = 2 * hip_tau_max * t_extend  # both hips, reaction pitches body back
# Actually hip extension pushes wheels DOWN and body UP. The reaction
# torque on the body from hip extension depends on geometry...
# Simplified: some fraction goes to pitch-back angular momentum.
# Conservative: ~30% of hip impulse contributes to body pitch rotation.
L_hip_pitch = 0.3 * L_hip_kick

print(f"--- Bonus: hip extension pitch impulse ---")
print(f"  Extension time: {t_extend:.3f} s")
print(f"  Hip torque impulse: 2 x {hip_tau_max} x {t_extend} = {L_hip_kick:.3f} N.m.s")
print(f"  ~30% to pitch: {L_hip_pitch:.3f} N.m.s")
print(f"  Extra rotation (retracted, current flight): "
      f"{math.degrees(L_hip_pitch / I_retracted * flight_time_current):.0f} deg")
print(f"  Extra rotation (retracted, target flight):  "
      f"{math.degrees(L_hip_pitch / I_retracted * t_flight_target):.0f} deg")
print()

# ── Top speed check ──────────────────────────────────────────────────────
v_max = omega_noload * wheel_r
print(f"--- Limits ---")
print(f"  Wheel no-load speed:  {omega_noload:.1f} rad/s")
print(f"  Max robot speed:      {v_max:.1f} m/s ({v_max*3.6:.1f} km/h)")
print(f"  Wheel torque limit:   {wheel_tau_max:.2f} N.m each")
print(f"  Max ground force:     {2*wheel_tau_max/wheel_r:.1f} N vs weight {m_total*g:.1f} N")
print()

# ── Summary ──────────────────────────────────────────────────────────────
v_flip_current = theta_target * I_retracted * wheel_r / (flight_time_current * 2 * I_wheel)
v_flip_target = theta_target * I_retracted * wheel_r / (t_flight_target * 2 * I_wheel)

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  MOI extended:    {I_extended:.5f} kg.m2")
print(f"  MOI retracted:   {I_retracted:.5f} kg.m2")
print(f"  Speedup from retraction: {ratio:.1f}x")
print()
print(f"  With current jump (70 mm, {flight_time_current:.3f}s flight):")
print(f"    Need to drive at {v_flip_current:.1f} m/s ({v_flip_current*3.6:.0f} km/h) "
      f"{'- POSSIBLE' if v_flip_current < v_max else '- TOO FAST'}")
print()
print(f"  With target jump (283 mm, {t_flight_target:.3f}s flight):")
print(f"    Need to drive at {v_flip_target:.1f} m/s ({v_flip_target*3.6:.0f} km/h) "
      f"{'- POSSIBLE' if v_flip_target < v_max else '- TOO FAST'}")
print()
print(f"  Max theoretical speed: {v_max:.1f} m/s ({v_max*3.6:.0f} km/h)")
print(f"  Hip kick adds ~{math.degrees(L_hip_pitch / I_retracted * t_flight_target):.0f} deg extra")
