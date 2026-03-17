"""run_sim.py — Run baseline-1 two-leg simulation and save dense telemetry.

Runs the full 8-second sequence (balance -> crouch -> jump -> land) and logs
every physics step (2000 Hz) to telemetry.npz for offline analysis.

Usage:
    python simulation/mujoco/baseline1_leg_analysis/run_sim.py
Output:
    simulation/mujoco/baseline1_leg_analysis/telemetry.npz

Structural channels logged (all SI units — N, N·m):
  cfrc_{link}_{side}_{tx|ty|tz|fx|fy|fz}
      Raw 6-DOF constraint wrench on the subtree rooted at each body,
      in world frame (torque first, then force).  Logged for:
        fem = femur body   (subtree force at hip pivot A)
        tib = tibia body   (subtree force at knee pivot C, includes E equality)
        cpl = coupler body (net wrench at coupler F pivot + E equality, ≈0 static)
        whl = wheel_asm    (pure wheel-axle bearing force)

  fax_{link}_{side}   — Component of cfrc force along the link axis [N]
  flat_{link}_{side}  — Component of cfrc force perpendicular to link axis [N]
                        (drives mid-span bending; combined with fax → interaction)

  fbear_{joint}_{side} — Radial bearing load at each 608 pin [N]
        A = hip pivot      (cfrc_int[femur], accurate — no equality on femur)
        C = knee pivot     (cfrc_int[tibia] minus E force = F_C; see note below)
        E = 4-bar closure  (equality efc_force magnitude, accurate)
        F = coupler pivot  (≈ fbear_E for two-force member; cfrc_int[coupler] → 0)
        W = wheel axle     (cfrc_int[wheel_asm], accurate — no equality on wheel)

  grf_{side}           — Ground-reaction normal force at wheel contact [N]

Note on C and F bearings:
  cfrc_int[tibia]  = F_C + F_E_on_tibia   (C force is NOT cfrc_int[tibia] alone)
  cfrc_int[coupler]= F_F + F_E_on_coupler ≈ 0 in quasi-static (two-force member)
  fbear_E gives the equality force; fbear_C = |cfrc_int[tibia][3:6] - F_E_vec|,
  fbear_F ≈ fbear_E (equal-and-opposite for a massless two-force coupler).
"""
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_config import *
from physics import solve_ik, get_equilibrium_pitch, build_xml, build_assets

_DIR           = os.path.dirname(os.path.abspath(__file__))
TELEMETRY_PATH = os.path.join(_DIR, "telemetry.npz")

STATE_NEUTRAL = 0
STATE_CROUCH  = 1
STATE_JUMP    = 2


def run():
    p   = ROBOT
    xml = build_xml(p)
    model = mujoco.MjModel.from_xml_string(xml, assets=build_assets())
    data  = mujoco.MjData(model)

    # Fix 4-bar equality constraint anchors for both legs
    for eq_name in ["4bar_close_L", "4bar_close_R"]:
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
        model.eq_data[eq_id, 0:3] = [-p['Lc'], 0.0, 0.0]
        model.eq_data[eq_id, 3:6] = [0.0, 0.0, p['L_stub']]

    def jqp(n): return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def jdof(n): return model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def bid(n):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    def gid(n):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    def eid(n):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, n)

    s_root   = jqp("root_free")
    s_hip_L  = jqp("hip_L");         s_hip_R  = jqp("hip_R")
    s_hF_L   = jqp("hinge_F_L");     s_hF_R   = jqp("hinge_F_R")
    s_knee_L = jqp("knee_joint_L");  s_knee_R = jqp("knee_joint_R")
    d_root   = jdof("root_free")
    d_pitch  = d_root + 4
    d_hip_L  = jdof("hip_L");        d_hip_R  = jdof("hip_R")
    d_whl_L  = jdof("wheel_spin_L"); d_whl_R  = jdof("wheel_spin_R")

    # ── Body IDs ──────────────────────────────────────────────────────────────
    box_bid       = bid("box")
    femur_bid_L   = bid("femur_L");    femur_bid_R   = bid("femur_R")
    tibia_bid_L   = bid("tibia_L");    tibia_bid_R   = bid("tibia_R")
    coupler_bid_L = bid("coupler_L");  coupler_bid_R = bid("coupler_R")
    wheel_bid_L   = bid("wheel_asm_L"); wheel_bid_R  = bid("wheel_asm_R")

    # ── Geom IDs (for contact filtering) ─────────────────────────────────────
    tire_gid_L = gid("wheel_tire_geom_L")
    tire_gid_R = gid("wheel_tire_geom_R")

    # ── Equality constraint IDs ───────────────────────────────────────────────
    eq_id_L = eid("4bar_close_L")
    eq_id_R = eid("4bar_close_R")

    # ── Structural logging helpers ────────────────────────────────────────────
    Lc  = p['Lc']
    _EQ_TYPE = mujoco.mjtConstraint.mjCNSTR_EQUALITY

    def _cfrc(b_id):
        """6D constraint wrench [Tx,Ty,Tz,Fx,Fy,Fz] for subtree in world frame."""
        return data.cfrc_int[b_id].copy()

    def _link_axis_fem(side_L):
        """Unit vector femur A→C in world frame."""
        prox = femur_bid_L if side_L else femur_bid_R
        dist = tibia_bid_L  if side_L else tibia_bid_R
        dv = data.xpos[dist] - data.xpos[prox]
        n = np.linalg.norm(dv)
        return dv / n if n > 1e-6 else np.array([1., 0., 0.])

    def _link_axis_tib(side_L):
        """Unit vector tibia C→W in world frame."""
        prox = tibia_bid_L   if side_L else tibia_bid_R
        dist = wheel_bid_L   if side_L else wheel_bid_R
        dv = data.xpos[dist] - data.xpos[prox]
        n = np.linalg.norm(dv)
        return dv / n if n > 1e-6 else np.array([0., 0., -1.])

    def _link_axis_cpl(side_L):
        """Unit vector coupler F→E in world frame.
        E is at local [-Lc, 0, 0] → world direction = R @ [-1, 0, 0]."""
        c_bid = coupler_bid_L if side_L else coupler_bid_R
        R = data.xmat[c_bid].reshape(3, 3)
        return R @ np.array([-1., 0., 0.])  # unit length (R is rotation matrix)

    def _axial_lateral(F3, axis):
        """Decompose force vector F3 into axial and lateral components."""
        F_ax  = float(np.dot(F3, axis))
        F_lat = float(np.linalg.norm(F3 - F_ax * axis))
        return F_ax, F_lat

    def _bearing_radial(cfrc6):
        """Radial load on 608 bearing (rotation axis ≈ Y): sqrt(Fx²+Fz²) [N]."""
        F = cfrc6[3:6]
        return float(math.sqrt(F[0]**2 + F[2]**2))

    def _grf(tire_g_id):
        """Sum of normal contact forces on one wheel tyre geom [N]."""
        fn = 0.0
        f6 = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 == tire_g_id or c.geom2 == tire_g_id:
                mujoco.mj_contactForce(model, data, i, f6)
                fn += f6[0]
        return float(fn)

    def _eq_force_mag(eq_id_val):
        """Scalar magnitude of equality constraint force at E [N].
        Iterates efc rows for this equality ID and returns ||lambda||.
        For a connect constraint (3 translational DOFs), the 3 scalar
        efc_force values are the world-frame XYZ force components."""
        fvec = []
        for i in range(data.nefc):
            if data.efc_type[i] == _EQ_TYPE and data.efc_id[i] == eq_id_val:
                fvec.append(data.efc_force[i])
        return float(np.linalg.norm(fvec)) if fvec else 0.0

    def _eq_force_vec(eq_id_val):
        """3D force vector from equality constraint [N, world frame approx].
        Used to subtract E contribution from cfrc_int[tibia] to isolate F_C."""
        fvec = np.zeros(3)
        k = 0
        for i in range(data.nefc):
            if data.efc_type[i] == _EQ_TYPE and data.efc_id[i] == eq_id_val:
                if k < 3:
                    fvec[k] = data.efc_force[i]
                k += 1
        return fvec

    # ── Init pose ─────────────────────────────────────────────────────────────
    ik0 = solve_ik(Q_NEUTRAL, p)
    if not ik0: raise RuntimeError("IK failed at Q_NEUTRAL")
    for s_hF, s_hip, s_knee in [
        (s_hF_L, s_hip_L, s_knee_L),
        (s_hF_R, s_hip_R, s_knee_R),
    ]:
        data.qpos[s_hF]  = ik0['q_coupler_F']
        data.qpos[s_hip] = ik0['q_hip']
        data.qpos[s_knee]= ik0['q_knee']
    mujoco.mj_forward(model, data)
    wz = (data.xpos[wheel_bid_L][2] + data.xpos[wheel_bid_R][2]) / 2.0
    data.qpos[s_root + 2] += WHEEL_R - wz
    theta = get_equilibrium_pitch(p, Q_NEUTRAL)
    data.qpos[s_root + 3] = math.cos(theta / 2)
    data.qpos[s_root + 4] = 0.0
    data.qpos[s_root + 5] = math.sin(theta / 2)
    data.qpos[s_root + 6] = 0.0
    mujoco.mj_forward(model, data)

    # ── Controller state ───────────────────────────────────────────────────────
    pitch_integral     = 0.0
    odo_x              = 0.0
    grounded           = True
    leg_state          = STATE_NEUTRAL
    current_hip_target = Q_NEUTRAL
    jump_triggered     = False
    jump_start_t       = 0.0
    crouch_start_t     = 0.0
    was_airborne       = False
    land_t             = -999.0
    max_height_m       = 0.0
    liftoff_t          = None
    flight_duration    = None

    # ── Telemetry channels ─────────────────────────────────────────────────────
    # Kinematic / control channels (existing)
    base_channels = [
        't', 'pitch', 'pitch_rate', 'hip_q', 'hip_omega',
        'wheel_vel', 'wheel_z', 'box_x', 'box_z',
        'u_hip', 'u_wheel', 'pitch_ff',
        'leg_state', 'grounded',
    ]

    # Structural channels — generated systematically
    struct_channels = []
    for side in ('L', 'R'):
        # Raw 6-DOF constraint wrench for each link body
        for link in ('fem', 'tib', 'cpl', 'whl'):
            for comp in ('tx', 'ty', 'tz', 'fx', 'fy', 'fz'):
                struct_channels.append(f'cfrc_{link}_{side}_{comp}')
        # Axial / lateral force decomposition (links only, not wheel body)
        for link in ('fem', 'tib', 'cpl'):
            struct_channels.append(f'fax_{link}_{side}')
            struct_channels.append(f'flat_{link}_{side}')
        # Radial bearing loads at each 608 pin
        for joint in ('A', 'C', 'E', 'F', 'W'):
            struct_channels.append(f'fbear_{joint}_{side}')
        # Ground-reaction normal force at wheel contact
        struct_channels.append(f'grf_{side}')

    channels = base_channels + struct_channels

    N = int(SIM_DURATION_S / model.opt.timestep) + 200
    rec = {k: np.zeros(N) for k in channels}
    idx = 0

    print(f"Running simulation  ({SIM_DURATION_S:.0f} s, dt={model.opt.timestep*1000:.1f} ms) ...")

    # ── Physics loop ──────────────────────────────────────────────────────────
    while data.time < SIM_DURATION_S:
        sim_t = data.time

        q = data.xquat[box_bid]
        pitch_true      = math.asin(max(-1.0, min(1.0, 2*(q[0]*q[2] - q[3]*q[1]))))
        pitch_rate_true = data.qvel[d_pitch]
        pitch      = pitch_true      + np.random.normal(0, PITCH_NOISE_STD_RAD)
        pitch_rate = pitch_rate_true + np.random.normal(0, PITCH_RATE_NOISE_STD_RAD_S)

        hip_q     = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0
        hip_omega = (data.qvel[d_hip_L]  + data.qvel[d_hip_R])  / 2.0
        wheel_vel = (data.qvel[d_whl_L]  + data.qvel[d_whl_R])  / 2.0

        accel_noisy = data.sensor("accel").data + np.random.normal(0, ACCEL_NOISE_STD, 3)
        accel_mag   = np.linalg.norm(accel_noisy)
        if   accel_mag < 3.0: grounded = False
        elif accel_mag > 7.0: grounded = True

        wheel_z_now  = data.xpos[wheel_bid_L][2]
        airborne_now = wheel_z_now > WHEEL_R + 0.003
        if jump_triggered and airborne_now and liftoff_t is None:
            liftoff_t = sim_t
        if jump_triggered and liftoff_t is not None and not airborne_now and land_t < jump_start_t:
            land_t = sim_t
            flight_duration = sim_t - liftoff_t

        pitch_ff = get_equilibrium_pitch(p, hip_q)
        if grounded:
            vel_est = (wheel_vel + pitch_rate) * WHEEL_R
            odo_x  += vel_est * model.opt.timestep
            pitch_fb = np.clip(
                -(POSITION_KP * odo_x + VELOCITY_KP * vel_est),
                -MAX_PITCH_CMD, MAX_PITCH_CMD)
            target_pitch = pitch_ff + pitch_fb
        else:
            target_pitch   = 0.0
            pitch_integral = 0.0

        pitch_error    = pitch - target_pitch
        pitch_integral = np.clip(
            pitch_integral + pitch_error * model.opt.timestep, -1.0, 1.0)
        u_bal = (PITCH_KP * pitch_error
                 + PITCH_KI * pitch_integral
                 + PITCH_KD * pitch_rate)
        u_bal_clipped = np.clip(u_bal, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
        data.ctrl[2] = u_bal_clipped
        data.ctrl[3] = u_bal_clipped

        if sim_t > CROUCH_START_S and leg_state == STATE_NEUTRAL and not jump_triggered:
            leg_state      = STATE_CROUCH
            crouch_start_t = sim_t
        if sim_t > JUMP_TRIGGER_S and leg_state == STATE_CROUCH and not jump_triggered:
            leg_state      = STATE_JUMP
            jump_start_t   = sim_t
            jump_triggered = True

        if leg_state == STATE_JUMP:
            ramp_in     = min(1.0, (sim_t - jump_start_t) / JUMP_RAMP_S)
            ramp_out    = min(1.0, max(0.0, (hip_q - Q_EXT) / JUMP_RAMPDOWN))
            speed_scale = max(0.0, 1.0 - abs(hip_omega) / OMEGA_MAX)
            u_hip = -HIP_TORQUE_LIMIT * ramp_in * ramp_out * speed_scale
            if (hip_q <= Q_EXT + 0.05) or (
                    not grounded and (sim_t - jump_start_t) > 0.05):
                leg_state          = STATE_NEUTRAL
                current_hip_target = Q_NEUTRAL
        else:
            if leg_state == STATE_CROUCH:
                frac = min(1.0, (sim_t - crouch_start_t) / CROUCH_DURATION_S)
                current_hip_target = Q_NEUTRAL + frac * (Q_RET - Q_NEUTRAL)
            u_hip = HIP_KP_SUSP * (current_hip_target - hip_q) - HIP_KD_SUSP * hip_omega
        u_hip_clipped = np.clip(u_hip, -HIP_TORQUE_LIMIT, HIP_TORQUE_LIMIT)
        data.ctrl[0] = u_hip_clipped
        data.ctrl[1] = u_hip_clipped

        # ── Log ──────────────────────────────────────────────────────────────
        if idx < N:
            # Base kinematic/control channels
            rec['t'][idx]          = sim_t
            rec['pitch'][idx]      = pitch_true
            rec['pitch_rate'][idx] = pitch_rate_true
            rec['hip_q'][idx]      = hip_q
            rec['hip_omega'][idx]  = hip_omega
            rec['wheel_vel'][idx]  = wheel_vel
            rec['wheel_z'][idx]    = wheel_z_now
            rec['box_x'][idx]      = data.xpos[box_bid][0]
            rec['box_z'][idx]      = data.xpos[box_bid][2]
            rec['u_hip'][idx]      = data.ctrl[0]
            rec['u_wheel'][idx]    = data.ctrl[2]
            rec['pitch_ff'][idx]   = pitch_ff
            rec['leg_state'][idx]  = float(leg_state)
            rec['grounded'][idx]   = 1.0 if grounded else 0.0

            # ── Structural channels (per side) ────────────────────────────
            for side_L, side_ch, f_bid, t_bid, c_bid, w_bid, eq_id, tire_g in [
                (True,  'L', femur_bid_L, tibia_bid_L, coupler_bid_L, wheel_bid_L, eq_id_L, tire_gid_L),
                (False, 'R', femur_bid_R, tibia_bid_R, coupler_bid_R, wheel_bid_R, eq_id_R, tire_gid_R),
            ]:
                s = side_ch

                # 6D constraint wrenches [Tx,Ty,Tz,Fx,Fy,Fz] in world frame
                cfrc_fem = _cfrc(f_bid)
                cfrc_tib = _cfrc(t_bid)
                cfrc_cpl = _cfrc(c_bid)
                cfrc_whl = _cfrc(w_bid)

                for comp_i, comp_name in enumerate(('tx','ty','tz','fx','fy','fz')):
                    rec[f'cfrc_fem_{s}_{comp_name}'][idx] = cfrc_fem[comp_i]
                    rec[f'cfrc_tib_{s}_{comp_name}'][idx] = cfrc_tib[comp_i]
                    rec[f'cfrc_cpl_{s}_{comp_name}'][idx] = cfrc_cpl[comp_i]
                    rec[f'cfrc_whl_{s}_{comp_name}'][idx] = cfrc_whl[comp_i]

                # Axial/lateral decompositions
                ax_fem = _link_axis_fem(side_L)
                ax_tib = _link_axis_tib(side_L)
                ax_cpl = _link_axis_cpl(side_L)

                fax_f, flat_f = _axial_lateral(cfrc_fem[3:6], ax_fem)
                fax_t, flat_t = _axial_lateral(cfrc_tib[3:6], ax_tib)
                fax_c, flat_c = _axial_lateral(cfrc_cpl[3:6], ax_cpl)

                rec[f'fax_fem_{s}'][idx]  = fax_f
                rec[f'flat_fem_{s}'][idx] = flat_f
                rec[f'fax_tib_{s}'][idx]  = fax_t
                rec[f'flat_tib_{s}'][idx] = flat_t
                rec[f'fax_cpl_{s}'][idx]  = fax_c
                rec[f'flat_cpl_{s}'][idx] = flat_c

                # Bearing radial loads
                # A: force at hip pivot — pure (femur subtree, no equality)
                rec[f'fbear_A_{s}'][idx] = _bearing_radial(cfrc_fem)

                # E: equality constraint force magnitude (accurate)
                F_E_mag = _eq_force_mag(eq_id)
                rec[f'fbear_E_{s}'][idx] = F_E_mag

                # F: coupler pivot — for two-force member, |F_F| ≈ |F_E|
                #    cfrc_int[coupler] ≈ 0 (sum of equal-and-opposite end forces)
                #    so we use fbear_E as the best estimate
                rec[f'fbear_F_{s}'][idx] = F_E_mag

                # C: knee pivot — subtract E from cfrc_int[tibia] to isolate F_C
                #    F_C = cfrc_int[tibia][3:6] - F_E_vec (E acts on tibia subtree too)
                F_E_vec = _eq_force_vec(eq_id)
                F_C_vec = cfrc_tib[3:6] - F_E_vec
                rec[f'fbear_C_{s}'][idx] = float(np.linalg.norm(F_C_vec))

                # W: wheel axle — pure (wheel_asm subtree, no equality)
                rec[f'fbear_W_{s}'][idx] = _bearing_radial(cfrc_whl)

                # Ground reaction force (normal component, wheel contact)
                rec[f'grf_{s}'][idx] = _grf(tire_g)

            idx += 1

        mujoco.mj_step(model, data)

        max_height_m = max(max_height_m, data.xpos[wheel_bid_L][2] - WHEEL_R)
        if abs(pitch_true) > 1.2:
            print("FAIL: fell over (|pitch| > 1.2 rad)"); break
        if jump_triggered and sim_t > jump_start_t + 4.0:
            break

    # Trim and save
    n = idx
    np.savez(TELEMETRY_PATH, **{k: rec[k][:n] for k in channels})

    print(f"\nTelemetry saved -> {TELEMETRY_PATH}")
    print(f"Steps logged    : {n}  ({n * model.opt.timestep:.2f} s sim time)")
    print(f"Channels logged : {len(channels)}  ({len(struct_channels)} structural)")
    print(f"Max jump height : {max_height_m * 1000:.1f} mm")
    if liftoff_t:
        print(f"Liftoff         : t = {liftoff_t:.3f} s")
    if flight_duration:
        print(f"Flight duration : {flight_duration * 1000:.0f} ms")
    if land_t > 0:
        print(f"Landing         : t = {land_t:.3f} s")


if __name__ == "__main__":
    run()
