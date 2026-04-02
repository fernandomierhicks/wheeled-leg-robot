"""sim_loop.py — THE single simulation core for all scenarios.

Ported from latency_sensitivity/scenarios.py (_run_sim_loop + init_sim +
_build_model_and_data).  All parameters come from SimParams; no module-level
mutable globals.

Usage:
    # Headless optimizer
    metrics = run(params, scenario_config)

    # Replay with callbacks
    metrics = run(params, scenario_config, callbacks=[telemetry_recorder])

    # Sandbox with command queue
    metrics = run(params, scenario_config, callbacks=[live_chart], command_queue=q)
"""
import collections
import math
from dataclasses import replace as _dc_replace
import numpy as np
import mujoco

from master_sim_jump.params import SimParams
from master_sim_jump.scenarios.base import ScenarioConfig, WorldConfig
from master_sim_jump.physics import (build_xml, build_assets, solve_ik,
                                     get_equilibrium_pitch, compute_com_x_from_wheel)
from master_sim_jump.models.battery import BatteryModel
from master_sim_jump.models.motor import motor_taper, motor_currents
from master_sim_jump.models.latency import LatencyBuffer
from master_sim_jump.controllers.lqr import (
    compute_gain_table, lqr_torque, compute_AB_table, interpolate_AB,
    discretize_AB,
)
from master_sim_jump.controllers.velocity_pi import VelocityPI
from master_sim_jump.controllers.yaw_pi import YawPI
from master_sim_jump.controllers.hip import (
    hip_position_torque, hip_impedance_torque, roll_leveling_offsets,
    knee_spring_torque,
)
from master_sim_jump.controllers.jump import JumpController, RobotMode


# ── Shared MuJoCo lookup helpers ─────────────────────────────────────────────

def _jnt_qposadr(model, name: str) -> int:
    return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]


# ── Model construction ───────────────────────────────────────────────────────

def build_model_and_data(params: SimParams,
                         world: WorldConfig = None) -> tuple:
    """Construct MuJoCo model + data from SimParams and WorldConfig."""
    if world is None:
        world = WorldConfig()
    xml = build_xml(
        robot=params.robot,
        motors=params.motors,
        obstacle_height=world.obstacle_height,
        bumps=list(world.bumps) if world.bumps else None,
        sandbox_obstacles=list(world.sandbox_obstacles) if world.sandbox_obstacles else None,
        prop_bodies=list(world.prop_bodies) if world.prop_bodies else None,
        floor_size=world.floor_size[:2] if world.floor_size else None,
    )
    assets = build_assets()
    model = mujoco.MjModel.from_xml_string(xml, assets)
    data = mujoco.MjData(model)
    return model, data


def init_sim(model, data, params: SimParams,
             q_hip_init: float = None) -> None:
    """Reset and place robot at q_hip_init (defaults to Q_NOM)."""
    robot = params.robot
    p = robot.as_dict()
    if q_hip_init is None:
        q_hip_init = robot.Q_NOM

    mujoco.mj_resetData(model, data)

    # Fix 4-bar closure anchors
    for side in ('L', 'R'):
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                   f"4bar_close_{side}")
        if eq_id >= 0:
            model.eq_data[eq_id, 0:3] = [-p['Lc'], 0.0, 0.0]
            model.eq_data[eq_id, 3:6] = [0.0, 0.0, p['L_stub']]

    s_root   = _jnt_qposadr(model, "root_free")
    s_hF_L   = _jnt_qposadr(model, "hinge_F_L"); s_hF_R   = _jnt_qposadr(model, "hinge_F_R")
    s_hip_L  = _jnt_qposadr(model, "hip_L");     s_hip_R  = _jnt_qposadr(model, "hip_R")
    s_knee_L = _jnt_qposadr(model, "knee_joint_L"); s_knee_R = _jnt_qposadr(model, "knee_joint_R")

    ik = solve_ik(q_hip_init, p)
    if ik is None:
        raise RuntimeError(f"IK failed at q_hip={q_hip_init:.3f}")

    for s_hF, s_hip, s_knee in [
        (s_hF_L, s_hip_L, s_knee_L),
        (s_hF_R, s_hip_R, s_knee_R),
    ]:
        data.qpos[s_hF]   = ik['q_coupler_F']
        data.qpos[s_hip]  = ik['q_hip']
        data.qpos[s_knee] = ik['q_knee']

    mujoco.mj_forward(model, data)

    wheel_bid_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_asm_L")
    wheel_bid_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_asm_R")
    wz = (data.xpos[wheel_bid_L][2] + data.xpos[wheel_bid_R][2]) / 2.0
    data.qpos[s_root + 2] += robot.wheel_r - wz

    theta = get_equilibrium_pitch(robot, q_hip_init,
                                   m_spring=params.gains.knee_spring.m_spring)
    data.qpos[s_root + 3] = math.cos(theta / 2.0)
    data.qpos[s_root + 4] = 0.0
    data.qpos[s_root + 5] = math.sin(theta / 2.0)
    data.qpos[s_root + 6] = 0.0

    mujoco.mj_forward(model, data)


# ── Sensor helpers ───────────────────────────────────────────────────────────

def get_pitch_and_rate(data, box_bid: int, d_pitch: int) -> tuple:
    """Return (pitch_rad, pitch_rate_rad_s) from body world quaternion."""
    q = data.xquat[box_bid]
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q[0]*q[2] - q[3]*q[1]))))
    return pitch, float(data.qvel[d_pitch])


def get_roll_and_rate(data, box_bid: int, d_roll: int) -> tuple:
    """Return (roll_rad, roll_rate_rad_s) from body world quaternion."""
    q = data.xquat[box_bid]
    roll = math.atan2(
        2.0 * (q[0]*q[1] + q[2]*q[3]),
        1.0 - 2.0 * (q[1]**2 + q[2]**2))
    return roll, float(data.qvel[d_roll])


# ── State predictor ─────────────────────────────────────────────────────────

def predict_state(pitch: float, pitch_rate: float,
                  sensor_delay_s: float, l_eff: float) -> tuple:
    """One Euler step of the linearised inverted pendulum forward by sensor_delay_s.

    Dynamics: θ̈ = (g/l) * θ  (unstable, α > 0).
    Used to partially cancel phase lag introduced by the sensor delay buffer.

    Parameters
    ----------
    pitch, pitch_rate : delayed sensor readings (rad, rad/s)
    sensor_delay_s    : how far ahead to integrate [s]
    l_eff             : effective pendulum length [m]

    Returns
    -------
    (pitch_pred, pitch_rate_pred) : forward-predicted state
    """
    pitch_pred      = pitch      + pitch_rate * sensor_delay_s
    pitch_rate_pred = pitch_rate + (9.81 / l_eff * pitch) * sensor_delay_s
    return pitch_pred, pitch_rate_pred


def smith_predict(pitch: float, pitch_rate: float,
                  tau_hist: collections.deque,
                  dt_ctrl: float, l_eff: float, B1: float) -> tuple:
    """Smith predictor: step forward through stored torque history one tick at a time.

    For each stored torque u_i (oldest first), one Euler step:
        θ     ← θ  + θ̇ * dt
        θ̇    ← θ̇ + (g/l * θ  +  B1 * u_i) * dt

    B1 is B[1,0] of the linearised model — maps wheel torque → pitch angular accel.
    Including the torque history prevents the phase over-correction that occurs when
    predict_state is used alone (which ignores control effort during the delay window).
    """
    g_over_l = 9.81 / l_eff
    p, pdot = pitch, pitch_rate
    for tau in tau_hist:          # deque iterates oldest → newest
        p_new    = p    + pdot * dt_ctrl
        pdot_new = pdot + (g_over_l * p + B1 * tau) * dt_ctrl
        p, pdot = p_new, pdot_new
    return p, pdot


# ── Disturbance application — single source of truth ────────────────────────

def apply_disturbance(data, t: float, scenario: ScenarioConfig,
                      box_bid: int, wheel_bid_L: int, wheel_bid_R: int):
    """Apply scenario disturbance forces to the correct bodies.

    Called by sim_loop.run() AND the replay viewer so both paths produce
    identical physics.  Returns the (target_body_id, force_value) actually
    applied by dist_fn (for visualisation), or (None, 0.0) if inactive.
    """
    dist_fn      = scenario.dist_fn
    roll_dist_fn = scenario.roll_dist_fn
    dist_target  = scenario.dist_target

    fz_roll = roll_dist_fn(t) if roll_dist_fn else 0.0
    fd = dist_fn(t) if dist_fn else 0.0
    vis_bid, vis_force = None, 0.0

    if dist_fn and abs(fd) > 1e-9:
        vis_force = fd
    if dist_target == "wheel_L":
        data.xfrc_applied[wheel_bid_L, 2] = fd + fz_roll
        if vis_force:
            vis_bid = wheel_bid_L
    elif dist_target == "wheel_R":
        data.xfrc_applied[wheel_bid_R, 2] = fd
        data.xfrc_applied[wheel_bid_L, 2] = fz_roll
        if vis_force:
            vis_bid = wheel_bid_R
    else:
        data.xfrc_applied[box_bid, 0] = fd
        data.xfrc_applied[wheel_bid_L, 2] = fz_roll
        if vis_force:
            vis_bid = box_bid

    return vis_bid, vis_force


# ── SimController — single source of truth for the control cascade ──────────

class SimController:
    """Encapsulates all mutable controller state.

    Both ``run()`` (headless scenarios) and ``sandbox()`` (interactive viewer)
    call ``tick()`` for each control step, guaranteeing identical physics.

    Usage:
        ctrl = SimController(model, data, params, rng_seed=42)
        tick = ctrl.tick(model, data, v_target_ms=0.5, omega_target=0.0)
    """

    def __init__(self, model, data, params: SimParams, rng_seed=None):
        self.params = params
        self.rng = np.random.default_rng(rng_seed)
        self._lookup_addresses(model)
        self._init_controllers(params, model)

    # ── Address lookups (cached once per model) ──────────────────────────────

    def _lookup_addresses(self, model):
        def _jdof(n):
            return model.jnt_dofadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        def _act(n):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        def _bid(n):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

        d_root = _jdof("root_free")
        self.d_pitch = d_root + 4
        self.d_roll  = d_root + 3
        self.d_yaw   = d_root + 5
        self.s_root  = _jnt_qposadr(model, "root_free")

        self.s_hip_L = _jnt_qposadr(model, "hip_L"); self.s_hip_R = _jnt_qposadr(model, "hip_R")
        self.d_hip_L = _jdof("hip_L");               self.d_hip_R = _jdof("hip_R")
        self.d_whl_L = _jdof("wheel_spin_L")
        self.d_whl_R = _jdof("wheel_spin_R")

        self.act_hip_L   = _act("hip_act_L")
        self.act_hip_R   = _act("hip_act_R")
        self.act_wheel_L = _act("wheel_act_L")
        self.act_wheel_R = _act("wheel_act_R")

        self.box_bid     = _bid("box")
        self.wheel_bid_L = _bid("wheel_asm_L")
        self.wheel_bid_R = _bid("wheel_asm_R")
        self.tibia_bid_L = _bid("tibia_L")
        self.tibia_bid_R = _bid("tibia_R")

    # ── Controller objects (re-created on reset) ─────────────────────────────

    def _init_controllers(self, params, model):
        robot = params.robot
        ctrl_steps = params.timing.ctrl_steps
        self.dt_ctrl = model.opt.timestep * ctrl_steps

        self.K_table = compute_gain_table(robot, params.gains.lqr,
                                          m_spring=params.gains.knee_spring.m_spring)
        self.vel_pi = VelocityPI(params.gains.velocity_pi, self.dt_ctrl)
        self.yaw_pi_ctrl = YawPI(params.gains.yaw_pi, self.dt_ctrl)
        self.prev_theta_ref = 0.0
        self.prev_v_target = 0.0

        n_sens = round(params.latency.sensor_delay_s / self.dt_ctrl) \
            if params.latency.sensor_delay_s > 0 else 0
        n_act = round(params.latency.actuator_delay_s / self.dt_ctrl) \
            if params.latency.actuator_delay_s > 0 else 0
        self.n_sens = n_sens
        self.n_act = n_act
        pitch0 = get_equilibrium_pitch(robot, robot.Q_NOM,
                                       m_spring=params.gains.knee_spring.m_spring)
        self.sens_buf = LatencyBuffer(n_sens, (pitch0, 0.0, 0.0, 9.81))
        self.ctrl_buf = LatencyBuffer(n_act, (0.0, 0.0))

        ik0 = solve_ik(robot.Q_NOM, robot.as_dict())
        self._l_eff_nom = abs(ik0['W_z']) if ik0 else 0.2

        # FF3: CoM X offset at nominal hip angle (reference for delta computation)
        _com_x_nom = compute_com_x_from_wheel(robot, robot.Q_NOM,
                                              m_spring=params.gains.knee_spring.m_spring)
        self._com_x_nom = _com_x_nom if _com_x_nom is not None else 0.0

        # Body mass (excluding wheels) — same as lqr.py for consistency
        self._m_b = (robot.m_box
                     + 2 * (robot.m_femur + robot.m_tibia + robot.m_coupler
                            + robot.m_bearing)
                     + 2 * robot.motor_mass
                     + 2 * params.gains.knee_spring.m_spring)

        # Matrix predictor: discretise A, B at nominal leg position and
        # pre-compute the n_sens-step prediction coefficients.
        #   x̂(t) = A_d^n * x(t-n) + sum_{k=0}^{n-1} A_d^{n-1-k} * B_d * u(t-n+k)
        # where tau_hist[k] = u(t-n+k) (oldest → newest).
        AB_table = compute_AB_table(robot,
                                    m_spring=params.gains.knee_spring.m_spring)
        A_nom, B_nom = interpolate_AB(AB_table, robot.Q_NOM, robot)
        self._B1_nom = float(B_nom[1, 0])          # kept for legacy reference
        if n_sens > 0:
            A_d, B_d = discretize_AB(A_nom, B_nom, self.dt_ctrl)
            self._Ad_n   = np.linalg.matrix_power(A_d, n_sens)
            self._BdPow  = [np.linalg.matrix_power(A_d, n_sens - 1 - k) @ B_d
                            for k in range(n_sens)]
        else:
            self._Ad_n  = None
            self._BdPow = None
        # Ring buffer of last n_sens symmetric torque commands (oldest → newest)
        self._tau_hist = collections.deque([0.0] * n_sens, maxlen=max(n_sens, 1))

        self.jump_ctrl = JumpController(robot, params.gains.jump, self.dt_ctrl)

        self.battery = BatteryModel(params.battery)
        self.battery.reset()
        self.v_batt = params.battery.V_nom

        # Adaptive suspension scale is now owned by JumpController (via mode_out.susp_scale)

    def reset(self, model, data):
        """Reset all controller state (e.g. after sandbox auto-restart)."""
        self._init_controllers(self.params, model)

    def _patch_gains_subfield(self, subfield: str, field: str, value: float):
        """Replace one field on params.gains.<subfield>, persist to params, and return new sub-gains."""
        new_sub = _dc_replace(getattr(self.params.gains, subfield), **{field: value})
        self.params = _dc_replace(self.params,
                                  gains=_dc_replace(self.params.gains, **{subfield: new_sub}))
        return new_sub

    def update_velocity_pi_gain(self, field: str, value: float):
        """Hot-swap a single VelocityPI gain field (preserves integrator + survives reset)."""
        self.vel_pi.gains = self._patch_gains_subfield("velocity_pi", field, value)

    def update_lqr_gain(self, field: str, value: float):
        """Hot-swap a single LQR cost weight and recompute the gain table."""
        self._patch_gains_subfield("lqr", field, value)
        self.K_table = compute_gain_table(self.params.robot, self.params.gains.lqr,
                                          m_spring=self.params.gains.knee_spring.m_spring)

    def update_yaw_pi_gain(self, field: str, value: float):
        """Hot-swap a single YawPI gain field (preserves integrator + survives reset)."""
        self.yaw_pi_ctrl.gains = self._patch_gains_subfield("yaw_pi", field, value)

    def update_suspension_gain(self, field: str, value: float):
        """Hot-swap a single Suspension gain field (survives reset)."""
        self._patch_gains_subfield("suspension", field, value)

    def update_jump_gain(self, field: str, value: float):
        """Hot-swap a single JumpGains field; also updates jump_ctrl.gains live."""
        new_gains = self._patch_gains_subfield("jump", field, value)
        self.jump_ctrl.gains = new_gains

    def update_robot_geom(self, field: str, value: float):
        """Hot-swap Q_EXT or Q_RET and recompute LQR gain table."""
        new_robot = _dc_replace(self.params.robot, **{field: value})
        self.params = _dc_replace(self.params, robot=new_robot)
        self.K_table = compute_gain_table(new_robot, self.params.gains.lqr,
                                          m_spring=self.params.gains.knee_spring.m_spring)

    # ── THE control tick — called by run() and sandbox() ─────────────────────

    def tick(self, model, data, *,
             v_target_ms: float = 0.0,
             omega_target: float = 0.0,
             theta_ref_cmd: float = 0.0,
             q_hip_target: float = None,
             dq_hip_target: float = 0.0,
             use_lqr: bool = True,
             use_velocity_pi: bool = True,
             use_yaw_pi: bool = True,
             use_impedance: bool = True,
             use_roll_leveling: bool = True,
             use_suspension: bool = True,
             use_ff1: bool = True,
             use_ff2: bool = True,
             use_ff3: bool = True,
             use_ff4: bool = True,
             use_knee_spring: bool = False,
             jump_active: bool = False) -> dict:
        """Execute one control tick.  Writes actuator commands to ``data.ctrl``.

        Returns a telemetry dict with all values needed for metrics, callbacks,
        and live visualisation.
        """
        params = self.params
        robot  = params.robot
        rng    = self.rng

        if q_hip_target is None:
            q_hip_target = robot.Q_NOM
        q_hip_target = float(np.clip(q_hip_target, robot.Q_EXT, robot.Q_RET))

        # ── Sensors ──────────────────────────────────────────────────────────
        pitch_true, pitch_rate_true = get_pitch_and_rate(
            data, self.box_bid, self.d_pitch)
        pitch      = pitch_true      + rng.normal(0, params.noise.pitch_std_rad)
        pitch_rate = pitch_rate_true + rng.normal(0, params.noise.pitch_rate_std_rad_s)
        wheel_vel  = (data.qvel[self.d_whl_L] + data.qvel[self.d_whl_R]) / 2.0
        az_imu     = float(data.cacc[self.box_bid, 5])        + rng.normal(0, params.noise.accel_std)
        ax_imu     = float(data.cacc[self.box_bid, 3])        + rng.normal(0, params.noise.accel_std)
        ay_imu     = float(data.cacc[self.box_bid, 4])        + rng.normal(0, params.noise.accel_std)
        gx_imu     = float(data.cvel[self.box_bid, 0])        + rng.normal(0, params.noise.pitch_rate_std_rad_s)
        gy_imu     = float(data.cvel[self.box_bid, 1])        + rng.normal(0, params.noise.pitch_rate_std_rad_s)
        gz_imu     = float(data.cvel[self.box_bid, 2])        + rng.normal(0, params.noise.pitch_rate_std_rad_s)
        hip_q_avg  = (data.qpos[self.s_hip_L] + data.qpos[self.s_hip_R]) / 2.0
        pitch_ff   = get_equilibrium_pitch(robot, hip_q_avg,
                                           m_spring=params.gains.knee_spring.m_spring)

        # ── Velocity reference + direction-reversal guard ────────────────────
        v_ref_rads = v_target_ms / robot.wheel_r

        if (use_velocity_pi and self.prev_v_target != 0.0 and
                math.copysign(1, v_target_ms) != math.copysign(1, self.prev_v_target)):
            self.vel_pi.reset()
        self.prev_v_target = v_target_ms

        # ── Sensor delay buffer ──────────────────────────────────────────────
        _pitch_d, _pitch_rate_d, _wheel_vel_d, _az_d = self.sens_buf.push(
            (pitch, pitch_rate, wheel_vel, az_imu))

        # ── Jump controller (az undelayed — raw sample for responsive liftoff detection) ──
        if jump_active:
            hip_dq_avg = (data.qvel[self.d_hip_L] + data.qvel[self.d_hip_R]) / 2.0
            mode_out = self.jump_ctrl.update(
                data.time, hip_q_avg, hip_dq_avg,
                az_imu, params.gains.suspension, _pitch_d)
            q_hip_target = mode_out.q_hip_target
            dq_hip_target = mode_out.dq_hip_target
        else:
            mode_out = None

        # Capture delayed (pre-predictor) values for diagnostics
        _pitch_delayed     = float(_pitch_d)
        _pitch_rate_delayed = float(_pitch_rate_d)
        _wheel_vel_delayed = float(_wheel_vel_d)

        # ── Matrix predictor — ZOH-discretised 3-state prediction ──────────────
        # Propagates the delayed state x(t-n) forward n steps using the exact
        # discrete A_d, B_d and the stored torque history, giving x̂(t).
        # Replaces the scalar Smith predictor which ignored wheel_vel coupling.
        if self.n_sens > 0:
            x_del = np.array([_pitch_d, _pitch_rate_d, _wheel_vel_d])
            x_pred = self._Ad_n @ x_del
            for k, tau in enumerate(self._tau_hist):
                x_pred += self._BdPow[k].ravel() * tau
            _pitch_d, _pitch_rate_d, _wheel_vel_d = x_pred

        # ── Velocity PI (uses delayed wheel_vel — matches real CAN telemetry) ─
        v_measured_ms = _wheel_vel_d * robot.wheel_r

        # Freeze VelocityPI during CROUCH/EXTEND to prevent velocity error
        # from commanding a lean that fights the jump impulse.
        _jump_launching = (jump_active and mode_out is not None and
                           mode_out.mode in (RobotMode.CROUCH, RobotMode.EXTEND))

        if use_velocity_pi and not _jump_launching:
            theta_ref = self.vel_pi.update(v_target_ms, v_measured_ms)
            _d_max = params.gains.velocity_pi.theta_ref_rate_limit * self.dt_ctrl
            theta_ref = float(np.clip(
                theta_ref,
                self.prev_theta_ref - _d_max,
                self.prev_theta_ref + _d_max))
        elif _jump_launching:
            # Hold theta_ref at pre-jump value, don't touch integrator
            theta_ref = self.prev_theta_ref
        else:
            self.vel_pi.reset()
            theta_ref  = theta_ref_cmd
            # v_ref_rads passes through to LQR even without VelocityPI
        self.prev_theta_ref = theta_ref

        # ── FF3: CoM shift compensation (pitch offset) ───────────────────────
        theta_ff3 = 0.0
        ff3_alpha = params.gains.feedforward.ff3_alpha
        if use_ff3 and ff3_alpha > 0.0:
            _com_x = compute_com_x_from_wheel(robot, hip_q_avg,
                                              m_spring=params.gains.knee_spring.m_spring)
            if _com_x is not None:
                _delta_com_x = _com_x - self._com_x_nom
                _ik_ff3 = solve_ik(hip_q_avg, robot.as_dict())
                _l_eff_ff3 = abs(_ik_ff3['W_z']) if _ik_ff3 else self._l_eff_nom
                if _l_eff_ff3 > 0.01:
                    theta_ff3 = ff3_alpha * math.atan2(_delta_com_x, _l_eff_ff3)

        # ── FF4: Centripetal turn coupling (pitch offset) ──────────────────
        theta_ff4 = 0.0
        ff4_alpha = params.gains.feedforward.ff4_alpha
        if use_ff4 and ff4_alpha > 0.0:
            _yaw_rate = data.qvel[self.d_yaw]
            _v_fwd = wheel_vel * robot.wheel_r
            theta_ff4 = -ff4_alpha * _v_fwd * _yaw_rate / 9.81

        # ── LQR controller (symmetric torque) ────────────────────────────────
        if use_lqr:
            tau_sym = lqr_torque(
                _pitch_d, _pitch_rate_d, _wheel_vel_d, hip_q_avg,
                self.K_table, robot, params.motors.wheel,
                v_ref=v_ref_rads, theta_ref=theta_ref + theta_ff3 + theta_ff4)
        else:
            tau_sym = 0.0

        # ── Yaw PI (differential torque) ─────────────────────────────────────
        yaw_rate = data.qvel[self.d_yaw]
        omega_cmd_max = self.yaw_pi_ctrl.gains.omega_cmd_max
        omega_target = max(-omega_cmd_max, min(omega_cmd_max, omega_target))
        if use_yaw_pi:
            tau_yaw = self.yaw_pi_ctrl.update(omega_target, yaw_rate)
        else:
            self.yaw_pi_ctrl.reset()
            tau_yaw = 0.0

        # ── Wheel hold: velocity damping during CROUCH+EXTEND ───────────────
        # Resists wheel rotation so hip extension energy goes into vertical
        # impulse rather than rolling the robot.  Goes to zero as wheels stop.
        if _jump_launching:
            _K_hold = params.gains.jump.wheel_hold_gain
            tau_sym += -_K_hold * _wheel_vel_d
            # Reverse feedforward torque during EXTEND only — counteracts
            # hip reaction that pushes wheels forward during explosive extension.
            if mode_out.mode == RobotMode.EXTEND:
                tau_sym += params.gains.jump.wheel_ff_torque

        # Store torque in Smith predictor history (symmetric, before yaw split)
        self._tau_hist.append(tau_sym)

        # ── Actuator delay buffer + motor taper → wheel ctrl ─────────────────
        _tau_cmd_L = tau_sym - tau_yaw
        _tau_cmd_R = tau_sym + tau_yaw
        _tau_L_d, _tau_R_d = self.ctrl_buf.push((_tau_cmd_L, _tau_cmd_R))
        data.ctrl[self.act_wheel_L] = motor_taper(
            _tau_L_d, data.qvel[self.d_whl_L],
            self.v_batt, params.motors, params.battery)
        data.ctrl[self.act_wheel_R] = motor_taper(
            _tau_R_d, data.qvel[self.d_whl_R],
            self.v_batt, params.motors, params.battery)

        # ── Adaptive suspension scale (owned by JumpController) ─────────────
        # susp_scale is 1.0 normally, freefall_scale in FLYING, ramping in LANDING.
        _susp_scale = mode_out.susp_scale if mode_out is not None else 1.0

        _eff_suspension = _dc_replace(
            params.gains.suspension,
            K_s=params.gains.suspension.K_s * _susp_scale,
            B_s=params.gains.suspension.B_s * _susp_scale,
        )

        # ── Hip control ──────────────────────────────────────────────────────
        roll_true, roll_rate = get_roll_and_rate(
            data, self.box_bid, self.d_roll)

        # Determine effective hip mode: jump controller overrides per-phase
        if jump_active and mode_out is not None:
            eff_hip_mode = mode_out.hip_mode          # phase-dependent
        elif use_impedance:
            eff_hip_mode = "impedance"
        else:
            eff_hip_mode = "position"

        if eff_hip_mode == "torque_override":
            # EXTEND phase: direct torque bypass (symmetric, no roll leveling)
            q_nom_L = q_nom_R = q_hip_target
            data.ctrl[self.act_hip_L] = mode_out.hip_torque_override
            data.ctrl[self.act_hip_R] = mode_out.hip_torque_override

        elif eff_hip_mode == "impedance":
            # Soft spring-damper with roll leveling
            roll_meas = roll_true + rng.normal(0, params.noise.roll_std_rad)
            if use_roll_leveling or (jump_active and mode_out is not None):
                q_nom_L, q_nom_R = roll_leveling_offsets(
                    roll_meas, roll_rate, q_hip_target,
                    _eff_suspension, robot)
            else:
                q_nom_L = q_nom_R = q_hip_target
            for s_hip, d_hip, act_hip, q_nom_leg in [
                (self.s_hip_L, self.d_hip_L, self.act_hip_L, q_nom_L),
                (self.s_hip_R, self.d_hip_R, self.act_hip_R, q_nom_R),
            ]:
                if not use_suspension:
                    data.ctrl[act_hip] = 0.0
                    continue
                tau_hip = hip_impedance_torque(
                    data.qpos[s_hip], data.qvel[d_hip], q_nom_leg,
                    _eff_suspension, params.motors.hip)
                data.ctrl[act_hip] = tau_hip

        else:
            # Stiff PD position servo (non-jump position mode, or FLYING phase)
            q_nom_L = q_nom_R = q_hip_target
            for s_hip, d_hip, act_hip in [
                (self.s_hip_L, self.d_hip_L, self.act_hip_L),
                (self.s_hip_R, self.d_hip_R, self.act_hip_R),
            ]:
                if not use_suspension:
                    data.ctrl[act_hip] = 0.0
                    continue
                data.ctrl[act_hip] = hip_position_torque(
                    data.qpos[s_hip], data.qvel[d_hip],
                    q_hip_target, params.motors.hip,
                    dq_target=dq_hip_target)

        # ── Knee spring (conditional torsional spring) ─────────────────────
        if use_knee_spring:
            spring = params.gains.knee_spring
            q_engage = robot.Q_NOM + spring.engage_offset
            tau_spring_L = knee_spring_torque(
                float(data.qpos[self.s_hip_L]), float(data.qvel[self.d_hip_L]),
                q_engage, spring)
            tau_spring_R = knee_spring_torque(
                float(data.qpos[self.s_hip_R]), float(data.qvel[self.d_hip_R]),
                q_engage, spring)
        else:
            tau_spring_L = 0.0
            tau_spring_R = 0.0
        data.qfrc_applied[self.d_hip_L] = tau_spring_L
        data.qfrc_applied[self.d_hip_R] = tau_spring_R

        # ── Feedforward: shared IK for current leg geometry ─────────────────
        _need_ff_ik = ((use_ff1 and params.gains.feedforward.ff1_alpha > 0.0)
                       or (use_ff2 and params.gains.feedforward.ff2_alpha > 0.0))
        if _need_ff_ik:
            _ik_ff = solve_ik(hip_q_avg, robot.as_dict())
            _l_eff_ff = abs(_ik_ff['W_z']) if _ik_ff else self._l_eff_nom
        else:
            _l_eff_ff = self._l_eff_nom

        # ── FF1: Hip reaction torque cancellation ─────────────────────────────
        tau_ff1 = 0.0
        ff1_alpha = params.gains.feedforward.ff1_alpha
        if use_ff1 and ff1_alpha > 0.0 and _l_eff_ff > 0.01:
            tau_hip_total = (float(data.ctrl[self.act_hip_L])
                            + float(data.ctrl[self.act_hip_R]))
            tau_ff1 = -ff1_alpha * tau_hip_total * (robot.wheel_r / _l_eff_ff)
            data.ctrl[self.act_wheel_L] += tau_ff1 / 2.0
            data.ctrl[self.act_wheel_R] += tau_ff1 / 2.0

        # ── FF2: Gravity compensation torque on wheels ────────────────────────
        # Pendulum gravity torque = m_b * g * l_eff * sin(pitch)
        tau_ff2 = 0.0
        ff2_alpha = params.gains.feedforward.ff2_alpha
        if use_ff2 and ff2_alpha > 0.0 and _l_eff_ff > 0.01:
            tau_ff2 = ff2_alpha * self._m_b * 9.81 * _l_eff_ff * math.sin(pitch)
            data.ctrl[self.act_wheel_L] += tau_ff2 / 2.0
            data.ctrl[self.act_wheel_R] += tau_ff2 / 2.0

        # ── Battery step ─────────────────────────────────────────────────────
        self.v_batt = self.battery.step(self.dt_ctrl, motor_currents(
            float(data.ctrl[self.act_wheel_L]),
            float(data.ctrl[self.act_wheel_R]),
            float(data.ctrl[self.act_hip_L]),
            float(data.ctrl[self.act_hip_R]),
            params.motors, params.battery.I_quiescent))

        # ── Fall detection ───────────────────────────────────────────────────
        fell = abs(pitch_true) > params.thresholds.fall_rad

        # ── Telemetry dict (superset of what callbacks + sandbox need) ────────
        return dict(
            t=data.time,
            pitch=pitch_true, pitch_rate=pitch_rate_true, pitch_ff=pitch_ff,
            roll=roll_true, roll_rate=roll_rate,
            wheel_vel=wheel_vel,
            v_target=v_target_ms, v_measured=v_measured_ms,
            theta_ref=theta_ref,
            tau_sym=tau_sym, tau_yaw=tau_yaw, tau_ff1=tau_ff1, tau_ff2=tau_ff2, theta_ff3=theta_ff3, theta_ff4=theta_ff4,
            tau_whl_L=float(data.ctrl[self.act_wheel_L]),
            tau_whl_R=float(data.ctrl[self.act_wheel_R]),
            tau_hip_L=float(data.ctrl[self.act_hip_L]),
            tau_hip_R=float(data.ctrl[self.act_hip_R]),
            tau_spring_L=tau_spring_L,
            tau_spring_R=tau_spring_R,
            hip_q_L=float(data.qpos[self.s_hip_L]),
            hip_q_R=float(data.qpos[self.s_hip_R]),
            hip_q_avg=hip_q_avg,
            q_nom_L=q_nom_L, q_nom_R=q_nom_R,
            v_batt=self.v_batt,
            batt_soc=self.battery.soc_pct,
            batt_temp=self.battery.temperature_c,
            i_total=self.battery.i_total,
            yaw_rate=yaw_rate,
            omega_tgt=omega_target,
            pos_x=float(data.qpos[self.s_root]),
            wheel_z_L=float(data.xpos[self.wheel_bid_L][2]),
            wheel_z_R=float(data.xpos[self.wheel_bid_R][2]),
            fell=fell,
            # ── Diagnostic: delay / predictor intermediates ─────────────
            pitch_noisy=pitch,
            pitch_rate_noisy=pitch_rate,
            pitch_delayed=_pitch_delayed,
            pitch_rate_delayed=_pitch_rate_delayed,
            wheel_vel_delayed=_wheel_vel_delayed,
            pitch_predicted=float(_pitch_d),
            pitch_rate_predicted=float(_pitch_rate_d),
            wheel_vel_predicted=float(_wheel_vel_d),
            tau_cmd_L=_tau_cmd_L,
            tau_cmd_R=_tau_cmd_R,
            tau_delayed_L=float(_tau_L_d),
            tau_delayed_R=float(_tau_R_d),
            mode=mode_out.mode.name if mode_out else "BALANCE",
            az_imu=az_imu,
            ax_imu=ax_imu,
            ay_imu=ay_imu,
            gx_imu=gx_imu,
            gy_imu=gy_imu,
            gz_imu=gz_imu,
            susp_scale=_susp_scale,
        )


# ── Main simulation loop ────────────────────────────────────────────────────

def run(params: SimParams, scenario: ScenarioConfig,
        callbacks: list = None, command_queue=None,
        rng_seed: int = None) -> dict:
    """THE single simulation loop — all scenarios funnel through here.

    Parameters
    ----------
    params        : SimParams (immutable, from optimizer or DEFAULT_PARAMS)
    scenario      : ScenarioConfig (what to run)
    callbacks     : list of callable(tick_data_dict) for live telemetry
    command_queue : queue.Queue for runtime commands (sandbox mode)
    rng_seed      : reproducible noise seed

    Returns
    -------
    dict of all metrics (ISE, RMS, tracking errors, settle time, survival, etc.)
    """
    robot = params.robot

    # ── Build model + init ───────────────────────────────────────────────────
    model, data = build_model_and_data(params, scenario.world)
    init_sim(model, data, params)

    # Apply scenario-specific initial conditions (e.g., pitch step for S1)
    if scenario.init_fn is not None:
        scenario.init_fn(model, data, params)

    # ── Controller (single source of truth) ──────────────────────────────────
    ctrl = SimController(model, data, params, rng_seed=rng_seed)

    # ── Which controllers are active (from scenario — single source of truth) ─
    flags = scenario.tick_flags

    # ── Resolve profile callables ────────────────────────────────────────────
    v_profile_fn     = scenario.v_profile or (lambda t: 0.0)
    theta_ref_fn     = scenario.theta_ref_profile  # may be None
    omega_profile_fn = scenario.omega_profile  # may be None
    hip_profile_fn   = scenario.hip_profile    # may be None
    hip_vel_fn       = getattr(scenario, 'hip_vel_profile', None)
    use_theta_ref_correction = scenario.use_theta_ref_correction

    ctrl_steps = params.timing.ctrl_steps
    dt = model.opt.timestep * ctrl_steps

    # ── Thresholds ────────────────────────────────────────────────────────────
    th = params.thresholds
    liftoff_threshold = robot.wheel_r + th.liftoff_margin_m

    # ── Metric accumulators ──────────────────────────────────────────────────
    duration = scenario.duration
    pitch_sq_sum      = 0.0
    ise_pitch         = 0.0
    ise_pitch_rate    = 0.0
    pitch_rate_sq_sum = 0.0
    roll_sq_sum       = 0.0
    max_roll          = 0.0
    vel_sq_sum        = 0.0
    yaw_sq_sum        = 0.0
    max_pitch         = 0.0
    wheel_travel_m    = 0.0
    wheel_liftoff_s   = 0.0
    peak_body_z_m     = 0.0
    peak_wheel_z_m    = 0.0
    hip_track_sq_sum  = 0.0
    hip_rate_sq_sum   = 0.0
    prev_tau_hip_L    = 0.0
    prev_tau_hip_R    = 0.0
    hip_cmd_rate_sq   = 0.0
    prev_q_nom_L      = None
    prev_q_nom_R      = None
    liftoff_kill      = False
    vel_error_kill    = False
    n_samples         = 0
    n_vel             = 0
    n_yaw             = 0
    survived_s        = duration
    settle_time       = duration
    settled           = False
    settle_start      = None

    # Transient tracking
    prev_v_target     = 0.0
    transient_end     = -1.0
    transient_lag_sum = 0.0

    # Yaw error start
    yaw_err_start = getattr(params.scenarios, 'yaw_err_start', 1.0)

    # ── Simulation loop ─────────────────────────────────────────────────────
    step = 0
    while data.time < duration:
        if step % ctrl_steps == 0:
            v_target_ms = v_profile_fn(data.time)
            theta_ref_c = theta_ref_fn(data.time) if theta_ref_fn else 0.0
            omega_tgt   = omega_profile_fn(data.time) if omega_profile_fn else 0.0
            q_hip_sym   = hip_profile_fn(data.time) if hip_profile_fn else robot.Q_NOM
            dq_hip_tgt  = hip_vel_fn(data.time) if hip_vel_fn else 0.0

            # ── Jump trigger ──────────────────────────────────────────────
            if (scenario.jump_time is not None and
                    data.time >= scenario.jump_time):
                ctrl.jump_ctrl.trigger()

            tick = ctrl.tick(
                model, data,
                v_target_ms=v_target_ms,
                theta_ref_cmd=theta_ref_c,
                omega_target=omega_tgt,
                q_hip_target=q_hip_sym,
                dq_hip_target=dq_hip_tgt,
                **flags)

            # ── Unpack for metrics ───────────────────────────────────────────
            pitch_true      = tick['pitch']
            pitch_rate_true = tick['pitch_rate']
            pitch_ff        = tick['pitch_ff']
            roll_true       = tick['roll']
            wheel_vel       = tick['wheel_vel']
            v_measured_ms   = tick['v_measured']
            yaw_rate        = tick['yaw_rate']

            # ── Transient detection ──────────────────────────────────────────
            if abs(v_target_ms - prev_v_target) > 0.05 and data.time >= th.vel_err_start_s:
                transient_end = data.time + th.transient_window_s
            prev_v_target = v_target_ms

            # ── Yaw tracking error ───────────────────────────────────────────
            if omega_profile_fn and data.time >= yaw_err_start:
                yaw_sq_sum += (omega_tgt - yaw_rate) ** 2
                n_yaw += 1

            # ── Metrics ──────────────────────────────────────────────────────
            theta_ref_corr = tick['theta_ref'] if use_theta_ref_correction else 0.0
            theta_ff3_corr = tick.get('theta_ff3', 0.0)
            theta_ff4_corr = tick.get('theta_ff4', 0.0)
            pitch_err_deg = math.degrees(abs(pitch_true - pitch_ff - theta_ref_corr - theta_ff3_corr - theta_ff4_corr))
            pitch_err_rad = pitch_true - pitch_ff - theta_ref_corr - theta_ff3_corr - theta_ff4_corr
            pitch_sq_sum     += pitch_err_deg ** 2
            ise_pitch        += pitch_err_rad ** 2 * dt
            ise_pitch_rate   += pitch_rate_true ** 2 * dt
            pitch_rate_sq_sum += math.degrees(pitch_rate_true) ** 2
            roll_deg_now      = math.degrees(roll_true)
            roll_sq_sum      += roll_deg_now ** 2
            max_roll          = max(max_roll, abs(roll_deg_now))
            max_pitch         = max(max_pitch, pitch_err_deg)
            vel_est           = (wheel_vel + pitch_rate_true) * robot.wheel_r
            wheel_travel_m   += abs(vel_est) * dt
            hip_track_sq_sum += (tick['hip_q_avg'] - q_hip_sym) ** 2
            tau_hip_L = tick['tau_hip_L']
            tau_hip_R = tick['tau_hip_R']
            hip_rate_sq_sum += ((tau_hip_L - prev_tau_hip_L) / dt) ** 2
            hip_rate_sq_sum += ((tau_hip_R - prev_tau_hip_R) / dt) ** 2
            prev_tau_hip_L = tau_hip_L
            prev_tau_hip_R = tau_hip_R
            q_nom_L = tick['q_nom_L']
            q_nom_R = tick['q_nom_R']
            if prev_q_nom_L is not None:
                hip_cmd_rate_sq += ((q_nom_L - prev_q_nom_L) / dt) ** 2
                hip_cmd_rate_sq += ((q_nom_R - prev_q_nom_R) / dt) ** 2
            prev_q_nom_L = q_nom_L
            prev_q_nom_R = q_nom_R
            n_samples        += 1

            if data.time >= th.vel_err_start_s:
                vel_sq_sum += (v_target_ms - v_measured_ms) ** 2
                n_vel += 1

            if data.time < transient_end:
                transient_lag_sum += abs(v_target_ms - v_measured_ms) * dt

            wz_L = data.xpos[ctrl.wheel_bid_L][2]
            wz_R = data.xpos[ctrl.wheel_bid_R][2]

            if wz_L > liftoff_threshold or wz_R > liftoff_threshold:
                wheel_liftoff_s += dt

            # ── Peak heights (for jump optimizer) ─────────────────────
            body_z_now  = float(data.xpos[ctrl.box_bid][2])
            wheel_z_now = (wz_L + wz_R) / 2.0
            if body_z_now  > peak_body_z_m:  peak_body_z_m  = body_z_now
            if wheel_z_now > peak_wheel_z_m: peak_wheel_z_m = wheel_z_now

            # ── Liftoff early-termination (skip when jump is active) ──────
            if (scenario.max_liftoff_s is not None and
                    scenario.jump_time is None and
                    wheel_liftoff_s > scenario.max_liftoff_s):
                survived_s = data.time
                liftoff_kill = True
                break

            # ── Velocity-error early-termination ─────────────────────────
            if (scenario.max_vel_error_ms is not None and
                    abs(v_target_ms - v_measured_ms) > scenario.max_vel_error_ms):
                survived_s = data.time
                vel_error_kill = True
                break

            if not settled:
                if pitch_err_deg < th.settle_deg:
                    if settle_start is None:
                        settle_start = data.time
                    elif data.time - settle_start >= th.settle_window_s:
                        settle_time = settle_start
                        settled = True
                else:
                    settle_start = None

            if tick['fell']:
                survived_s = data.time
                break

            # ── Callbacks (replay/sandbox telemetry) ─────────────────────────
            if callbacks:
                for cb in callbacks:
                    cb(tick)

        # ── Disturbance impulse ──────────────────────────────────────────────
        apply_disturbance(data, data.time, scenario,
                          ctrl.box_bid, ctrl.wheel_bid_L, ctrl.wheel_bid_R)

        mujoco.mj_step(model, data)
        step += 1

    # ── Compute final metrics ────────────────────────────────────────────────
    rms_pitch_deg      = math.sqrt(pitch_sq_sum / max(1, n_samples))
    rms_pitch_rate_dps = math.sqrt(pitch_rate_sq_sum / max(1, n_samples))
    rms_roll_deg       = math.sqrt(roll_sq_sum  / max(1, n_samples))
    hip_track_rms_rad  = math.sqrt(hip_track_sq_sum / max(1, n_samples))
    hip_rate_rms       = math.sqrt(hip_rate_sq_sum / max(1, 2 * n_samples))
    hip_cmd_rate_rms   = math.sqrt(hip_cmd_rate_sq / max(1, 2 * max(1, n_samples - 1)))
    rms_vel_ms         = math.sqrt(vel_sq_sum / max(1, n_vel)) if n_vel > 0 else 0.0
    yaw_track_rms_rads = math.sqrt(yaw_sq_sum / max(1, n_yaw)) if n_yaw > 0 else 0.0
    final_x            = data.qpos[ctrl.s_root]
    fell               = survived_s < duration - 0.05 or liftoff_kill or vel_error_kill

    return dict(
        rms_pitch_deg           = round(rms_pitch_deg,          4),
        ise_pitch               = round(ise_pitch,              6),
        ise_pitch_rate          = round(ise_pitch_rate,         6),
        rms_pitch_rate_dps      = round(rms_pitch_rate_dps,    4),
        rms_roll_deg            = round(rms_roll_deg,           4),
        max_roll_deg            = round(max_roll,               4),
        vel_track_rms_ms        = round(rms_vel_ms,             4),
        transient_lag_ms        = round(transient_lag_sum,      4),
        yaw_track_rms_rads      = round(yaw_track_rms_rads,    4),
        max_pitch_deg           = round(max_pitch,              4),
        wheel_travel_m          = round(wheel_travel_m,         4),
        wheel_liftoff_s         = round(wheel_liftoff_s,        4),
        peak_body_z_m           = round(peak_body_z_m,          4),
        peak_wheel_z_m          = round(peak_wheel_z_m,         4),
        hip_track_rms_rad       = round(hip_track_rms_rad,      6),
        hip_rate_rms            = round(hip_rate_rms,            4),
        hip_cmd_rate_rms        = round(hip_cmd_rate_rms,        4),
        final_x_m               = round(final_x,               3),
        settle_time_s           = round(settle_time,            3),
        survived_s              = round(survived_s,             3),
        fell                    = fell,
        status                  = "FAIL" if fell else "PASS",
        fail_reason             = ("wheels off ground" if liftoff_kill
                                   else "vel error exceeded" if vel_error_kill
                                   else "fell over" if fell else ""),
    )
