"""vis_scenario1.py — 1_LQR_pitch_step: MuJoCo viewer + live matplotlib.

Pitch reference line = get_equilibrium_pitch() — the angle the robot needs
to hold to balance in place at the current hip position.

Usage:
    python vis_scenario1.py
"""
import math
import sys
import time
import os
import multiprocessing as mp
import queue as _queue

import mujoco
import mujoco.viewer
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sim_config import (
    ROBOT, Q_NOM, WHEEL_R,
    HIP_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT,
    LEG_K_S, LEG_B_S,
    CTRL_STEPS,
    PITCH_STEP_RAD,
)
from physics import build_xml, build_assets, get_equilibrium_pitch
from scenarios import (
    init_sim, lqr_torque,
    FALL_THRESHOLD, SETTLE_THRESHOLD, SETTLE_WINDOW, BALANCE_DURATION,
)

TELEMETRY_HZ = 50   # how often to push data to the live plot


# ---------------------------------------------------------------------------
# Live matplotlib process
# ---------------------------------------------------------------------------
def _plot_process(q: mp.Queue, duration: float, pitch_ff_nominal: float):
    """Runs in a separate process — receives telemetry and updates live plot."""
    import matplotlib
    matplotlib.use("TkAgg")        # explicit backend avoids hangs on Windows
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.ion()
    fig = plt.figure(figsize=(11, 6))
    fig.suptitle(
        "1_LQR_pitch_step  —  +5° perturbation, VelocityPI OFF",
        fontsize=11, fontweight='bold')
    gs = gridspec.GridSpec(2, 1, hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Pitch panel
    ax1.set_xlim(0, duration)
    ax1.set_ylabel("Pitch (°)")
    ax1.set_xlabel("Time (s)")
    ax1.set_title("Pitch  —  dashed = equilibrium reference", fontsize=9)
    ax1.axhline(math.degrees(pitch_ff_nominal), color='k', linewidth=1.0,
                linestyle='--', label=f'pitch_ff = {math.degrees(pitch_ff_nominal):.2f}°')
    ax1.axhline(math.degrees(pitch_ff_nominal) + SETTLE_THRESHOLD, color='gray',
                linewidth=0.7, linestyle=':', label=f'±{SETTLE_THRESHOLD}° band')
    ax1.axhline(math.degrees(pitch_ff_nominal) - SETTLE_THRESHOLD, color='gray',
                linewidth=0.7, linestyle=':')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    line_pitch, = ax1.plot([], [], color='royalblue', linewidth=1.5, label='pitch')
    line_ff,    = ax1.plot([], [], color='green', linewidth=0.8, linestyle='--',
                           label='pitch_ff (live)')

    # Torque panel
    ax2.set_xlim(0, duration)
    ax2.set_ylim(-WHEEL_TORQUE_LIMIT * 1.15, WHEEL_TORQUE_LIMIT * 1.15)
    ax2.set_ylabel("Wheel torque (N·m)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Wheel torque command", fontsize=9)
    ax2.axhline( WHEEL_TORQUE_LIMIT, color='gray', linewidth=0.7, linestyle=':',
                 label=f'±{WHEEL_TORQUE_LIMIT} N·m limit')
    ax2.axhline(-WHEEL_TORQUE_LIMIT, color='gray', linewidth=0.7, linestyle=':')
    ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    line_tau, = ax2.plot([], [], color='tomato', linewidth=1.5)

    t_buf     = []
    pitch_buf = []
    ff_buf    = []
    tau_buf   = []

    settle_vline = None

    while True:
        # Drain all pending items from queue
        updated = False
        try:
            while True:
                item = q.get(timeout=0.03)
                if item is None:          # sentinel — sim finished
                    plt.ioff()
                    plt.show(block=True)
                    return
                t, pitch_deg, ff_deg, tau, settled_t = item
                t_buf.append(t)
                pitch_buf.append(pitch_deg)
                ff_buf.append(ff_deg)
                tau_buf.append(tau)
                if settled_t is not None and settle_vline is None:
                    settle_vline = ax1.axvline(settled_t, color='green',
                                               linewidth=1.2, linestyle='--',
                                               label=f'settled t={settled_t:.3f}s')
                    ax1.legend(fontsize=8, loc='upper right')
                updated = True
        except _queue.Empty:
            pass

        if updated and len(t_buf) > 1:
            ta = np.array(t_buf)
            line_pitch.set_data(ta, np.array(pitch_buf))
            line_ff.set_data(ta, np.array(ff_buf))
            line_tau.set_data(ta, np.array(tau_buf))
            ax1.relim(); ax1.autoscale_view()
            ax2.relim()
            ax2.set_ylim(-WHEEL_TORQUE_LIMIT * 1.15, WHEEL_TORQUE_LIMIT * 1.15)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        plt.pause(0.01)


# ---------------------------------------------------------------------------
# Main — simulation + viewer
# ---------------------------------------------------------------------------
def main():
    xml    = build_xml(ROBOT)
    assets = build_assets()
    model  = mujoco.MjModel.from_xml_string(xml, assets)
    data   = mujoco.MjData(model)
    init_sim(model, data)

    def _jqp(n):  return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def _jdof(n): return model.jnt_dofadr [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def _act(n):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    def _bid(n):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

    # Apply +5° pitch perturbation
    s_root = _jqp("root_free")
    pitch_eq = get_equilibrium_pitch(ROBOT, Q_NOM)
    theta    = pitch_eq + PITCH_STEP_RAD
    data.qpos[s_root + 3] = math.cos(theta / 2.0)
    data.qpos[s_root + 4] = 0.0
    data.qpos[s_root + 5] = math.sin(theta / 2.0)
    data.qpos[s_root + 6] = 0.0
    mujoco.mj_forward(model, data)

    d_root      = _jdof("root_free");      d_pitch = d_root + 4
    s_hip_L     = _jqp("hip_L");          s_hip_R = _jqp("hip_R")
    d_hip_L     = _jdof("hip_L");         d_hip_R = _jdof("hip_R")
    d_whl_L     = _jdof("wheel_spin_L");  d_whl_R = _jdof("wheel_spin_R")
    act_hip_L   = _act("hip_act_L");      act_hip_R   = _act("hip_act_R")
    act_wheel_L = _act("wheel_act_L");    act_wheel_R = _act("wheel_act_R")
    box_bid     = _bid("box")

    dt            = model.opt.timestep * CTRL_STEPS
    tel_every     = max(1, round(1.0 / (dt * TELEMETRY_HZ)))

    q             = mp.Queue(maxsize=2000)
    plot_proc     = mp.Process(target=_plot_process,
                               args=(q, BALANCE_DURATION, pitch_eq), daemon=True)
    plot_proc.start()

    print(f"1_LQR_pitch_step  — perturbation +{math.degrees(PITCH_STEP_RAD):.1f}°")
    print(f"Equilibrium pitch = {math.degrees(pitch_eq):.3f}°")

    settled      = False
    settle_start = None
    settle_time  = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 1.4
        viewer.cam.azimuth   = 135.0
        viewer.cam.elevation = -15.0

        step       = 0
        wall_start = time.time()

        while data.time < BALANCE_DURATION and viewer.is_running():
            if step % CTRL_STEPS == 0:
                q_quat     = data.xquat[box_bid]
                pitch_true = math.asin(max(-1.0, min(1.0,
                    2.0 * (q_quat[0] * q_quat[2] - q_quat[3] * q_quat[1]))))
                pitch_rate = data.qvel[d_pitch]
                wheel_vel  = (data.qvel[d_whl_L] + data.qvel[d_whl_R]) / 2.0
                hip_q_avg  = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0
                pitch_ff   = get_equilibrium_pitch(ROBOT, hip_q_avg)

                tau_wheel = lqr_torque(pitch_true, pitch_rate, wheel_vel, hip_q_avg,
                                       v_ref=0.0, theta_ref=0.0)
                data.ctrl[act_wheel_L] = tau_wheel
                data.ctrl[act_wheel_R] = tau_wheel

                for s_hip, d_hip, act_hip in [
                    (s_hip_L, d_hip_L, act_hip_L),
                    (s_hip_R, d_hip_R, act_hip_R),
                ]:
                    q_hip  = data.qpos[s_hip]
                    dq_hip = data.qvel[d_hip]
                    data.ctrl[act_hip] = np.clip(
                        -(LEG_K_S * (q_hip - Q_NOM) + LEG_B_S * dq_hip),
                        -HIP_IMPEDANCE_TORQUE_LIMIT, HIP_IMPEDANCE_TORQUE_LIMIT)

                # Settle detection
                pitch_err_deg = abs(math.degrees(pitch_true - pitch_ff))
                if not settled:
                    if pitch_err_deg < SETTLE_THRESHOLD:
                        if settle_start is None:
                            settle_start = data.time
                        elif data.time - settle_start >= SETTLE_WINDOW:
                            settle_time = settle_start
                            settled = True
                            print(f"  Settled at t = {settle_time:.3f} s")
                    else:
                        settle_start = None

                # Push telemetry at reduced rate
                if step % (CTRL_STEPS * tel_every) == 0:
                    try:
                        q.put_nowait((
                            data.time,
                            math.degrees(pitch_true),
                            math.degrees(pitch_ff),
                            tau_wheel,
                            settle_time if (settled and settle_time is not None
                                            and step // (CTRL_STEPS * tel_every) ==
                                            round(settle_time / (dt * tel_every))) else None,
                        ))
                    except Exception:
                        pass

                if abs(pitch_true) > FALL_THRESHOLD:
                    print(f"  FELL at t = {data.time:.2f} s")
                    break

            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1

            # Real-time pacing
            ahead = data.time - (time.time() - wall_start)
            if ahead > 0:
                time.sleep(ahead)

    q.put(None)   # sentinel — trigger final static plot
    plot_proc.join()


if __name__ == "__main__":
    mp.freeze_support()
    main()
