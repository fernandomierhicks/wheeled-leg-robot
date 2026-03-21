"""Phase 0 verification — Steps 0.4 and 0.5.

Step 0.4: Load MJCF model with body welded in air, open MuJoCo viewer.
Step 0.5: Sweep hip joint Q_RET -> Q_NOM -> Q_EXT over 3 seconds.

Run:  python -m master_sim.verify_phase0
"""
import time
import math
import numpy as np
import mujoco
import mujoco.viewer

from master_sim.defaults import DEFAULT_PARAMS
from master_sim.physics import build_xml, build_assets, solve_ik, init_sim


def main():
    robot = DEFAULT_PARAMS.robot

    # Build model with body welded (fixed in air)
    xml = build_xml(robot, weld_body=True)
    assets = build_assets()
    model = mujoco.MjModel.from_xml_string(xml, assets)
    data = mujoco.MjData(model)

    print("Step 0.4: Model loaded successfully!")
    print(f"  Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")
    print(f"  Q_RET={robot.Q_RET:.4f}, Q_NOM={robot.Q_NOM:.4f}, Q_EXT={robot.Q_EXT:.4f}")

    # Initialise 4-bar closure anchors + joint angles from IK at Q_NOM
    init_sim(model, data, robot, q_hip_init=robot.Q_NOM)
    print("  init_sim OK — 4-bar anchors set, joints at Q_NOM via IK")

    # Verify IK at key positions
    p = robot.as_dict()
    for label, q in [("Q_RET", robot.Q_RET), ("Q_NOM", robot.Q_NOM), ("Q_EXT", robot.Q_EXT)]:
        ik = solve_ik(q, p)
        if ik:
            print(f"  IK({label}={math.degrees(q):.1f} deg): W_z={ik['W_z']*1000:.1f} mm, KR={ik['KR_ratio']:.3f}")
        else:
            print(f"  IK({label}): SINGULARITY!")

    # Actuator + joint indices
    act_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act_L")
    act_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act_R")
    jnt_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_L")
    jnt_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_R")
    qa_L = model.jnt_qposadr[jnt_hip_L]
    qa_R = model.jnt_qposadr[jnt_hip_R]
    da_L = model.jnt_dofadr[jnt_hip_L]
    da_R = model.jnt_dofadr[jnt_hip_R]

    # PD gains — same as HIP_POSITION_KP/KD in sim_config
    KP = 50.0
    KD = 3.0

    print("\nStep 0.5: Starting hip sweep (loops every 6s)...")
    print("  0-2s: Q_NOM -> Q_RET  (retract)")
    print("  2-4s: Q_RET -> Q_EXT  (extend)")
    print("  4-6s: Q_EXT -> Q_NOM  (return)")
    print("\nOpening MuJoCo viewer... Close window when done.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t_start = time.time()
        PERIOD = 6.0

        while viewer.is_running():
            t_sweep = (time.time() - t_start) % PERIOD

            # Target hip angle (smooth linear ramp between keyframes)
            if t_sweep < 2.0:
                frac = t_sweep / 2.0
                q_target = robot.Q_NOM + frac * (robot.Q_RET - robot.Q_NOM)
            elif t_sweep < 4.0:
                frac = (t_sweep - 2.0) / 2.0
                q_target = robot.Q_RET + frac * (robot.Q_EXT - robot.Q_RET)
            else:
                frac = (t_sweep - 4.0) / 2.0
                q_target = robot.Q_EXT + frac * (robot.Q_NOM - robot.Q_EXT)

            # PD torque on hip actuators — MuJoCo constraint solver closes the 4-bar
            for qa, da, act in [(qa_L, da_L, act_hip_L), (qa_R, da_R, act_hip_R)]:
                err = q_target - data.qpos[qa]
                dq = data.qvel[da]
                data.ctrl[act] = np.clip(KP * err - KD * dq, -7.0, 7.0)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)

    print("Viewer closed. Phase 0 verification complete.")


if __name__ == "__main__":
    main()
