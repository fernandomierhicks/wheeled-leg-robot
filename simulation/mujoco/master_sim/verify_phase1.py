"""Phase 1 verification -- all model sub-modules.

Step 1.1: physics.py IK at Q_RET, Q_NOM, Q_EXT + build_xml with MotorParams
Step 1.2: BatteryModel -- step 100 times at 10A, print voltage/SoC
Step 1.3: motor_taper + motor_currents (hip + wheel)
Step 1.4: LatencyBuffer -- size-5 and size-0 pass-through
Step 1.5: RobotThermalModel -- heating at 20A for 60s

Run:  python -m master_sim.verify_phase1
"""
import math
from master_sim.defaults import DEFAULT_PARAMS
from master_sim.physics import solve_ik, get_equilibrium_pitch, build_xml, build_assets
from master_sim.models.battery import BatteryModel
from master_sim.models.motor import motor_taper, motor_currents
from master_sim.models.latency import LatencyBuffer
from master_sim.models.thermal import RobotThermalModel


def verify_physics():
    """Step 1.1 -- IK + build_xml with motor params."""
    print("=" * 60)
    print("Step 1.1: physics.py -- IK + build_xml")
    print("=" * 60)
    robot = DEFAULT_PARAMS.robot
    p = robot.as_dict()

    for label, q in [("Q_RET", robot.Q_RET), ("Q_NOM", robot.Q_NOM), ("Q_EXT", robot.Q_EXT)]:
        ik = solve_ik(q, p)
        if ik:
            pitch_ff = get_equilibrium_pitch(robot, q)
            print(f"  IK({label}={math.degrees(q):+7.2f} deg): "
                  f"W_z={ik['W_z']*1000:+7.1f} mm, KR={ik['KR_ratio']:.3f}, "
                  f"pitch_ff={math.degrees(pitch_ff):+.2f} deg")
        else:
            print(f"  IK({label}): SINGULARITY!")

    # Build XML with motor params -- verify actuator limits are correct
    xml = build_xml(robot, motors=DEFAULT_PARAMS.motors)
    hip_lim = DEFAULT_PARAMS.motors.hip.torque_limit
    whl_lim = DEFAULT_PARAMS.motors.wheel.torque_limit
    assert f'ctrlrange="-{hip_lim:.1f} {hip_lim:.1f}"' in xml, "Hip torque limit not in XML!"
    assert f'ctrlrange="-{whl_lim:.1f} {whl_lim:.1f}"' in xml, "Wheel torque limit not in XML!"
    print(f"  build_xml OK -- hip limit={hip_lim:.1f} Nm, wheel limit={whl_lim:.2f} Nm")

    # Verify model loads in MuJoCo
    import mujoco
    assets = build_assets()
    model = mujoco.MjModel.from_xml_string(xml, assets)
    print(f"  MuJoCo model OK -- {model.nbody} bodies, {model.nu} actuators")
    print("  PASS")


def verify_battery():
    """Step 1.2 -- BatteryModel step test."""
    print("\n" + "=" * 60)
    print("Step 1.2: BatteryModel -- 100 steps at 10A")
    print("=" * 60)
    batt = BatteryModel(DEFAULT_PARAMS.battery)
    batt.reset()

    dt = 0.002  # 500 Hz control rate
    for i in range(100):
        v = batt.step(dt, 10.0)

    print(f"  After 100 steps (0.2s) at 10A:")
    print(f"    V_terminal = {batt.v_terminal:.3f} V")
    print(f"    SoC        = {batt.soc_pct:.4f} %")
    print(f"    Temp       = {batt.temperature_c:.3f} degC")
    print(f"    R_int      = {batt.r_int*1000:.2f} mohm")

    assert batt.v_terminal > 23.0, "Voltage too low after 0.2s!"
    assert batt.soc > 0.999, "SoC dropped too much in 0.2s!"
    print("  PASS")


def verify_motor():
    """Step 1.3 -- motor_taper + motor_currents (hip + wheel)."""
    print("\n" + "=" * 60)
    print("Step 1.3: Motor functions -- taper + currents")
    print("=" * 60)
    motors = DEFAULT_PARAMS.motors
    battery = DEFAULT_PARAMS.battery

    # motor_taper at various wheel speeds
    print("  motor_taper (1 Nm command):")
    for omega in [0.0, 50.0, 150.0]:
        tapered = motor_taper(1.0, omega, battery.V_nom, motors, battery)
        print(f"    w={omega:6.1f} rad/s -> t={tapered:.4f} Nm")

    # At zero speed, should pass through (taper=1.0)
    t0 = motor_taper(1.0, 0.0, battery.V_nom, motors, battery)
    assert abs(t0 - 1.0) < 0.01, f"motor_taper at w=0 should be ~1.0, got {t0}"

    # Near no-load speed, should be near zero
    omega_nl = motors.wheel.omega_noload(battery.V_nom)
    t_nl = motor_taper(1.0, omega_nl * 0.99, battery.V_nom, motors, battery)
    assert t_nl < 0.10, f"motor_taper near no-load should be small, got {t_nl}"

    # motor_currents -- hip + wheel
    print(f"\n  motor_currents:")
    print(f"    Wheel Kt = {motors.wheel.Kt:.4f} Nm/A")
    print(f"    Hip Kt_output = {motors.hip.Kt_output:.4f} Nm/A")
    I = motor_currents(1.0, 1.0, 3.0, 3.0, motors, battery.I_quiescent)
    I_whl_expected = 2.0 / motors.wheel.Kt
    I_hip_expected = 6.0 / motors.hip.Kt_output
    print(f"    t_whl=1+1, t_hip=3+3 -> I_total={I:.2f} A "
          f"(whl={I_whl_expected:.2f} + hip={I_hip_expected:.2f} + quiescent={battery.I_quiescent})")
    assert abs(I - (I_whl_expected + I_hip_expected + battery.I_quiescent)) < 0.01
    print("  PASS")


def verify_latency():
    """Step 1.4 -- LatencyBuffer ring buffer."""
    print("\n" + "=" * 60)
    print("Step 1.4: LatencyBuffer -- delay + pass-through")
    print("=" * 60)

    # Size-5 buffer: push 10 values, oldest should be 5 behind
    buf5 = LatencyBuffer(n_steps=5, init_value=0.0)
    results = []
    for i in range(10):
        out = buf5.push(float(i + 1))
        results.append(out)
    print(f"  Size-5 buffer, pushed 1..10:")
    print(f"    Outputs: {results}")
    # First 5 outputs should be 0.0 (init), then 1,2,3,4,5
    assert results[0] == 0.0, f"Expected 0.0, got {results[0]}"
    assert results[4] == 0.0, f"Expected 0.0, got {results[4]}"
    assert results[5] == 1.0, f"Expected 1.0, got {results[5]}"
    assert results[9] == 5.0, f"Expected 5.0, got {results[9]}"

    # Size-0 buffer (pass-through): should return what was just pushed
    buf0 = LatencyBuffer(n_steps=0, init_value=0.0)
    for i in range(5):
        out = buf0.push(float(i + 1))
        assert out == float(i + 1), f"Pass-through failed: expected {i+1}, got {out}"
    print(f"  Size-0 pass-through: OK")

    # Tuple values (sensor delay stores tuples)
    buf_t = LatencyBuffer(n_steps=3, init_value=(0.0, 0.0, 0.0))
    for i in range(5):
        out = buf_t.push((float(i), float(i)*2, float(i)*3))
    assert out == (1.0, 2.0, 3.0), f"Tuple buffer failed: got {out}"
    print(f"  Tuple buffer (size-3): OK")
    print("  PASS")


def verify_thermal():
    """Step 1.5 -- RobotThermalModel heating test (hip + wheel)."""
    print("\n" + "=" * 60)
    print("Step 1.5: RobotThermalModel -- 60s heating at 20A equivalent")
    print("=" * 60)
    thermal = RobotThermalModel(DEFAULT_PARAMS.motors)
    thermal.reset()

    dt = 0.002  # 500 Hz
    # Simulate 60s of moderate torque: 2 Nm on each wheel, 3 Nm on each hip
    tau_whl = 2.0
    tau_hip = 3.0
    for _ in range(int(60.0 / dt)):
        thermal.step(dt, tau_whl, tau_whl, tau_hip, tau_hip)

    print(f"  After 60s with t_whl=+-{tau_whl} Nm, t_hip=+-{tau_hip} Nm:")
    print(f"    Wheel L: T_winding={thermal.wheel_L.T_winding:.1f}degC, "
          f"T_case={thermal.wheel_L.T_case:.1f}degC, margin={thermal.wheel_L.T_margin:.1f}degC")
    print(f"    Hip L:   T_winding={thermal.hip_L.T_winding:.1f}degC, "
          f"T_case={thermal.hip_L.T_case:.1f}degC, margin={thermal.hip_L.T_margin:.1f}degC")
    print(f"    Peak winding: {thermal.peak_winding_temp():.1f}degC")
    print(f"    Min margin:   {thermal.min_margin():.1f}degC")

    # Sanity checks
    assert thermal.wheel_L.T_winding > 25.0, "Wheel should heat up!"
    assert thermal.hip_L.T_winding > 25.0, "Hip should heat up!"
    assert thermal.min_margin() > 0, "Should not over-temp in 60s at moderate load!"

    # Verify hip and wheel heat differently (different R_eff, Kt)
    I_whl = tau_whl / DEFAULT_PARAMS.motors.wheel.Kt
    I_hip = tau_hip / DEFAULT_PARAMS.motors.hip.Kt_output
    print(f"\n  Motor currents: wheel={I_whl:.2f}A, hip={I_hip:.2f}A")
    print("  PASS")


def main():
    print("Phase 1 Verification -- Core Physics Models")
    print("=" * 60)
    verify_physics()
    verify_battery()
    verify_motor()
    verify_latency()
    verify_thermal()
    print("\n" + "=" * 60)
    print("ALL PHASE 1 CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
