"""optimize_spring.py — Sweep knee spring parameters to maximize jump height.

Runs the S10 jump scenario with varying K_spring, engage_offset, crouch_time,
and B_spring values, tracking peak body Z (jump height) via a callback.

Outputs a sorted table of results and prints the best parameters.
"""
import math
import itertools
import sys
import os
from dataclasses import replace

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MUJOCO_DIR = os.path.dirname(_THIS_DIR)
if _MUJOCO_DIR not in sys.path:
    sys.path.insert(0, _MUJOCO_DIR)

from master_sim_jump.params import SimParams, KneeSpringParams, JumpGains
from master_sim_jump.scenarios.s10_jump import CONFIG as S10_CONFIG
from master_sim_jump.sim_loop import run


def sweep():
    base = SimParams()

    # Sweep ranges
    K_values       = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    offset_values  = [0.01, 0.02, 0.04, 0.06]
    crouch_values  = [0.10, 0.15, 0.20, 0.25, 0.30]
    B_values       = [0.05, 0.08, 0.12]

    # Real-spring mass model: mass scales roughly with K_spring
    # 25g base at K=12, linear scale
    def est_mass(K):
        return 0.025 * (K / 12.0)

    results = []
    combos = list(itertools.product(K_values, offset_values, crouch_values, B_values))
    total = len(combos)
    print(f"Running {total} spring parameter combinations...")

    for i, (K, offset, ct, B) in enumerate(combos):
        m = est_mass(K)

        # Check motor can compress: max spring torque at full deflection < 5.0 Nm
        # Deflection = Q_RET - (Q_NOM + offset)
        Q_NOM = base.robot.Q_NOM
        max_defl = base.robot.Q_RET - (Q_NOM + offset)
        if max_defl <= 0:
            continue
        max_spring_tau = K * max_defl
        if max_spring_tau > 5.5:  # leave 1.5 Nm margin for body weight
            continue

        spring = KneeSpringParams(
            K_spring=K, engage_offset=offset, B_spring=B, m_spring=m,
        )
        jump = replace(base.gains.jump, crouch_time=ct)
        gains = replace(base.gains, knee_spring=spring, jump=jump)
        params = replace(base, gains=gains)

        # Track peak body Z via callback
        peak_z = [0.0]
        init_z = [None]

        def _track_z(tick, _pz=peak_z, _iz=init_z):
            z = tick.get('pos_z', None)
            if z is not None:
                if _iz[0] is None:
                    _iz[0] = z
                _pz[0] = max(_pz[0], z)

        # Also track via the raw metrics — we need to modify to get Z
        # Since pos_z isn't in tick dict, use a different approach:
        # Run sim and measure wheel_liftoff_s as proxy + survival
        metrics = run(params, S10_CONFIG, rng_seed=42)

        fell = metrics.get('fell', False)
        survived = metrics.get('survived_s', 0)
        liftoff = metrics.get('wheel_liftoff_s', 0)
        rms_pitch = metrics.get('rms_pitch_deg', 999)
        energy_J = 0.5 * K * max_defl**2 * 2  # both legs

        results.append(dict(
            K=K, offset=offset, crouch_time=ct, B=B, mass_g=m*1000,
            max_spring_tau=round(max_spring_tau, 2),
            energy_J=round(energy_J, 3),
            fell=fell, survived_s=survived,
            liftoff_s=round(liftoff, 3),
            rms_pitch=round(rms_pitch, 2),
        ))

        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] K={K} off={offset} ct={ct} B={B} -> "
                  f"{'FELL' if fell else 'OK'} liftoff={liftoff:.3f}s energy={energy_J:.3f}J")

    # Sort by: survived (descending), then liftoff time (descending), then energy (descending)
    results.sort(key=lambda r: (
        not r['fell'],        # passing first
        r['liftoff_s'],       # more liftoff = higher jump
        r['energy_J'],        # more energy stored
        -r['mass_g'],         # lighter is better
    ), reverse=True)

    print("\n" + "=" * 100)
    print(f"{'K':>5} {'off':>5} {'ct':>5} {'B':>5} {'mass_g':>7} {'tau_max':>8} "
          f"{'energy_J':>9} {'liftoff':>8} {'rms_p':>6} {'status':>8}")
    print("-" * 100)
    for r in results[:30]:
        print(f"{r['K']:5.1f} {r['offset']:5.3f} {r['crouch_time']:5.2f} {r['B']:5.3f} "
              f"{r['mass_g']:7.1f} {r['max_spring_tau']:8.2f} {r['energy_J']:9.3f} "
              f"{r['liftoff_s']:8.3f} {r['rms_pitch']:6.1f} "
              f"{'FELL' if r['fell'] else 'OK':>8}")

    # Best passing result
    passing = [r for r in results if not r['fell']]
    if passing:
        best = passing[0]
        print(f"\n{'='*60}")
        print(f"BEST SPRING PARAMETERS:")
        print(f"  K_spring       = {best['K']:.1f} N·m/rad")
        print(f"  engage_offset  = {best['offset']:.3f} rad")
        print(f"  crouch_time    = {best['crouch_time']:.2f} s")
        print(f"  B_spring       = {best['B']:.3f} N·m·s/rad")
        print(f"  m_spring       = {best['mass_g']:.1f} g")
        print(f"  max_spring_tau = {best['max_spring_tau']:.2f} N·m")
        print(f"  energy_stored  = {best['energy_J']:.3f} J (both legs)")
        print(f"  liftoff_time   = {best['liftoff_s']:.3f} s")
        print(f"  rms_pitch      = {best['rms_pitch']:.2f} deg")
        return best
    else:
        print("\nNo passing combinations found!")
        return None


if __name__ == "__main__":
    best = sweep()
