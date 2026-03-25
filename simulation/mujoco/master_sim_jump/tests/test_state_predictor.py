"""test_state_predictor.py — Unit tests for predict_state().

The predictor is a single Euler step of the linearised inverted pendulum:
    θ̈ = (g/l) * θ

Run from repo root:
    python -m pytest simulation/mujoco/master_sim/tests/test_state_predictor.py -v

No MuJoCo install required.
"""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from master_sim_jump.sim_loop import predict_state

G = 9.81
L = 0.20   # representative effective leg length [m]
ALPHA = G / L   # = 49.05 rad/s²  (unstable pole)


# ── Test 1: equilibrium stays at zero ────────────────────────────────────────

def test_equilibrium_is_fixed_point():
    p, pr = predict_state(0.0, 0.0, sensor_delay_s=0.001, l_eff=L)
    assert p  == 0.0
    assert pr == 0.0


# ── Test 2: known numerical values ───────────────────────────────────────────

def test_pure_pitch_no_rate():
    """θ=0.1 rad, θ̇=0 → only rate should grow (acceleration term)."""
    dt = 0.001
    theta0, omega0 = 0.1, 0.0
    p, pr = predict_state(theta0, omega0, dt, L)

    assert math.isclose(p,  theta0 + omega0 * dt,         rel_tol=1e-12)
    assert math.isclose(pr, omega0 + ALPHA * theta0 * dt, rel_tol=1e-12)


def test_pure_rate_no_pitch():
    """θ=0, θ̇=0.5 rad/s → only pitch should change (velocity term)."""
    dt = 0.001
    theta0, omega0 = 0.0, 0.5
    p, pr = predict_state(theta0, omega0, dt, L)

    assert math.isclose(p,  omega0 * dt, rel_tol=1e-12)
    assert math.isclose(pr, omega0,      rel_tol=1e-12)   # acceleration = 0 when θ=0


def test_general_case():
    """Both terms active; check against hand-computed values."""
    dt = 0.002
    theta0, omega0 = 0.05, 0.3
    p, pr = predict_state(theta0, omega0, dt, L)

    expected_p  = theta0 + omega0 * dt
    expected_pr = omega0 + ALPHA * theta0 * dt
    assert math.isclose(p,  expected_p,  rel_tol=1e-12)
    assert math.isclose(pr, expected_pr, rel_tol=1e-12)


# ── Test 3: predictor reduces lag compared to raw delayed reading ─────────────
#
# Simulate ground truth as θ(t) = θ₀ * cosh(√α * t)  (analytical solution for
# zero initial rate).  The delayed reading is the state at t=0; the predictor
# should bring it closer to the true state at t = sensor_delay_s.

def _true_state(theta0, omega0, t, alpha=ALPHA):
    """Analytical solution for θ̈ = α θ with IC (θ₀, ω₀):
       θ(t)  = θ₀ cosh(√α t) + (ω₀/√α) sinh(√α t)
       θ̇(t) = θ₀ √α sinh(√α t) + ω₀ cosh(√α t)
    """
    sq = math.sqrt(alpha)
    st, ct = math.sinh(sq * t), math.cosh(sq * t)
    return theta0 * ct + (omega0 / sq) * st, theta0 * sq * st + omega0 * ct


def test_predictor_reduces_pitch_error():
    """With nonzero rate, pitch prediction should track the true state better."""
    sd = 0.005       # 5 ms delay
    theta0, omega0 = 0.1, 0.5   # falling with positive rate

    theta_true, _ = _true_state(theta0, omega0, sd)

    raw_error  = abs(theta_true - theta0)
    pitch_pred, _ = predict_state(theta0, omega0, sd, L)
    pred_error = abs(theta_true - pitch_pred)

    assert pred_error < raw_error, (
        f"Predictor made pitch error worse: raw={raw_error:.6f}, pred={pred_error:.6f}")


def test_predictor_reduces_pitch_rate_error():
    """With nonzero pitch, rate prediction should track the true state better."""
    sd = 0.005
    theta0, omega0 = 0.1, 0.0   # pitch but no initial rate

    _, rate_true = _true_state(theta0, omega0, sd)

    raw_rate_error = abs(rate_true - omega0)
    _, rate_pred = predict_state(theta0, omega0, sd, L)
    pred_rate_error = abs(rate_true - rate_pred)

    assert pred_rate_error < raw_rate_error, (
        f"Predictor made rate error worse: raw={raw_rate_error:.6f}, pred={pred_rate_error:.6f}")


# ── Test 4: longer l_eff → smaller acceleration correction ───────────────────

def test_longer_leg_smaller_correction():
    """Higher l_eff → lower α → smaller rate correction for same pitch."""
    dt = 0.001
    theta0, omega0 = 0.1, 0.0

    _, pr_short = predict_state(theta0, omega0, dt, l_eff=0.10)   # α = 98.1
    _, pr_long  = predict_state(theta0, omega0, dt, l_eff=0.40)   # α = 24.5

    assert pr_short > pr_long, "Shorter leg (higher α) should give larger rate correction"
