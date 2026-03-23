Here's what I found after a deep review of the delay/predictor pipeline:

Verdict: No smoking-gun code bug found
The buffer indexing, predictor math, torque history timing, and discretization are all correct. I verified the full chain:

LatencyBuffer — delay is exactly n_steps ticks, ring-buffer logic is sound
Predictor formula — x̂(t) = A_d^n · x(t−n) + Σ A_d^(n−1−k) · B_d · u(t−n+k) matches the ZOH discrete dynamics exactly
Torque history ordering — deque iterates oldest→newest, _BdPow[k] indices match correctly
Timing — tau_hist is read BEFORE the new torque is appended (line 394 before line 432), so the history contains [u(t−n), ..., u(t−1)] as required
One real (but small) bug found
Predictor operates on raw sensor state instead of deviation state (sim_loop.py:392):


x_del = np.array([_pitch_d, _pitch_rate_d, _wheel_vel_d])  # RAW values
x_pred = self._Ad_n @ x_del  # A_d describes DEVIATION dynamics
The A/B matrices are linearized around equilibrium (pitch = pitch_ff, pitch_rate = 0, wheel_vel = 0). The predictor should subtract pitch_ff before propagating and add it back after. The error is (A_d^n − I) · [pitch_ff, 0, 0], which injects a spurious pitch_rate of ~n · α · dt · pitch_ff. Numerically this is ~0.003 rad/s for n=2 — too small to explain the cliff alone.

Likely root cause: not a code bug but a stability margin issue
The predictor uses a fixed linear model (A_d, B_d at Q_NOM), while the actual MuJoCo plant has:

Nonlinear dynamics (sin θ, 4-bar kinematics)
Motor taper (predictor stores tau_sym but plant sees motor_taper(tau_sym, ω, ...))
Hip-angle-dependent dynamics (A/B change with leg position)
These mismatches mean the predictor doesn't perfectly cancel the delay. For n=1, the residual error is tolerable. For n=2, the prediction horizon doubles and so does every model-mismatch error. If your gains are already near the stability boundary (tuned aggressively for 0ms), even a 2× increase in predictor error can tip the system over — producing the sharp cliff you see.

Suggested next steps
Quick diagnostic: temporarily disable the predictor entirely (set n_sens = 0 in the if block) and run with 2ms delay. If the robot still balances (even poorly), the predictor is making things worse, not better — confirming model mismatch is the issue.

Fix the deviation-state bug — small but free to fix. Want me to do it?

If the cliff persists: the real fix is either re-tuning gains with delay in the loop, or making the predictor adaptive (update A_d/B_d with current hip angle each tick).