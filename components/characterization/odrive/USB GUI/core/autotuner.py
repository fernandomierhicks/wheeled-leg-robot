"""autotuner.py — Relay-feedback autotuner for ODrive velocity & position loops.

Implements the Åström–Hägglund relay method (Automatica, 1984) to identify
the ultimate gain (Ku) and ultimate period (Tu) of the velocity plant, then
computes PID gains using classical tuning rules.

The relay test works as follows:
  1. Motor is put in torque-control mode at a target velocity of 0.
  2. A bang-bang relay applies +d or −d torque depending on whether
     vel_estimate is below or above the setpoint.
  3. The resulting limit-cycle oscillation is measured for amplitude (a)
     and period (Tu).
  4. Ultimate gain: Ku = 4d / (π · a)
  5. Gains are computed from Ku/Tu using the selected tuning rule.

Reference:
  Åström, K.J. & Hägglund, T. (1984). "Automatic tuning of simple
  regulators with specifications on phase and amplitude margins."
  Automatica, 20(5), 645–651.
"""

import logging
import math
import time

from PySide6.QtCore import QThread, Signal

log = logging.getLogger("odrive_gui")

# ODrive 0.5.6 enums
AXIS_STATE_IDLE = 1
AXIS_STATE_CLOSED_LOOP = 8
CONTROL_MODE_TORQUE = 1
INPUT_MODE_PASSTHROUGH = 1

# Minimum number of full cycles before we compute gains
MIN_CYCLES = 4
# Maximum number of cycles to collect (stop after this many)
MAX_CYCLES = 20


# ── Tuning rules ─────────────────────────────────────────────────────────────

def ziegler_nichols_pi(Ku: float, Tu: float) -> dict:
    """Classic Ziegler–Nichols PI rule."""
    return {
        "vel_gain": 0.45 * Ku,
        "vel_integrator_gain": 0.54 * Ku / Tu,
        "rule": "Ziegler-Nichols PI",
    }


def ziegler_nichols_pid(Ku: float, Tu: float) -> dict:
    """Classic Ziegler–Nichols PID rule."""
    return {
        "vel_gain": 0.60 * Ku,
        "vel_integrator_gain": 1.2 * Ku / Tu,
        "rule": "Ziegler-Nichols PID",
    }


def tyreus_luyben_pi(Ku: float, Tu: float) -> dict:
    """Tyreus–Luyben PI — less aggressive, less overshoot."""
    return {
        "vel_gain": Ku / 3.2,
        "vel_integrator_gain": Ku / (3.2 * 2.2 * Tu),
        "rule": "Tyreus-Luyben PI",
    }


TUNING_RULES = {
    "ZN-PI": ziegler_nichols_pi,
    "ZN-PID": ziegler_nichols_pid,
    "TL-PI": tyreus_luyben_pi,
}


# ── Relay test worker ────────────────────────────────────────────────────────

class RelayTestWorker(QThread):
    """Run relay-feedback test in a background thread.

    Signals:
        progress(str)           — status text for the UI
        sample(float, float)    — (t_relative, vel_estimate) for live plotting
        relay_output(float, float) — (t_relative, torque_cmd) for live plotting
        finished(dict)          — results dict with Ku, Tu, recommended gains
        error(str)              — error message if test fails
    """
    progress = Signal(str)
    sample = Signal(float, float)
    relay_output = Signal(float, float)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, manager, axis_idx: int, relay_amplitude: float,
                 vel_target: float = 0.0, timeout: float = 15.0,
                 hysteresis: float = 0.0):
        super().__init__()
        self._mgr = manager
        self._axis_idx = axis_idx
        self._relay_amp = abs(relay_amplitude)
        self._vel_target = vel_target
        self._timeout = timeout
        self._hysteresis = hysteresis  # deadband around zero-crossing
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        axis = self._mgr.get_axis(self._axis_idx)
        if axis is None:
            self.error.emit("Not connected or invalid axis")
            return

        # ── Pre-checks ───────────────────────────────────────────────────
        try:
            state = getattr(axis, "current_state", 0)
            ax_err = getattr(axis, "error", 0)
            mot_err = getattr(axis.motor, "error", 0)
            enc_err = getattr(axis.encoder, "error", 0)
        except Exception as e:
            self.error.emit(f"Cannot read axis state: {e}")
            return

        if ax_err or mot_err or enc_err:
            self.error.emit(
                f"Axis has errors (ax=0x{ax_err:X} mot=0x{mot_err:X} "
                f"enc=0x{enc_err:X}). Clear errors first."
            )
            return

        # ── Enter torque mode ────────────────────────────────────────────
        try:
            axis.controller.config.control_mode = CONTROL_MODE_TORQUE
            axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
            axis.controller.input_torque = 0.0
            if state != AXIS_STATE_CLOSED_LOOP:
                axis.requested_state = AXIS_STATE_CLOSED_LOOP
                time.sleep(0.1)
            state = getattr(axis, "current_state", 0)
            if state != AXIS_STATE_CLOSED_LOOP:
                self.error.emit(f"Failed to enter closed-loop (state={state})")
                return
        except Exception as e:
            self.error.emit(f"Setup error: {e}")
            return

        self.progress.emit("Relay running — waiting for oscillation…")
        log.info("Autotuner: relay test started, amp=%.4f Nm, target=%.4f t/s",
                 self._relay_amp, self._vel_target)

        # ── Relay loop ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        zero_crossings = []       # timestamps of upward zero-crossings
        peak_highs = []           # peak velocity above target
        peak_lows = []            # peak velocity below target
        prev_above = None
        local_max = -1e9
        local_min = 1e9
        sample_interval = 0.005   # 200 Hz sampling
        vel_limit = self._mgr.safe_get(
            f"axis{self._axis_idx}.controller.config.vel_limit", 20.0
        )

        try:
            while not self._stop_flag:
                now = time.perf_counter()
                t_rel = now - t0

                if t_rel > self._timeout:
                    self.progress.emit("Timeout — not enough oscillation detected")
                    break

                # Read velocity
                vel = float(getattr(axis.encoder, "vel_estimate", 0.0))
                err = vel - self._vel_target

                # Safety: abort if velocity exceeds limit
                if abs(vel) > vel_limit * 0.9:
                    self.error.emit(
                        f"Velocity {vel:.2f} near limit {vel_limit:.1f} — aborting"
                    )
                    axis.controller.input_torque = 0.0
                    axis.requested_state = AXIS_STATE_IDLE
                    return

                # Relay decision with hysteresis
                if err > self._hysteresis:
                    torque = -self._relay_amp
                    currently_above = True
                elif err < -self._hysteresis:
                    torque = self._relay_amp
                    currently_above = False
                else:
                    # Inside deadband — hold previous output
                    torque = -self._relay_amp if prev_above else self._relay_amp
                    currently_above = prev_above

                # Write torque
                axis.controller.input_torque = torque

                # Emit samples for live chart
                self.sample.emit(t_rel, vel)
                self.relay_output.emit(t_rel, torque)

                # Track peaks
                if currently_above:
                    local_max = max(local_max, vel)
                    if local_min < 1e8:
                        peak_lows.append(local_min)
                        local_min = 1e9
                else:
                    local_min = min(local_min, vel)
                    if local_max > -1e8:
                        peak_highs.append(local_max)
                        local_max = -1e9

                # Detect upward zero-crossing (below → above)
                if prev_above is not None and currently_above and not prev_above:
                    zero_crossings.append(now)
                    n = len(zero_crossings)
                    self.progress.emit(
                        f"Relay running — {n} cycle(s) detected…"
                    )
                    if n >= MAX_CYCLES:
                        break

                prev_above = currently_above
                time.sleep(sample_interval)

        except Exception as e:
            self.error.emit(f"Relay loop error: {e}")
            # Fall through to cleanup
        finally:
            # ── Cleanup: zero torque, go idle ────────────────────────────
            try:
                axis.controller.input_torque = 0.0
                axis.requested_state = AXIS_STATE_IDLE
            except Exception:
                pass

        if self._stop_flag:
            self.progress.emit("Autotuner stopped by user")
            return

        # ── Analyse results ──────────────────────────────────────────────
        n_crossings = len(zero_crossings)
        if n_crossings < MIN_CYCLES:
            self.error.emit(
                f"Only {n_crossings} zero-crossings detected "
                f"(need ≥{MIN_CYCLES}). Increase relay amplitude or timeout."
            )
            return

        # Period: average time between successive zero-crossings
        periods = [zero_crossings[i+1] - zero_crossings[i]
                   for i in range(n_crossings - 1)]
        Tu = sum(periods) / len(periods)

        # Amplitude: average of (peak_high − peak_low) / 2, discarding first cycle
        n_peaks = min(len(peak_highs), len(peak_lows))
        if n_peaks < 2:
            self.error.emit("Could not measure oscillation amplitude")
            return
        # Skip first peak (transient)
        amps = [(peak_highs[i] - peak_lows[i]) / 2.0
                for i in range(1, n_peaks)]
        a = sum(amps) / len(amps)

        if a < 1e-6:
            self.error.emit("Oscillation amplitude too small to measure")
            return

        # Ultimate gain: Ku = 4d / (π · a)
        Ku = (4.0 * self._relay_amp) / (math.pi * a)

        log.info("Autotuner: Tu=%.4f s, a=%.4f t/s, Ku=%.5f", Tu, a, Ku)

        # Compute gains for all rules
        results = {
            "Tu": Tu,
            "Ku": Ku,
            "amplitude": a,
            "n_cycles": n_crossings,
            "relay_amp": self._relay_amp,
            "gains": {},
        }
        for name, func in TUNING_RULES.items():
            results["gains"][name] = func(Ku, Tu)

        self.progress.emit(
            f"Done — Ku={Ku:.5f}, Tu={Tu:.4f}s, "
            f"osc amplitude={a:.4f} t/s, {n_crossings} cycles"
        )
        self.finished.emit(results)


# ── Validation step worker ───────────────────────────────────────────────────

class ValidationWorker(QThread):
    """Run a velocity square-wave test with the proposed gains and measure
    settling time, overshoot, and steady-state error.

    Signals:
        progress(str)
        sample(float, float)     — (t_rel, vel_estimate)
        setpoint(float, float)   — (t_rel, vel_setpoint) for overlay
        finished(dict)           — metrics dict
        error(str)
    """
    progress = Signal(str)
    sample = Signal(float, float)
    setpoint = Signal(float, float)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, manager, axis_idx: int,
                 vel_gain: float, vel_int_gain: float,
                 test_vel: float = 1.0, period: float = 3.0,
                 n_steps: int = 4, settle_pct: float = 0.05):
        super().__init__()
        self._mgr = manager
        self._axis_idx = axis_idx
        self._vel_gain = vel_gain
        self._vel_int_gain = vel_int_gain
        self._test_vel = test_vel
        self._period = period
        self._n_steps = n_steps
        self._settle_pct = settle_pct  # 5% settling band
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        axis = self._mgr.get_axis(self._axis_idx)
        if axis is None:
            self.error.emit("Not connected")
            return

        # Apply proposed gains
        try:
            cc = axis.controller.config
            cc.control_mode = 2  # VELOCITY
            cc.input_mode = INPUT_MODE_PASSTHROUGH
            cc.vel_gain = self._vel_gain
            cc.vel_integrator_gain = self._vel_int_gain
            axis.controller.input_vel = 0.0
            if getattr(axis, "current_state", 0) != AXIS_STATE_CLOSED_LOOP:
                axis.requested_state = AXIS_STATE_CLOSED_LOOP
                time.sleep(0.1)
        except Exception as e:
            self.error.emit(f"Setup error: {e}")
            return

        self.progress.emit("Validation: running square-wave test…")

        t0 = time.perf_counter()
        sample_dt = 0.005  # 200 Hz
        step_results = []

        try:
            for step_i in range(self._n_steps):
                if self._stop_flag:
                    break

                # Alternate between +vel and −vel
                target = self._test_vel if (step_i % 2 == 0) else -self._test_vel
                axis.controller.input_vel = target

                step_t0 = time.perf_counter()
                samples = []

                while not self._stop_flag:
                    now = time.perf_counter()
                    t_step = now - step_t0
                    t_rel = now - t0

                    if t_step >= self._period:
                        break

                    vel = float(getattr(axis.encoder, "vel_estimate", 0.0))
                    samples.append((t_step, vel))
                    self.sample.emit(t_rel, vel)
                    self.setpoint.emit(t_rel, target)

                    time.sleep(sample_dt)

                # Analyse this step
                if len(samples) > 10:
                    vels = [s[1] for s in samples]
                    times = [s[0] for s in samples]

                    # Overshoot
                    if target > 0:
                        peak = max(vels)
                        overshoot = (peak - target) / target if target != 0 else 0
                    else:
                        peak = min(vels)
                        overshoot = (target - peak) / abs(target) if target != 0 else 0

                    # Settling time (first time all subsequent samples within band)
                    band = abs(target) * self._settle_pct
                    settle_time = self._period  # default: never settled
                    for i in range(len(samples)):
                        if all(abs(vels[j] - target) < band
                               for j in range(i, min(i + 20, len(samples)))):
                            settle_time = times[i]
                            break

                    # Steady-state error (mean of last 20% of step)
                    tail = vels[int(len(vels) * 0.8):]
                    ss_error = abs(sum(tail) / len(tail) - target)

                    step_results.append({
                        "target": target,
                        "overshoot_pct": overshoot * 100,
                        "settling_time": settle_time,
                        "ss_error": ss_error,
                    })

                self.progress.emit(
                    f"Validation: step {step_i+1}/{self._n_steps} done"
                )

        except Exception as e:
            self.error.emit(f"Validation error: {e}")
        finally:
            try:
                axis.controller.input_vel = 0.0
                axis.requested_state = AXIS_STATE_IDLE
            except Exception:
                pass

        if self._stop_flag:
            self.progress.emit("Validation stopped by user")
            return

        if not step_results:
            self.error.emit("No step data collected")
            return

        # Aggregate
        avg_overshoot = sum(r["overshoot_pct"] for r in step_results) / len(step_results)
        avg_settle = sum(r["settling_time"] for r in step_results) / len(step_results)
        avg_ss_err = sum(r["ss_error"] for r in step_results) / len(step_results)

        result = {
            "steps": step_results,
            "avg_overshoot_pct": avg_overshoot,
            "avg_settling_time": avg_settle,
            "avg_ss_error": avg_ss_err,
            "vel_gain": self._vel_gain,
            "vel_integrator_gain": self._vel_int_gain,
        }

        self.progress.emit(
            f"Validation done — overshoot={avg_overshoot:.1f}%, "
            f"settling={avg_settle:.3f}s, SS err={avg_ss_err:.4f} t/s"
        )
        self.finished.emit(result)
