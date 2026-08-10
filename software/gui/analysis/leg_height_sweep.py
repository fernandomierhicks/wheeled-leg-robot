"""Leg-height (alpha) sweep analysis — shared by the Log Analyzer tab and the CLI.

A tuning run that walks the robot through several leg heights holds each one for
a while, and every question worth asking about it ("is the trim right here?",
"why did it oscillate up there?") is a *per-height* question. Whole-file
statistics average the heights together and say nothing.

So this module does one thing: split a run into leg-height plateaus, then report
each plateau on its own. It is Qt-free and imports only wlog_metrics, so the GUI
tab and `tools/balance_trim_sweep.py` compute identical numbers.

What each piece is for:

- `alpha_plateaus()`  — the segmentation. Stretches where gain_sched_alpha is
  flat, which is where the robot is actually settled at a height rather than
  moving between two.
- `plateau_report()`  — the per-plateau table: balance point (with the
  equilibrium gate below), pitch-error band decomposition, saturation duties.
- `band_rms()`        — spectral split. The band a disturbance lives in says
  which loop owns it: 0.3-1.5 Hz is the velocity PI, 1.5-4 Hz the LQR pitch
  loop, above ~8 Hz the leg/hip structure. A number that grows in exactly one
  band across leg heights names the culprit.
- `rate_limit_duty()` — how much of the time vel_pi_rate_lim is slewing
  theta_ref. A rate limiter inside a feedback loop is a describing-function
  limit-cycle source, and it is invisible in any RMS statistic.
- `fit_trim_schedule()` — least-squares fit of measured balance points to the
  firmware's own scheduled_pitch_trim() form, so the result is three numbers
  that can be typed straight into params.

**The equilibrium gate is the whole game for the balance point.** A robot that
is not accelerating needs no wheel torque, so `tau_sym ~ 0` AND wheels not
drifting => the pitch it is holding IS the balance point. Both halves matter:
an oscillating robot averages `tau_sym` to nearly zero while still drifting,
so torque alone will happily certify a window that is not in equilibrium.
Windows that fail the gate are reported with `equilibrium=False` rather than
dropped, because "it never settled here" is itself the finding when a run went
unstable at the top of the stroke.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .wlog_metrics import DecodedRun

STATE_RUNNING = 3

# Plateau detection
ALPHA_FLAT_TOL = 0.01    # alpha peak-to-peak inside the window counting as "flat"
ALPHA_FLAT_WINDOW_S = 1.0
MIN_PLATEAU_S = 4.0

# Equilibrium gate (shared with tools/balance_trim_sweep.py)
TAU_EQ_NM = 0.005        # |mean tau_sym| above this => still accelerating
DRIFT_EQ_TPS = 0.05      # |mean wheel velocity| above this => still creeping
V_STILL_MS = 0.02        # |v_ref| below this counts as "not commanded to move"
OMEGA_STILL = 0.02       # |omega_cmd_rds| likewise [rad/s]
TAIL_FRAC = 0.5          # balance point uses the LAST half of a plateau only

# Which loop owns which band. Boundaries are the loops' own separation, not
# round numbers: the velocity PI runs an order of magnitude below the LQR.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("dc", 0.01, 0.3),        # slow drift / operator input
    ("vel_pi", 0.3, 1.5),     # outer velocity loop
    ("lqr", 1.5, 4.0),        # inner pitch loop
    ("mid", 4.0, 10.0),
    ("struct", 10.0, 50.0),   # leg / hip structural modes
)

_PSD_NFFT = 4096


def welch_psd(x: np.ndarray, fs: float, nfft: int = _PSD_NFFT
              ) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD, 50 % overlap, Hann window. Mean is removed first so
    the DC bin never swamps the bands that matter."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0), np.zeros(0)
    x = x - x.mean()
    n = min(nfft, x.size)
    step = max(n // 2, 1)
    segments = [x[i:i + n] for i in range(0, x.size - n + 1, step)]
    if not segments:
        segments = [np.pad(x, (0, n - x.size))]
    window = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    for seg in segments:
        acc += np.abs(np.fft.rfft(seg * window)) ** 2
    acc *= 2.0 / (len(segments) * (window ** 2).sum() * fs)
    return np.fft.rfftfreq(n, 1.0 / fs), acc


def band_rms(x: np.ndarray, fs: float, lo: float, hi: float,
             nfft: int = _PSD_NFFT) -> float:
    """RMS of x restricted to [lo, hi) Hz, in x's own units."""
    freqs, psd = welch_psd(x, fs, nfft)
    if freqs.size == 0:
        return 0.0
    sel = (freqs >= lo) & (freqs < hi)
    if sel.sum() < 2:
        return 0.0
    return float(np.sqrt(np.trapezoid(psd[sel], freqs[sel])))


def band_split(x: np.ndarray, fs: float, nfft: int = _PSD_NFFT) -> dict[str, float]:
    """band_rms over every entry of BANDS, keyed by band name."""
    freqs, psd = welch_psd(x, fs, nfft)
    out: dict[str, float] = {}
    for name, lo, hi in BANDS:
        sel = (freqs >= lo) & (freqs < hi)
        out[name] = (float(np.sqrt(np.trapezoid(psd[sel], freqs[sel])))
                     if sel.sum() >= 2 else 0.0)
    return out


def dominant_peak(x: np.ndarray, fs: float, lo: float = 0.2, hi: float = 30.0,
                  nfft: int = _PSD_NFFT) -> tuple[float, float]:
    """(frequency, RMS within +-20 % of it) of the strongest peak in [lo, hi)."""
    freqs, psd = welch_psd(x, fs, nfft)
    sel = (freqs >= lo) & (freqs < hi)
    if sel.sum() < 2:
        return 0.0, 0.0
    peak_hz = float(freqs[sel][int(np.argmax(psd[sel]))])
    around = (freqs > peak_hz * 0.8) & (freqs < peak_hz * 1.25)
    rms = (float(np.sqrt(np.trapezoid(psd[around], freqs[around])))
           if around.sum() >= 2 else 0.0)
    return peak_hz, rms


def _contiguous(mask: np.ndarray, fs: float, min_s: float) -> list[tuple[int, int]]:
    """Inclusive index ranges where mask holds for at least min_s."""
    out: list[tuple[int, int]] = []
    start = None
    for i, good in enumerate(mask):
        if good and start is None:
            start = i
        elif not good and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return [(a, b) for a, b in out if (b - a + 1) / fs >= min_s]


def alpha_plateaus(alpha: np.ndarray, fs: float,
                   tol: float = ALPHA_FLAT_TOL,
                   window_s: float = ALPHA_FLAT_WINDOW_S,
                   min_s: float = MIN_PLATEAU_S) -> list[tuple[int, int]]:
    """Inclusive index ranges over which alpha is flat to within `tol`.

    A sliding `window_s` window is flat when both of these hold:

      - it is not *drifting* — the window's two half-means agree to within
        `tol`. This is what separates a plateau from a ramp, and it works no
        matter how small the per-sample step is, which a derivative test does
        not: a 60 s crawl across the whole stroke has per-sample steps at the
        1e-5 level and would sail through any per-sample threshold.
      - it is not *spread out* — 4 sigma inside the window is within `tol`.

    Both are computed from running sums, so the cost is linear in the trace
    however wide the window; a 500 Hz run of several minutes stays instant
    enough to recompute on a GUI redraw.

    Deliberately not peak-to-peak: alpha is reconstructed from hip encoders and
    carries noise, and one stray sample would otherwise cut a perfectly good
    plateau in half. Sigma tolerates the noise while still rejecting a ramp,
    whose spread grows with the window just as its drift does.

    Resolution follows from `tol`/`window_s`: with the defaults, anything
    moving slower than roughly 0.01 alpha/s reads as a plateau. That is ~100 s
    to cross the stroke — far slower than any real height change, and the
    limit is a floor on what "held still" can mean, not a bug.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    width = max(int(round(window_s * fs)), 2)
    if alpha.size < width:
        return []

    # Centre first: the running sums below lose precision to cancellation when
    # every term shares a large common offset.
    centred = alpha - alpha.mean()
    csum = np.concatenate([[0.0], np.cumsum(centred)])
    csum_sq = np.concatenate([[0.0], np.cumsum(centred ** 2)])
    starts = np.arange(alpha.size - width + 1)
    half = width // 2

    mean_all = (csum[starts + width] - csum[starts]) / width
    mean_sq = (csum_sq[starts + width] - csum_sq[starts]) / width
    sigma = np.sqrt(np.maximum(mean_sq - mean_all ** 2, 0.0))
    mean_first = (csum[starts + half] - csum[starts]) / half
    mean_second = (csum[starts + width] - csum[starts + half]) / (width - half)

    flat_start = (np.abs(mean_second - mean_first) < tol) & (4.0 * sigma < tol)

    # A window being flat makes every sample in it part of a plateau.
    flat = np.zeros(alpha.size, dtype=bool)
    for i in np.flatnonzero(flat_start):
        flat[i:i + width] = True
    return _contiguous(flat, fs, min_s)


def rate_limit_duty(theta_ref: np.ndarray, fs: float,
                    rate_lim: float | None = None) -> dict:
    """How hard vel_pi_rate_lim is working on theta_ref.

    `rate_lim` [rad/s] comes from the .PARAMS sidecar when available. Without
    it, max|dtheta/dt| is used instead: the limiter makes that an exact read of
    the setting whenever it saturates at all, and a lower bound otherwise, so
    the duty is still meaningful — `inferred` says which happened.

    `mean_run_s` separates the two very different things a saturated limiter
    can mean. Runs of a few milliseconds are the limiter smoothing encoder
    noise on the PI output, which is what it is for. Runs approaching half an
    oscillation period mean theta_ref has become a slew-rate-limited triangle
    wave and the limiter is now setting the loop's phase — a limit cycle.
    """
    theta_ref = np.asarray(theta_ref, dtype=np.float64)
    if theta_ref.size < 3:
        return {"rate_lim": float(rate_lim or 0.0), "inferred": rate_lim is None,
                "duty_frac": 0.0, "mean_run_s": 0.0, "max_run_s": 0.0}
    slew = np.diff(theta_ref) * fs
    inferred = rate_lim is None
    limit = float(np.abs(slew).max()) if inferred else float(rate_lim)
    if limit <= 0.0:
        return {"rate_lim": limit, "inferred": inferred,
                "duty_frac": 0.0, "mean_run_s": 0.0, "max_run_s": 0.0}

    at_limit = np.abs(slew) >= 0.99 * limit
    signed = np.sign(slew) * at_limit
    runs, current, prev = [], 0, 0.0
    for value in signed:
        if value != 0 and value == prev:
            current += 1
        else:
            if current:
                runs.append(current)
            current = 1 if value != 0 else 0
        prev = value
    if current:
        runs.append(current)

    return {
        "rate_lim": limit,
        "inferred": inferred,
        "duty_frac": float(at_limit.mean()),
        "mean_run_s": float(np.mean(runs) / fs) if runs else 0.0,
        "max_run_s": float(np.max(runs) / fs) if runs else 0.0,
    }


@dataclass
class PlateauMetrics:
    """One leg-height plateau. Angles are radians; convert at the display edge."""
    index: int
    i0: int
    i1: int
    t0: float
    t1: float
    duration_s: float
    alpha: float
    # Balance point, measured over the tail (see TAIL_FRAC)
    balance_rad: float
    xcheck_rad: float          # mean(theta_ref + pitch_trim); == balance at equilibrium
    equilibrium: bool
    mean_tau_nm: float
    drift_turns_s: float
    # Applied schedule, and the error it leaves behind
    applied_trim_rad: float
    trim_error_rad: float      # mean theta_ref: what the velocity PI had to add
    # Regulation quality
    rms_pitch_err_rad: float
    max_pitch_err_rad: float
    pitch_err_bands: dict[str, float] = field(default_factory=dict)
    theta_ref_bands: dict[str, float] = field(default_factory=dict)
    peak_hz: float = 0.0
    # Actuation
    rms_tau_sym_nm: float = 0.0
    torque_sat_frac: float = 0.0
    theta_bwd_sat_frac: float = 0.0
    theta_fwd_sat_frac: float = 0.0
    rate_limit: dict = field(default_factory=dict)
    # Hips
    hip_sag_l_rad: float = 0.0
    hip_sag_r_rad: float = 0.0
    mean_hip_torque_nm: float = 0.0
    rms_wheel_turns_s: float = 0.0


def plateau_report(run: DecodedRun,
                   tol: float = ALPHA_FLAT_TOL,
                   min_s: float = MIN_PLATEAU_S,
                   torque_limit_nm: float | None = None,
                   rate_lim: float | None = None,
                   theta_max_fwd: float | None = None,
                   theta_max_bwd: float | None = None,
                   ) -> list[PlateauMetrics]:
    """Per-leg-height metrics for every plateau in the RUNNING part of `run`.

    `torque_limit_nm`, `rate_lim` and `theta_max_*` come from the .PARAMS
    sidecar. Only the torque limit is recovered from the run itself when
    missing (max|tau_sym| is exactly the clamp whenever it is ever reached);
    the saturation fractions for the others are reported as 0 rather than
    guessed, since a wrong clamp value invents saturation that never happened.
    """
    if not run.has_gain_sched_alpha:
        return []
    f = run.fields
    fs = float(run.sample_rate_hz)
    running = np.flatnonzero(f["robot_state"].astype(np.int64) == STATE_RUNNING)
    if running.size < int(min_s * fs):
        return []

    def sub(name: str) -> np.ndarray:
        return f[name][running]

    alpha = sub("gain_sched_alpha")
    pitch = sub("pitch_rad")
    theta_ref = sub("theta_ref")
    trim = sub("pitch_trim_rad")
    tau_sym = sub("tau_sym")
    wheel = 0.5 * (sub("wm_l_vel_turns_s") + sub("wm_r_vel_turns_s"))
    v_ref = sub("v_ref")
    omega = sub("omega_cmd_rds")
    t_running = run.t_s[running]

    if torque_limit_nm is None:
        peak = float(np.abs(tau_sym).max())
        torque_limit_nm = peak if peak > 0 else None

    out: list[PlateauMetrics] = []
    for n, (a, b) in enumerate(alpha_plateaus(alpha, fs, tol=tol, min_s=min_s)):
        whole = slice(a, b + 1)
        tail = slice(b - int((b - a + 1) * TAIL_FRAC), b + 1)

        pitch_err = pitch[whole] - theta_ref[whole] - trim[whole]
        mean_tau = float(tau_sym[tail].mean())
        drift = float(wheel[tail].mean())
        still = (np.abs(v_ref[tail]).max() < V_STILL_MS
                 and np.abs(omega[tail]).max() < OMEGA_STILL)

        metrics = PlateauMetrics(
            index=n, i0=int(running[a]), i1=int(running[b]),
            t0=float(t_running[a]), t1=float(t_running[b]),
            duration_s=(b - a + 1) / fs,
            alpha=float(alpha[whole].mean()),
            balance_rad=float(pitch[tail].mean()),
            xcheck_rad=float((theta_ref[tail] + trim[tail]).mean()),
            equilibrium=bool(still and abs(mean_tau) <= TAU_EQ_NM
                             and abs(drift) <= DRIFT_EQ_TPS),
            mean_tau_nm=mean_tau,
            drift_turns_s=drift,
            applied_trim_rad=float(trim[whole].mean()),
            trim_error_rad=float(theta_ref[tail].mean()),
            rms_pitch_err_rad=float(np.sqrt((pitch_err ** 2).mean())),
            max_pitch_err_rad=float(np.abs(pitch_err).max()),
            pitch_err_bands=band_split(pitch_err, fs),
            theta_ref_bands=band_split(theta_ref[whole], fs),
            peak_hz=dominant_peak(pitch_err, fs)[0],
            rms_tau_sym_nm=float(np.sqrt((tau_sym[whole] ** 2).mean())),
            rate_limit=rate_limit_duty(theta_ref[whole], fs, rate_lim),
            rms_wheel_turns_s=float(np.sqrt((wheel[whole] ** 2).mean())),
        )
        if torque_limit_nm:
            metrics.torque_sat_frac = float(
                np.mean(np.abs(tau_sym[whole]) >= 0.98 * torque_limit_nm))
        if theta_max_bwd:
            metrics.theta_bwd_sat_frac = float(
                np.mean(theta_ref[whole] <= -0.995 * theta_max_bwd))
        if theta_max_fwd:
            metrics.theta_fwd_sat_frac = float(
                np.mean(theta_ref[whole] >= 0.995 * theta_max_fwd))
        for key, name in (("hip_sag_l_rad", "hip_l"), ("hip_sag_r_rad", "hip_r")):
            cmd, pos = f[f"{name}_cmd_pos_rad"][running], f[f"{name}_pos_rad"][running]
            setattr(metrics, key, float((cmd[whole] - pos[whole]).mean()))
        metrics.mean_hip_torque_nm = 0.5 * float(
            sub("hip_l_torque_nm")[whole].mean() + sub("hip_r_torque_nm")[whole].mean())
        out.append(metrics)
    return out


def fit_trim_schedule(alphas, balances) -> dict:
    """Fit measured balance points to the firmware's own trim schedule.

    control_safety.h computes
        trim(a) = trim_ret + a*(trim_ext - trim_ret) + trim_curve*a*(1 - a)
    so fitting in that basis returns three numbers that go straight into
    lqr_pitch_trim_ret / _ext / _curve with no further algebra.

    With fewer than three points the quadratic term is dropped rather than
    fitted to noise. `alpha_span` is reported because trim_ext is the value at
    a=1: unless the sweep actually reached full extension it is an
    extrapolation, and the quadratic basis extrapolates hard.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    balances = np.asarray(balances, dtype=np.float64)
    if alphas.size < 2:
        raise ValueError("need at least two leg heights to fit a trim schedule")

    use_curve = alphas.size >= 3
    columns = [1.0 - alphas, alphas]
    if use_curve:
        columns.append(alphas * (1.0 - alphas))
    design = np.column_stack(columns)
    solution, *_ = np.linalg.lstsq(design, balances, rcond=None)
    residuals = design @ solution - balances
    return {
        "trim_ret": float(solution[0]),
        "trim_ext": float(solution[1]),
        "trim_curve": float(solution[2]) if use_curve else 0.0,
        "residual_rad": [float(r) for r in residuals],
        "max_residual_rad": float(np.abs(residuals).max()),
        "n_points": int(alphas.size),
        "alpha_span": (float(alphas.min()), float(alphas.max())),
        "extrapolated": bool(alphas.max() < 0.9),
    }
