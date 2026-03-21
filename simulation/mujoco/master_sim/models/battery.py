"""battery.py — Realistic 6S LiPo battery model.

Ported from latency_sensitivity/battery_model.py.
All parameters come from BatteryParams dataclass — no module-level globals.
"""
import math
import numpy as np

from master_sim.params import BatteryParams

# ── OCV–SoC lookup table (6S LiPo, empirical fit) ───────────────────────────
_SOC_PTS  = np.array([0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00])
_VOCV_PTS = np.array([18.0, 19.8, 21.0, 22.2, 23.7, 24.6, 25.2])  # [V]


def _ocv(soc: float) -> float:
    """Open-circuit voltage [V] as a function of state of charge [0-1]."""
    return float(np.interp(soc, _SOC_PTS, _VOCV_PTS))


class BatteryModel:
    """First-principles 6S LiPo pack model.

    Call ``step(dt, I_demand)`` at the control rate; returns V_terminal.

    Usage::
        batt = BatteryModel(params.battery)
        batt.reset()
        v_term = batt.step(dt, I_total)
    """

    def __init__(self, params: BatteryParams) -> None:
        self._p = params
        self._capacity_coulombs = params.capacity_Ah * 3600.0
        self.reset()

    def reset(self) -> None:
        """Restore initial conditions (call at the start of each scenario)."""
        self._soc = float(self._p.SoC_init)
        self._temp = float(self._p.temp_init_C)
        self._v_term = _ocv(self._soc)
        self._i_total = 0.0
        self._r_int = self._calc_r_int(self._soc, self._temp)
        self._p_heat = 0.0

    def step(self, dt: float, I_demand: float) -> float:
        """Advance battery state by ``dt`` seconds under load ``I_demand`` [A].

        Returns the new terminal voltage [V].
        """
        I = max(0.0, I_demand)

        r = self._calc_r_int(self._soc, self._temp)
        self._r_int = r

        v_ocv = _ocv(self._soc)
        v_term = v_ocv - I * r
        v_term = max(self._p.V_cutoff, v_term)
        self._v_term = v_term

        self._soc -= I * dt / self._capacity_coulombs
        self._soc = max(0.0, min(1.0, self._soc))

        p_heat = I * I * r
        p_cool = self._p.cool_W_per_C * (self._temp - self._p.temp_init_C)
        self._temp += dt * (p_heat - p_cool) / self._p.thermal_mass
        self._p_heat = p_heat

        self._i_total = I
        return v_term

    # ── Properties (read-only telemetry) ─────────────────────────────────────

    @property
    def v_terminal(self) -> float:
        return self._v_term

    @property
    def soc(self) -> float:
        return self._soc

    @property
    def soc_pct(self) -> float:
        return self._soc * 100.0

    @property
    def temperature_c(self) -> float:
        return self._temp

    @property
    def i_total(self) -> float:
        return self._i_total

    @property
    def r_int(self) -> float:
        return self._r_int

    @property
    def p_heat(self) -> float:
        return self._p_heat

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _calc_r_int(self, soc: float, temp_c: float) -> float:
        """R_int = R0 * soc_factor * temp_factor."""
        soc_factor = 1.0 + self._p.K_soc * (1.0 - soc) ** 2
        temp_factor = math.exp(self._p.K_temp * (self._p.temp_ref_C - temp_c))
        return self._p.R0 * soc_factor * temp_factor
