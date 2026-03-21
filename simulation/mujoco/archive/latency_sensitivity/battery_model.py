"""battery_model.py — Realistic 6S LiPo battery model for LQR_Control_optimization.

Models:
  - OCV-SoC: 7-point lookup table fitted to 6S LiPo chemistry
  - Internal resistance: SoC-dependent + Arrhenius temperature dependence
  - Terminal voltage sag: V_term = V_ocv - I * R_int
  - Capacity depletion: Coulomb counting
  - Thermal dynamics: 1st-order heat model (I²R heating, passive cooling)

All parameters read from sim_config.py; no magic numbers in this file.
"""
import math
import numpy as np

from sim_config import (
    BATT_CAPACITY_AH,
    BATT_V_FULL, BATT_V_NOM, BATT_V_CUTOFF,
    BATT_R0, BATT_K_SOC, BATT_K_TEMP, BATT_TEMP_REF_C,
    BATT_THERMAL_MASS, BATT_COOL_W_PER_C,
    BATT_TEMP_INIT_C, BATT_SOC_INIT,
    BATT_I_QUIESCENT,
)

# ── OCV–SoC lookup table (6S LiPo, empirical fit) ───────────────────────────
# Voltage per pack (6 cells in series).  3.0 V/cell cutoff → 18.0 V pack min.
_SOC_PTS  = np.array([0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00])
_VOCV_PTS = np.array([18.0, 19.8, 21.0, 22.2, 23.7, 24.6, 25.2])  # [V]


def _ocv(soc: float) -> float:
    """Open-circuit voltage [V] as a function of state of charge [0-1]."""
    return float(np.interp(soc, _SOC_PTS, _VOCV_PTS))


class BatteryModel:
    """First-principles 6S LiPo pack model.

    Call ``step(dt, I_demand)`` at the control rate; it returns V_terminal and
    updates the internal state (SoC, temperature, R_int).

    Properties expose all internal state for telemetry logging.
    """

    def __init__(self) -> None:
        self._capacity_coulombs = BATT_CAPACITY_AH * 3600.0  # [C]
        self.reset()

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Restore initial conditions (call at the start of each scenario)."""
        self._soc   = float(BATT_SOC_INIT)
        self._temp  = float(BATT_TEMP_INIT_C)
        self._v_term = _ocv(self._soc)   # start with no-load voltage
        self._i_total = 0.0
        self._r_int   = self._calc_r_int(self._soc, self._temp)
        self._p_heat  = 0.0

    def step(self, dt: float, I_demand: float) -> float:
        """Advance battery state by ``dt`` seconds under load ``I_demand`` [A].

        ``I_demand`` is the total current drawn (all motors + quiescent).
        Returns the new terminal voltage [V].
        """
        # ── Clamp current to physical limits ─────────────────────────────────
        I = max(0.0, I_demand)   # battery only sources current (no regen for now)

        # ── Internal resistance (SoC + temperature) ───────────────────────────
        r = self._calc_r_int(self._soc, self._temp)
        self._r_int = r

        # ── Terminal voltage sag ──────────────────────────────────────────────
        v_ocv = _ocv(self._soc)
        v_term = v_ocv - I * r
        v_term = max(BATT_V_CUTOFF, v_term)   # hard floor
        self._v_term = v_term

        # ── SoC depletion (Coulomb counting) ──────────────────────────────────
        self._soc -= I * dt / self._capacity_coulombs
        self._soc  = max(0.0, min(1.0, self._soc))

        # ── Thermal dynamics ──────────────────────────────────────────────────
        p_heat = I * I * r                                    # resistive heating [W]
        p_cool = BATT_COOL_W_PER_C * (self._temp - BATT_TEMP_INIT_C)  # passive cooling
        self._temp += dt * (p_heat - p_cool) / BATT_THERMAL_MASS
        self._p_heat = p_heat

        self._i_total = I
        return v_term

    # ── Properties (read-only telemetry) ─────────────────────────────────────

    @property
    def v_terminal(self) -> float:
        """Terminal voltage [V] after sag."""
        return self._v_term

    @property
    def soc(self) -> float:
        """State of charge [0–1]."""
        return self._soc

    @property
    def soc_pct(self) -> float:
        """State of charge [%]."""
        return self._soc * 100.0

    @property
    def temperature_c(self) -> float:
        """Battery temperature [°C]."""
        return self._temp

    @property
    def i_total(self) -> float:
        """Total current drawn in last step [A]."""
        return self._i_total

    @property
    def r_int(self) -> float:
        """Internal resistance in last step [Ω]."""
        return self._r_int

    @property
    def p_heat(self) -> float:
        """Resistive heat dissipation in last step [W]."""
        return self._p_heat

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _calc_r_int(soc: float, temp_c: float) -> float:
        """R_int = R0 * soc_factor * temp_factor.

        soc_factor:  quadratic rise toward SoC = 0 (electrolyte depletion).
        temp_factor: Arrhenius-like — colder pack → higher resistance.
                     Warm pack (> T_ref) → slightly lower resistance.
        """
        soc_factor  = 1.0 + BATT_K_SOC  * (1.0 - soc) ** 2
        temp_factor = math.exp(BATT_K_TEMP * (BATT_TEMP_REF_C - temp_c))
        return BATT_R0 * soc_factor * temp_factor
