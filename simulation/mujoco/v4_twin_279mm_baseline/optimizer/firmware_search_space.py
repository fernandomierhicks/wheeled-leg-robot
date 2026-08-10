"""Firmware-gain search spaces, derived and bounded by the protocol schema."""

from __future__ import annotations

from .search_space import ParamSpec, SearchSpace
from ..twin.params_control import PARAMS_BY_NAME


def _space(*names: str) -> SearchSpace:
    return SearchSpace(params={
        name: ParamSpec(PARAMS_BY_NAME[name].min, PARAMS_BY_NAME[name].max,
                        zero_ok=PARAMS_BY_NAME[name].min == 0.0, scale="linear")
        for name in names
    })


LQR_FIRMWARE_SPACE = _space(
    "lqr_k_pitch_ret", "lqr_k_pitch_ext", "lqr_k_rate_ret", "lqr_k_rate_ext",
    "lqr_k_vel", "lqr_pitch_trim_ret", "lqr_pitch_trim_ext", "lqr_barrier_k",
    "lqr_barrier_th_ret", "lqr_barrier_th_ext", "theta_max_fwd_ret",
    "theta_max_bwd_ret", "theta_max_fwd_ext", "theta_max_bwd_ext",
)
VELOCITY_PI_FIRMWARE_SPACE = _space(
    "vel_pi_kp", "vel_pi_ki", "vel_pi_kff", "vel_pi_rate_lim", "vel_pi_int_max",
)
YAW_PI_FIRMWARE_SPACE = _space(
    "yaw_pi_kp", "yaw_pi_ki", "yaw_pi_torque_max", "yaw_pi_int_max",
)
ROLL_FIRMWARE_SPACE = _space("roll_kp", "roll_kd", "roll_ki", "roll_int_max")
INTEGRATED_FIRMWARE_SPACE = SearchSpace(params={
    **LQR_FIRMWARE_SPACE.params,
    **VELOCITY_PI_FIRMWARE_SPACE.params,
    **YAW_PI_FIRMWARE_SPACE.params,
    **ROLL_FIRMWARE_SPACE.params,
})

FIRMWARE_SPACE_BY_STAGE = {
    "lqr": LQR_FIRMWARE_SPACE,
    "vel_pi": VELOCITY_PI_FIRMWARE_SPACE,
    "yaw_pi": YAW_PI_FIRMWARE_SPACE,
    "roll": ROLL_FIRMWARE_SPACE,
    "integrated": INTEGRATED_FIRMWARE_SPACE,
}
