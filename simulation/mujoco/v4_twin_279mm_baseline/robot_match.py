"""Build the default MuJoCo plant from the latest real-robot evidence.

The control snapshot remains in the GUI's interchange format.  Plant evidence
and its caveats live in ``robot_match.json`` so assumptions do not disappear
inside Python constants.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

from v4_twin_279mm_baseline.params import SimParams
from v4_twin_279mm_baseline.twin.tools.param_snapshot import load_snapshot


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
MATCH_REPORT = PACKAGE_DIR / "robot_match.json"
CONTROLLER_SNAPSHOT_ENV = "WLR_CONTROLLER_SNAPSHOT"


def load_robot_match(path: Path = MATCH_REPORT) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_latest_firmware_params(report_path: Path = MATCH_REPORT) -> dict[str, float]:
    report = load_robot_match(report_path)
    frozen_snapshot = os.environ.get(CONTROLLER_SNAPSHOT_ENV)
    snapshot_path = (Path(frozen_snapshot) if frozen_snapshot
                     else REPO_ROOT / report["control_export"])
    firmware = load_snapshot(snapshot_path)
    lock = report.get("control_snapshot")
    if lock:
        actual = control_snapshot_sha256(firmware)
        expected = str(lock["sha256"])
        if actual != expected or len(firmware) != int(lock["parameter_count"]):
            raise RuntimeError(
                "controller snapshot changed since robot_match.json was fitted; "
                "regenerate the report before running the matched twin. "
                f"Checked {snapshot_path}")
    return firmware


def control_snapshot_sha256(values: dict[str, float]) -> str:
    """Stable digest used to keep plant fitting from changing controller data."""
    payload = json.dumps(
        {name: float(values[name]) for name in sorted(values)},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_robot_matched_params(base: SimParams | None = None,
                               report_path: Path = MATCH_REPORT) -> SimParams:
    """Return immutable parameters matched to the latest export/log evidence."""
    base = base or SimParams()
    report = load_robot_match(report_path)
    firmware = load_latest_firmware_params(report_path)

    geometry = report["geometry"]
    robot = replace(
        base.robot,
        calib_backoff_rad=firmware["calib_backoff_rad"],
        box_cg_x=float(geometry["box_cg_x_m"]),
        box_cg_z=float(geometry["box_cg_z_m"]),
        battery_cg_x=float(geometry["battery_cg_x_m"]),
        battery_cg_z=float(geometry["battery_cg_z_m"]),
        m_femur=float(geometry["effective_link_mass_kg_each"]["femur"]),
        m_tibia=float(geometry["effective_link_mass_kg_each"]["tibia"]),
        m_coupler=float(geometry["effective_link_mass_kg_each"]["coupler"]),
    )

    drive = report["wheel_drive"]
    wheel = replace(
        base.motors.wheel,
        KV=float(drive["motor_kv_rpm_per_v"]),
        current_limit=float(drive["odrive_current_limit_a"]),
        odrive_torque_constant=float(
            drive["odrive_configured_torque_constant_nm_per_a"]),
    )
    hip_drive = report.get("hip_drive", {})
    hip = replace(
        base.motors.hip,
        torque_scale_ret=float(hip_drive.get("torque_scale_ret", 1.0)),
        torque_scale_ext=float(hip_drive.get("torque_scale_ext", 1.0)),
    )
    motors = replace(base.motors, wheel=wheel, hip=hip)

    return replace(
        base,
        robot=robot,
        motors=motors,
        firmware_params=tuple(sorted(firmware.items())),
        robot_match_source=str(report_path),
    )
