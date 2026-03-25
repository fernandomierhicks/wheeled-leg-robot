"""baseline.py — Write optimized gains directly into params.py.

params.py is the single source of truth.  After an optimizer run, this module
patches the dataclass default values in-place so every future import of
SimParams() already contains the best gains.  A timestamped backup of params.py
is kept before each write.

    python -m master_sim.optimizer.baseline --show   # show current gains in params.py
    python -m master_sim.optimizer.baseline --revert # restore previous params.py
"""
from __future__ import annotations

import datetime
import pathlib
import re
import shutil

_PACKAGE = pathlib.Path(__file__).parent.parent.resolve()
PARAMS_PY = _PACKAGE / "params.py"
LOGS_DIR = _PACKAGE / "logs"
BACKUP_DIR = LOGS_DIR / "params_backups"

# Maps search-space keys → (dataclass name, field name) in params.py
_GAINS_MAP = {
    "Q_PITCH":      ("LQRGains", "Q_pitch"),
    "Q_PITCH_RATE": ("LQRGains", "Q_pitch_rate"),
    "Q_VEL":        ("LQRGains", "Q_vel"),
    "R":            ("LQRGains", "R"),
    "KP_V":         ("VelocityPIGains", "Kp"),
    "KI_V":         ("VelocityPIGains", "Ki"),
    "KFF_V":        ("VelocityPIGains", "Kff"),
    "KP_YAW":       ("YawPIGains", "Kp"),
    "KI_YAW":       ("YawPIGains", "Ki"),
    "LEG_K_S":      ("SuspensionGains", "K_s"),
    "LEG_B_S":      ("SuspensionGains", "B_s"),
    "LEG_K_ROLL":   ("SuspensionGains", "K_roll"),
    "LEG_D_ROLL":   ("SuspensionGains", "D_roll"),
    "JUMP_CROUCH_TIME":  ("JumpGains", "crouch_time"),
    "JUMP_MAX_TORQUE":   ("JumpGains", "max_torque"),
    "JUMP_RAMP_UP_S":    ("JumpGains", "ramp_up_s"),
    "JUMP_RAMP_DOWN_RAD":("JumpGains", "ramp_down_rad"),
}

# Reverse map: (class, field) → search-space key
_REVERSE_MAP = {v: k for k, v in _GAINS_MAP.items()}


def _format_float(value: float) -> str:
    """Format a float for params.py — use scientific notation for very small values."""
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.4e}"
    return f"{value:.6g}"


def _backup_params() -> pathlib.Path:
    """Back up current params.py with timestamp. Returns backup path."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"params_{ts}.py"
    shutil.copy2(PARAMS_PY, backup_path)
    return backup_path


def _patch_field(source: str, class_name: str, field_name: str, value: float) -> str:
    """Replace a single dataclass field default in params.py source text.

    Matches patterns like:
        Q_pitch: float = 0.6902
        Q_vel: float = 1.228e-06
    within the correct class block.
    """
    # Find the class block
    class_pattern = rf'(class\s+{class_name}\b.*?\n)'
    class_match = re.search(class_pattern, source)
    if not class_match:
        raise ValueError(f"Class {class_name} not found in params.py")

    class_start = class_match.start()

    # Find the field within this class (search from class start)
    # Pattern: field_name: float = <number>
    field_pattern = rf'(\s+{field_name}:\s+float\s*=\s*)([-+]?\d[\d.eE+-]*)'
    remaining = source[class_start:]
    field_match = re.search(field_pattern, remaining)
    if not field_match:
        raise ValueError(f"Field {class_name}.{field_name} not found in params.py")

    abs_start = class_start + field_match.start(2)
    abs_end = class_start + field_match.end(2)

    new_val = _format_float(value)
    return source[:abs_start] + new_val + source[abs_end:]


def save_baseline_gains(best_params: dict, best_fitness: float,
                        scenario: str, source: str = "optimizer") -> pathlib.Path:
    """Write best gains directly into params.py dataclass defaults.

    Parameters
    ----------
    best_params : dict
        Search-space keys → float (e.g. {"Q_PITCH": 0.46, ...}).
    best_fitness : float
        Fitness value achieved.
    scenario : str
        Scenario that produced these gains.
    source : str
        Label for provenance.

    Returns
    -------
    Path to the backup file.
    """
    backup_path = _backup_params()

    text = PARAMS_PY.read_text(encoding="utf-8")

    for key, value in best_params.items():
        if key not in _GAINS_MAP:
            print(f"[BASELINE] WARNING: unknown key {key!r}, skipping")
            continue
        class_name, field_name = _GAINS_MAP[key]
        text = _patch_field(text, class_name, field_name, value)

    PARAMS_PY.write_text(text, encoding="utf-8")

    n = len(best_params)
    print(f"[BASELINE] Wrote {n} gains into params.py (fitness={best_fitness:.4f})")
    print(f"[BASELINE] Scenario: {scenario} | Source: {source}")
    print(f"[BASELINE] Backup: {backup_path.name}")

    return backup_path


def revert_gains() -> bool:
    """Restore the most recent params.py backup.

    Returns True if a backup was restored, False if none exist.
    """
    if not BACKUP_DIR.exists():
        return False

    backups = sorted(BACKUP_DIR.glob("params_*.py"))
    if not backups:
        return False

    latest = backups[-1]
    shutil.copy2(latest, PARAMS_PY)
    try:
        latest.unlink()
    except PermissionError:
        pass  # Dropbox/antivirus file lock — backup stays
    print(f"[BASELINE] Reverted params.py from {latest.name}")
    return True


def show_gains():
    """Print current gain values from params.py."""
    from master_sim_jump.params import SimParams
    p = SimParams()

    print("Current gains in params.py:")
    print()
    print(f"{'Param':<16} {'Value':>14}")
    print("-" * 34)

    for cand_key, (class_name, field_name) in _GAINS_MAP.items():
        # Navigate the dataclass hierarchy
        gains_attr = {
            "LQRGains": "lqr",
            "VelocityPIGains": "velocity_pi",
            "YawPIGains": "yaw_pi",
            "SuspensionGains": "suspension",
            "JumpGains": "jump",
        }[class_name]
        value = getattr(getattr(p.gains, gains_attr), field_name)
        print(f"{cand_key:<16} {value:>14.8g}")

    # Show backups
    if BACKUP_DIR.exists():
        backups = sorted(BACKUP_DIR.glob("params_*.py"))
        if backups:
            print(f"\n{len(backups)} backup(s) available for --revert")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Manage baseline gains in params.py.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--show", action="store_true", help="Show current gains")
    group.add_argument("--revert", action="store_true", help="Restore previous params.py")
    args = ap.parse_args()

    if args.show:
        show_gains()
    elif args.revert:
        if not revert_gains():
            print("No backups available to revert to.")


if __name__ == "__main__":
    main()
