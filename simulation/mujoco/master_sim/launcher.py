"""launcher.py — GUI launcher for master_sim visualizer.

Launches viz/visualizer.py in sandbox or replay mode.

Usage:
    python launcher.py
"""

import subprocess
import sys
import tkinter as tk
from pathlib import Path

VISUALIZER = Path(__file__).resolve().parent / "viz" / "visualizer.py"
MASTER_SIM_DIR = Path(__file__).resolve().parent.parent  # parent of master_sim for module imports

SCENARIOS = [
    ("s01_lqr_pitch_step",        "S1 — LQR Pitch Step"),
    ("s02_leg_height_gain_sched", "S2 — Leg Height Gain Sched"),
    ("s03_vel_pi_disturbance",    "S3 — VelPI Disturbance"),
    ("s04_vel_pi_staircase",      "S4 — VelPI Staircase"),
    ("s05_vel_pi_leg_cycling",    "S5 — VelPI Leg Cycling"),
    ("s06_yaw_pi_turn",           "S6 — Yaw PI Turn"),
    ("s07_drive_turn",            "S7 — Drive + Turn"),
    ("s08_terrain_compliance",    "S8 — Terrain Compliance"),
]


def launch(args: list[str]):
    subprocess.Popen(
        [sys.executable, str(VISUALIZER)] + args,
        cwd=str(MASTER_SIM_DIR),
    )


def launch_scenario(name: str):
    launch(["--mode", "replay", "--scenario", name])


def launch_sandbox():
    launch(["--mode", "sandbox"])


def main():
    root = tk.Tk()
    root.title("Sim Launcher")
    root.resizable(False, False)

    # Sandbox button — prominent at top
    tk.Button(
        root,
        text="Sandbox",
        font=("Segoe UI", 12, "bold"),
        width=36,
        height=2,
        bg="#4CAF50",
        fg="white",
        activebackground="#388E3C",
        activeforeground="white",
        command=launch_sandbox,
    ).pack(padx=10, pady=(10, 5))

    # Separator
    tk.Frame(root, height=2, bd=1, relief="sunken").pack(fill="x", padx=10, pady=5)

    # Scenario buttons
    for name, display in SCENARIOS:
        tk.Button(
            root,
            text=display,
            font=("Segoe UI", 10),
            width=36,
            anchor="w",
            command=lambda n=name: launch_scenario(n),
        ).pack(padx=10, pady=2)

    # Bottom padding
    tk.Frame(root, height=8).pack()

    root.mainloop()


if __name__ == "__main__":
    main()
