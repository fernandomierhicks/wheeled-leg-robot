"""launcher.py — Persistent GUI launcher for master_sim visualizer.

Spawns a single visualizer process and sends live switch commands via mp.Queue.
The launcher stays open — clicking buttons switches scenarios without restarting.

Usage:
    python launcher.py
"""

import multiprocessing as mp
import sys
import os
import tkinter as tk
from pathlib import Path

# Ensure master_sim is importable when running this file directly
_MUJOCO_DIR = str(Path(__file__).resolve().parent.parent)
if _MUJOCO_DIR not in sys.path:
    sys.path.insert(0, _MUJOCO_DIR)

from master_sim.viz.visualizer import run_unified

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


switch_q: mp.Queue = None
viz_proc: mp.Process = None


def on_click(scenario_key: str):
    """Handle button click — spawn or switch."""
    global viz_proc, switch_q
    if viz_proc is None or not viz_proc.is_alive():
        # First click or process died — spawn fresh
        switch_q = mp.Queue(maxsize=8)
        viz_proc = mp.Process(
            target=run_unified,
            args=(scenario_key,),
            kwargs={"switch_q": switch_q},
        )
        viz_proc.start()
    else:
        # Already running — send switch command
        try:
            switch_q.put_nowait(("SWITCH", scenario_key))
        except Exception:
            pass


def main():
    mp.freeze_support()

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
        command=lambda: on_click("sandbox"),
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
            command=lambda n=name: on_click(n),
        ).pack(padx=10, pady=2)

    # Bottom padding
    tk.Frame(root, height=8).pack()

    root.mainloop()

    # Clean up on launcher exit
    if viz_proc is not None and viz_proc.is_alive():
        viz_proc.terminate()
        viz_proc.join(timeout=2)


if __name__ == "__main__":
    main()
