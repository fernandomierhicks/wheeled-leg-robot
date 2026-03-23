"""launcher.py — Persistent GUI launcher for master_sim visualizer + optimizer pipeline.

Spawns a single visualizer process and sends live switch commands via mp.Queue.
The launcher stays open — clicking buttons switches scenarios without restarting.
Also provides a pipeline launcher panel to run optimizer steps from the GUI.

Usage:
    python launcher.py
"""

import multiprocessing as mp
import subprocess
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
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
    ("s09_integrated",            "S9 — Integrated"),
]

PIPELINE_STEPS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]


switch_q: mp.Queue = None
viz_proc: mp.Process = None
pipeline_proc: subprocess.Popen = None


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


def _launch_pipeline(start_var, end_var, hours_var, workers_var, patience_var,
                     tol_var, fresh_var, random_var, status_label):
    """Build CLI args and launch pipeline.py as a subprocess."""
    global pipeline_proc

    # Check if already running
    if pipeline_proc is not None and pipeline_proc.poll() is None:
        messagebox.showwarning("Pipeline Running",
                               "A pipeline is already running.\n"
                               "Wait for it to finish or close its terminal window.")
        return

    cmd = [sys.executable, "-m", "master_sim.optimizer.pipeline"]

    start = start_var.get()
    end = end_var.get()
    if start and start != "(all)":
        cmd += ["--start", start]
    if end and end != "(all)":
        cmd += ["--end", end]

    minutes = hours_var.get().strip()
    if minutes:
        try:
            mins = float(minutes)
        except ValueError:
            messagebox.showerror("Invalid", f"Minutes must be a number, got: {minutes}")
            return
        cmd += ["--hours", str(mins / 60.0)]

    workers = workers_var.get().strip()
    if workers:
        try:
            int(workers)
        except ValueError:
            messagebox.showerror("Invalid", f"Workers must be an integer, got: {workers}")
            return
        cmd += ["--workers", workers]

    patience = patience_var.get().strip()
    if patience:
        try:
            int(patience)
        except ValueError:
            messagebox.showerror("Invalid", f"Patience must be an integer, got: {patience}")
            return
        cmd += ["--patience", patience]

    tol = tol_var.get().strip()
    if tol:
        try:
            float(tol)
        except ValueError:
            messagebox.showerror("Invalid", f"Tolerance must be a number, got: {tol}")
            return
        cmd += ["--tol", tol]

    # Auto-fresh when running a single step (user clearly wants to re-optimize)
    if fresh_var.get() or (start != "(all)" and start == end):
        cmd.append("--fresh")
    if random_var.get():
        cmd.append("--random")

    cmd_str = " ".join(cmd)
    status_label.config(text=f"Running: {cmd_str}")

    # Launch in a new console window so user can see live output.
    # Wrap with cmd /K so the window stays open after the pipeline finishes.
    cwd = str(Path(__file__).resolve().parent.parent)
    wrapped = ["cmd", "/K"] + cmd
    pipeline_proc = subprocess.Popen(
        wrapped,
        cwd=cwd,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    status_label.config(text=f"PID {pipeline_proc.pid}  |  {cmd_str}")


def main():
    mp.freeze_support()

    root = tk.Tk()
    root.title("Sim Launcher")
    root.resizable(False, False)

    # ── Left frame: Visualizer ───────────────────────────────────────────────
    left = tk.Frame(root)
    left.pack(side="left", fill="both", padx=(10, 5), pady=10)

    tk.Label(left, text="Visualizer", font=("Segoe UI", 11, "bold")).pack(pady=(0, 5))

    # Sandbox button — prominent at top
    tk.Button(
        left,
        text="Sandbox",
        font=("Segoe UI", 12, "bold"),
        width=30,
        height=2,
        bg="#4CAF50",
        fg="white",
        activebackground="#388E3C",
        activeforeground="white",
        command=lambda: on_click("sandbox"),
    ).pack(pady=(0, 5))

    tk.Frame(left, height=2, bd=1, relief="sunken").pack(fill="x", pady=5)

    for name, display in SCENARIOS:
        tk.Button(
            left,
            text=display,
            font=("Segoe UI", 10),
            width=30,
            anchor="w",
            command=lambda n=name: on_click(n),
        ).pack(pady=2)

    # ── Vertical separator ───────────────────────────────────────────────────
    tk.Frame(root, width=2, bd=1, relief="sunken").pack(side="left", fill="y", pady=10)

    # ── Right frame: Pipeline ────────────────────────────────────────────────
    right = tk.Frame(root)
    right.pack(side="left", fill="both", padx=(5, 10), pady=10)

    tk.Label(right, text="Optimizer Pipeline", font=("Segoe UI", 11, "bold")).pack(pady=(0, 8))

    # Step range
    range_frame = tk.Frame(right)
    range_frame.pack(fill="x", pady=2)

    tk.Label(range_frame, text="Start:", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
    start_var = tk.StringVar(value="(all)")
    start_combo = ttk.Combobox(range_frame, textvariable=start_var,
                               values=["(all)"] + PIPELINE_STEPS,
                               state="readonly", width=8)
    start_combo.grid(row=0, column=1, padx=(4, 12))

    tk.Label(range_frame, text="End:", font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w")
    end_var = tk.StringVar(value="(all)")
    end_combo = ttk.Combobox(range_frame, textvariable=end_var,
                             values=["(all)"] + PIPELINE_STEPS,
                             state="readonly", width=8)
    end_combo.grid(row=0, column=3, padx=4)

    # Numeric params
    params_frame = tk.Frame(right)
    params_frame.pack(fill="x", pady=(8, 2))

    tk.Label(params_frame, text="Min/step:", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
    hours_var = tk.StringVar(value="60")
    tk.Entry(params_frame, textvariable=hours_var, width=8).grid(row=0, column=1, padx=4)

    tk.Label(params_frame, text="Workers:", font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w", padx=(8, 0))
    workers_var = tk.StringVar(value="8")
    tk.Entry(params_frame, textvariable=workers_var, width=8).grid(row=0, column=3, padx=4)

    params_frame2 = tk.Frame(right)
    params_frame2.pack(fill="x", pady=2)

    tk.Label(params_frame2, text="Patience:", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
    patience_var = tk.StringVar(value="200")
    tk.Entry(params_frame2, textvariable=patience_var, width=8).grid(row=0, column=1, padx=4)

    tk.Label(params_frame2, text="Tolerance:", font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w", padx=(8, 0))
    tol_var = tk.StringVar(value="1e-4")
    tk.Entry(params_frame2, textvariable=tol_var, width=8).grid(row=0, column=3, padx=4)

    # Flags
    flags_frame = tk.Frame(right)
    flags_frame.pack(fill="x", pady=(8, 2))

    fresh_var = tk.BooleanVar(value=False)
    tk.Checkbutton(flags_frame, text="Fresh (delete existing CSVs)",
                   variable=fresh_var, font=("Segoe UI", 9)).pack(anchor="w")

    random_var = tk.BooleanVar(value=False)
    tk.Checkbutton(flags_frame, text="Random seed (instead of defaults)",
                   variable=random_var, font=("Segoe UI", 9)).pack(anchor="w")

    # Status label
    status_label = tk.Label(right, text="Ready", font=("Segoe UI", 8),
                            fg="gray", anchor="w", wraplength=280)
    status_label.pack(fill="x", pady=(8, 4))

    # Launch button
    tk.Button(
        right,
        text="Launch Pipeline",
        font=("Segoe UI", 12, "bold"),
        width=24,
        height=2,
        bg="#1976D2",
        fg="white",
        activebackground="#1565C0",
        activeforeground="white",
        command=lambda: _launch_pipeline(
            start_var, end_var, hours_var, workers_var,
            patience_var, tol_var, fresh_var, random_var, status_label,
        ),
    ).pack(pady=(4, 0))

    # ── Quick presets ────────────────────────────────────────────────────────
    tk.Frame(right, height=2, bd=1, relief="sunken").pack(fill="x", pady=8)
    tk.Label(right, text="Quick Presets", font=("Segoe UI", 9, "bold")).pack(anchor="w")

    def _set_preset(start, end, hours, fresh=False, random=False):
        start_var.set(start)
        end_var.set(end)
        hours_var.set(str(hours))
        fresh_var.set(fresh)
        random_var.set(random)

    presets_frame = tk.Frame(right)
    presets_frame.pack(fill="x", pady=2)

    tk.Button(presets_frame, text="Smoke Test", font=("Segoe UI", 9),
              command=lambda: _set_preset("(all)", "(all)", 1)
              ).pack(side="left", padx=2)
    tk.Button(presets_frame, text="LQR Only", font=("Segoe UI", 9),
              command=lambda: _set_preset("S1", "S2", 60)
              ).pack(side="left", padx=2)
    tk.Button(presets_frame, text="VelPI Only", font=("Segoe UI", 9),
              command=lambda: _set_preset("S3", "S5", 60)
              ).pack(side="left", padx=2)
    tk.Button(presets_frame, text="Full Fresh", font=("Segoe UI", 9),
              command=lambda: _set_preset("(all)", "(all)", 120, fresh=True)
              ).pack(side="left", padx=2)

    root.mainloop()

    # Clean up on launcher exit
    if viz_proc is not None and viz_proc.is_alive():
        viz_proc.terminate()
        viz_proc.join(timeout=2)


if __name__ == "__main__":
    main()
