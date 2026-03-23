"""progress_ui.py — Floating tkinter progress window with play/pause button.

Runs in a daemon thread. The optimizer calls update_from_progress(ESProgress)
each generation; the window polls every 300 ms and redraws.

Usage
-----
    from master_sim.optimizer.progress_ui import ProgressUI

    ui = ProgressUI("LQR Optimizer — S2")
    # pass ui.update_from_progress as progress_fn to ESOptimizer
    opt = ESOptimizer(..., progress_fn=ui.update_from_progress)
    result = opt.run(...)
    ui.finish()
"""
from __future__ import annotations

import threading
import tkinter as tk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from master_sim.optimizer.es_engine import ESProgress


# ── Colour palette ───────────────────────────────────────────────────────────
BG       = "#1e1e2e"
FG       = "#cdd6f4"
FG_DIM   = "#6c7086"
ACCENT   = "#89b4fa"    # blue  — bar fill
ACCENT2  = "#a6e3a1"    # green — best fitness
BAR_BG   = "#313244"
BTN_PLAY = "#a6e3a1"    # green — running
BTN_PAUSE = "#f38ba8"   # red   — paused
FONT_MONO = ("Consolas", 10)
FONT_BIG  = ("Consolas", 12, "bold")
FONT_SM   = ("Consolas", 9)


class ProgressUI:
    """Floating progress window with play/pause for the ES optimizer.

    Parameters
    ----------
    title : str
        Window title.
    all_defaults : dict, optional
        All 12 gain defaults {name: float}.  When provided a live gains
        panel is shown — active gains update each generation, inactive
        ones stay dimmed at their default value.
    active_names : set, optional
        Names of gains currently being optimised (subset of all_defaults).
    """

    def __init__(self, title: str = "Optimizer Progress",
                 all_defaults: dict | None = None,
                 active_names: set | None = None):
        self._lock = threading.Lock()
        self._all_defaults = all_defaults or {}
        self._active_names = active_names or set()
        self._state: dict = dict(
            pct=0.0, elapsed_s=0, remaining_s=0,
            n_evals=0, gen=0,
            best_fit=float("inf"), best_params="—",
            status="starting", success_rate=0.0,
            gens_without_improvement=0,
            current_params={},
            last_tried={},
        )
        self._paused = threading.Event()       # set = paused
        self._done = threading.Event()
        self._closed = threading.Event()
        self._auto_close_s: float | None = None  # set by finish(auto_close=True)
        self._t = threading.Thread(target=self._run_gui, args=(title,), daemon=True)
        self._t.start()

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    def wait_if_paused(self):
        """Block the calling thread while paused. Call between generations."""
        while self._paused.is_set() and not self._closed.is_set():
            self._paused.wait(timeout=0.3)
            if not self._paused.is_set():
                break

    def update_from_progress(self, p: "ESProgress") -> None:
        """Thread-safe state update — pass as progress_fn to ESOptimizer."""
        with self._lock:
            self._state.update(
                pct=p.pct,
                elapsed_s=p.elapsed_s,
                remaining_s=p.remaining_s,
                n_evals=p.n_evals,
                gen=p.gen,
                best_fit=p.best_fitness,
                best_params=_params_str(p.best_params),
                status=p.status,
                success_rate=p.success_rate,
                gens_without_improvement=p.gens_without_improvement,
                n_restarts=p.n_restarts,
                current_params=dict(p.best_params),
                last_tried=dict(p.last_tried),
            )

    def finish(self, auto_close: bool = True, linger_s: float = 3.0):
        """Signal that optimization is complete.

        Parameters
        ----------
        auto_close : bool
            If True (default), the window closes automatically after *linger_s*
            seconds.  Set False to keep the window open until the user closes it.
        linger_s : float
            Seconds to keep the window visible before auto-closing.
        """
        with self._lock:
            self._state["status"] = "done"
            self._state["pct"] = 100.0
            self._state["remaining_s"] = 0
        self._done.set()
        if auto_close:
            self._auto_close_s = linger_s
        # Wait for window to close (auto or manual) so tkinter vars are
        # cleaned up on the GUI thread (avoids cross-thread StringVar.__del__)
        self._closed.wait()

    def wait_closed(self):
        """Block until the user closes the window."""
        self._closed.wait()

    # ── Private / GUI thread ─────────────────────────────────────────────────

    def _snap(self) -> dict:
        with self._lock:
            return dict(self._state)

    def _run_gui(self, title: str):
        root = tk.Tk()
        root.title(title)
        root.configure(bg=BG)
        root.resizable(False, False)
        root.attributes("-topmost", True)

        PAD = 14
        W = 440

        # ── Title ────────────────────────────────────────────────────────────
        tk.Label(root, text=title, font=FONT_BIG, bg=BG, fg=ACCENT
                 ).pack(pady=(PAD, 4), padx=PAD, anchor="w")

        # ── Progress bar ─────────────────────────────────────────────────────
        bar_frame = tk.Frame(root, bg=BG)
        bar_frame.pack(fill="x", padx=PAD, pady=(0, 4))

        canvas = tk.Canvas(bar_frame, height=18, bg=BAR_BG,
                           highlightthickness=0, relief="flat")
        canvas.pack(fill="x")
        bar_rect = canvas.create_rectangle(0, 0, 0, 18, fill=ACCENT, width=0)
        bar_label = canvas.create_text(W // 2, 9, text="0%",
                                       fill=FG, font=FONT_MONO, anchor="center")

        # ── Info grid ────────────────────────────────────────────────────────
        info_frame = tk.Frame(root, bg=BG)
        info_frame.pack(fill="x", padx=PAD, pady=(0, 4))

        def _row(parent, label, row, col=0):
            tk.Label(parent, text=label, font=FONT_SM, bg=BG, fg=FG_DIM,
                     anchor="w").grid(row=row, column=col * 2,
                                      sticky="w", padx=(0, 4))
            var = tk.StringVar(value="—")
            tk.Label(parent, textvariable=var, font=FONT_SM, bg=BG, fg=FG,
                     anchor="w").grid(row=row, column=col * 2 + 1, sticky="w")
            return var

        v_elapsed   = _row(info_frame, "Elapsed",    0, 0)
        v_remaining = _row(info_frame, "Remaining",  0, 1)
        v_evals     = _row(info_frame, "Evals",      1, 0)
        v_gen       = _row(info_frame, "Generation", 1, 1)

        # ── Best fitness ─────────────────────────────────────────────────────
        sep = tk.Frame(root, height=1, bg=BAR_BG)
        sep.pack(fill="x", padx=PAD, pady=(2, 6))

        best_frame = tk.Frame(root, bg=BG)
        best_frame.pack(fill="x", padx=PAD, pady=(0, 4))

        tk.Label(best_frame, text="Best fitness", font=FONT_SM,
                 bg=BG, fg=FG_DIM, anchor="w").pack(anchor="w")
        v_best_fit = tk.StringVar(value="—")
        tk.Label(best_frame, textvariable=v_best_fit, font=FONT_BIG,
                 bg=BG, fg=ACCENT2, anchor="w").pack(anchor="w")

        tk.Label(best_frame, text="Best params", font=FONT_SM,
                 bg=BG, fg=FG_DIM, anchor="w").pack(anchor="w", pady=(4, 0))
        v_best_params = tk.StringVar(value="—")
        tk.Label(best_frame, textvariable=v_best_params, font=FONT_MONO,
                 bg=BG, fg=FG, anchor="w", wraplength=W - PAD * 2
                 ).pack(anchor="w")

        # ── Live gains panel ────────────────────────────────────────────────
        gain_best_vars: dict[str, tk.StringVar] = {}
        gain_try_vars: dict[str, tk.StringVar] = {}
        if self._all_defaults:
            sep_gains = tk.Frame(root, height=1, bg=BAR_BG)
            sep_gains.pack(fill="x", padx=PAD, pady=(6, 4))

            gains_frame = tk.Frame(root, bg=BG)
            gains_frame.pack(fill="x", padx=PAD, pady=(0, 4))

            tk.Label(gains_frame, text="Controller Gains", font=FONT_SM,
                     bg=BG, fg=FG_DIM, anchor="w").pack(anchor="w")

            gains_grid = tk.Frame(gains_frame, bg=BG)
            gains_grid.pack(fill="x", pady=(2, 0))

            # Group gains by controller for visual clarity
            _GAIN_GROUPS = [
                ("LQR",        ["Q_PITCH", "Q_PITCH_RATE", "Q_VEL", "R"]),
                ("Velocity PI", ["KP_V", "KI_V", "KFF_V"]),
                ("Yaw PI",     ["KP_YAW", "KI_YAW"]),
                ("Suspension", ["LEG_K_S", "LEG_B_S", "LEG_K_ROLL", "LEG_D_ROLL"]),
            ]

            # Column headers
            tk.Label(gains_grid, text="", font=FONT_SM, bg=BG, width=2
                     ).grid(row=0, column=0)
            tk.Label(gains_grid, text="", font=FONT_SM, bg=BG
                     ).grid(row=0, column=1)
            tk.Label(gains_grid, text="Best", font=FONT_SM, bg=BG, fg=FG_DIM,
                     anchor="e", width=12).grid(row=0, column=2, sticky="e")
            tk.Label(gains_grid, text="Trying", font=FONT_SM, bg=BG, fg=FG_DIM,
                     anchor="e", width=12).grid(row=0, column=3, sticky="e", padx=(8, 0))

            row_idx = 1
            for group_label, keys in _GAIN_GROUPS:
                # Only show groups that have at least one key in all_defaults
                group_keys = [k for k in keys if k in self._all_defaults]
                if not group_keys:
                    continue

                tk.Label(gains_grid, text=group_label, font=FONT_SM,
                         bg=BG, fg=ACCENT, anchor="w"
                         ).grid(row=row_idx, column=0, columnspan=4,
                                sticky="w", pady=(4, 0))
                row_idx += 1

                for k in group_keys:
                    is_active = k in self._active_names
                    name_fg = FG if is_active else FG_DIM
                    val_fg = ACCENT2 if is_active else FG_DIM
                    marker = "\u25b6" if is_active else " "

                    tk.Label(gains_grid, text=marker, font=FONT_SM,
                             bg=BG, fg=ACCENT2, width=2
                             ).grid(row=row_idx, column=0, sticky="w")
                    tk.Label(gains_grid, text=k, font=FONT_SM,
                             bg=BG, fg=name_fg, anchor="w"
                             ).grid(row=row_idx, column=1, sticky="w", padx=(0, 8))

                    default_str = f"{self._all_defaults[k]:.6g}"

                    # Best column — updates for active gains
                    best_var = tk.StringVar(value=default_str)
                    gain_best_vars[k] = best_var
                    tk.Label(gains_grid, textvariable=best_var, font=FONT_MONO,
                             bg=BG, fg=val_fg, anchor="e", width=12
                             ).grid(row=row_idx, column=2, sticky="e")

                    # Trying column — only for active gains
                    if is_active:
                        try_var = tk.StringVar(value=default_str)
                        gain_try_vars[k] = try_var
                        tk.Label(gains_grid, textvariable=try_var, font=FONT_MONO,
                                 bg=BG, fg=FG, anchor="e", width=12
                                 ).grid(row=row_idx, column=3, sticky="e", padx=(8, 0))
                    else:
                        tk.Label(gains_grid, text="—", font=FONT_MONO,
                                 bg=BG, fg=FG_DIM, anchor="e", width=12
                                 ).grid(row=row_idx, column=3, sticky="e", padx=(8, 0))

                    row_idx += 1

        # ── Play / Pause button ──────────────────────────────────────────────
        sep2 = tk.Frame(root, height=1, bg=BAR_BG)
        sep2.pack(fill="x", padx=PAD, pady=(6, 4))

        btn_frame = tk.Frame(root, bg=BG)
        btn_frame.pack(fill="x", padx=PAD, pady=(0, 4))

        btn_var = tk.StringVar(value="Pause")
        btn = tk.Button(
            btn_frame, textvariable=btn_var, font=FONT_MONO,
            bg=BTN_PLAY, fg="#1e1e2e", width=10, relief="flat",
            activebackground=BTN_PAUSE,
        )
        btn.pack(side="left")

        def _toggle_pause():
            if self._paused.is_set():
                self._paused.clear()
                btn_var.set("Pause")
                btn.configure(bg=BTN_PLAY)
            else:
                self._paused.set()
                btn_var.set("Resume")
                btn.configure(bg=BTN_PAUSE)

        btn.configure(command=_toggle_pause)

        # ── Status bar ───────────────────────────────────────────────────────
        v_status = tk.StringVar(value="Starting...")
        tk.Label(root, textvariable=v_status, font=FONT_SM,
                 bg=BG, fg=FG_DIM, anchor="w"
                 ).pack(anchor="w", padx=PAD, pady=(0, PAD))

        root.update_idletasks()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        ww = root.winfo_reqwidth()
        wh = root.winfo_reqheight()
        root.geometry(f"+{(sw - ww) // 2}+{(sh - wh) // 2}")

        def _fmt_time(secs: float) -> str:
            secs = max(0, int(secs))
            return f"{secs // 60:02d}:{secs % 60:02d}"

        def _poll():
            s = self._snap()
            pct = max(0.0, min(100.0, s["pct"]))

            # Bar
            canvas.update_idletasks()
            bw = canvas.winfo_width()
            fill_w = int(bw * pct / 100.0)
            canvas.coords(bar_rect, 0, 0, fill_w, 18)
            canvas.itemconfig(bar_label, text=f"{pct:.1f}%")
            canvas.coords(bar_label, bw // 2, 9)

            # Info
            v_elapsed.set(_fmt_time(s["elapsed_s"]))
            v_remaining.set(
                _fmt_time(s["remaining_s"]) if s["status"] != "done" else "00:00"
            )
            v_evals.set(str(s["n_evals"]))
            v_gen.set(str(s["gen"]))

            # Best
            bf = s["best_fit"]
            v_best_fit.set(f"{bf:.5f}" if bf < 1e9 else "—")
            v_best_params.set(s["best_params"])

            # Live gains — Best column (current parent)
            cp = s.get("current_params", {})
            for k, var in gain_best_vars.items():
                if k in cp:
                    var.set(f"{cp[k]:.6g}")

            # Live gains — Trying column (last generation's best child)
            lt = s.get("last_tried", {})
            for k, var in gain_try_vars.items():
                if k in lt:
                    var.set(f"{lt[k]:.6g}")

            # Status
            if s["status"] == "done":
                btn_var.set("Done")
                btn.configure(state="disabled", bg=FG_DIM)
                if self._auto_close_s is not None:
                    secs = max(0, int(self._auto_close_s))
                    v_status.set(f"Done — closing in {secs}s")
                    self._auto_close_s -= 0.3  # poll interval
                    if self._auto_close_s <= 0:
                        _on_close()
                        return
                else:
                    v_status.set("Optimization complete — you may close this window")
            elif self._paused.is_set():
                v_status.set("PAUSED — click Resume to continue")
            else:
                sr = s.get("success_rate", 0.0)
                stag = s.get("gens_without_improvement", 0)
                restarts = s.get("n_restarts", 0)
                v_status.set(f"Running  |  success rate {sr:.0%}  |  stagnant {stag}  |  restarts {restarts}")

            if not self._closed.is_set():
                root.after(300, _poll)

        root.after(300, _poll)

        def _on_close():
            # Delete all StringVars on the GUI thread to avoid cross-thread errors
            for var in gain_best_vars.values():
                var.set("")
            for var in gain_try_vars.values():
                var.set("")
            gain_best_vars.clear()
            gain_try_vars.clear()
            self._closed.set()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", _on_close)
        root.mainloop()
        self._closed.set()


def _params_str(p: dict) -> str:
    if not p or not isinstance(p, dict):
        return "—"
    return "  ".join(f"{k}={v:.4g}" for k, v in p.items())
