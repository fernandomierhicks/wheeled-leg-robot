"""progress_window.py — Floating progress window for the LQR optimizer.

Runs a small always-on-top tkinter window in a daemon thread.
The optimizer calls win.update(**state) each generation; the window
polls every 300 ms and redraws itself.  No extra dependencies.
"""
import threading
import time
import tkinter as tk


# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#1e1e2e"
FG        = "#cdd6f4"
FG_DIM    = "#6c7086"
ACCENT    = "#89b4fa"   # blue  — bar fill
ACCENT2   = "#a6e3a1"   # green — best fitness
BAR_BG    = "#313244"
FONT_MONO = ("Consolas", 10)
FONT_BIG  = ("Consolas", 12, "bold")
FONT_SM   = ("Consolas", 9)


class ProgressWindow:
    """Floating progress window driven by the optimizer.

    Usage
    -----
    win = ProgressWindow("LQR Optimizer — Scenario 1")
    win.update(pct=42.0, remaining_s=178, n_evals=336, gen=42,
               best_fit=1.692, best_gains="Q=[3.98,0.70,0.0001] R=5.11",
               status="running")
    # ... optimizer finishes ...
    win.finish()          # marks done, window stays open until user closes it
    win.wait_closed()     # optional: block until the user closes the window
    """

    def __init__(self, title: str = "Optimizer Progress"):
        self._lock   = threading.Lock()
        self._state  = dict(
            pct=0.0, elapsed_s=0, remaining_s=0,
            n_evals=0, gen=0,
            best_fit=float("inf"), best_gains="—",
            status="starting",
        )
        self._done       = threading.Event()
        self._closed     = threading.Event()
        self._t = threading.Thread(target=self._run, args=(title,), daemon=True)
        self._t.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, **kwargs):
        """Thread-safe state update called by the optimizer."""
        with self._lock:
            self._state.update(kwargs)

    def finish(self):
        """Signal that optimization is complete (window stays open)."""
        self.update(status="done", pct=100.0, remaining_s=0)
        self._done.set()

    def wait_closed(self):
        """Block until the user closes the window."""
        self._closed.wait()

    # ── Private / GUI thread ──────────────────────────────────────────────────

    def _snap(self):
        with self._lock:
            return dict(self._state)

    def _run(self, title: str):
        root = tk.Tk()
        root.title(title)
        root.configure(bg=BG)
        root.resizable(False, False)
        root.attributes("-topmost", True)

        PAD = 14
        W   = 420

        # ── Title row ─────────────────────────────────────────────────────────
        tk.Label(root, text=title, font=FONT_BIG, bg=BG, fg=ACCENT
                 ).pack(pady=(PAD, 4), padx=PAD, anchor="w")

        # ── Progress bar (canvas) ─────────────────────────────────────────────
        bar_frame = tk.Frame(root, bg=BG)
        bar_frame.pack(fill="x", padx=PAD, pady=(0, 4))

        canvas = tk.Canvas(bar_frame, height=18, bg=BAR_BG,
                           highlightthickness=0, relief="flat")
        canvas.pack(fill="x")
        bar_rect  = canvas.create_rectangle(0, 0, 0, 18, fill=ACCENT, width=0)
        bar_label = canvas.create_text(W // 2, 9, text="0%",
                                       fill=FG, font=FONT_MONO, anchor="center")

        # ── Info grid ─────────────────────────────────────────────────────────
        info_frame = tk.Frame(root, bg=BG)
        info_frame.pack(fill="x", padx=PAD, pady=(0, 4))

        def _row(parent, label, row, col=0):
            tk.Label(parent, text=label, font=FONT_SM, bg=BG, fg=FG_DIM,
                     anchor="w").grid(row=row, column=col*2,
                                      sticky="w", padx=(0, 4))
            var = tk.StringVar(value="—")
            tk.Label(parent, textvariable=var, font=FONT_SM, bg=BG, fg=FG,
                     anchor="w").grid(row=row, column=col*2+1, sticky="w")
            return var

        v_elapsed   = _row(info_frame, "Elapsed",    0, 0)
        v_remaining = _row(info_frame, "Remaining",  0, 1)
        v_evals     = _row(info_frame, "Evals",      1, 0)
        v_gen       = _row(info_frame, "Generation", 1, 1)

        # ── Best fitness ──────────────────────────────────────────────────────
        sep = tk.Frame(root, height=1, bg=BAR_BG)
        sep.pack(fill="x", padx=PAD, pady=(2, 6))

        best_frame = tk.Frame(root, bg=BG)
        best_frame.pack(fill="x", padx=PAD, pady=(0, 4))

        tk.Label(best_frame, text="Best fitness", font=FONT_SM,
                 bg=BG, fg=FG_DIM, anchor="w").pack(anchor="w")
        v_best_fit = tk.StringVar(value="—")
        tk.Label(best_frame, textvariable=v_best_fit, font=FONT_BIG,
                 bg=BG, fg=ACCENT2, anchor="w").pack(anchor="w")

        tk.Label(best_frame, text="Best gains", font=FONT_SM,
                 bg=BG, fg=FG_DIM, anchor="w").pack(anchor="w", pady=(4, 0))
        v_best_gains = tk.StringVar(value="—")
        tk.Label(best_frame, textvariable=v_best_gains, font=FONT_MONO,
                 bg=BG, fg=FG, anchor="w", wraplength=W - PAD*2
                 ).pack(anchor="w")

        # ── Status bar ────────────────────────────────────────────────────────
        sep2 = tk.Frame(root, height=1, bg=BAR_BG)
        sep2.pack(fill="x", padx=PAD, pady=(6, 0))

        v_status = tk.StringVar(value="starting…")
        tk.Label(root, textvariable=v_status, font=FONT_SM,
                 bg=BG, fg=FG_DIM, anchor="w"
                 ).pack(anchor="w", padx=PAD, pady=(4, PAD))

        root.update_idletasks()
        # Centre the window on screen
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
            v_remaining.set(_fmt_time(s["remaining_s"]) if s["status"] != "done" else "00:00")
            v_evals.set(str(s["n_evals"]))
            v_gen.set(str(s["gen"]))

            # Best
            bf = s["best_fit"]
            v_best_fit.set(f"{bf:.5f}" if bf < 1e9 else "—")
            v_best_gains.set(s["best_gains"])

            # Status
            if s["status"] == "done":
                v_status.set("Optimization complete — you may close this window")
                v_status_color = ACCENT2
            elif s["status"] == "starting":
                v_status_color = FG_DIM
                v_status.set("Starting up…")
            else:
                v_status_color = FG_DIM
                sr = s.get("success_rate", None)
                sr_str = f"  |  success rate {sr:.0%}" if sr is not None else ""
                v_status.set(f"Running{sr_str}")

            if not self._closed.is_set():
                root.after(300, _poll)

        root.after(300, _poll)
        root.protocol("WM_DELETE_WINDOW", lambda: (self._closed.set(), root.destroy()))
        root.mainloop()
        self._closed.set()
