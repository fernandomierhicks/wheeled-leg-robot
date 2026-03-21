"""es_engine.py — Generic (1+lambda)-ES optimizer.

Takes a SearchSpace + eval_fn and runs a (1+lambda) Evolution Strategy
with adaptive step sizes, patience-based early stopping, and multiprocessing.

Usage
-----
    from master_sim.optimizer.search_space import LQR_SPACE
    from master_sim.optimizer.es_engine import ESOptimizer

    def eval_fn(candidate: dict) -> dict:
        # ... run sim, return dict with 'fitness' and 'status' keys ...
        return {"fitness": 1.23, "status": "PASS", ...}

    opt = ESOptimizer(
        search_space=LQR_SPACE,
        eval_fn=eval_fn,
        csv_path="logs/S2_leg_height_gain_sched.csv",
    )
    result = opt.run(hours=1.0)
"""
from __future__ import annotations

import math
import multiprocessing
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from master_sim.optimizer.search_space import SearchSpace


# ── Hyper-parameter defaults ─────────────────────────────────────────────────

@dataclass
class ESConfig:
    """(1+lambda)-ES hyper-parameters."""
    lambda_: int = 8                # offspring per generation
    sigma_init: float = 1.00        # initial step size in log10 space
    sigma_min: float = 0.01
    sigma_max: float = 1.00
    success_target: float = 0.20    # 1/5 success rule
    adapt_window: int = 10          # generations for success rate estimate
    patience: int = 200             # gens without improvement → early stop
    tol: float = 1e-4               # relative improvement threshold
    n_workers: int | None = None    # None → min(lambda_, cpu_count)
    rng_seed: int | None = None


# ── Progress snapshot ────────────────────────────────────────────────────────

@dataclass
class ESProgress:
    """Snapshot passed to the progress callback each generation."""
    gen: int = 0
    n_evals: int = 0
    best_fitness: float = float("inf")
    parent_fitness: float = float("inf")
    best_gen: int = 0
    best_params: Dict[str, float] = field(default_factory=dict)
    success_rate: float = 0.0
    gens_without_improvement: int = 0
    elapsed_s: float = 0.0
    remaining_s: float = 0.0
    pct: float = 0.0
    status: str = "running"


# ── Top-level worker function (must be at module level for pickling) ─────────

def _worker_wrapper(args):
    """Multiprocessing wrapper — unpacks (eval_fn, candidate) and calls eval_fn."""
    eval_fn, candidate, label = args
    try:
        result = eval_fn(candidate)
        result.setdefault("label", label)
        return result
    except Exception as e:
        return {"fitness": 9999.0, "status": "FAIL", "fail_reason": str(e), "label": label}


# ── ESOptimizer ──────────────────────────────────────────────────────────────

class ESOptimizer:
    """Generic (1+lambda)-ES optimizer.

    Parameters
    ----------
    search_space : SearchSpace
        Defines parameter names, bounds, and log-space sampling.
    eval_fn : callable(candidate_dict) -> dict
        Evaluates one candidate. Must return dict with at least
        'fitness' (float, lower is better) and 'status' ("PASS"/"FAIL").
        Must be picklable for multiprocessing (top-level function or lambda-free).
    csv_path : str, optional
        Path for CSV logging (passed through to run_log if provided).
    config : ESConfig, optional
        Hyper-parameters (defaults are sensible for this project).
    progress_fn : callable(ESProgress) -> None, optional
        Called each generation with current state (for UI updates).
    pause_fn : callable() -> None, optional
        Called between generations. Should block while paused.
        Use ProgressUI.wait_if_paused for play/pause support.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        eval_fn: Callable[[Dict[str, float]], dict],
        csv_path: str | None = None,
        config: ESConfig | None = None,
        progress_fn: Callable[[ESProgress], None] | None = None,
        pause_fn: Callable[[], None] | None = None,
    ):
        self.space = search_space
        self.eval_fn = eval_fn
        self.csv_path = csv_path
        self.cfg = config or ESConfig()
        self.progress_fn = progress_fn
        self.pause_fn = pause_fn

        # Resolve worker count
        if self.cfg.n_workers is None:
            self.cfg.n_workers = min(self.cfg.lambda_, multiprocessing.cpu_count())

        # RNG
        self._rng = np.random.default_rng(self.cfg.rng_seed)

    # ── Public API ───────────────────────────────────────────────────────────

    def run(
        self,
        hours: float | None = None,
        max_iters: int | None = None,
        seed_params: Dict[str, float] | None = None,
        seed_fitness: float | None = None,
    ) -> dict:
        """Run the optimizer.

        Parameters
        ----------
        hours : float, optional
            Wall-clock time limit. Default 1.0 if max_iters is also None.
        max_iters : int, optional
            Generation count limit.
        seed_params : dict, optional
            Starting point (linear-space values). If None, random_init.
        seed_fitness : float, optional
            Known fitness of seed_params (skips initial evaluation).

        Returns
        -------
        dict with keys: 'best_params', 'best_fitness', 'n_evals', 'n_gens',
                        'elapsed_s', 'stopped_reason'.
        """
        if hours is None and max_iters is None:
            hours = 1.0

        # ── Initialise parent ────────────────────────────────────────────────
        if seed_params is not None:
            parent = self.space.clamp(seed_params)
        else:
            parent = self.space.random_init(self._rng)

        if seed_fitness is not None:
            parent_fit = seed_fitness
        else:
            # Evaluate seed
            result = self.eval_fn(parent)
            parent_fit = float(result.get("fitness", float("inf")))
            self._log_eval(parent, result, "seed", 0)

        # ── State ────────────────────────────────────────────────────────────
        sigmas = self.space.init_sigmas(self.cfg.sigma_init)
        success_window: deque = deque(maxlen=self.cfg.adapt_window)

        best_fit = parent_fit
        best_params = dict(parent)
        best_gen = 0
        gen = 0
        n_evals = 0 if seed_fitness is not None else 1
        gens_without_improvement = 0
        prev_best_for_patience = best_fit

        t_start = time.perf_counter()
        deadline = (t_start + hours * 3600.0) if hours else None
        stopped_reason = "completed"

        self._print_banner(hours, max_iters)

        # ── Main loop ────────────────────────────────────────────────────────
        with multiprocessing.Pool(processes=self.cfg.n_workers) as pool:
            while True:
                # Honour play/pause
                if self.pause_fn is not None:
                    self.pause_fn()

                # Check termination conditions
                if deadline and time.perf_counter() >= deadline:
                    stopped_reason = "time_limit"
                    break
                if max_iters is not None and gen >= max_iters:
                    stopped_reason = "max_iters"
                    break
                if gens_without_improvement >= self.cfg.patience:
                    stopped_reason = "patience"
                    print(f"\nEarly stop: no improvement in {self.cfg.patience} generations.")
                    break

                # Generate offspring
                children = [
                    self.space.sample_offspring(parent, sigmas, self._rng)
                    for _ in range(self.cfg.lambda_)
                ]

                labels = [f"g{gen:06d}_c{i}" for i in range(self.cfg.lambda_)]

                # Evaluate in parallel
                args = [(self.eval_fn, child, lbl) for child, lbl in zip(children, labels)]
                rows = pool.map(_worker_wrapper, args)
                n_evals += self.cfg.lambda_

                # Log evaluations
                for child, row, lbl in zip(children, rows, labels):
                    self._log_eval(child, row, lbl, gen)

                # Find best offspring
                gen_best_fit = float("inf")
                gen_best_params = None
                for child, row in zip(children, rows):
                    fit = float(row.get("fitness", float("inf")))
                    if row.get("status") == "PASS" and fit < gen_best_fit:
                        gen_best_fit = fit
                        gen_best_params = child

                improved = gen_best_params is not None and gen_best_fit < parent_fit
                success_window.append(1 if improved else 0)

                if improved:
                    parent = gen_best_params
                    parent_fit = gen_best_fit
                    if gen_best_fit < best_fit:
                        best_fit = gen_best_fit
                        best_params = dict(gen_best_params)
                        best_gen = gen

                # Convergence check
                rel_improvement = (prev_best_for_patience - best_fit) / (prev_best_for_patience + 1e-12)
                if rel_improvement > self.cfg.tol:
                    gens_without_improvement = 0
                    prev_best_for_patience = best_fit
                else:
                    gens_without_improvement += 1

                # Adapt step sizes (1/5 success rule)
                if len(success_window) >= self.cfg.adapt_window:
                    sr = sum(success_window) / len(success_window)
                    for k in sigmas:
                        if sr > self.cfg.success_target:
                            sigmas[k] = min(sigmas[k] * 1.22, self.cfg.sigma_max)
                        else:
                            sigmas[k] = max(sigmas[k] / 1.22, self.cfg.sigma_min)

                gen += 1

                # ── Progress reporting ───────────────────────────────────────
                elapsed_s = time.perf_counter() - t_start
                if deadline:
                    total_s = hours * 3600.0
                    pct = min(100.0, elapsed_s / total_s * 100.0)
                    remain_s = max(0.0, total_s - elapsed_s)
                else:
                    pct = min(100.0, gen / max_iters * 100.0)
                    remain_s = 0.0

                sr = sum(success_window) / len(success_window) if success_window else 0.0

                progress = ESProgress(
                    gen=gen,
                    n_evals=n_evals,
                    best_fitness=best_fit,
                    parent_fitness=parent_fit,
                    best_gen=best_gen,
                    best_params=dict(parent),
                    success_rate=sr,
                    gens_without_improvement=gens_without_improvement,
                    elapsed_s=elapsed_s,
                    remaining_s=remain_s,
                    pct=pct,
                    status="running",
                )
                if self.progress_fn is not None:
                    self.progress_fn(progress)

                if gen % 5 == 0:
                    self._print_progress(progress)

        # ── Final summary ────────────────────────────────────────────────────
        elapsed_s = time.perf_counter() - t_start
        elapsed_min = elapsed_s / 60.0

        print(f"\n{'=' * 72}")
        print(f"Optimization complete: {gen} gens, {n_evals} evals, {elapsed_min:.1f} min")
        print(f"Best fitness: {best_fit:.5f}  (gen {best_gen})  stopped: {stopped_reason}")
        print(f"Best params: {_params_to_str(best_params)}")

        if self.progress_fn is not None:
            self.progress_fn(ESProgress(
                gen=gen, n_evals=n_evals,
                best_fitness=best_fit, parent_fitness=parent_fit,
                best_gen=best_gen, best_params=best_params,
                elapsed_s=elapsed_s, pct=100.0, status="done",
            ))

        return {
            "best_params": best_params,
            "best_fitness": best_fit,
            "n_evals": n_evals,
            "n_gens": gen,
            "elapsed_s": elapsed_s,
            "stopped_reason": stopped_reason,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _log_eval(self, candidate: dict, result: dict, label: str, gen: int) -> None:
        """Log one evaluation to CSV if csv_path is set."""
        if self.csv_path is None:
            return
        from master_sim.optimizer.run_log import log_run
        row = dict(label=f"evo_{label}", gen=gen)
        row.update(candidate)
        row.update(result)
        log_run(row, self.csv_path)

    def _print_banner(self, hours, max_iters):
        print("=" * 72)
        print(f"(1+{self.cfg.lambda_})-ES  |  {self.space.dim}D  |  workers={self.cfg.n_workers}")
        print(f"  params: {list(self.space.names)}")
        print(f"  patience={self.cfg.patience}  tol={self.cfg.tol:.1e}")
        if hours:
            print(f"  time limit: {hours:.2f} h")
        else:
            print(f"  max iters: {max_iters}")
        print("=" * 72)

    @staticmethod
    def _print_progress(p: ESProgress):
        filled = int(p.pct / 100 * 40)
        bar = "[" + "#" * filled + "-" * (40 - filled) + "]"
        remain_str = f"{int(p.remaining_s // 60):02d}:{int(p.remaining_s % 60):02d}"
        elapsed_min = p.elapsed_s / 60.0
        print(f"\n{bar} {p.pct:5.1f}%  remain={remain_str}  "
              f"evals={p.n_evals}  elapsed={elapsed_min:.1f}min  "
              f"stagnant={p.gens_without_improvement}/{200}")
        print(f"  best={p.best_fitness:.5f}  parent={p.parent_fitness:.5f}  best@gen={p.best_gen}")
        print(f"  params: {_params_to_str(p.best_params)}")


def _params_to_str(p: dict) -> str:
    return "  ".join(f"{k}={v:.4g}" for k, v in p.items())
