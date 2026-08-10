"""Robust (1+lambda)-ES tuning of real firmware gains over a plant ensemble."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import statistics
import sys
import tempfile

from .es_engine import ESConfig, ESOptimizer
from .firmware_search_space import FIRMWARE_SPACE_BY_STAGE
from ..twin.params_control import PARAMS_BY_NAME
from ..twin.params_plant import plant_ensemble
from ..twin.runtime import run_scenario
from ..twin.scenario import Scenario


ROOT = Path(__file__).resolve().parents[4]
GUI_ROOT = ROOT / "software" / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))
from analysis.wlog_metrics import evaluate  # noqa: E402


def evaluate_candidate(candidate: dict[str, float], *, scenario_path: str,
                       ensemble_size: int, ensemble_seed: int) -> dict:
    scenario = Scenario.load(Path(scenario_path))
    metrics_stage = scenario.stage if scenario.stage in ("lqr", "vel_pi", "yaw_pi") else "lqr"
    scores: list[float] = []
    safety_reasons: list[str] = []
    with tempfile.TemporaryDirectory(prefix="wlr_twin_eval_") as directory:
        for index, plant in enumerate(plant_ensemble(ensemble_size, ensemble_seed)):
            path = Path(directory) / f"plant_{index}.WLOG"
            run_scenario(scenario, path, plant=plant, control_overrides=candidate,
                         seed=ensemble_seed + index)
            result = evaluate(path, metrics_stage)
            scores.append(float(result["fitness"]))
            safety_reasons.extend(f"plant {index}: {reason}"
                                  for reason in result["safety_reasons"])
    return {
        "fitness": max(scores), "median_fitness": statistics.median(scores),
        "status": "PASS" if not safety_reasons else "FAIL",
        "plant_scores": scores, "safety_reasons": safety_reasons,
    }


def _snapshot(params: dict[str, float]) -> dict:
    return {
        f"0x{PARAMS_BY_NAME[name].id:04X}": {"name": name, "value": value}
        for name, value in params.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--stage", choices=tuple(FIRMWARE_SPACE_BY_STAGE), default="lqr")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--lambda", dest="lambda_", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenario = Scenario.load(args.scenario)
    space = FIRMWARE_SPACE_BY_STAGE[args.stage]
    seed_params = {name: PARAMS_BY_NAME[name].default for name in space.names}
    evaluator = partial(evaluate_candidate, scenario_path=str(args.scenario.resolve()),
                        ensemble_size=args.ensemble_size, ensemble_seed=args.seed)
    optimizer = ESOptimizer(
        space, evaluator,
        config=ESConfig(lambda_=args.lambda_, n_workers=min(args.lambda_, args.ensemble_size),
                        rng_seed=args.seed, sigma_init=0.08),
    )
    result = optimizer.run(max_iters=args.generations, seed_params=seed_params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_snapshot(result["best_params"]), indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({**result, "output": str(args.output),
                      "scenario": scenario.name, "ensemble_size": args.ensemble_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
