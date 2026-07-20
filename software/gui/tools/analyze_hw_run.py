"""analyze_hw_run.py — analyze one hardware .wlog run: metrics + fitness + safety verdict.

Thin CLI wrapper around analysis/wlog_metrics.py — this is what Claude invokes
in the Phase-1 hardware tuning loop's automated per-trial step (tuning.md
§Test protocol step 7). Prints one JSON summary to stdout; verdict + reasons
to stderr.

Usage:
    python analyze_hw_run.py <path.wlog> --stage {lqr,vel_pi,yaw_pi} [--wheel-vel-limit N]

Exit codes: 0 = safe, 2 = unsafe (safety-maxima violation), 1 = couldn't analyze.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # software/gui/
from analysis.wlog_metrics import DEFAULT_WHEEL_VEL_LIMIT_TURNS_S, STAGES, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wlog_path", type=Path)
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument(
        "--wheel-vel-limit", type=float, default=DEFAULT_WHEEL_VEL_LIMIT_TURNS_S,
        help=f"PARAM_WHEEL_VEL_LIMIT_TURNS_S at capture time (default {DEFAULT_WHEEL_VEL_LIMIT_TURNS_S}, "
             "the firmware compile-time default — pass the live value if it's been changed)",
    )
    args = ap.parse_args()

    try:
        result = evaluate(args.wlog_path, args.stage, args.wheel_vel_limit)
    except (ValueError, OSError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))

    verdict = "SAFE" if result["safety_ok"] else "UNSAFE"
    print(f"\n{verdict} — fitness={result['fitness']} (stage={args.stage})", file=sys.stderr)
    for reason in result["safety_reasons"]:
        print(f"  - {reason}", file=sys.stderr)

    sys.exit(0 if result["safety_ok"] else 2)


if __name__ == "__main__":
    main()
