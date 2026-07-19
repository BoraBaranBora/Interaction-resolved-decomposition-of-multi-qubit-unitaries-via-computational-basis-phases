from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from control_optimization import ControlOptimizer, load_control_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gradient-based refinement of the ZZZ and XZZ NV-center pulses."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument(
        "--gradient-check",
        action="store_true",
        help="Compare selected autograd entries with central finite differences and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_control_config(args.config)
    if args.resume_from is not None:
        config = replace(config, resume_from=args.resume_from)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)

    optimizer = ControlOptimizer(config, project_root)
    if args.gradient_check:
        raw = optimizer._load_initial_raw()
        result = optimizer.gradient_check(raw)
        for check in result["checks"]:
            print(
                f"index={check['index']:3d}  analytic={check['analytic']:+.6e}  "
                f"finite_difference={check['finite_difference']:+.6e}  "
                f"relative_error={check['relative_error']:.3e}"
            )
        return
    optimizer.run()


if __name__ == "__main__":
    main()
