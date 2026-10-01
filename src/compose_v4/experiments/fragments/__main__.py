"""Verified fragment assets, isolated CPU generation, and selector replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import EvidenceError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="repository or source-export root"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    assets = commands.add_parser(
        "assets", help="verify assets or copy one hash-checked local asset"
    )
    assets.add_argument("--assets", type=Path, required=True)
    assets.add_argument("--install", help="asset name from experiments/fragments/assets.json")
    assets.add_argument("--source", type=Path, help="existing local asset; never moved")
    assets.add_argument("--task", help="verify only this task's required assets")
    assets.add_argument(
        "--no-evaluator",
        action="store_true",
        help="check generation inputs without third-party evaluator files",
    )
    generate = commands.add_parser("generate", help="run a local fragment sampler")
    generate.add_argument("--task", required=True)
    generate.add_argument("--assets", type=Path, required=True)
    generate.add_argument("--python", type=Path, default=Path(sys.executable))
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--attempts", type=int, required=True, help="1..100 slots per prompt/seed"
    )
    generate.add_argument("--seed", type=int, action="append", dest="seeds")
    generate.add_argument("--prompt", action="append", dest="prompts")
    generate.add_argument(
        "--no-metrics",
        action="store_true",
        help="save molecules/traces only; do not evaluate benchmark metrics",
    )
    selection = commands.add_parser(
        "selection-ablation", help="replay learned and uniform selectors on saved offer panels"
    )
    selection.add_argument("--input", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    selection.add_argument("--seed", type=int, required=True)
    selection.add_argument("--assets", type=Path, help="verified evaluator assets for metrics")
    args = parser.parse_args(argv)
    try:
        if args.command == "assets":
            from .assets import install_asset, load_registry, verify_assets

            registry = load_registry(args.root)
            if args.install:
                if args.source is None:
                    parser.error("--install requires --source")
                print(install_asset(registry, args.assets, args.install, args.source))
            else:
                if args.source is not None:
                    parser.error("--source requires --install")
                checked = verify_assets(
                    registry,
                    args.assets,
                    args.task,
                    include_evaluator=not args.no_evaluator,
                )
                print(f"Verified {len(checked)} local assets. No files changed.")
            return 0
        if args.command == "generate":
            from .generation import run_generation

            result = run_generation(
                root=args.root,
                assets=args.assets,
                task=args.task,
                output=args.output,
                python=args.python,
                attempts=args.attempts,
                seeds=args.seeds,
                prompts=args.prompts,
                evaluate=not args.no_metrics,
            )
            print(f"Saved complete fragment {args.command} output to {result}")
            return 0
        if args.command == "selection-ablation":
            from .selection_ablation import run

            path = run(
                root=args.root,
                input_path=args.input,
                output_path=args.output,
                seed=args.seed,
                assets=args.assets,
            )
            print(f"Saved fixed-panel selector comparison to {path}")
            return 0
    except (EvidenceError, OSError, ImportError, ValueError) as exc:
        parser.exit(1, f"Fragment verification failed: {exc}\n")
    except (KeyError, TypeError) as exc:
        parser.exit(1, f"Fragment artifact has a malformed field: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
