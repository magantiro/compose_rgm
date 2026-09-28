"""Fragment tables, verified local assets, and isolated CPU generation. No cloud jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evidence import EvidenceError, load_evidence
from .reduction import recompute_intervals, reduce_evidence
from .reporting import provenance, publish


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="repository or source-export root"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "tables"):
        child = commands.add_parser(command)
        child.add_argument(
            "--recompute-intervals",
            action="store_true",
            help="repeat historical prompt bootstrap; requires numpy==1.26.4",
        )
        if command == "tables":
            child.add_argument(
                "--output",
                type=Path,
                required=True,
                help="new directory; existing files are never replaced",
            )
    assets = commands.add_parser(
        "assets", help="verify assets or copy one hash-checked local asset"
    )
    assets.add_argument("--assets", type=Path, required=True)
    assets.add_argument("--install", help="asset name from experiments/fragments/assets.json")
    assets.add_argument("--source", type=Path, help="existing local asset; never moved")
    assets.add_argument("--task", help="verify only this task's required assets")
    for name in ("generate", "parity"):
        child = commands.add_parser(
            name,
            help="run local frozen samplers"
            if name == "generate"
            else "compare identity-selected saved attempts",
        )
        child.add_argument("--task", required=True)
        child.add_argument("--assets", type=Path, required=True)
        child.add_argument("--python", type=Path, default=Path(sys.executable))
        child.add_argument("--output", type=Path, required=True)
        if name == "generate":
            child.add_argument(
                "--attempts", type=int, required=True, help="1..100 slots per prompt/seed"
            )
            child.add_argument("--seed", type=int, action="append", dest="seeds")
            child.add_argument("--prompt", action="append", dest="prompts")
            child.add_argument(
                "--no-metrics",
                action="store_true",
                help="save molecules/traces only; do not evaluate benchmark metrics",
            )
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
                checked = verify_assets(registry, args.assets, args.task)
                print(f"Verified {len(checked)} local assets. No files changed.")
            return 0
        if args.command in {"generate", "parity"}:
            from .generation import run_generation

            result = run_generation(
                root=args.root,
                assets=args.assets,
                task=args.task,
                output=args.output,
                python=args.python,
                attempts=getattr(args, "attempts", 1),
                seeds=getattr(args, "seeds", None),
                prompts=getattr(args, "prompts", None),
                parity=args.command == "parity",
                evaluate=not getattr(args, "no_metrics", False),
            )
            print(f"Saved complete fragment {args.command} output to {result}")
            return 0
        evidence = load_evidence(args.root)
        report = reduce_evidence(evidence)
        if args.recompute_intervals:
            recompute_intervals(report)
        if args.command == "tables":
            publish(args.output, report, provenance(evidence, report))
            print(f"Saved fragment tables and provenance to {args.output.absolute()}")
        else:
            print(
                "Verified 7 artifact hashes and complete saved metric panels for 4 independent tasks."
            )
            print("Generation, model weights and raw molecular evaluation were not checked.")
    except (EvidenceError, OSError, ImportError, ValueError) as exc:
        parser.exit(1, f"Fragment verification failed: {exc}\n")
    except (KeyError, TypeError) as exc:
        parser.exit(1, f"Fragment artifact has a malformed field: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
