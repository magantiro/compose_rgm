"""Verify or reduce frozen fragment metrics; never launch generation or cloud jobs."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args(argv)
    try:
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
    except (EvidenceError, OSError, ImportError) as exc:
        parser.exit(1, f"Fragment verification failed: {exc}\n")
    except (KeyError, TypeError) as exc:
        parser.exit(1, f"Fragment artifact has a malformed field: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
