"""Materialize the bounded Process-V2 T1 unique-state panel.

This command performs no optimization.  It requires an existing Process-V2
Active8 completion and Gate-0 PASS, streams train decision shards, publishes one
content-authenticated panel, and exits nonzero on any insufficiency or mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
    ProcessV2T1PanelError,
    build_process_v2_t1_panel,
    open_process_v2_t1_source,
    write_process_v2_t1_panel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active8-run-root", type=Path, required=True)
    parser.add_argument("--gate-zero-decision", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    try:
        source = open_process_v2_t1_source(
            args.active8_run_root,
            gate_zero_decision_path=args.gate_zero_decision,
            artifact_root=args.artifact_root,
            repo_root=args.repo_root,
        )
        panel = build_process_v2_t1_panel(source, scratch_dir=args.scratch_dir)
        path = write_process_v2_t1_panel(
            panel, output_root=args.output_root, source=source
        )
    except ProcessV2T1PanelError as error:
        print(f"Process-V2 T1 panel refused: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "panel_path": str(path),
                "panel_sha256": panel["panel_sha256"],
                "entry_count": len(panel["entries"]),
                "family_counts": panel["family_counts"],
                "capability_cell_counts": panel["capability_cell_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
