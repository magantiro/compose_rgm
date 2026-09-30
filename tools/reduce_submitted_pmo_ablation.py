"""Recompute the submitted 14-cell PMO A/B table from its pinned saved reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from compose_v4.experiments.pmo_submitted_ablation import (
    EXPECTED_COMPLETED,
    AblationError,
    reduce_ablation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=str, required=True, help="Saved reduction JSON path or '-' for stdin"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root"
    )
    args = parser.parse_args()
    try:
        raw = sys.stdin.buffer.read() if args.input == "-" else Path(args.input).read_bytes()
        manifest = json.loads((args.root / "experiments/paper/manifest.json").read_text())
        source = next(row for row in manifest["sources"] if row["id"] == "pmo_structured_proposals")
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != source["artifact_sha256"]:
            raise AblationError(f"input SHA-256 {actual_sha} != pinned {source['artifact_sha256']}")
        result = reduce_ablation(json.loads(raw), EXPECTED_COMPLETED)
        result["source_sha256"] = actual_sha
        if [
            round(result["aggregate"][key]["mean"], 3)
            for key in ("A_top10", "B_top10", "A_auc", "B_auc")
        ] != [0.621, 0.558, 0.503, 0.465]:
            raise AblationError("recomputed aggregate does not match submitted Table 18 rounding")
    except (OSError, json.JSONDecodeError, StopIteration, AblationError) as exc:
        parser.exit(2, f"PMO submitted-ablation input error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
