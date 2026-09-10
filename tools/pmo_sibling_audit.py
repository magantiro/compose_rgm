"""Audit same-parent alternatives using saved chronological predictions only."""

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_sibling_audit import TOLERANCE, audit_rows, joined_rows

ROOT = Path(__file__).resolve().parents[1]
INPUT = "diagnostics/pmo_chronological/result.json"
EXPECTED = "6e164790e344eaeb41dff735fcb14ef17f1928211bf5a8c07f2b7118dc665873"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to replace existing result: {args.output}")
    start = perf_counter()
    hashes = {INPUT: verify_file(ROOT / INPUT, EXPECTED)}
    previous = json.loads((ROOT / INPUT).read_text())
    if previous["decision"] != "no_predictor_qualified":
        raise ValueError("unexpected predecessor decision")
    arms = []
    for arm in previous["arms"]:
        name = arm["case"]["arm"]
        path = f"diagnostics/pmo_archive_pilot_100/{name}.json"
        hashes[path] = verify_file(ROOT / path, previous["input_hashes"][path])
        original = json.loads((ROOT / path).read_text())
        if original["case"] != arm["case"]:
            raise ValueError(f"arm identity mismatch: {path}")
        rows = joined_rows(arm["predictions"], original)
        arms.append({"case": arm["case"], **audit_rows(rows)})
    dependencies = [
        "docs/PMO_SIBLING_AUDIT.md",
        "src/compose_v4/experiments/pmo_sibling_audit.py",
        "src/compose_v4/experiments/continuation_profile.py",
        "tools/pmo_sibling_audit.py",
    ]
    report = {
        "schema_version": "pmo_sibling_audit_v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "input_hashes": hashes,
        "implementation_hashes": {p: sha256_file(ROOT / p) for p in dependencies},
        "configuration": {
            "tolerance": TOLERANCE,
            "group": ["arm", "exact_parent_id", "fit_through_query"],
        },
        "software": {"python": platform.python_version()},
        "hardware": {
            "platform": platform.platform(),
            "device": "cpu",
            "precision": "python_float64",
        },
        "random_sampling": False,
        "seed": None,
        "new_oracle_calls": 0,
        "new_fits": 0,
        "split": "same_parent_same_snapshot_within_inspected_development_arm",
        "predecessor_decision": previous["decision"],
        "decision": "diagnostic_only_no_guidance_authorized",
        "seconds": perf_counter() - start,
        "arms": arms,
    }
    publish_json(args.output, report)
    for arm in arms:
        print(
            arm["case"]["arm"],
            json.dumps(
                {k: arm[k] for k in ("prediction_count", "parent_count", "primary", "mixed_cycle")}
            ),
        )
    print(
        json.dumps({k: report[k] for k in ("seconds", "new_oracle_calls", "new_fits", "decision")})
    )


if __name__ == "__main__":
    main()
