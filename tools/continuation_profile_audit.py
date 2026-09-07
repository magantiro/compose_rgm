"""Inventory a downloaded failed profile without executing molecular chemistry.

This checks stored payload identities, not scientific equivalence to a new
evaluator. Complete rows remain candidates for validated reuse, not authority
to interpret missing branches as zero-probability successors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

from compose_v4.experiments.continuation_profile import canonical_bytes, publish_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input.resolve()
    if not directory.is_dir():
        parser.error(f"profile directory missing: {directory}")
    metadata = {
        name: json.loads((directory / f"{name}.json").read_text())
        for name in ("launch", "progress", "failure")
    }
    run_id = metadata["launch"]["run_id"]
    if directory.name != run_id or metadata["progress"]["run_id"] != run_id:
        raise ValueError("profile directory and launch/progress run identities disagree")
    if metadata["progress"]["complete"] or metadata["progress"]["phase"] != "failed":
        raise ValueError("this audit expects a failed, incomplete profile")
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    manifest = {str(p.relative_to(directory)): sha256_file(p) for p in files}
    rows, laws = [], []
    for kind, collection in (("rows", rows), ("laws", laws)):
        for path in sorted((directory / kind).glob("*.json")):
            payload = json.loads(path.read_text())
            expected_schema = f"continuation_profile_{'row' if kind == 'rows' else 'law'}_v1"
            if payload["schema_version"] != expected_schema or payload["snapshot_id"] != run_id:
                raise ValueError(f"unexpected schema or snapshot identity: {path}")
            if hashlib.sha256(canonical_bytes(payload)).hexdigest() != path.stem:
                raise ValueError(f"content-address mismatch: {path}")
            if len(payload["marks"]) != len(payload["probabilities"]):
                raise ValueError(f"unaligned marks/probabilities: {path}")
            if kind == "rows" and len(payload["successors"]) != len(payload["probabilities"]):
                raise ValueError(f"unaligned successor row: {path}")
            collection.append(payload)
    attempts = []
    for path in sorted((directory / "attempts").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload["snapshot_id"] != run_id:
            raise ValueError(f"attempt snapshot mismatch: {path}")
        attempts.extend(payload["attempts"])
    if len(rows) != metadata["progress"]["completed_rows"]:
        raise ValueError("downloaded row count does not match durable progress")
    root_rows = [row for row in rows if row["source"]["step"] == 0]
    if len(root_rows) != 1:
        raise ValueError("expected one complete root row")
    report = {
        "schema_version": "continuation_failed_profile_audit_v1",
        "evidence_role": "computed inventory of failed production development profile",
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "helper_sha256": sha256_file(
            Path(__file__).resolve().parents[1]
            / "src/compose_v4/experiments/continuation_profile.py"
        ),
        "input": {
            "local_path": str(directory),
            "volume": "compose-v4-artifacts",
            "remote_path": f"/continuation_profile/{run_id}",
            "sha256": manifest,
        },
        "configuration": {
            "seed": None,
            "seed_reason": "deterministic inventory, no sampling",
            "workers": 1,
            "exclusions": [],
        },
        "software": {"python": platform.python_version()},
        "metadata": metadata,
        "counts": {
            "complete_rows": len(rows),
            "complete_laws": len(laws),
            "root_successors": len(root_rows[0]["successors"]),
            "rows_by_step": dict(sorted(Counter(row["source"]["step"] for row in rows).items())),
            "empty_rows": sum(not row["successors"] for row in rows),
            "public_calls_reported": metadata["progress"]["total_public_executor_calls"],
            "attempt_receipts": len(attempts),
            "receipt_gap": metadata["progress"]["total_public_executor_calls"] - len(attempts),
            "recorded_attempts_by_phase": dict(
                sorted(Counter(a["phase"] for a in attempts).items())
            ),
            "input_bytes": sum(p.stat().st_size for p in files),
            "new_executor_calls": 0,
            "new_oracle_calls": 0,
        },
        "decision": "preserve completed payloads; do not rerun the exhausted exact tree or promote a partial backup",
        "limitations": [
            "Stored hash checks are not direct evaluator equivalence checks.",
            "Legacy interrupted-call accounting is incomplete; no missing receipt was synthesized.",
            "Empty rows concern sampled precursor states, not global ring reachability.",
            "Missing cache rows must cause an explicit miss, never a fabricated empty reference row.",
        ],
    }
    digest = publish_json(args.output, report)
    print(
        json.dumps(
            {"output": str(args.output), "sha256": digest, "counts": report["counts"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
