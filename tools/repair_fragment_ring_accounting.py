"""Correct only ring-capability labels; preserve every sampled panel and RNG state."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from run_fragment_joint_prior_gate import prompt_ring_count


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_attempt(attempt, core_rings):
    result = json.loads(json.dumps(attempt))
    changes = 0
    for record in result["offered"]:
        if "capabilities" not in record:
            continue
        corrected = prompt_ring_count(record["endpoint"]) > core_rings
        changes += record["capabilities"]["new_ring"] != corrected
        record["capabilities"]["new_ring"] = corrected
    if result["committed_smiles"]:
        corrected = prompt_ring_count(result["committed_smiles"]) > core_rings
        changes += result["selected_capabilities"]["new_ring"] != corrected
        result["selected_capabilities"]["new_ring"] = corrected
    return result, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    summary_path = args.source_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["mode"] != "support" or len(summary["row_sha256"]) != 20:
        raise ValueError("requires complete support run, regardless of pass/fail")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent, prefix=".ring-report-") as temp:
        stage = Path(temp) / "complete"
        (stage / "attempts").mkdir(parents=True)
        records = []
        for name, expected in sorted(summary["row_sha256"].items()):
            path = args.source_dir / name
            if sha(path) != expected:
                raise ValueError(f"sealed row changed: {path}")
            row = json.loads(path.read_text())
            core_rings = prompt_ring_count(row["start_smiles"])
            for attempt in row["attempts"]:
                relative = (
                    f"attempts/{row['task']}_{row['drug']}_{attempt['attempt_index']:03d}.json"
                )
                original = args.source_dir / relative
                if json.loads(original.read_text()) != attempt:
                    raise ValueError("attempt differs from sealed row")
                corrected, changes = repair_attempt(attempt, core_rings)
                destination = stage / relative
                destination.write_text(json.dumps(corrected, indent=2, sort_keys=True) + "\n")
                records.append(
                    {
                        "path": relative,
                        "source_sha256": sha(original),
                        "corrected_sha256": sha(destination),
                        "labels_changed": changes,
                    }
                )
        receipt = {
            "schema": "fragment_ring_capability_accounting_correction_v1",
            "role": "reporting-only correction; no regeneration, reselection, scoring or RNG changes",
            "source_summary": str(summary_path.resolve()),
            "source_summary_sha256": sha(summary_path),
            "code_sha256": sha(Path(__file__)),
            "attempts": records,
            "next_step": "rerun same support reducer under corrected ring counter, without resampling completed attempts",
        }
        (stage / "accounting_correction.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        stage.rename(args.output_dir)
    print(
        json.dumps(
            {"attempts": len(records), "labels_changed": sum(r["labels_changed"] for r in records)}
        )
    )


if __name__ == "__main__":
    main()
