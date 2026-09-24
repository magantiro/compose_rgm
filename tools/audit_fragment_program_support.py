"""No-score distribution audit of complete programs versus saved local attempts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, rdBase
from run_fragment_attachment_library_pilot import _atomic_json

from compose_v4.benchmark.training_attachment_fragments import physical_sha256


def ring_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid emitted molecule in support audit")
    return mol.GetRingInfo().NumRings()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary_path = args.support_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if len(summary["shard_hashes"]) != 20:
        raise ValueError("distribution audit requires the entire twenty-prompt support preview")
    inputs = {str(summary_path.resolve()): physical_sha256(summary_path)}
    inputs[str(Path(__file__).resolve())] = physical_sha256(Path(__file__))
    tasks = defaultdict(lambda: {"baseline": Counter(), "program": Counter()})
    families, refusals = Counter(), Counter()
    for relative, expected in sorted(summary["shard_hashes"].items()):
        path = args.support_dir / relative
        if physical_sha256(path) != expected:
            raise ValueError(f"support shard changed: {path}")
        inputs[str(path.resolve())] = expected
        row = json.loads(path.read_text())
        baseline_path = (
            args.baseline_dir / "shards" / f"{row['task']}__{row['drug']}__baseline.json"
        )
        baseline = json.loads(baseline_path.read_text())
        if (baseline["task"], baseline["drug"]) != (row["task"], row["drug"]):
            raise ValueError("baseline prompt identity mismatch")
        inputs[str(baseline_path.resolve())] = physical_sha256(baseline_path)
        initial_rings = ring_count(row["start_smiles"])
        for arm, records in (
            ("baseline", baseline["attempt_records"][:2]),
            ("program", row["attempts"]),
        ):
            if len(records) != 2:
                raise ValueError(
                    "support comparison requires first two attempts of each exact prompt"
                )
            counts = tasks[row["task"]][arm]
            for attempt in records:
                counts["attempts"] += 1
                smiles = attempt["committed_smiles"]
                if smiles is None:
                    counts["no_output"] += 1
                    continue
                counts["outputs"] += 1
                counts["outputs_adding_ring"] += ring_count(smiles) > initial_rings
                if arm == "baseline":
                    counts["accepted_local_events"] += attempt["events"]
                    counts["coordinated_program_proposals"] += 0
                    continue
                cap = attempt["selected_capabilities"]
                counts["selected_multi_primitive"] += cap["multi_primitive"]
                counts["selected_one_step"] += cap["one_step"]
                counts["selected_primitives"] += cap["primitives"]
                counts["selected_blocks"] += cap["program_blocks"]
                counts["selected_dependency_edges"] += cap["dependency_edges"]
                counts["selected_conflict_edges"] += cap["conflict_edges"]
                counts["selected_native_t4"] += cap["t4_refinement"]
                counts["selected_two_boundary"] += cap["two_boundary_program"]
        for attempt in row["attempts"]:
            for candidate in attempt["offered"]:
                if candidate["status"] != "model_supported":
                    refusals[candidate.get("reason", candidate["status"])] += 1
                    continue
                metadata = candidate["provenance"].get("t4_metadata", {})
                families.update(m["family"] for m in metadata.get("modules", ()))
    _atomic_json(
        args.output,
        {
            "schema": "fragment_complete_program_distribution_audit_v1",
            "inputs": inputs,
            "rdkit": rdBase.rdkitVersion,
            "selection": "first two saved attempts per arm per prompt, no quality selection",
            "tasks": {
                task: {arm: dict(c) for arm, c in arms.items()} for task, arms in tasks.items()
            },
            "model_supported_native_t4_families": dict(sorted(families.items())),
            "refusal_taxonomy": dict(sorted(refusals.items())),
            "interpretation": (
                "Local event counts are trajectory lengths, not coordinated-program lengths. "
                "Native T4 family counts describe invoked constructors, not an independently "
                "matched T4 optimization distribution. No docking, QED or SA was evaluated."
            ),
        },
    )
    print(
        json.dumps(
            {task: {arm: dict(c) for arm, c in arms.items()} for task, arms in tasks.items()}
        )
    )


if __name__ == "__main__":
    main()
