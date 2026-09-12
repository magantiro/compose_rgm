"""Convert locked complete programs and paid endpoint scores into development replay.

No fitting, new chemistry, new docking, or continuation-value labels are produced.
All examples from the two contexts belong to one original-source group. Multiple
program representations divide, rather than multiply, an endpoint's replay mass.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import EditProgram
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]


def normalize_endpoint_mass(records):
    """Equal source mass, then equal endpoint mass, then equal representation mass."""
    count = Counter((r["source_group"], r["endpoint"]) for r in records)
    endpoints = Counter(source for source, _ in count)
    sources = len(endpoints)
    return [
        {
            **r,
            "structural_replay_weight": 1
            / (sources * endpoints[r["source_group"]] * count[(r["source_group"], r["endpoint"])]),
        }
        for r in records
    ]


def prepare(root: Path, review_path: Path):
    review = unseal(review_path)
    for path, digest in review["input_sha256"].items():
        verify_file(root / path, digest)
    lock = unseal(root / "configs/t4_program_pool_lock.json")
    scores = {}
    for arm in review["arms"]:
        for row in arm["new_scored_endpoints"]:
            if row["smiles"] in scores and scores[row["smiles"]] != row["ds"]:
                raise ValueError("shared candidate has inconsistent docking labels")
            scores[row["smiles"]] = row["ds"]
    scores[review["winner"]["smiles"]] = review["winner"]["scores"][0]
    source_group = identity(
        {"original_benchmark_seed": lock["task"]["smiles"], "target": lock["task"]["target"]}
    )
    records, exclusions, input_pools = {}, [], {}
    for attempt, context in ((2, "original_seed"), (3, "post_linker")):
        folder = root / f"diagnostics/multi_site_proposal_probe/attempt_{attempt}"
        report = json.loads((folder / "result.json").read_text())
        for arm in report["arms"]:
            if arm["context"] != context:
                continue
            path = folder / arm["pool_path"]
            verify_file(path, arm["pool_sha256"])
            input_pools[str(path.relative_to(root))] = arm["pool_sha256"]
            pool = json.loads(gzip.decompress(path.read_bytes()))
            for row in pool["rows"]:
                origin = {
                    "path": str(path.relative_to(root)),
                    "attempt": row["attempt"],
                    "context": context,
                    "arm": arm["arm"],
                }
                if row["status"] != "complete":
                    exclusions.append({**origin, "reason": row["reason_code"], "task_label": None})
                    continue
                trace = row["receipt"]
                endpoint = trace["endpoint"]
                if endpoint not in scores:
                    exclusions.append(
                        {
                            **origin,
                            "reason": "no_paid_eligible_endpoint_label",
                            "endpoint": endpoint,
                            "task_label": None,
                        }
                    )
                    continue
                program = EditProgram.from_payload(row["program"])
                # Validate the typed payload and bind the existing exact trace. Its
                # verified prefix states are reused, not re-enumerated or reparsed.
                key = identity(
                    {
                        "source": trace["states"][0],
                        "program": program.payload(),
                        "assignment": trace["assignment"],
                        "actions": trace["actions"],
                    }
                )
                if key not in records:
                    records[key] = {
                        "example_id": key,
                        "source_group": source_group,
                        "split": "inspected_development_only",
                        "original_seed": lock["task"]["smiles"],
                        "source_state": trace["states"][0],
                        "program": program.payload(),
                        "assignment": trace["assignment"],
                        "executed_trace": trace,
                        "dependency_graph": row["graph"],
                        "endpoint": endpoint,
                        "oracle_label": {
                            "docking_score": scores[endpoint],
                            "docking_seed": 1701,
                            "reused_winner_control": endpoint == review["winner"]["smiles"],
                        },
                        "observed_policy_advantage": None,
                        "continuation_value_target": None,
                        "origins": [],
                    }
                records[key]["origins"].append(origin)
    bank = normalize_endpoint_mass([records[k] for k in sorted(records)])
    return {
        "schema_version": "scored_attachment_program_replay_v1",
        "records": bank,
        "exclusions": exclusions,
        "inputs": {
            "review_path": str(review_path),
            "review_sha256": sha256_file(review_path),
            "pool_sha256": input_pools,
        },
        "oracle_protocol": {
            "task": lock["task"],
            "docking": lock["docking"],
            "required_rdkit": lock["required_rdkit"],
        },
        "producer_sha256": sha256_file(Path(__file__)),
        "evidence_regime": "winner-informed, one-source development; same-source fit/evaluation cannot demonstrate transfer",
        "counts": {
            "program_representations": len(bank),
            "scored_endpoints": len({r["endpoint"] for r in bank}),
            "source_groups": len({r["source_group"] for r in bank}),
            "excluded_attempts": len(exclusions),
        },
        "new_oracle_calls": 0,
        "new_executor_calls": 0,
        "model_fitting": False,
        "sampling": "no sampling; deterministic conversion; structural weights do not assert policy gradients or score advantages",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("replay output already exists; reuse it or name a distinct revision")
    bank = prepare(ROOT, args.review)
    publish_json(args.output, bank)
    print(json.dumps(bank["counts"], indent=2))


if __name__ == "__main__":
    main()
