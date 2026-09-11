"""Summarize every locked continuation candidate and the paired fresh outcomes."""

import argparse
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.control.graph_geometry import topology
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    paths = [args.directory / "result_sealed.json"] + [
        args.directory / f"phase{phase}_scored.json" for phase in (1, 2, 3)
    ]
    result = unseal(paths[0])
    phases, candidates = {}, []
    for phase, path in enumerate(paths[1:], 1):
        slots = unseal(path)["slots"]
        rows = [c for row in slots for c in row if c is not None]
        candidates.extend(rows)
        phases[str(phase)] = {
            "attempts": 4 * len(slots),
            "completed": len(rows),
            "unique_canonical": len({c["smiles"] for c in rows}),
            "options": dict(Counter(c["bundle"]["option"] for c in rows)),
            "cycle_rank_increased": sum(c["structural_change"]["d_cycle_rank"] > 0 for c in rows),
            "cycle_rank_decreased": sum(c["structural_change"]["d_cycle_rank"] < 0 for c in rows),
            "ring_system_increased": sum(
                c["structural_change"]["d_ring_systems"] > 0 for c in rows
            ),
            "candidates": [
                {
                    k: c[k]
                    for k in (
                        "id",
                        "smiles",
                        "score",
                        "parent_smiles",
                        "parent_score",
                        "primitive_count",
                        "bundle",
                        "structural_change",
                        "topology",
                    )
                }
                for c in rows
            ],
        }
    pairs = []
    for choice in result["choices"]:
        pairs.append(
            {
                "source": choice["source"],
                "parent_score": choice["parent_score"],
                **{
                    arm: {
                        "selected_draw": d["selected"],
                        "immediate_score": d["fresh"]["selected_immediate_score"],
                        "witness_value": d["best_witness"][d["selected"]],
                        "fresh_mean": d["fresh"]["mean_return"],
                        "fresh_best": d["fresh"]["best_return"],
                        "abstained": d["abstained"],
                    }
                    for arm, d in choice["decisions"].items()
                },
            }
        )
    unique = {c["smiles"]: c for c in candidates}
    best = max(unique.values(), key=lambda c: (c["score"], c["smiles"])) if unique else None
    witness_path = ROOT / "diagnostics/pmo_public_winner_recovery/current_best.json"
    witness = json.loads(witness_path.read_text())["result"]
    target = decode_state(witness["states"][-1])
    if canonical_state_key(target) != result["public_development_reference"]["smiles"]:
        raise ValueError("witness endpoint differs from locked public target")
    output = {
        "schema_version": "pmo_continuation_audit_v1",
        "input_sha256": {str(p): sha256_file(p) for p in [*paths, witness_path]},
        "implementation_sha256": sha256_file(Path(__file__)),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "code_worktree_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "geometry_implementation_sha256": sha256_file(
            ROOT / "src/compose_v4/control/graph_geometry.py"
        ),
        "public_target_topology": topology(target),
        "configuration": {
            "all_locked_phases": [1, 2, 3],
            "seed": None,
            "deterministic_reduction": True,
        },
        "split": "answer-informed development; no new oracle calls or test generalization",
        "exclusions": [],
        "pairs": pairs,
        "phases": phases,
        "best_candidate": best,
        "changed_choices": result["changed_choices"],
        "fresh_paired_differences": result["fresh_paired_differences"],
        "new_oracle_calls": result["new_oracle_calls"],
        "seconds": result["seconds"],
    }
    publish_json(args.directory / "audit.json", output)
    print(
        json.dumps(
            {
                k: output[k]
                for k in (
                    "pairs",
                    "changed_choices",
                    "fresh_paired_differences",
                    "new_oracle_calls",
                    "seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
