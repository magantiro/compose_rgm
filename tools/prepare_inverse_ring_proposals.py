"""Prepare exact-replayed ring-option demonstrations from inspected IVG winners.

No training or oracle call. Reuses the frozen source split from the previous
development diagnostic. Writes complete per-molecule restart units before its
final manifest, and verifies compatible units on resume.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.option_demonstrations import recognize_trace
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.experiments.continuation_profile import ExecutorMeter
from compose_v4.experiments.inverse_ring_demonstrations import inverse_ring_demonstrations
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.ivg_winner_paths import digest, implementation_closure, publish, sha
from tools.winner_option_proposal import source_id

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    if (args.output / "result.json").exists():
        raise ValueError("complete preparation exists; reuse it rather than regenerate")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("requires pinned RDKit 2024.03.5")
    start = perf_counter()
    audit = json.loads(args.audit.read_text())
    split = json.loads(args.split.read_text())
    if split["audit_sha256"] != sha(args.audit):
        raise ValueError("source split and winner audit have different identities")
    inputs = {
        str(args.audit.resolve()): sha(args.audit),
        str(args.split.resolve()): sha(args.split),
    }
    closure = implementation_closure(Path(__file__).resolve())
    unit_closure = implementation_closure(
        ROOT / "src/compose_v4/experiments/inverse_ring_demonstrations.py"
    )
    groups, exclusions = {}, []
    for pair in audit["pairs"]:
        if pair["status"] != "witness_found":
            exclusions.append({"pair_id": pair["pair_id"], "reason": pair["status"]})
            continue
        path = args.audit.parent / pair["receipt"]
        actual = sha(path)
        if actual != pair["receipt_sha256"]:
            raise ValueError(f"{path}: input hash mismatch")
        inputs[str(path.resolve())] = actual
        receipt = json.loads(gzip.decompress(path.read_bytes()))
        if digest(receipt["payload"]) != receipt["payload_sha256"]:
            raise ValueError(f"{path}: payload hash mismatch")
        state = receipt["payload"]["path"]["states"][-1]
        target = canonical_state_key(decode_state(state))
        if target != receipt["payload"]["path"]["target_2d"]:
            raise ValueError(f"{path}: final state is not the declared target")
        key = source_id(pair["source"])
        row = {
            "source_id": key,
            "role": split["source_roles"][key],
            "pair_id": pair["pair_id"],
            "receipt_sha256": actual,
        }
        group = groups.setdefault(target, {"state": state, "sources": []})
        group["sources"].append(row)
    # Assign winner identities before extracting any reusable demonstrations.
    # A winner shared with an excluded source cannot teach a training source.
    winner_roles = {}
    for target, group in groups.items():
        held = any(row["role"] != "train" for row in group["sources"])
        winner_roles[target] = "heldout_source_diagnostic" if held else "train"
        if held:
            exclusions.extend(
                {"pair_id": row["pair_id"], "reason": "winner_shared_with_heldout_source"}
                for row in group["sources"]
                if row["role"] == "train"
            )
            group["sources"] = [row for row in group["sources"] if row["role"] != "train"]
    publish(
        args.output / "roles.json",
        {"source_split_sha256": sha(args.split), "winner_roles": winner_roles},
    )
    rows, units = [], []
    counts = Counter()
    meter = ExecutorMeter(None)
    with meter.instrument():
        for index, (target, group) in enumerate(sorted(groups.items())):
            identity = digest(
                {
                    "state": group["state"],
                    "implementation": unit_closure,
                    "rdkit": rdBase.rdkitVersion,
                }
            )
            path = args.output / "units" / f"{identity}.json.gz"
            if path.exists():
                unit = json.loads(gzip.decompress(path.read_bytes()))
                if unit["identity"] != identity or unit["payload_sha256"] != digest(
                    unit["payload"]
                ):
                    raise ValueError(f"{path}: incompatible/corrupt restart unit")
            else:
                meter.phase = "inverse_ring_demonstrations"
                before, began = meter.calls, perf_counter()
                payload = inverse_ring_demonstrations(decode_state(group["state"]))
                unit = {
                    "identity": identity,
                    "payload": payload,
                    "payload_sha256": digest(payload),
                    "executor_calls": meter.calls - before,
                    "seconds": perf_counter() - began,
                }
                publish(path, unit, compressed=True)
            payload = unit["payload"]
            units.append(
                {
                    "target": target,
                    "path": str(path.relative_to(args.output)),
                    "sha256": sha(path),
                    "examples": len(payload["examples"]),
                    "executor_calls": unit["executor_calls"],
                    "seconds": unit["seconds"],
                }
            )
            counts.update(attempt["status"] for attempt in payload["attempts"])
            for example_index, example in enumerate(payload["examples"]):
                segments = [
                    {
                        "smiles": example["precursor"],
                        "option": example["option"],
                        "compound": True,
                        "start": 0,
                        "stop": example["forward"]["primitive_edits"],
                    }
                ]
                if example["redecoration"]["actions"]:
                    for segment in recognize_trace(
                        example["redecoration"]["states"], example["redecoration"]["actions"]
                    ):
                        smiles = canonical_state_key(
                            decode_state(example["redecoration"]["states"][segment.start])
                        )
                        segments.append(
                            {
                                "smiles": smiles,
                                "option": segment.option,
                                "compound": segment.compound,
                                "start": segment.start,
                                "stop": segment.stop,
                            }
                        )
                for source in group["sources"]:
                    for segment_index, segment in enumerate(segments):
                        rows.append(
                            {
                                **source,
                                **segment,
                                "decision_id": f"{identity}:{example_index}:{segment_index}:{source['source_id']}",
                                "features": molecule_features(segment["smiles"]).tolist(),
                                "unit_sha256": sha(path),
                                "target_used_for_preparation_only": target,
                                "lane": "inverse_derived_option",
                            }
                        )
            if (index + 1) % 10 == 0 or index + 1 == len(groups):
                print(
                    json.dumps(
                        {
                            "completed_winners": index + 1,
                            "total_winners": len(groups),
                            "verified_ring_examples": sum(unit["examples"] for unit in units),
                            "seconds": perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
    unique = {}
    for row in rows:
        key = (row["source_id"], row["smiles"], row["option"])
        if key in unique:
            unique[key]["aliases"].append(row["decision_id"])
        else:
            unique[key] = {**row, "aliases": [row["decision_id"]]}
    rows = list(unique.values())
    train_states = {row["smiles"] for row in rows if row["role"] == "train"}
    overlap = [
        row["decision_id"]
        for row in rows
        if row["role"] != "train" and row["smiles"] in train_states
    ]
    rows = [row for row in rows if row["decision_id"] not in set(overlap)]
    data_path = args.output / "demonstrations.json.gz"
    publish(
        data_path,
        {"schema_version": "winner_option_demonstrations_v1", "rows": rows},
        compressed=True,
    )
    report = {
        "schema_version": "inverse_ring_preparation_v1",
        "evidence": "winner-informed inverse reconstruction, not autonomous search",
        "inputs_sha256": dict(sorted(inputs.items())),
        "implementation_sha256": closure,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        "configuration": {
            "workers": 1,
            "cycles": [5, 6],
            "topologies": ["pendant", "fused"],
            "decoration": "strip/re-add terminal single-atom substituents only",
            "deterministic": True,
            "seed": None,
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "hardware": {
            "device": "CPU",
            "machine": platform.machine(),
            "precision": "integer graph arrays",
        },
        "access_basis_and_upstream": audit["access_basis_and_upstream"],
        "split_sha256": sha(args.split),
        "roles_sha256": sha(args.output / "roles.json"),
        "demonstrations_sha256": sha(data_path),
        "units": units,
        "exclusions": exclusions,
        "intermediate_overlap_exclusions": overlap,
        "counts": {
            "unique_winners": len(groups),
            "winners_with_reconstruction": sum(unit["examples"] > 0 for unit in units),
            "verified_ring_examples": sum(unit["examples"] for unit in units),
            "deduplicated_decisions": len(rows),
            "by_role": dict(sorted(Counter(row["role"] for row in rows).items())),
            "by_option": dict(sorted(Counter(row["option"] for row in rows).items())),
            "attempt_status": dict(sorted(counts.items())),
        },
        "costs": {
            "total_seconds": perf_counter() - start,
            "executor_calls_this_invocation": meter.calls,
            "new_docking_calls": 0,
            "reference_law_evaluations": 0,
        },
        "limitations": [
            "previously inspected winners, retrospective source split",
            "precursors are inverse-derived, not benchmark seeds",
            "only existing unrefined CNO five/six ring descriptors",
            "nonterminal substituent branches not stripped",
            "positive probability under a deployed learned enumerator remains to be checked",
            "no docking-value or future-value labels inferred",
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    publish(args.output / "result.json", report)
    print(json.dumps({"counts": report["counts"], "costs": report["costs"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", type=Path, default=ROOT / "diagnostics/ivg_winner_paths/audit.json"
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "diagnostics/winner_option_proposal/attempt_1/split.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
