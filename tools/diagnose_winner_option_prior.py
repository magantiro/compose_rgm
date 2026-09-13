"""Rescore saved real option rows without rerunning chemistry or calling docking.

The four known consecutive PARP1 program states are answer-known development
conditions. This computes a particular prefix's proposal probability from saved
HOW probabilities, not an autonomous success rate from the benchmark seed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.demonstration_prior import DemonstrationPrior
from tools.ivg_winner_paths import digest, implementation_closure, publish, sha

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    if args.output.exists():
        raise ValueError("preserve the existing route rescore")
    began = perf_counter()
    prepared = json.loads(args.prepared.read_text())
    data_path = args.prepared.parent / "demonstrations.json.gz"
    if sha(data_path) != prepared["demonstrations_sha256"]:
        raise ValueError("demonstration hash mismatch")
    rows = json.loads(gzip.decompress(data_path.read_bytes()))["rows"]
    prior = DemonstrationPrior.fit(
        [row for row in rows if row["role"] == "train"], training_identity=sha(data_path)
    )
    inputs = {
        str(args.prepared.resolve()): sha(args.prepared),
        str(data_path.resolve()): sha(data_path),
    }
    results, source_snapshots = [], []
    previous_exact_endpoint = None
    for index in range(4):
        directory = args.cases / f"case_{index}"
        lock_path, result_path = directory / "proposal_lock.json", directory / "result.json"
        inputs.update({str(path.resolve()): sha(path) for path in (lock_path, result_path)})
        wrapper = json.loads(lock_path.read_text())
        lock, result = wrapper["payload"], json.loads(result_path.read_text())
        if digest(lock) != wrapper["payload_sha256"] or result["option"] != lock["focal_option"]:
            raise ValueError(f"case {index}: lock identity mismatch")
        if not result["known_path_support"]["supported"]:
            raise ValueError(f"case {index}: no supported known prefix")
        plan_info = result["configuration"]["plan"]
        plan_path = ROOT / plan_info["path"]
        if sha(plan_path) != plan_info["sha256"]:
            raise ValueError(f"case {index}: compiled route identity mismatch")
        inputs[str(plan_path.resolve())] = sha(plan_path)
        plan = json.loads(plan_path.read_text())
        stages = next(
            attempt["stages"] for attempt in plan["attempts"] if attempt["status"] == "exact_winner"
        )
        stage = stages[index + 1]  # the first linker-remodel stage is conditioned on
        if lock["source"] != stage["states"][0] or (
            previous_exact_endpoint is not None and lock["source"] != previous_exact_endpoint
        ):
            raise ValueError(f"case {index}: exact persistent-slot route continuity failed")
        previous_exact_endpoint = stage["states"][-1]
        names, reference = tuple(lock["options"]), np.asarray(lock["option_prior"], dtype=float)
        selected = names.index(result["option"])
        q = prior.distribution(names, reference)
        how = float(result["known_path_support"]["exact_path_probability"])
        source_snapshots.append(result["configuration"]["expected_input_sha256"])
        results.append(
            {
                "case": index,
                "source": result["source"]["smiles"],
                "option": result["option"],
                "known_endpoint": result["known_continuation"]["smiles"],
                "applicable_options": names,
                "reference": reference.tolist(),
                "proposal": q.proposal,
                "reference_selected_option_probability": float(reference[selected]),
                "proposal_selected_option_probability": q.proposal[selected],
                "kl": q.kl,
                "conditional_how_probability": how,
                "reference_option_and_exact_prefix_probability": float(reference[selected]) * how,
                "proposal_option_and_exact_prefix_probability": q.proposal[selected] * how,
                "selected_option_probability_ratio": q.proposal[selected]
                / float(reference[selected]),
            }
        )
    if any(snapshot != source_snapshots[0] for snapshot in source_snapshots[1:]):
        raise ValueError("recorded HOW probabilities come from different frozen inputs")
    if any(first["known_endpoint"] != second["source"] for first, second in pairwise(results)):
        raise ValueError("recorded program states do not form the declared consecutive route")
    base = math.prod(row["reference_option_and_exact_prefix_probability"] for row in results)
    proposal = math.prod(row["proposal_option_and_exact_prefix_probability"] for row in results)
    report = {
        "schema_version": "winner_option_prior_rescore_v1",
        "evidence": "computed from locked production rows, not a sampled recovery experiment",
        "inputs_sha256": dict(sorted(inputs.items())),
        "implementation_sha256": implementation_closure(Path(__file__).resolve()),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        "configuration": {
            "prior_snapshot": prior.snapshot,
            "training_identity": prior.training_identity,
            "frequencies": prior.frequencies,
            "max_kl": 1.0,
            "base_floor": 0.1,
            "seed": None,
            "deterministic": True,
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "hardware": {"device": "CPU", "precision": "float64 probabilities"},
        "cases": results,
        "four_option_exact_prefix": {
            "reference": base,
            "proposal": proposal,
            "ratio": proposal / base,
        },
        "exact_source_continuity_checked": True,
        "decision": "prior_improves_this_conditional_route_probability"
        if proposal > base
        else "prior_does_not_improve_this_conditional_route",
        "costs": {
            "seconds": perf_counter() - began,
            "oracle_calls": 0,
            "molecular_replays": 0,
            "reference_law_evaluations": 0,
        },
        "limitations": [
            "conditional on the pre-remodeled linker state and fully mutable region at every option",
            "not probability from the benchmark seed or under actual WHERE selection",
            "only one specified exact-state route, not all routes to the winner",
            "historical HOW law held fixed for this counterfactual computation",
            "does not address docking-value ranking or survival under resampling",
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    publish(args.output, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "route_probability": report["four_option_exact_prefix"],
                "per_option_ratios": [row["selected_option_probability_ratio"] for row in results],
                "costs": report["costs"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "diagnostics/t4_option_decision_audit/attempt_1"
    )
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
