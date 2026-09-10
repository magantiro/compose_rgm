"""Reduce the fixed three-arm recovery diagnostic, preserving all null outcomes."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint
from compose_v4.experiments.t4_macro_beam import distance
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def distribution(values):
    return {
        "count": len(values),
        "min": float(min(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "max": float(max(values)) if values else None,
    }


def summarize(directory, label):
    lock = unseal(directory / "generation_lock.json")
    scored = unseal(directory / "scored_lock.json")
    result = json.loads((directory / "result.json").read_text())
    assert result["oracle_calls"] == lock["oracle_calls"] == 0
    assert lock["winner_used"] is False and result["replay_verified"] is True
    assert result["config"] == lock["config"] and lock["config"]["preserve_root"]
    assert result["candidates"] == scored["candidates"]
    attempts = {a["attempt_id"]: a for a in lock["attempts"]}
    candidates = {r["attempt_id"]: r for r in scored["candidates"]}
    assert len(attempts) == len(lock["attempts"]) <= 30
    assert set(candidates) == {k for k, a in attempts.items() if a["status"] == "complete"}
    assert acceptable_endpoint(result["source_prediction"])
    by_node = {identity(lock["root"]): result["source_prediction"]}
    by_node.update({identity(c["node"]): c for c in candidates.values()})
    transitions, benchmark_transitions, history_recovered = [], [], []
    ineligible_parent_attempts = 0
    for name, row in candidates.items():
        attempt = attempts[name]
        assert all(row[k] == v for k, v in attempt["candidate"].items())
        assert attempt["replayed_primitives"] > 0
        assert row["oracle_eligible"] == acceptable_endpoint(row)
        parent = by_node[identity(attempt["source"])]
        ineligible_parent_attempts += not parent["oracle_eligible"]
        if row["v"] == 0 and parent["v"] > 0:
            benchmark_transitions.append(name)
        if row["oracle_eligible"] and not parent["oracle_eligible"]:
            transitions.append(name)
        if row["oracle_eligible"] and any(
            not candidates[p]["oracle_eligible"] for p in row["chain"][:-1]
        ):
            history_recovered.append(name)
    unique = {r["smiles"]: r for r in candidates.values()}
    eligible = [r for r in unique.values() if r["oracle_eligible"]]
    novel = [r for r in eligible if not r["in_prior_archive"]]
    decisions = []
    for level in lock["levels"]:
        assert level["beam"][0]["node"] == lock["root"]
        assert level["beam"][0]["chain"] == []
        assert len({r["smiles"] for r in level["beam"]}) == len(level["beam"])
        d = level["decision"]["offspring"]
        if "first_slot_probabilities" not in d:
            decisions.append({"depth": level["depth"], "policy": d["policy"]})
            continue
        p, q = np.asarray(d["first_slot_reference"]), np.asarray(d["first_slot_probabilities"])
        assert np.isclose(p.sum(), 1) and np.isclose(q.sum(), 1)
        assert np.all(q >= 0.1 * p - 1e-12)
        kl = float(np.sum(q * np.log(q / p)))
        assert kl <= 1 + 1e-10
        assert np.isclose(kl, d["first_slot_kl_against_empirical_pool"])
        if label == "post_hoc":
            assert d["first_slot_values"] is None and np.array_equal(p, q)
        else:
            field = lock["config"]["retention_score"]
            assert d["first_slot_values"] == [candidates[a][field] for a in d["pool"]]
        decisions.append(
            {
                "depth": level["depth"],
                "kl": kl,
                "eta": d["eta"],
                "flat": bool(np.allclose(p, q, rtol=0, atol=1e-12)),
                "values": d["first_slot_values"],
            }
        )
    pairs = [distance(a["smiles"], b["smiles"]) for a, b in combinations(novel, 2)]
    return {
        "arm": label,
        "configuration": result["config"],
        "attempted": len(attempts),
        "completed": len(candidates),
        "failures": dict(
            Counter(a["status"] for a in attempts.values() if a["status"] != "complete")
        ),
        "unique_completed": len(unique),
        "unique_eligible": len(eligible),
        "unique_new_eligible": len(novel),
        "unique_new_eligible_cycle_adding": sum(
            r["cumulative_change"]["d_cycle_rank"] > 0 for r in novel
        ),
        "recovery_transitions": transitions,
        "completed_from_ineligible_parent": ineligible_parent_attempts,
        "benchmark_constraint_recovery_transitions": benchmark_transitions,
        "unique_recovered_molecules": len({candidates[n]["smiles"] for n in transitions}),
        "unique_new_recovered_molecules": len(
            {candidates[n]["smiles"] for n in transitions if not candidates[n]["in_prior_archive"]}
        ),
        "eligible_with_ineligible_ancestor": history_recovered,
        "new_eligible": [
            {
                "smiles": r["smiles"],
                "predicted_docking": r["predicted_docking"],
                "qed": r["qed"],
                "sa": r["sa"],
                "sim": r["sim"],
                "options": [candidates[a]["bundle"]["option"] for a in r["chain"]],
                "change": r["cumulative_change"],
            }
            for r in sorted(novel, key=lambda r: (r["predicted_docking"], r["smiles"]))
        ],
        "option_counts": dict(
            Counter(a["bundle"]["option"] for a in attempts.values() if a["bundle"])
        ),
        "retention": decisions,
        "intended_release": distribution([r["bundle"]["r_release"] for r in candidates.values()]),
        "realized_per_option": distribution(
            [r["structural_change"]["largest_changed_fraction"] for r in candidates.values()]
        ),
        "realized_new_eligible": distribution(
            [r["cumulative_change"]["largest_changed_fraction"] for r in novel]
        ),
        "candidate_similarity": distribution([r["sim"] for r in unique.values()]),
        "candidate_constraint_violation": distribution([r["v"] for r in unique.values()]),
        "closest_to_benchmark_constraints": [
            {
                **{
                    k: r[k] for k in ("smiles", "qed", "sa", "sim", "v", "oracle_eligible", "chain")
                },
                "retained_levels": [
                    level["depth"]
                    for level in lock["levels"]
                    if any(b["smiles"] == r["smiles"] for b in level["beam"])
                ],
                "exact_state_continuation_attempts": [
                    name
                    for name, attempt in attempts.items()
                    if identity(attempt["source"]) == identity(r["node"])
                ],
            }
            for r in sorted(unique.values(), key=lambda r: (r["v"], r["smiles"]))[:1]
        ],
        "new_eligible_pairwise_distance": distribution(pairs),
        "law_work": result["law_work"],
        "executor_calls": result["public_executor_calls_this_invocation"],
        "proposal_seconds": result["proposal_seconds_this_invocation"],
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_rss_native_units": result["peak_rss_native_units"],
        "software": result["software"],
        "source": result["source"],
        "source_prediction": result["source_prediction"],
        "snapshot_sha256": result["value_snapshot_sha256"],
        "generation_revision": result["code_revision"],
    }


def main():
    contract_path = ROOT / "configs/t4_constraint_recovery.json"
    contract = json.loads(contract_path.read_text())
    assert rdBase.rdkitVersion == contract["required_rdkit"]
    launch_path = HERE / "attempt_1/launch.json"
    launch = json.loads(launch_path.read_text())
    assert sha256_file(contract_path) == launch["task"]["contract_sha256"]
    inputs = [contract_path, launch_path]
    arms = []
    roots, first_levels, candidate_sets = [], [], []
    for case, label in enumerate(contract["case_labels"]):
        directory = HERE / f"attempt_1/case_{case}"
        inputs.extend(
            directory / f"{n}.json"
            for n in (
                "result",
                "generation_lock",
                "scored_lock",
                "runtime_gate",
                "law_cache_inventory",
            )
        )
        result = json.loads((directory / "result.json").read_text())
        assert result["configuration"] == contract and result["case_index"] == case
        assert result["code_revision"] == launch["task"]["image_revision"]["commit"]
        assert result["runtime_gate"]["input_sha256"] == contract["expected_input_sha256"]
        assert result["config"] == {**contract["search"], **contract["cases"][case]}
        lock = unseal(directory / "generation_lock.json")
        roots.append(lock["root"])
        first_levels.append(lock["levels"][0]["attempts"])
        candidate_sets.append({r["smiles"] for r in result["candidates"]})
        arms.append(summarize(directory, label))
    assert roots[0] == roots[1] == roots[2]
    assert first_levels[0] == first_levels[1] == first_levels[2]
    assert len({a["snapshot_sha256"] for a in arms}) == 1
    report = {
        "schema_version": "t4_constraint_recovery_summary_v1",
        "arms": arms,
        "candidate_overlap": [
            {
                "arms": [contract["case_labels"][i], contract["case_labels"][j]],
                "intersection": len(candidate_sets[i] & candidate_sets[j]),
                "union": len(candidate_sets[i] | candidate_sets[j]),
                "only_first": len(candidate_sets[i] - candidate_sets[j]),
                "only_second": len(candidate_sets[j] - candidate_sets[i]),
            }
            for i, j in combinations(range(len(arms)), 2)
        ],
        "oracle_calls": 0,
        "winner_used": False,
        "paired_max_attempts": 30,
        "paired_random_seed": contract["search"]["seed"],
        "input_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(inputs)},
        "analysis": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "claim_limit": "one inspected development root and one paired random seed; predictions are not new docking observations",
    }
    print(publish_json(HERE / "summary.json", report))
    for arm in arms:
        print(
            arm["arm"],
            "new eligible",
            arm["unique_new_eligible"],
            "new recovered",
            arm["unique_new_recovered_molecules"],
            "proposal seconds",
            round(arm["proposal_seconds"], 1),
        )


if __name__ == "__main__":
    main()
