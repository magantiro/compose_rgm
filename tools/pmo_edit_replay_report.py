"""Audit a locked contextual replay comparison without querying the oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.rewrite.kernel import canonical_state_key


def report(directory: Path) -> dict:
    store = Store(directory, lambda: None)
    result, lock, data = (store.read(k) for k in ("result", "candidate_lock", "prepared"))
    if result["status"] != "complete_development":
        raise ValueError("replay comparison is not complete")
    for name in ("configuration", "training", "candidate_lock"):
        if sha256_file(directory / f"{name}.json") != result[f"{name}_sha256"]:
            raise ValueError(f"replay {name} lock changed")
    if result["bank_sha256"] != store.read("bank")["bank_sha256"]:
        raise ValueError("replay bank changed")
    known = data["known"].copy()
    for row in result["oracle_rows"]:
        if (
            row["status"] != "complete"
            or row["smiles"] in known
            or row["lock_sha256"] != result["candidate_lock_sha256"]
        ):
            raise ValueError("replay query is repeated, incomplete, or not candidate-locked")
        known[row["smiles"]] = row["score"]
    attempted = lock["attempts"]
    if len(attempted) != 128 or any(
        sum(r["arm"] == arm for r in attempted) != 64 for arm in result["arms"]
    ):
        raise ValueError("matched attempt allocation changed")
    summaries = {}
    for arm, value in result["arms"].items():
        candidates = value["candidates"]
        expected = lock["candidates"][arm]
        if [r["id"] for r in candidates] != [r["id"] for r in expected]:
            raise ValueError("scored arm membership differs from its candidate lock")
        for candidate, frozen in zip(candidates, expected, strict=True):
            if {k: v for k, v in candidate.items() if k not in ("score", "gain")} != frozen:
                raise ValueError("candidate changed after locking")
            if (
                candidate["score"] != known[candidate["smiles"]]
                or abs(candidate["gain"] - (candidate["score"] - candidate["parent_score"])) > 1e-12
                or canonical_state_key(decode_search_state(candidate["node"]).graph)
                != candidate["smiles"]
            ):
                raise ValueError("candidate graph, score, or gain is inconsistent")
        archive = {r["smiles"]: r["score"] for r in data["parents"] + candidates}
        best = max(archive.values())
        top10 = float(np.mean(sorted(archive.values(), reverse=True)[:10]))
        if best != value["best"] or abs(top10 - value["top10_mean"]) > 1e-12:
            raise ValueError("reported best or top-ten mean does not reproduce")
        distinct = {r["smiles"]: r for r in candidates}
        summaries[arm] = {
            **{k: v for k, v in value.items() if k != "candidates"},
            "attempted": 64,
            "unique_improving": len({r["smiles"] for r in candidates if r["gain"] > 1e-12}),
            "duplicate_completed_draws": len(candidates) - len(distinct),
            "top_candidates": [
                {k: v for k, v in r.items() if k not in ("node", "chain")}
                for r in sorted(distinct.values(), key=lambda r: (-r["score"], r["smiles"]))[:3]
            ],
        }
    all_new = {r["smiles"] for arm in result["arms"].values() for r in arm["candidates"]} - set(
        data["known"]
    )
    if (
        all_new != {r["smiles"] for r in result["oracle_rows"]}
        or len(all_new) != result["new_oracle_calls"]
    ):
        raise ValueError("physical calls do not cover exactly the novel candidate union")
    return {
        "schema_version": "edit_replay_audit_v1",
        "source": str(directory / "result.json"),
        "source_sha256": sha256_file(directory / "result.json"),
        "producer_sha256": sha256_file(Path(__file__)),
        "configuration": result["configuration"],
        "checks": "sealed receipts, input locks, arm membership, exact graph identity, charged query union, best and top-ten reproduction",
        "initial_best": result["initial_best"],
        "arms": summaries,
        "costs": {
            key: result[key]
            for key in (
                "new_oracle_calls",
                "historical_prescreen_calls",
                "historical_development_physical_calls",
                "seconds",
                "fit_seconds",
                "context_seconds",
            )
        }
        | {"oracle_seconds": sum(r["oracle_seconds"] for r in result["oracle_rows"])},
        "decision": {
            "replay_primary_pass": summaries["replay"]["best"]
            > max(result["initial_best"], summaries["uniform"]["best"]),
            "interpretation": "Uniform proposals found the new development champion; replay improved top-ten mean but missed the prespecified best-score comparison. Do not promote or replicate this unchanged replay recipe as a passing controller.",
            "limitations": "One warm prescreened donor-channel comparison, not a full-controller improvement, official PMO AUC, or external superiority result.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.input)
    publish_json(args.output, result)
    print(json.dumps({"decision": result["decision"], "costs": result["costs"]}, indent=2))


if __name__ == "__main__":
    main()
