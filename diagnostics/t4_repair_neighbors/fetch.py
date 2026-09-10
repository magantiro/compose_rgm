"""Collect and audit existing repair receipts. Never launch scientific work."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import modal
from rdkit import rdBase

from compose_v4.experiments.continuation_profile import canonical_bytes, publish_json, sha256_file
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint
from compose_v4.experiments.t4_repair_neighbors import recovery_counts

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    destination = HERE / "attempt_1"
    launch_path = destination / "launch.json"
    if args.receipt:
        incoming = json.loads(args.receipt.read_text())
        if launch_path.exists() and incoming != json.loads(launch_path.read_text()):
            raise ValueError("refusing to mix repair launches")
        publish_json(launch_path, incoming)
    launch = json.loads(launch_path.read_text())
    contract_path = ROOT / "configs/t4_repair_neighbors.json"
    contract = json.loads(contract_path.read_text())
    assert rdBase.rdkitVersion == contract["required_rdkit"]
    assert launch["oracle_calls"] == launch["oracle_call_limit"] == contract["oracle_calls"] == 0
    assert sha256_file(contract_path) == launch["task"]["contract_sha256"]
    volume = modal.Volume.from_name(launch["volume"])
    remote_root = launch["volume_path"].lstrip("/")
    remote_hashes = {}

    def get(name, *, persist=True):
        remote = f"{remote_root}/{name}.json"
        try:
            data = b"".join(volume.read_file(remote))
        except FileNotFoundError:
            return None
        remote_hashes[remote] = hashlib.sha256(data).hexdigest()
        value = json.loads(data)
        if persist:
            publish_json(destination / f"{name}.json", value)
        if isinstance(value, dict) and "payload_sha256" in value:
            assert (
                hashlib.sha256(canonical_bytes(value["payload"])).hexdigest()
                == value["payload_sha256"]
            )
            return value["payload"]
        return value

    result = get("result")
    failure, heartbeat = get("failure"), get("heartbeat")
    if failure:
        raise RuntimeError(f"repair worker failed: {failure}")
    if result is None:
        print(json.dumps({"status": "pending", "heartbeat": heartbeat}, sort_keys=True))
        return
    assert result["oracle_calls"] == 0 and result["winner_used"] is False
    assert result["code_revision"] == launch["task"]["image_revision"]["commit"]
    assert result["configuration"] == contract
    assert result["runtime_gate"]["input_sha256"] == contract["expected_input_sha256"]
    get("runtime_gate")
    get("law_cache_inventory")
    panel = get("panel_lock")
    assert len(panel["roots"]) == len(result["roots"]) == contract["panel"]["count"]
    summaries = []
    all_new = set()
    new_predictions = {}
    prior_path = ROOT / "diagnostics/t4_constraint_recovery/attempt_1/case_2/result.json"
    prior = json.loads(prior_path.read_text())
    for index, source in enumerate(panel["roots"]):
        generation = get(f"roots/{index:02d}/generation_lock", persist=False)
        scored = get(f"roots/{index:02d}/scored_lock", persist=False)
        root = result["roots"][index]
        assert generation is not None and scored is not None
        assert root["root"] == source
        assert generation["source"] == source["node"]["graph"]
        assert generation["source_smiles"] == source["smiles"]
        assert source["v"] > 0 and not source["oracle_eligible"]
        assert len({r["smiles"] for r in scored}) == len(scored) == len(generation["products"])
        for generated, row in zip(generation["products"], scored, strict=True):
            assert all(row[k] == v for k, v in generated.items())
            assert row["smiles"] != source["smiles"] and row["witnesses"]
            assert row["oracle_eligible"] == acceptable_endpoint(row)
        counts = generation["counts"]
        assert sum(len(r["witnesses"]) for r in scored) == counts.get(
            "positive_mass_marks", 0
        ) - counts.get("canonical_self_marks", 0)
        eligible = [r for r in scored if r["oracle_eligible"]]
        assert root["eligible_products"] == eligible
        known = {r["smiles"] for r in scored if r["in_prior_archive"]}
        # The previous run's exact incumbent is held in its committed result.
        computed = recovery_counts(scored, known, prior["source"]["smiles"])
        assert all(root[k] == v for k, v in computed.items())
        new = [r for r in eligible if not r["in_prior_archive"]]
        all_new.update(r["smiles"] for r in new)
        new_predictions.update({r["smiles"]: r["predicted_docking"] for r in new})
        summaries.append(
            {
                "root_index": index,
                "root_smiles": source["smiles"],
                "root_violation": source["v"],
                "root_similarity": source["sim"],
                "attempt_id": source["attempt_id"],
                "option": source["bundle"]["option"],
                **computed,
                "new_cycle_rank_deltas": dict(
                    sorted(
                        Counter(
                            r["topology"]["cycle_rank"] - source["topology"]["cycle_rank"]
                            for r in new
                        ).items()
                    )
                ),
                "new_ring_system_deltas": dict(
                    sorted(
                        Counter(
                            r["topology"]["n_ring_systems"] - source["topology"]["n_ring_systems"]
                            for r in new
                        ).items()
                    )
                ),
                "new_eligible_products": [
                    {
                        k: r[k]
                        for k in (
                            "smiles",
                            "qed",
                            "sa",
                            "sim",
                            "predicted_docking",
                            "topology",
                            "witnesses",
                        )
                    }
                    for r in new
                ],
            }
        )
    report = {
        "schema_version": "t4_repair_neighbors_summary_v1",
        "roots": summaries,
        "unique_new_eligible_across_roots": len(all_new),
        "roots_with_new_eligible": sum(r["new_eligible"] > 0 for r in summaries),
        "eligible_root_product_pairs": sum(r["eligible"] for r in summaries),
        "all_root_product_pairs": sum(r["unique_products"] for r in summaries),
        "incumbent_predicted_docking": prior["source_prediction"]["predicted_docking"],
        "best_new_predicted_docking": min(new_predictions.values(), default=None),
        "new_predicted_better_than_incumbent": sum(
            p < prior["source_prediction"]["predicted_docking"] for p in new_predictions.values()
        ),
        "oracle_calls": 0,
        "generation_revision": result["code_revision"],
        "run_id": launch["task"]["run_id"],
        "completed_at_utc": result["completed_at_utc"],
        "elapsed_seconds": result["elapsed_seconds"],
        "proposal_seconds": result["proposal_seconds_this_invocation"],
        "law_work": result["law_work"],
        "executor_calls": result["executor_calls_this_invocation"],
        "remote_input_sha256": dict(sorted(remote_hashes.items())),
        "local_input_sha256": {
            str(p.relative_to(ROOT)): sha256_file(p)
            for p in (
                contract_path,
                launch_path,
                ROOT / "diagnostics/t4_constraint_recovery/attempt_1/case_2/result.json",
            )
        },
        "analysis": {
            "script_sha256": sha256_file(Path(__file__)),
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "claim_limit": "three selected development intermediates; support availability, not controller sampling success or observed docking improvement",
    }
    publish_json(HERE / "summary.json", report)
    print(
        json.dumps(
            {
                "status": "complete",
                "unique_new_eligible": len(all_new),
                "roots": [
                    {
                        k: r[k]
                        for k in (
                            "root_index",
                            "unique_products",
                            "eligible",
                            "returns_to_incumbent",
                        )
                    }
                    for r in summaries
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
