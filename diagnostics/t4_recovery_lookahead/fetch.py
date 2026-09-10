"""Collect fixed lookahead rows and replay one paired retention decision."""

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
from pathlib import Path

import modal
import numpy as np
from rdkit import rdBase

from compose_v4.control.graph_geometry import topology
from compose_v4.experiments.continuation_profile import canonical_bytes, publish_json, sha256_file
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint
from compose_v4.experiments.t4_recovery_lookahead import compare_retention, decision_pool
from compose_v4.experiments.t4_repair_neighbors import recovery_counts
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    destination = HERE / "attempt_1"
    receipt_path = destination / "launch.json"
    if args.receipt:
        incoming = json.loads(args.receipt.read_text())
        if receipt_path.exists() and json.loads(receipt_path.read_text()) != incoming:
            raise ValueError("refusing to mix lookahead launches")
        publish_json(receipt_path, incoming)
    launch = json.loads(receipt_path.read_text())
    contract_path = ROOT / "configs/t4_recovery_lookahead.json"
    contract = json.loads(contract_path.read_text())
    assert rdBase.rdkitVersion == contract["required_rdkit"]
    assert launch["oracle_calls"] == contract["oracle_calls"] == 0
    assert [c["case_index"] for c in launch["cases"]] == [0, 1, 2]
    assert sha256_file(contract_path) == launch["task"]["contract_sha256"]
    volume = modal.Volume.from_name(launch["volume"])
    remote_hashes = {}

    def get(path, local):
        try:
            data = b"".join(volume.read_file(path))
        except FileNotFoundError:
            return None
        remote_hashes[path] = hashlib.sha256(data).hexdigest()
        value = json.loads(data)
        publish_json(local, value)
        if isinstance(value, dict) and "payload_sha256" in value:
            assert (
                hashlib.sha256(canonical_bytes(value["payload"])).hexdigest()
                == value["payload_sha256"]
            )
            return value["payload"]
        return value

    results, status = [], []
    for case in launch["cases"]:
        remote = case["volume_path"].lstrip("/")
        local = destination / f"case_{case['case_index']}"
        result = get(f"{remote}/result.json", local / "result.json")
        failure = get(f"{remote}/failure.json", local / "failure.json")
        heartbeat = get(f"{remote}/heartbeat.json", local / "heartbeat.json")
        if failure:
            raise RuntimeError(f"worker {case['case_index']} failed: {failure}")
        results.append(result)
        status.append(
            {"case": case["case_index"], "complete": result is not None, "heartbeat": heartbeat}
        )
    if any(r is None for r in results):
        print(json.dumps(status, sort_keys=True))
        return
    assets = {}
    for name in ("panel", "lock", "archive", "value_snapshot"):
        assets[name] = get(contract[name]["path"], destination / f"{name}.json")
        assert remote_hashes[contract[name]["path"]] == contract[name]["sha256"]
    lock, snapshot, archive = assets["lock"], assets["value_snapshot"], assets["archive"]
    pool = decision_pool(assets["panel"]["candidates"], 9)
    incumbent = lock["levels"][1]["beam"][0]
    assert incumbent["node"] == lock["root"] and incumbent["chain"] == []
    known = {r["smiles"] for r in archive["archive"]}
    repairs, census, work, reused = {}, [], [], []
    for case, result in zip(launch["cases"], results, strict=True):
        i = case["case_index"]
        remote, local = case["volume_path"].lstrip("/"), destination / f"case_{i}"
        assert result["configuration"] == contract and result["oracle_calls"] == 0
        assert result["code_revision"] == launch["task"]["image_revision"]["commit"]
        assert result["runtime_gate"]["input_sha256"] == contract["expected_input_sha256"]
        assert [r["root"] for r in result["roots"]] == pool[i::3]
        for name in ("runtime_gate", "law_cache_inventory", "panel_lock"):
            assert get(f"{remote}/{name}.json", local / f"{name}.json") is not None
        for j, root in enumerate(result["roots"]):
            prefix = f"roots/{j:02d}"
            generated = get(
                f"{remote}/{prefix}/generation_lock.json", local / f"{prefix}/generation_lock.json"
            )
            rows = get(f"{remote}/{prefix}/scored_lock.json", local / f"{prefix}/scored_lock.json")
            reuse = get(f"{remote}/{prefix}/reuse.json", local / f"{prefix}/reuse.json")
            if reuse:
                reused.append(root["root"]["attempt_id"])
            assert generated["source"] == root["root"]["node"]["graph"]
            assert len(rows) == len(generated["products"]) == len({r["smiles"] for r in rows})
            for row, raw in zip(rows, generated["products"], strict=True):
                assert all(row[k] == v for k, v in raw.items())
                assert row["oracle_eligible"] == acceptable_endpoint(row)
                assert row["in_prior_archive"] == (row["smiles"] in known)
            counts = recovery_counts(rows, known, incumbent["smiles"])
            assert all(root[k] == v for k, v in counts.items())
            repairs[root["root"]["smiles"]] = rows
            census.append({"attempt_id": root["root"]["attempt_id"], **counts})
        work.append(
            {
                "worker": i,
                "elapsed_seconds": result["elapsed_seconds"],
                "proposal_seconds": result["proposal_seconds_this_invocation"],
                "law_work": result["law_work"],
                "executor_calls": result["executor_calls_this_invocation"],
                "peak_rss_native_units": result["peak_rss_native_units"],
                "completed_at_utc": result["completed_at_utc"],
            }
        )
    comparison = compare_retention(pool, repairs, incumbent, snapshot, lock["config"])
    old = lock["levels"][1]["decision"]["offspring"]
    replayed = comparison["arms"]["immediate"]["decision"]["offspring"]
    assert (
        old["pool"] == replayed["pool"] and old["selected_indices"] == replayed["selected_indices"]
    )
    assert np.allclose(
        old["first_slot_probabilities"], replayed["first_slot_probabilities"], rtol=0, atol=1e-12
    )
    # Lock retained decisions before verifying the returned continuations.
    publish_json(destination / "selection_lock.json", comparison)
    system = editing_v2_semantic_rewrite_system()
    selected = []
    for arm, decision in comparison["arms"].items():
        for source in pool:
            if source["attempt_id"] not in decision["retained"]:
                continue
            product = comparison["values"][source["smiles"]]["endpoint"]
            if product is None:
                continue
            witness = product["witnesses"][0]
            codec = action_codec_v4 if witness["schema_version"] == 4 else action_codec
            graph = decode_state(source["node"]["graph"])
            replay = system.apply(graph, *codec.decode_action(witness))
            assert encode_state(replay) == product["state"] and acceptable_endpoint(product)
            selected.append(
                {
                    "arm": arm,
                    "source_attempt_id": source["attempt_id"],
                    "source_smiles": source["smiles"],
                    "smiles": product["smiles"],
                    "predicted_docking": product["predicted_docking"],
                    "qed": product["qed"],
                    "sa": product["sa"],
                    "sim": product["sim"],
                    "remaining_budget": source["node"]["budget"] - 1,
                    "witness": witness,
                    "source_topology": topology(graph),
                    "product_topology": topology(replay),
                    "exact_replay_verified": True,
                }
            )
    inputs = [contract_path, *sorted(destination.rglob("*.json"))]
    report = {
        "schema_version": "t4_recovery_lookahead_summary_v1",
        "oracle_calls": 0,
        "pool_count": len(pool),
        "census": census,
        "reused_roots": reused,
        "comparison": comparison,
        "selected_continuations": selected,
        "worker_work": work,
        "input_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in inputs},
        "remote_sha256": remote_hashes,
        "analysis": {
            "script_sha256": sha256_file(Path(__file__)),
            "planner_sha256": sha256_file(Path(inspect.getfile(compare_retention))),
            "executor_module_sha256": sha256_file(
                Path(inspect.getfile(editing_v2_semantic_rewrite_system))
            ),
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "claim_limit": "one inspected fixed decision, not a closed-loop optimizer trial; one-step max-value planning, not a Doob expectation; no observed docking gain",
    }
    print(publish_json(HERE / "summary.json", report))
    for arm, decision in comparison["arms"].items():
        print(
            arm,
            "recoverable selection mass",
            decision["recoverable_first_slot_probability"],
            "new eligible continuations",
            decision["new_eligible_continuations"],
            "retained",
            decision["retained"],
        )


if __name__ == "__main__":
    main()
