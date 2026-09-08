"""Verify the identity repair against every saved failed-program closure product.

Read-only molecular work: no model, oracle, or sampling. Extract exact states
from physically verified ledgers; never reconstruct replay states from SMILES.
Publish to a new directory, preserving the original negative audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import rdBase

from compose_v4.control import macro_engine
from compose_v4.experiments.continuation_profile import canonical_bytes, publish_json, sha256_file
from compose_v4.experiments.t4_matched_pilot import ARMS, unseal
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "t4_append_contract_fixture_v1"


def extract_fixture(root: Path, ledger_root: Path) -> dict:
    inventory_path = root / "remote_inventory.json"
    diagnosis_path = root / "closure_diagnosis.json"
    inventory = json.loads(inventory_path.read_text())
    diagnosis = json.loads(diagnosis_path.read_text())
    if diagnosis["minimum_new_system_atoms"] != macro_engine.APPEND_MIN_NEW_ATOMS:
        raise ValueError("saved diagnosis and production append minimum disagree")
    hashes = {str(p): sha256_file(p) for p in (inventory_path, diagnosis_path)}
    if hashes[str(inventory_path)] != diagnosis["remote_inventory_sha256"]:
        raise ValueError("remote inventory differs from saved diagnosis")
    rows, census = [], {}
    for arm in ARMS:
        path = ledger_root / f"{arm}.json"
        digest = sha256_file(path)
        if digest != inventory["files"][f"{arm}/executor_attempts_0.json"]["sha256"]:
            raise ValueError(f"ledger hash mismatch: {path}")
        hashes[str(path)] = digest
        lock_path = root / arm / "candidate_lock.json"
        hashes[str(lock_path)] = sha256_file(lock_path)
        if hashes[str(lock_path)] != inventory["files"][f"{arm}/candidate_lock.json"]["sha256"]:
            # Input paths may move; the physical identity, not path spelling, governs reuse.
            raise ValueError(f"candidate lock differs from saved diagnosis: {lock_path}")
        lock = unseal(lock_path)
        endpoints = {
            canonical_bytes(t["product"]): t["bundle_id"]
            for work in lock["work"]
            for t in work["sampled_transitions"]
            if t["option"] == "build_ring_system" and t["step"] == 8
        }
        expected = Counter(
            canonical_bytes([bundle, row["mark"], row["product"]])
            for bundle, values in diagnosis["rows"][arm].items()
            for row in values
        )
        labels = {
            canonical_bytes([bundle, row["mark"], row["product"]]): row
            for bundle, values in diagnosis["rows"][arm].items()
            for row in values
        }
        counts = Counter()
        for attempt in json.loads(path.read_text())["attempts"]:
            counts["all_recorded_attempts"] += 1
            bundle = endpoints.get(canonical_bytes(attempt["source"]))
            if bundle is None:
                counts["excluded_other_source"] += 1
                continue
            if attempt["mark"].get("executor_rule") not in ("cycle_close", "bond_insert"):
                counts["excluded_other_family"] += 1
                continue
            if attempt["status"] != "executed":
                counts["excluded_no_executed_product"] += 1
                continue
            product_key = canonical_state_key(decode_state(attempt["product"]))
            key = canonical_bytes([bundle, attempt["mark"], product_key])
            if expected[key] <= 0:
                raise ValueError(f"saved diagnosis omits or differs from a closure: {arm}/{bundle}")
            expected[key] -= 1
            counts["included_closure_products"] += 1
            rows.append(
                {
                    "arm": arm,
                    "bundle_id": bundle,
                    "source": attempt["source"],
                    "mark": attempt["mark"],
                    "product": attempt["product"],
                    "expected_accept": labels[key]["exact_slot_contract"],
                    "historical_accept": labels[key]["smiles_contract"],
                }
            )
        if any(expected.values()):
            raise ValueError(f"ledger lacks products from saved diagnosis: {arm}")
        census[arm] = dict(sorted(counts.items()))
    return {
        "schema_version": SCHEMA,
        "input_sha256": hashes,
        "source_revision": diagnosis["source_revision"],
        "label_basis": "independent pre-repair exact-slot graph diagnosis, not affinity labels",
        "minimum_new_system_atoms": macro_engine.APPEND_MIN_NEW_ATOMS,
        "census": census,
        "rows": rows,
    }


def audit_fixture(fixture: dict) -> dict:
    if fixture["schema_version"] != SCHEMA:
        raise ValueError("unknown saved closure fixture schema")
    if fixture["minimum_new_system_atoms"] != macro_engine.APPEND_MIN_NEW_ATOMS:
        raise ValueError("saved fixture and production append minimum disagree")
    system = editing_v2_rewrite_system()
    replayed, rows = {}, []
    times = Counter()
    for row in fixture["rows"]:
        before, product = (decode_state(row[k]) for k in ("source", "product"))
        before_key, product_key = canonical_state_key(before), canonical_state_key(product)
        key = hashlib.sha256(canonical_bytes([row["source"], row["mark"]])).hexdigest()
        if key not in replayed:
            family, mark = decode_action(row["mark"])
            replayed[key] = encode_state(system.apply(before, family, mark))
        if replayed[key] != row["product"]:
            raise ValueError(f"executor replay differs from saved product: {key}")
        started = time.perf_counter()
        legacy = macro_engine.append_system_closure(before_key)(product_key)
        times["legacy_smiles_contract_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        repaired = macro_engine.state_contract_for("append_system", before)(product)
        times["exact_state_contract_seconds"] += time.perf_counter() - started
        if legacy != row["historical_accept"]:
            raise ValueError(f"legacy contract no longer reproduces historical decision: {key}")
        rows.append(
            {
                "arm": row["arm"],
                "bundle_id": row["bundle_id"],
                "source_mark_sha256": key,
                "product": product_key,
                "expected_accept": row["expected_accept"],
                "legacy_accept": legacy,
                "repaired_accept": repaired,
                "exact_replay_matches": True,
            }
        )
    metrics = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        tp = sum(r["repaired_accept"] and r["expected_accept"] for r in selected)
        fp = sum(r["repaired_accept"] and not r["expected_accept"] for r in selected)
        fn = sum(not r["repaired_accept"] and r["expected_accept"] for r in selected)
        tn = len(selected) - tp - fp - fn
        metrics[arm] = {
            "closure_products": len(selected),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "coverage": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
        }
    return {
        "schema_version": "t4_append_contract_audit_v1",
        "metrics": metrics,
        "rows": rows,
        "all_expected_decisions_match": all(
            r["repaired_accept"] == r["expected_accept"] for r in rows
        ),
        "unique_source_mark_replays": len(replayed),
        "new_model_calls": 0,
        "new_oracle_calls": 0,
        "timing": {
            **times,
            "scope": "predicate construction and evaluation; excludes decoding, canonicalization and executor replay",
        },
        "evidence_scope": "fixed saved counterfactual products, not completed programs, generated coverage, or docking improvement",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("ledger_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"use a new immutable output directory: {args.output}")
    started = time.perf_counter()
    fixture = extract_fixture(args.root, args.ledger_root)
    result = audit_fixture(fixture)
    result.update(
        {
            "fixture_payload_sha256": hashlib.sha256(canonical_bytes(fixture)).hexdigest(),
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "implementation_sha256": {
                str(p): sha256_file(p) for p in (Path(__file__), Path(macro_engine.__file__))
            },
            "config": {
                "minimum_new_system_atoms": macro_engine.APPEND_MIN_NEW_ATOMS,
                "seed": None,
                "selection": "all saved executed closure products at all six failed growth endpoints",
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "rdkit": rdBase.rdkitVersion,
                "networkx": nx.__version__,
            },
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "workers": 1,
                "gpu": False,
                "precision": "integer graph topology; no model tensors",
            },
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    publish_json(args.output / "fixture.json", fixture)
    publish_json(args.output / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, sort_keys=True, indent=2))
    if not result["all_expected_decisions_match"]:
        raise SystemExit("contract regression: saved negative result published")


if __name__ == "__main__":
    main()
