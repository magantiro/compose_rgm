"""Retrospective T4 controller audit. No oracle, model fitting, or runtime changes."""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr

CODE = Path("/private/tmp/compose-t4-complete-region-policy-20260916")
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(CODE / "src"), str(CODE)]
from compose_v4.control.docking_value import identity
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

INPUTS = {}
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def read(path):
    blob = path.read_bytes()
    INPUTS[str(path)] = hashlib.sha256(blob).hexdigest()
    obj = json.loads(gzip.decompress(blob) if path.suffix == ".gz" else blob)
    if isinstance(obj, dict) and "payload" in obj and "payload_sha256" in obj:
        if identity(obj["payload"]) != obj["payload_sha256"]:
            raise ValueError(f"Payload hash mismatch: {path}")
        return obj["payload"]
    return obj


def stats(values):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": len(a),
        "min": float(a.min()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p90": float(np.quantile(a, 0.9)),
        "max": float(a.max()),
    }


@lru_cache(maxsize=100000)
def molinfo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unparseable historical endpoint: {smiles}")
    return (
        FPGEN.GetFingerprint(mol),
        MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
        mol.GetNumHeavyAtoms(),
        mol.GetRingInfo().NumRings(),
    )


@lru_cache(maxsize=100000)
def state_smiles(serialized):
    return canonical_state_key(decode_state(json.loads(serialized)))


def descriptor(entry):
    trace = entry["trace"]
    change = trace["actual_changes"]
    p = entry.get("provenance", {})
    src = state_smiles(json.dumps(entry["source_state"], sort_keys=True))
    src_info, dst_info = molinfo(src), molinfo(entry["endpoint"])
    return {
        "input_smiles": src,
        "input_state_sha256": identity(entry["source_state"]),
        "primitive_count": len(trace["actions"]),
        "region_count_net_source": change["changed_site_count"],
        "changed_slots": change["changed_original_slots"],
        "changed_slot_count": len(change["changed_original_slots"]),
        "net_deleted": change["deleted_original_atoms"],
        "net_created": change["surviving_new_atoms"],
        "rule_counts": dict(Counter(a["executor_rule"] for a in trace["actions"])),
        "block_labels": [b["label"] for b in entry["program"]["blocks"]],
        "input_endpoint_similarity": DataStructs.TanimotoSimilarity(src_info[0], dst_info[0]),
        "ring_count_change": dst_info[3] - src_info[3],
        "same_input_scaffold": src_info[1] == dst_info[1],
        "genealogical_parent": p.get("entry_id"),
        "parent_score": p.get("parent_measured_score"),
        "parent_probability": p.get("parent_probability"),
        "channel": p.get("channel"),
        "planner_channel": p.get("planner_channel"),
        "ancestral_primitives": entry.get("ancestral_primitive_edits"),
    }


def lineage(candidate, entries, scored, qmap):
    result, seen = [], set()
    while candidate:
        key = candidate.get("entry_id", candidate.get("candidate_id"))
        if key in seen:
            raise ValueError("Archive genealogy cycle")
        seen.add(key)
        desc = descriptor(candidate)
        parent = entries.get(desc["genealogical_parent"])
        result.append(
            {
                "entry_id": key,
                "endpoint": candidate["endpoint"],
                "score": scored.get(candidate["endpoint"]),
                "query": qmap.get(candidate["endpoint"]),
                "direct_edit_of_measured_parent": None
                if parent is None
                else candidate["source_state"] == parent["trace"]["states"][-1],
                **desc,
            }
        )
        candidate = parent
    return list(reversed(result))


def main():
    began = time.monotonic()
    rows, units = [], []
    for checkpoint in sorted((OUT / "raw").glob("*_r*_checkpoint.json.gz")):
        name = checkpoint.name.removesuffix("_checkpoint.json.gz")
        arm, protein, seed, rep = name.split("_")
        cell = f"{protein}_{seed}"
        saved = read(checkpoint)
        search = saved["search"]
        resultfile = checkpoint.with_name(name + "_result.json")
        result = read(resultfile) if resultfile.exists() else None
        query_count = saved["query_count"]
        curve = saved["curve"]
        entries = search["entries"] if search else {}
        observations = search["observations"] if search else {}
        by_endpoint = defaultdict(list)
        for key, entry in entries.items():
            by_endpoint[entry["endpoint"]].append((key, entry))
        endpoint_scores = defaultdict(list)
        for observation in observations.values():
            endpoint_scores[observation["endpoint"]].append(observation["score"])
        endpoint_means = {s: float(np.mean(v)) for s, v in endpoint_scores.items()}
        qmap, attempt_counts, channel_status, bootstrap = {}, Counter(), defaultdict(Counter), []
        calls_seen, rounds_seen = 0, []
        for path in sorted((OUT / "raw").glob(name + "_rounds_round_*_batch.json.gz")):
            batch_record = read(path)
            batch = batch_record["batch"]
            rounds_seen.append(batch_record["round"])
            for c in batch["candidates"]:
                calls_seen += 1
                if calls_seen <= query_count:
                    qmap[c["endpoint"]] = calls_seen
            for a in batch["attempts"]:
                status = a.get("status", "missing_status")
                attempt_counts[status] += 1
                channel_status[a.get("planner_channel", a.get("channel", "unassigned"))][
                    status
                ] += 1
            if batch_record["bootstrap"]:
                bootstrap.append(
                    {
                        "round": batch_record["round"],
                        "id": batch["batch_id"],
                        "attempts": len(batch["attempts"]),
                        "candidates": len(batch["candidates"]),
                        "attempt_identity": identity(batch["attempts"]),
                        "proposal_seconds": batch.get("proposal_seconds"),
                    }
                )
        qmap_valid = rounds_seen == list(range(saved["next_round"])) and calls_seen == query_count
        if not qmap_valid:
            # Never infer call indices from a partial round download.
            qmap = {}
        config = search["configuration"] if search else None
        scored_count = 0
        for receipt, obs in sorted(observations.items()):
            candidates = by_endpoint[obs["endpoint"]]
            if not candidates:
                raise ValueError(f"Scored endpoint lacks archive construction: {name}")
            key, entry = candidates[0]
            d = descriptor(entry)
            row = {
                "run": name,
                "arm": arm,
                "cell": cell,
                "target": protein,
                "replicate": int(rep[1:]),
                "protocol": obs["oracle_protocol"],
                "receipt_id": receipt,
                "endpoint": obs["endpoint"],
                "score": obs["score"],
                "entry_id": key,
                "query": qmap.get(obs["endpoint"]),
                "query_verified": qmap_valid,
                "checkpoint": str(checkpoint),
                "entry_alternatives": len(candidates),
                **d,
            }
            rows.append(row)
            scored_count += 1
        champion = saved["champion"]
        line = (
            []
            if champion is None
            else lineage(champion["candidate"], entries, endpoint_means, qmap)
        )
        units.append(
            {
                "run": name,
                "arm": arm,
                "cell": cell,
                "replicate": int(rep[1:]),
                "query_count": query_count,
                "measured_rows": scored_count,
                "archive_entries": len(entries),
                "failed_endpoints": len(search["failed_endpoints"]) if search else 0,
                "final_result_available": result is not None,
                "termination": result.get("termination")
                if result
                else "final_result_missing_checkpoint_only",
                "rounds": saved["next_round"],
                "rounds_recovered": len(rounds_seen),
                "query_mapping_verified": qmap_valid,
                "best": None if champion is None else champion["score"],
                "champion_query": None if champion is None else champion["query_index"],
                "ivg_mean": None if result is None else result.get("ivg_mean"),
                "curve": {
                    str(k): next((r["best_score"] for r in curve if r["query"] == k), None)
                    for k in (1, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000)
                },
                "config": config,
                "champion_lineage": line,
                "attempt_status": dict(attempt_counts),
                "channel_attempt_status": {k: dict(v) for k, v in channel_status.items()},
                "bootstrap": bootstrap,
                "allocator": search.get("dynamic_v21", {}).get("allocator_state")
                if search
                else None,
            }
        )
        print(
            json.dumps(
                {
                    k: units[-1][k]
                    for k in (
                        "run",
                        "query_count",
                        "measured_rows",
                        "best",
                        "query_mapping_verified",
                    )
                }
            ),
            flush=True,
        )

    receipts = {r["receipt_id"] for r in rows}
    if len(receipts) != len(rows):
        raise ValueError("Duplicate receipt across recovered units: reconcile before counting")
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    summaries = {}
    for arm, rs in by_arm.items():
        summaries[arm] = {
            "observations": len(rs),
            "unique_protocol_endpoints": len({(r["protocol"], r["endpoint"]) for r in rs}),
            "with_call_index": sum(r["query"] is not None for r in rs),
            "with_parent_score": sum(r["parent_score"] is not None for r in rs),
            "with_parent_entry": sum(r["genealogical_parent"] is not None for r in rs),
            "with_exact_source_program_trace": len(rs),
            "primitives": stats([r["primitive_count"] for r in rs]),
            "net_source_regions": stats([r["region_count_net_source"] for r in rs]),
            "net_created": stats([r["net_created"] for r in rs]),
            "net_deleted": stats([r["net_deleted"] for r in rs]),
            "changed_atoms": stats([r["changed_slot_count"] for r in rs]),
            "channels": dict(Counter(r["channel"] for r in rs)),
        }
    neighborhood = []
    # Protocol contains target/source/constraints. Keep replicates separate too;
    # the protocol hash alone does not identify the docking random seed.
    by_protocol = defaultdict(list)
    for row in rows:
        by_protocol[(row["protocol"], row["replicate"])].append(row)
    repeated = []
    for (protocol, replicate), rs in sorted(by_protocol.items()):
        endpoints = defaultdict(list)
        for r in rs:
            endpoints[r["endpoint"]].append(r)
        for endpoint, erows in endpoints.items():
            if len(erows) > 1:
                values = [r["score"] for r in erows]
                repeated.append(
                    {
                        "protocol": protocol,
                        "replicate": replicate,
                        "cell": erows[0]["cell"],
                        "endpoint": endpoint,
                        "observations": len(values),
                        "scores": values,
                        "range": max(values) - min(values),
                        "arms": [r["arm"] for r in erows],
                    }
                )
        unique = [
            {**v[0], "score": float(np.mean([r["score"] for r in v]))} for v in endpoints.values()
        ]
        if len(unique) < 20:
            continue
        anchor = min(unique, key=lambda r: r["score"])
        others = [r for r in unique if r["endpoint"] != anchor["endpoint"]]
        afp, ascaffold, _, _ = molinfo(anchor["endpoint"])
        sims = np.asarray(
            DataStructs.BulkTanimotoSimilarity(afp, [molinfo(r["endpoint"])[0] for r in others])
        )
        gaps = np.asarray([r["score"] - anchor["score"] for r in others])
        rho = None if len(set(sims)) < 2 else float(spearmanr(1 - sims, gaps).statistic)
        bins = {}
        for cut in (0.6, 0.7, 0.8, 0.9):
            selected = gaps[sims >= cut]
            bins[str(cut)] = {
                **stats(selected),
                "within_0_5": int(np.sum(selected <= 0.5)),
                "gap_at_least_1": int(np.sum(selected >= 1)),
            }
        same_scaffold = [
            r["score"] - anchor["score"] for r in others if molinfo(r["endpoint"])[1] == ascaffold
        ]
        same_input = [r for r in others if r["input_state_sha256"] == anchor["input_state_sha256"]]
        same_where = [
            r["score"] - anchor["score"]
            for r in same_input
            if r["changed_slots"] == anchor["changed_slots"]
        ]
        same_modules = [
            r["score"] - anchor["score"]
            for r in same_input
            if r["block_labels"] == anchor["block_labels"]
        ]
        neighborhood.append(
            {
                "protocol": protocol,
                "cell": anchor["cell"],
                "replicate": anchor["replicate"],
                "unique_endpoints": len(unique),
                "anchor_score": anchor["score"],
                "anchor_arm": anchor["arm"],
                "anchor_endpoint": anchor["endpoint"],
                "rho_distance_vs_gap": rho,
                "similarity_bins": bins,
                "same_scaffold_gap": stats(same_scaffold),
                "same_exact_input_and_where_gap": stats(same_where),
                "same_exact_input_and_module_labels_gap": stats(same_modules),
            }
        )
    summary = {
        "schema": "t4_strategy_reset_audit_v1",
        "new_oracle_calls": 0,
        "new_model_fits": 0,
        "source_code_revision": subprocess.check_output(
            ["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True
        ).strip(),
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "hardware": platform.platform(),
        },
        "inputs_sha256": INPUTS,
        "units": units,
        "by_arm": summaries,
        "neighborhoods": neighborhood,
        "repeated_identical_endpoints": repeated,
        "limitations": [
            "Retrospective adaptive samples, not causal evidence or a held-out test.",
            "Winner-neighborhood anchors chosen post hoc; similarity association is optimistic and not validation of a utility predictor.",
            "Descriptor fingerprints use the recorded RDKit version; no new benchmark docking or compiler gate is claimed.",
            "Net source-connected edit regions are not dependency-region decompositions.",
            "Unfinished units retain only checkpoint observations; missing terminal artifacts are explicit.",
        ],
        "seconds": time.monotonic() - began,
    }
    (OUT / "scored_rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    )
    (OUT / "audit.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "units": len(units),
                "seconds": summary["seconds"],
                "by_arm": summaries,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
