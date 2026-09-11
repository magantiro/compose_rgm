"""Audit the locked replacement probe without any new oracle or model calls."""

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.control.docking_value import identity
from compose_v4.control.graph_geometry import structural_displacement
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.inference_package import software


def report(path):
    r = json.loads(path.read_text())
    snapshot = path.with_name("source_snapshot.json")
    archive = path.with_name("source_snapshot.tar.gz")
    spawn = path.with_name("spawn.json")
    manifest = json.loads(snapshot.read_text())
    verify_file(archive, manifest["archive_sha256"])
    receipt = json.loads(spawn.read_text())
    if receipt["run_id"] != r["run_id"] or receipt["task"]["image_revision"] != r["image_revision"]:
        raise ValueError("run/source snapshot mismatch")
    c = r["configuration"]
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("result contract changed")
    with tarfile.open(archive) as tar:
        for p, digest in manifest["files"].items():
            if hashlib.sha256(tar.extractfile(p).read()).hexdigest() != digest:
                raise ValueError(f"source snapshot hash mismatch: {p}")
        for p, digest in r["image_revision"]["serialized_sources"].items():
            if manifest["files"].get(p) != digest:
                raise ValueError(f"executed source missing in snapshot: {p}")
        for key in ("prepared", "protocol", "snapshot_authorization"):
            p = c[key]["path"]
            if manifest["files"][p] != c[key]["sha256"]:
                raise ValueError(f"contract input differs: {key}")
        data = json.loads(tar.extractfile(c["prepared"]["path"]).read())
    if len(r["workers"]) != c["particles"] or not all(w["replay_verified"] for w in r["workers"]):
        raise ValueError("incomplete or unverified worker census")
    history = dict(data["observed"])
    for q in r["oracle_rows"]:
        if q["smiles"] in history:
            raise ValueError("duplicate or unnecessarily repeated oracle query")
        history[q["smiles"]] = q["score"]
    if (
        len(r["oracle_rows"]) != r["new_oracle_calls"]
        or r["new_oracle_calls"] > c["new_oracle_limit"]
    ):
        raise ValueError("oracle census mismatch")
    parents = {p["smiles"]: p for p in r["initial_parents"]}
    rows = []
    for p in r["candidates"]:
        before, after = (
            decode_search_state(parents[p["parent_smiles"]]["node"]),
            decode_search_state(p["node"]),
        )
        computed = structural_displacement(before.graph, after.graph, before.lineage, after.lineage)
        if computed != p["structural_change"] or history[p["smiles"]] != p["score"]:
            raise ValueError("candidate geometry or actual score mismatch")
        rows.append(
            {
                "smiles": p["smiles"],
                "parent_smiles": p["parent_smiles"],
                "parent_score": p["parent_score"],
                "score": p["score"],
                "score_delta": p["score"] - p["parent_score"],
                "option": p["bundle"]["option"],
                "primitive_steps": p["primitive_count"],
                "intended_release": p["bundle"]["r_release"],
                "realized_coherent": computed["largest_changed_fraction"],
                "cycle_rank_delta": computed["d_cycle_rank"],
                "ring_system_delta": computed["d_ring_systems"],
                "heavy_atoms_before": before.graph.n_real_atoms,
                "heavy_atoms_after": after.graph.n_real_atoms,
            }
        )
    unique = sorted({p["smiles"] for p in rows})
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [fp.GetFingerprint(Chem.MolFromSmiles(s)) for s in unique]
    distances = [
        1 - DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
        for i in range(len(unique))
        for j in range(i)
    ]
    attempted = [a for w in r["workers"] for a in w["attempts"]]
    replacement_attempts = [
        a for a in attempted if a["bundle"] and a["bundle"]["option"].startswith("replace_region:")
    ]
    replacements = [p for p in rows if p["option"].startswith("replace_region:")]
    return {
        "schema_version": "region_replacement_report_v1",
        "run_id": r["run_id"],
        "inputs": {str(p): sha256_file(p) for p in (path, snapshot, spawn, archive)},
        "analyzer_sha256": sha256_file(Path(__file__)),
        "analysis_software": software(),
        "executed_revision": r["image_revision"],
        "configuration": c,
        "attempts": len(attempted),
        "completed": len(rows),
        "failures": [a for a in attempted if a["status"] != "complete"],
        "replacement_attempts": len(replacement_attempts),
        "replacement_completions": len(replacements),
        "replacement_improvements_over_parent": sum(p["score_delta"] > 0 for p in replacements),
        "unique_candidates": len(unique),
        "archive_metrics": r["archive_metrics"],
        "initial_best": max(p["score"] for p in parents.values()),
        "new_oracle_calls": r["new_oracle_calls"],
        "wall_seconds": r["seconds"],
        "oracle_seconds": sum(q["oracle_seconds"] for q in r["oracle_rows"]),
        "sum_worker_seconds": sum(w["seconds"] for w in r["workers"]),
        "sum_law_seconds": sum(w["law_work"]["law_seconds"] for w in r["workers"]),
        "driver_io_timings": r["io_timings"],
        "mean_pairwise_morgan_distance": float(np.mean(distances)) if distances else None,
        "option_counts": dict(Counter(p["option"] for p in rows)),
        "candidates": rows,
        "evidence": "remote exact primitive replay; independently recomputed endpoint displacement and score ledger; no local neural qualification",
        "interpretation": "single warm-development proposal batch, not a paired optimizer comparison or PMO AUC result",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = report(args.result)
    publish_json(args.output, value)
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "attempts",
                    "completed",
                    "replacement_attempts",
                    "replacement_completions",
                    "archive_metrics",
                    "new_oracle_calls",
                    "wall_seconds",
                )
            }
        )
    )
    for row in value["candidates"]:
        if row["option"].startswith("replace_region:"):
            print(json.dumps(row))


if __name__ == "__main__":
    main()
