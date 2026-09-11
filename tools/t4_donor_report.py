"""Audit one locked T4 transfer batch without new docking or neural inference."""

import argparse
import hashlib
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.inference_package import software
from compose_v4.experiments.t4_donor_probe import candidate_lock


def spread(values):
    return (
        dict(zip(("min", "median", "max"), map(float, np.quantile(values, [0, 0.5, 1]))))
        if values
        else None
    )


def validate_locked_row(expected, actual):
    """Allow property arithmetic roundoff, never a changed eligibility decision."""
    for key, value in expected.items():
        if key in ("qed", "sa", "sim", "v"):
            if not math.isclose(value, actual[key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"locked property differs: {key}")
        elif value != actual[key]:
            raise ValueError(f"docked candidate differs from rederived lock: {key}")


def chemistry(rows):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in sorted({r["smiles"] for r in rows})]
    distances = [
        1 - DataStructs.TanimotoSimilarity(fp, previous)
        for i, fp in enumerate(fps)
        for previous in fps[:i]
    ]
    return {
        "canonical_count": len(fps),
        "mean_pairwise_morgan_distance": float(np.mean(distances)) if distances else None,
        "options": dict(Counter(r["bundle"]["option"] for r in rows)),
        "intended_release": spread([r["bundle"]["r_release"] for r in rows]),
        "realized_largest_changed_relative_to_parent_size": spread(
            [r["structural_change"]["largest_changed_fraction"] for r in rows]
        ),
        "ring_system_deltas": dict(Counter(r["structural_change"]["d_ring_systems"] for r in rows)),
        "cycle_rank_deltas": dict(Counter(r["structural_change"]["d_cycle_rank"] for r in rows)),
        "heavy_atom_deltas": dict(Counter(r["structural_change"]["d_heavy"] for r in rows)),
        "deleted_atoms": spread([r["structural_change"]["n_deleted"] for r in rows]),
        "inserted_atoms": spread([r["structural_change"]["n_inserted"] for r in rows]),
        "scale_note": "largest_changed_fraction describes changed connected material present in the product, not all removed material; inspect deleted/inserted counts and topology alongside it",
    }


def report(path):
    result = json.loads(path.read_text())
    c = result["configuration"]
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("T4 configuration self-hash mismatch")
    archive = path.with_name("source_snapshot.tar.gz")
    with tarfile.open(archive, "r:gz") as tar:
        for relative, digest in result["image_revision"]["serialized_sources"].items():
            if hashlib.sha256(tar.extractfile(relative).read()).hexdigest() != digest:
                raise ValueError(f"executed source archive mismatch: {relative}")
        prepared = tar.extractfile(c["prepared"]["path"]).read()
        if hashlib.sha256(prepared).hexdigest() != c["prepared"]["sha256"]:
            raise ValueError("prepared T4 archive differs from the lock")
        data = json.loads(prepared)
    lock = candidate_lock(c, data, result["workers"])
    rows = result["docked"]
    if (
        result["new_oracle_calls"] != len(rows)
        or len(rows) != len(lock["take"])
        or len(rows) > c["new_oracle_limit"]
        or [r["index"] for r in rows] != list(range(len(rows)))
        or result["selected_counts"] != lock["selected_counts"]
    ):
        raise ValueError("T4 attempt census or allocation differs from the frozen batch")
    for expected, actual in zip(lock["take"], rows, strict=True):
        validate_locked_row(expected, actual)
        if actual["ds"] is not None and not math.isfinite(actual["ds"]):
            raise ValueError("nonfinite docking score")
    if len({r["candidate_lock_sha256"] for r in rows}) != 1:
        raise ValueError("docking rows do not share one lock")
    arms = {}
    for arm in c["arms"]:
        proposed = [r for r in lock["pool"] if r["arm"] == arm]
        selected = [r for r in rows if arm in r["arms"]]
        selected_ids = {
            (p["slot"], p["id"]) for r in selected for p in r["origins"] if p["arm"] == arm
        }
        selected_proposals = [r for r in proposed if (r["slot"], r["id"]) in selected_ids]
        workers = [
            w
            for w in result["workers"]
            if w.get("proposal_policy_sha256") == c["proposal_ids"][arm]
        ]
        arms[arm] = {
            "proposal_attempts": len(workers),
            "completed_options": len(proposed),
            "eligible_options": sum(r["oracle_eligible"] for r in proposed),
            "endpoint_exclusions_nonexclusive": {
                "qed": sum(r["qed"] < c["qed_min"] for r in proposed),
                "sa": sum(r["sa"] > c["sa_max"] for r in proposed),
                "similarity": sum(r["sim"] < c["delta"] for r in proposed),
                "medchem_only": sum(r["v"] == 0 and not r["oracle_eligible"] for r in proposed),
                "prior_or_duplicate": sum(r["already_observed_or_duplicate"] for r in proposed),
            },
            "novel_eligible_options": sum(
                r["oracle_eligible"] and not r["already_observed_or_duplicate"] for r in proposed
            ),
            "docking_attempts": len(selected),
            "docking_failures": sum(r["ds"] is None for r in selected),
            "score_distribution": spread([r["ds"] for r in selected if r["ds"] is not None]),
            "proposed_chemistry": chemistry(proposed),
            "docked_chemistry": chemistry(selected_proposals),
            "worker_seconds": sum(w["seconds"] for w in workers),
            "initialization_seconds": sum(w["initialization_seconds"] for w in workers),
            "law_seconds": sum(w["law_work"]["law_seconds"] for w in workers),
            "attempt_statuses": dict(Counter(a["status"] for w in workers for a in w["attempts"])),
        }
    controls = {
        role: [r["ds"] for r in rows if r.get("role") == role] for role in ("seed", "incumbent")
    }
    if any(len(values) != c["anchor_repeats"] for values in controls.values()):
        raise ValueError("missing fixed docking controls")
    return {
        "schema_version": "t4_donor_report_v1",
        "run_id": result["run_id"],
        "source_revision": result["image_revision"]["commit"],
        "contract": c,
        "inputs_sha256": {str(p): sha256_file(p) for p in (path, archive)},
        "analyzer_sha256": sha256_file(Path(__file__)),
        "property_replay_tolerance": 1e-12,
        "allocation_and_eligibility_replay_exact": True,
        "analysis_software": software(),
        "executed_software": result["software"],
        "wall_seconds": result["seconds"],
        "new_oracle_calls": result["new_oracle_calls"],
        "historical_oracle_attempts": result["historical_oracle_attempts"],
        "docking_seconds_sum": sum(r["docking_seconds"] for r in rows),
        "arms": arms,
        "control_scores": controls,
        "docked": rows,
        "interpretation": "warm exposed single-batch transfer; two control repeats do not estimate precise docking uncertainty; not an autonomous full-campaign comparison or external SOTA",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = report(args.result)
    publish_json(args.output, value)
    print(json.dumps({k: value[k] for k in ("new_oracle_calls", "wall_seconds", "control_scores")}))
    for arm, row in value["arms"].items():
        print(json.dumps({"arm": arm, **row}))
