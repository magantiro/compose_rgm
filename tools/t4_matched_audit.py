"""Read-only, hash-verified report of a completed paired development audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.experiments.continuation_profile import canonical_bytes
from compose_v4.experiments.t4_matched_pilot import ARMS, unseal, verify_pair


def report(root: Path) -> dict:
    result = json.loads((root / "result.json").read_text())
    locks = {arm: unseal(root / arm / "candidate_lock.json") for arm in ARMS}
    verify_pair(locks)
    barrier = json.loads((root / "oracle_barrier.json").read_text())
    inputs = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.json"))
        if p.name not in ("review.json", "verification.json")
    }
    arms = {}
    for arm in ARMS:
        lock, outcome = locks[arm], result["arms"][arm]
        identity = inputs[f"{arm}/candidate_lock.json"]
        if (
            any(
                identity != binding[arm]
                for binding in (
                    result["candidate_lock_sha256"],
                    barrier["candidate_lock_sha256"],
                )
            )
            or outcome["candidate_lock_sha256"] != identity
        ):
            raise ValueError(f"{arm}: candidate lock hash mismatch")
        if lock["locked_at_utc"] > barrier["locked_at_utc"]:
            raise ValueError("oracle barrier predates a candidate lock")
        docked = outcome["docked"]
        if canonical_bytes(
            [{k: v for k, v in c.items() if k != "ds"} for c in docked]
        ) != canonical_bytes(lock["take"]):
            raise ValueError("docked candidates differ from frozen selection")
        trace = [t for w in lock["work"] for t in w["sampled_transitions"]]
        if any(
            not 0 < t["sample_probability"] <= 1 or not -1e-12 <= t["kl"] <= 1 + 1e-8 for t in trace
        ):
            raise ValueError("recorded sampled probability or KL violates the frozen bounds")
        programs = [
            b for b in lock["bundles"] if b["option"] in ("build_ring_system", "build_fused_ring")
        ]
        arms[arm] = {
            "docking_attempts": outcome["oracle_attempts"],
            "docking_failures": outcome["oracle_failures"],
            "feasible": outcome["n_feasible"],
            "best": outcome["best"],
            "selected_options": dict(sorted(Counter(b["option"] for b in lock["bundles"]).items())),
            "docked_options": outcome["selected_options"],
            "candidate_count": lock["n_candidates"],
            "unique_pool": len(lock["pool"]),
            "unique_docked": len({c["smiles"] for c in docked}),
            "docked_bundles": len({c["bundle_id"] for c in docked}),
            "pool_diversity": lock["candidate_diversity"],
            "docked_diversity": lock["selected_diversity"],
            "proposal_seconds": lock["proposal_seconds"],
            "docking_seconds": outcome["docking_seconds"],
            "executor_calls": lock["total_public_executor_calls"],
            "law_enumerations": sum(w["law_enumerations"] for w in lock["work"]),
            "sampled_transitions": len(trace),
            "nontrivial_reference_rows": sum(t["reference_support_size"] > 1 for t in trace),
            "mean_recorded_kl": float(np.mean([t["kl"] for t in trace])) if trace else None,
            "compound_programs": [
                {
                    k: b[k]
                    for k in (
                        "bundle_id",
                        "option",
                        "max_search_step",
                        "program_complete",
                        "halt_counts",
                    )
                }
                for b in programs
            ],
            "docked_topology_changes": [
                {
                    k: c[k]
                    for k in (
                        "smiles",
                        "option",
                        "r_release",
                        "r_coherent",
                        "d_cycle_rank",
                        "d_ring_systems",
                        "d_heavy",
                    )
                }
                for c in docked
                if c["d_cycle_rank"] or c["d_ring_systems"]
            ],
            "intended_release_min_median_max": [
                float(f([c["r_release"] for c in docked])) for f in (np.min, np.median, np.max)
            ]
            if docked
            else None,
            "realized_coherent_min_median_max": [
                float(f([c["r_coherent"] for c in docked])) for f in (np.min, np.median, np.max)
            ]
            if docked
            else None,
            "by_option": lock["selected_option_diagnostics"],
        }
    return {
        "schema_version": "t4_matched_review_v1",
        "source_revision": result["code_revision"],
        "review_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "review_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": inputs,
        "configuration": result["configuration"],
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "status": result["status"],
        "paired_outer_plan_verified": True,
        "arms": arms,
        "elapsed_seconds": result["elapsed_seconds"],
        "limitations": [
            "one inspected development source",
            "cold-start structural committor, not docking-value learning",
            "inherited unseeded OpenBabel/QuickVina oracle; no paired noise-control claim",
            "same compute ceilings, not identical realized work",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(report(parser.parse_args().root), sort_keys=True, indent=2, allow_nan=False))
