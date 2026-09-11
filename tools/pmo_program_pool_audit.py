"""Score an already locked candidate pool to separate proposal and ranking failure."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_branch_policy import _frozen_save
from compose_v4.experiments.pmo_macro_probe import DurableScores, make_oracle
from compose_v4.experiments.t4_matched_pilot import _stamp

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = Store(args.prior, lambda: None)
    result, pools, selection = prior.read("result"), prior.read("pools"), prior.read("selection")
    if result["status"] != "complete_development" or result["new_oracle_calls"] != 30:
        raise ValueError("expected the completed 30-call preference comparison")
    if (
        sha256_file(args.prior / "selection.json") != result["selection_sha256"]
        or sha256_file(args.prior / "pools.json") != selection["pools_sha256"]
    ):
        raise ValueError("original candidate/selection lock changed")
    old = {r["smiles"]: r["score"] for r in result["oracle_rows"]}
    all_smiles = sorted({c["smiles"] for p in pools["pools"] for c in p["candidates"]})
    new = [s for s in all_smiles if s not in old]
    if len(all_smiles) != 81 or len(new) != 51:
        raise ValueError("pool audit exceeds the exact 51-query scope")
    c = result["configuration"]
    versions = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        **{k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch", "PyTDC")},
    }
    if versions != c["software"]:
        raise ValueError("oracle audit environment differs from the qualified comparison")
    tdc_root = Path(importlib.util.find_spec("tdc").origin).parent
    if {str(p.relative_to(tdc_root)): sha256_file(p) for p in sorted(tdc_root.rglob("*.py"))} != c[
        "oracle_implementation_sha256"
    ] or sha256_file(ROOT / "src/compose_v4/experiments/pmo_macro_probe.py") != c[
        "implementation_sha256"
    ]["src/compose_v4/experiments/pmo_macro_probe.py"]:
        raise ValueError("oracle implementation changed since the original comparison")
    store = Store(args.output, lambda: None)
    lock = {
        "schema_version": "program_pool_audit_lock_v1",
        "prior": str(args.prior),
        "inputs_sha256": {
            name: sha256_file(args.prior / f"{name}.json")
            for name in ("result", "pools", "selection")
        },
        "queries": new,
        "query_limit": 51,
        "time_limit_seconds": 60,
        "role": "post-comparison candidate-pool diagnostic, not autonomous selection",
        "producer_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(ROOT / "docs/CONTROLLER_LIVE.md"),
        "software": versions,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "hardware": c["hardware"],
    }
    _frozen_save(store, "lock", lock)
    if store.read("result") is not None:
        print(json.dumps(store.read("result")))
        return
    started = perf_counter()
    scores = DurableScores(
        args.output, make_oracle("perindopril_mpo", ROOT, {}), 51, lambda: None, {}
    )
    scores.context = {"role": "unselected_pool_audit", "lock_path": str(args.output / "lock.json")}
    for smi in new:
        if perf_counter() - started >= 60:
            raise TimeoutError("pool audit deadline; completed labels are reusable")
        old[smi] = scores.score(smi)["desirability"]
    rows = []
    for pool, choice in zip(pools["pools"], selection["choices"], strict=True):
        candidates = pool["candidates"]
        if not candidates:
            continue
        values = np.array([old[r["smiles"]] for r in candidates])
        improves = values > pool["parent"]["score"] + 1e-12
        best = int(np.argmax(values))
        rows.append(
            {
                "parent_smiles": pool["parent"]["smiles"],
                "parent_score": pool["parent"]["score"],
                "candidates": len(values),
                "improving_candidates": int(improves.sum()),
                "pool_best": float(values[best]),
                "pool_best_candidate": candidates[best],
                "pool_best_utility_rank": 1
                + sum(u > choice["utilities"][best] for u in choice["utilities"]),
                "learned_mass_on_improvements": float(
                    np.asarray(choice["probabilities"])[improves].sum()
                ),
                "baseline_mass_on_improvements": float(
                    np.asarray(choice["reference"])[improves].sum()
                ),
                "learned_expected_score": float(values @ choice["probabilities"]),
                "baseline_expected_score": float(values @ choice["reference"]),
                "selected_scores": {arm: float(values[i]) for arm, i in choice["indices"].items()},
            }
        )
    audit = {
        "schema_version": "program_pool_audit_result_v1",
        "status": "complete_development",
        "lock_sha256": sha256_file(args.output / "lock.json"),
        "configuration": lock,
        "new_oracle_calls": len(scores.rows),
        "prior_comparison_calls": 30,
        "historical_prescreen_calls": result["historical_prescreen_calls"],
        "historical_development_physical_calls_before_comparison": result[
            "historical_development_physical_calls"
        ],
        "oracle_rows": scores.rows,
        "initial_best": result["initial_best"],
        "full_pool_best": max(old.values()),
        "full_pool_new_champion": max(old.values()) > result["initial_best"],
        "parents_with_available_improvements": sum(r["improving_candidates"] > 0 for r in rows),
        "rows": rows,
        "seconds": perf_counter() - started,
        "finished_at": _stamp(),
    }
    store.save("result", audit, durable=True)
    publish_json(args.output / "report.json", audit)
    print(
        json.dumps(
            {k: v for k, v in audit.items() if k not in ("configuration", "oracle_rows", "rows")},
            indent=2,
        )
    )
    print(
        json.dumps(
            [
                {k: v for k, v in r.items() if k != "pool_best_candidate"}
                for r in rows
                if r["improving_candidates"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
