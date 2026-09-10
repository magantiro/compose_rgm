"""One-step max-value planning on a fixed completed-option decision pool.

This is a finite-support planner, not a second successor law or Doob estimator.
The original region/option draws are locked. Its recovery probe spans all
positive-mass primitive support after the completed option, not just its region.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace

import numpy as np

from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.t4_macro_beam import BeamConfig, recovery_desirability, retain
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_repair_neighbors import run_remote as repair_remote
from compose_v4.rewrite.trace_shard import encode_state

KIND = "t4_recovery_lookahead"
CONTRACT_PATH = f"configs/{KIND}.json"


def enumerator_source_hash(source):
    """Bind unchanged function source without Python-version-dependent AST fields."""
    function = next(
        n for n in ast.parse(source).body if getattr(n, "name", "") == "enumerate_products"
    )
    return hashlib.sha256(ast.get_source_segment(source, function).encode()).hexdigest()


def decision_pool(candidates, count):
    pool = sorted(
        (r for r in candidates if r["attempt_id"].startswith("levels/01/")),
        key=lambda r: r["attempt_id"],
    )
    if count != 9 or len(pool) != count or len({r["smiles"] for r in pool}) != count:
        raise ValueError("lookahead requires the complete locked nine-candidate decision")
    if any(r["node"]["budget"] < 1 for r in pool):
        raise ValueError("one-step continuation exceeds a candidate's remaining budget")
    return pool


def best_new_endpoint(rows, payload):
    """Best eligible unobserved one-edit endpoint; known endpoints have zero acquisition."""
    eligible = [r for r in rows if r["oracle_eligible"] and not r["in_prior_archive"]]
    if not eligible:
        return {"one_step_value": 0.0, "endpoint": None, "new_eligible": 0}
    best = min(eligible, key=lambda r: (r["predicted_docking"], r["smiles"]))
    return {
        "one_step_value": recovery_desirability(0.0, best["predicted_docking"], payload, 0.1),
        "endpoint": best,
        "new_eligible": len({r["smiles"] for r in eligible}),
    }


def compare_retention(pool, repairs, incumbent, snapshot, saved_config):
    """Same saved decision and RNG, immediate versus one-step planned value."""
    by_smiles = {r["smiles"]: r for r in pool}
    if set(repairs) != set(by_smiles):
        raise ValueError("lookahead must evaluate every member of the decision pool")
    values = {s: best_new_endpoint(rows, snapshot) for s, rows in repairs.items()}
    config = BeamConfig(**saved_config)
    outputs = {}
    for arm in ("immediate", "lookahead"):
        current = (
            config if arm == "immediate" else replace(config, retention_score="one_step_value")
        )
        score = (lambda s: by_smiles[s]) if arm == "immediate" else (lambda s: values[s])
        rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, config.root_index, 1, 991])
        )
        kept, audit = retain(pool, current, rng, score, incumbent=incumbent)
        offspring = [r for r in kept if r["smiles"] != incumbent["smiles"]]
        endpoints = [
            values[r["smiles"]]["endpoint"]
            for r in offspring
            if values[r["smiles"]]["endpoint"] is not None
        ]
        q = audit["offspring"]["first_slot_probabilities"]
        indices = audit["offspring"]["pool"]
        sources = {r["attempt_id"]: r["smiles"] for r in pool}
        outputs[arm] = {
            "retained": [r["attempt_id"] for r in offspring],
            "decision": audit,
            "recoverable_first_slot_probability": sum(
                p
                for p, a in zip(q, indices, strict=True)
                if values[sources[a]]["endpoint"] is not None
            ),
            "verified_continuations": endpoints,
            "new_eligible_continuations": len({r["smiles"] for r in endpoints}),
        }
    return {"arms": outputs, "values": values}


def run_remote(task, repo_root, artifact_root, volume, runtime_factory, validate_revision):
    case = task["case_index"]
    if type(case) is not int or not 0 <= case < 3:
        raise ValueError("lookahead preparation requires worker index 0..2")

    def select_roots(candidates, count):
        return decision_pool(candidates, count)[case::3]

    def cached_products(graph, compatible_law_dirs, config):
        directory = artifact_root / config["product_cache"]["path"]
        if directory not in compatible_law_dirs:
            return None
        verify_file(directory / "result.json", config["product_cache"]["result_sha256"])
        cached_result = json.loads((directory / "result.json").read_text())
        for name in ("archive", "value_snapshot"):
            if cached_result["configuration"][name] != config[name]:
                raise ValueError("cached repair scoring inputs differ")
        module = repo_root / "src/compose_v4/experiments/t4_repair_neighbors.py"
        if (
            enumerator_source_hash(module.read_text())
            != config["product_cache"]["enumerator_source_sha256"]
        ):
            raise ValueError("cached repair enumerator changed")
        exact = encode_state(graph)
        for root in cached_result["roots"]:
            path = directory / f"roots/{root['root_index']:02d}"
            if root["root"]["node"]["graph"] != exact:
                continue
            generated, scored = (
                unseal(path / "generation_lock.json"),
                unseal(path / "scored_lock.json"),
            )
            if generated["source"] != exact or {r["smiles"] for r in scored} != {
                r["smiles"] for r in generated["products"]
            }:
                raise ValueError("cached repair payload mismatch")
            return (
                generated,
                scored,
                {
                    "source_exact": exact,
                    "inputs": {
                        str(p): sha256_file(p)
                        for p in (
                            directory / "result.json",
                            path / "generation_lock.json",
                            path / "scored_lock.json",
                        )
                    },
                    "enumerator_source_sha256": config["product_cache"]["enumerator_source_sha256"],
                    "executor_and_model_compatible": True,
                },
            )
        return None

    return repair_remote(
        task,
        repo_root,
        artifact_root,
        volume,
        runtime_factory,
        validate_revision,
        contract_path=CONTRACT_PATH,
        run_kind=f"{KIND}/case_{case}",
        select_roots=select_roots,
        cached_products=cached_products,
    )
