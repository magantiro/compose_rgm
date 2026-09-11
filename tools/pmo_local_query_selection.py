"""Fixed-recipe, paid-label endpoint ranking and bounded prospective query test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import resource
import subprocess
import tarfile
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_branch_policy import _frozen_save
from compose_v4.experiments.pmo_chronological import RECIPE, kernel_prediction, state_features
from compose_v4.experiments.pmo_macro_probe import DurableScores, make_oracle
from compose_v4.experiments.t4_matched_pilot import _stamp, unseal
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/PMO_LOCAL_QUERY_SELECTION.md"


def partition(smiles):
    """Endpoint-disjoint, within-parent calibration. No outcome enters the split."""
    if len(set(smiles)) != len(smiles):
        raise ValueError("local query split contains duplicate canonical identities")
    train, calibration = [], []
    for s in sorted(smiles):
        byte = hashlib.sha256(f"local-query-v1/{s}".encode()).digest()[0]
        (calibration if byte % 5 == 0 else train).append(s)
    return {"train": train, "calibration": calibration}


def query_arms(smiles, predictions):
    if len(smiles) != len(predictions) or len(smiles) < 16 or len(set(smiles)) != len(smiles):
        raise ValueError("query arms require at least 16 distinct aligned candidates")
    if not np.isfinite(predictions).all():
        raise ValueError("nonfinite endpoint predictions")
    ordered = sorted(zip(smiles, predictions, strict=True), key=lambda r: (-r[1], r[0]))
    pool = sorted(smiles)
    indices = np.random.default_rng(20261005).choice(len(pool), 16, replace=False)
    return {
        "predicted": [s for s, _ in ordered[:16]],
        "uniform": [pool[int(i)] for i in indices],
    }


def metrics(selected, scores, threshold):
    values = [scores[s] for s in selected]
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "best": max(values),
        "parent_improvers": sum(v > threshold for v in values),
        "improvement_precision": sum(v > threshold for v in values) / len(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    if args.output.resolve().is_relative_to(ROOT):
        parser.error("raw artifacts must remain outside the source tree")
    old_config = unseal(args.input / "configuration.json")
    old = unseal(args.input / "result.json")
    generation = unseal(args.input / "generation_lock.json")
    verify_file(args.input / "configuration.json", old["configuration_sha256"])
    for p, h in old_config["implementation"].items():
        if p.startswith(("src/compose_v4/chem/", "src/compose_v4/rewrite/")):
            verify_file(ROOT / p, h)
    software = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        **{k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch", "PyTDC")},
    }
    tdc = Path(importlib.util.find_spec("tdc").origin).parent
    tdc_hashes = {str(p.relative_to(tdc)): sha256_file(p) for p in sorted(tdc.rglob("*.py"))}
    if (
        software != old_config["software"]
        or tdc_hashes != old_config["oracle_implementation_sha256"]
    ):
        raise ValueError("local query experiment requires the qualified oracle/runtime")
    scores = {r["smiles"]: r["score"] for r in old["rows"]}
    if len(scores) != len(old["rows"]) or old["status"] != "complete_development":
        raise ValueError("local labels are duplicated or incomplete")
    split = partition(list(scores))
    store = Store(args.output, lambda: None)
    config = {
        "schema_version": "pmo_local_query_v1",
        "task": "perindopril_mpo",
        "new_call_limit": 32,
        "time_limit_seconds": 120,
        "selection_seed": 20261005,
        "recipe": RECIPE,
        "split": split,
        "input_sha256": {
            str(args.input / p): sha256_file(args.input / p)
            for p in (
                "configuration.json",
                "generation_lock.json",
                "query_lock.json",
                "result.json",
                "source_snapshot.tar.gz",
            )
        },
        "implementation": {
            p: sha256_file(ROOT / p)
            for p in sorted(
                set(old_config["implementation"]) | {PROTOCOL, "tools/pmo_local_query_selection.py"}
            )
        },
        "software": software,
        "oracle_implementation_sha256": tdc_hashes,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "hardware": {
            "device": "cpu",
            "precision": "float64",
            "threads": 1,
            "machine": platform.machine(),
        },
        "role": "exposed within-parent interpolation; not benchmark AUC or future value",
        "historical_prescreen_calls": old["historical_prescreen_calls"],
        "historical_development_physical_calls": old["historical_development_physical_calls"]
        + old["summary"]["new_calls"],
    }
    existing = store.read("configuration")
    if existing:
        config["code_revision"] = existing["code_revision"]
    _frozen_save(store, "configuration", config)
    snapshot_path = args.output / "source_snapshot.tar.gz"
    if not snapshot_path.exists():
        with tarfile.open(snapshot_path, "w:gz") as archive:
            for p in config["implementation"]:
                archive.add(ROOT / p, arcname=p, recursive=False)
    products = {r["smiles"]: r for r in generation["products"]}
    if not set(scores) <= set(products):
        raise ValueError("paid training labels are outside the locked neighborhood")
    threshold = old["summary"]["initial_score"]
    lock = store.read("query_lock")
    calibration = store.read("calibration")
    if calibration is None:
        # No fit or feature construction occurred before the split was published.
        all_smiles = sorted(products)
        index = {s: i for i, s in enumerate(all_smiles)}
        labels = np.array([scores.get(s, np.nan) for s in all_smiles])
        before_kernel = perf_counter()
        kernel, _ = state_features(all_smiles)
        kernel_seconds = perf_counter() - before_kernel
        train = [index[s] for s in split["train"]]
        test = [index[s] for s in split["calibration"]]
        before_fit = perf_counter()
        prediction, model = kernel_prediction(kernel, labels, train, test)
        ordered = sorted(
            zip(split["calibration"], prediction, strict=True), key=lambda r: (-r[1], r[0])
        )
        selected = [s for s, _ in ordered[:8]]
        measured = metrics(selected, scores, threshold)
        uniform = metrics(split["calibration"], scores, threshold)
        passes = measured["mean"] > uniform["mean"] and measured["parent_improvers"] > 0
        calibration = {
            "schema_version": "local_endpoint_calibration_v1",
            "configuration_sha256": sha256_file(args.output / "configuration.json"),
            "selected": measured,
            "uniform_expected": uniform,
            "positive_recall": measured["parent_improvers"] / uniform["parent_improvers"]
            if uniform["parent_improvers"]
            else None,
            "mae": float(np.mean(np.abs(prediction - labels[test]))),
            "predictions": [
                {"smiles": s, "predicted": float(p), "observed": scores[s]} for s, p in ordered
            ],
            "model": model,
            "kernel_seconds": kernel_seconds,
            "calibration_fit_seconds": perf_counter() - before_fit,
            "decision": "bounded_prospective_query_test"
            if passes
            else "stop_unchanged_local_regressor",
        }
        store.save("calibration", calibration, durable=True)
        if passes:
            unscored = sorted(set(products) - set(scores))
            prediction, model = kernel_prediction(
                kernel, labels, [index[s] for s in sorted(scores)], [index[s] for s in unscored]
            )
            arms = query_arms(unscored, prediction)
            lock = {
                "model": model,
                "training_smiles": sorted(scores),
                "all_smiles": all_smiles,
                "configuration_sha256": sha256_file(args.output / "configuration.json"),
                "calibration_sha256": sha256_file(args.output / "calibration.json"),
                "arms": arms,
                "predictions": dict(zip(unscored, map(float, prediction), strict=True)),
                "new": sorted(set().union(*map(set, arms.values()))),
                "prepared_at": _stamp(),
                "prepare_seconds": perf_counter() - started,
            }
            _frozen_save(store, "query_lock", lock)
    print(
        json.dumps(
            {
                k: calibration[k]
                for k in ("selected", "uniform_expected", "mae", "decision", "kernel_seconds")
            }
        ),
        flush=True,
    )
    if not args.score or calibration["decision"] != "bounded_prospective_query_test":
        return
    if store.read("result") is not None:
        print(json.dumps(store.read("result")["arms"]))
        return
    if (
        lock is None
        or len(lock["new"]) > config["new_call_limit"]
        or set(lock["new"]) & set(scores)
    ):
        raise ValueError("query lock missing, over budget, or includes paid training labels")
    system, source = editing_v2_semantic_rewrite_system(), decode_state(generation["source"])
    for s in lock["new"]:
        row = products[s]
        action = row["witnesses"][0]
        codec = action_codec_v4 if action["schema_version"] == 4 else action_codec
        endpoint = system.apply(source, *codec.decode_action(action))
        if encode_state(endpoint) != row["state"] or canonical_state_key(endpoint) != s:
            raise ValueError(f"locked candidate lacks exact primitive replay: {s}")
    if perf_counter() - started > config["time_limit_seconds"]:
        raise RuntimeError("local preparation exceeds bound; do not start oracle scoring")
    oracle = DurableScores(
        args.output,
        make_oracle(config["task"], ROOT, {}),
        config["new_call_limit"],
        lambda: None,
        {},
    )
    oracle.context = {
        "role": "prospective_local_query_comparison",
        "lock_path": str(args.output / "query_lock.json"),
    }
    values = [r["desirability"] for r in oracle.score_many(lock["new"])]
    scored = dict(zip(lock["new"], values, strict=True))
    arms = {arm: metrics(members, scored, threshold) for arm, members in lock["arms"].items()}
    passes = (
        arms["predicted"]["mean"] > arms["uniform"]["mean"]
        and arms["predicted"]["parent_improvers"] > arms["uniform"]["parent_improvers"]
    )
    result = {
        "schema_version": "local_query_comparison_v1",
        "status": "complete_development",
        "configuration_sha256": sha256_file(args.output / "configuration.json"),
        "query_lock_sha256": sha256_file(args.output / "query_lock.json"),
        "calibration_sha256": sha256_file(args.output / "calibration.json"),
        "arms": arms,
        "rows": [{**products[s], "score": scored[s]} for s in lock["new"]],
        "oracle_rows": oracle.rows,
        "new_calls": len(oracle.rows),
        "prior_champion": old["summary"]["best_neighbor"],
        "best": max(old["summary"]["best_neighbor"], max(scored.values())),
        "decision": "test_unchanged_model_on_new_parents"
        if passes
        else "stop_unchanged_local_regressor",
        "prepare_seconds": lock["prepare_seconds"],
        "scoring_and_replay_seconds": perf_counter() - started,
        "oracle_seconds": sum(q["oracle_seconds"] for q in oracle.rows),
        "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "completed_at": _stamp(),
    }
    store.save("result", result, durable=True)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("arms", "new_calls", "best", "decision", "scoring_and_replay_seconds")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
