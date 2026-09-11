"""Test an already-fitted endpoint selector on new cached parent neighborhoods."""

from __future__ import annotations

import argparse
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

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_branch_policy import _frozen_save
from compose_v4.experiments.pmo_chronological import state_features
from compose_v4.experiments.pmo_macro_probe import DurableScores, make_oracle
from compose_v4.experiments.t4_matched_pilot import _stamp, unseal
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from tools.pmo_cached_neighborhood import census
from tools.pmo_local_query_selection import metrics

ROOT = Path(__file__).resolve().parents[1]
WORKERS = (
    "64d73a1d8394c4f4fd2db5dddf56fd3e03c9d9670dbdc498428dd61e89f1d81e",
    "1a4dd4306be3a4c4d2224ebc8e25bc509f2ebbde494f0cffb741f4471e485448",
)


def transfer_arms(smiles, predictions, parent_index):
    if len(smiles) < 16 or len(smiles) != len(predictions) or len(set(smiles)) != len(smiles):
        raise ValueError("cross-parent allocation requires 16 distinct aligned candidates")
    if not np.isfinite(predictions).all() or parent_index not in (0, 1):
        raise ValueError("invalid predictions or parent index")
    ordered = sorted(zip(smiles, predictions, strict=True), key=lambda r: (-r[1], r[0]))
    pool = sorted(smiles)
    rng = np.random.default_rng(np.random.SeedSequence([20261007, parent_index]))
    return {
        "predicted": [s for s, _ in ordered[:16]],
        "uniform": [pool[int(i)] for i in rng.choice(len(pool), 16, replace=False)],
    }


def predict_frozen(lock, candidates):
    """Apply saved coefficients. No fit, label update or hyperparameter selection."""
    model = lock["model"]
    snapshot = {k: v for k, v in model.items() if k != "snapshot_sha256"}
    if identity(snapshot) != model["snapshot_sha256"]:
        raise ValueError("frozen endpoint model hash mismatch")
    train = lock["training_smiles"]
    if [lock["all_smiles"][i] for i in model["train_indices"]] != train:
        raise ValueError("frozen coefficient/training identity order mismatch")
    if set(train) & set(candidates) or len(set(candidates)) != len(candidates):
        raise ValueError("prediction candidates overlap paid training identities or repeat")
    kernel, _ = state_features(train + candidates)
    return model["mean"] + kernel[len(train) :, : len(train)] @ np.asarray(model["coefficients"])


def decision(measured):
    return (
        "integrate_local_channel_then_matched_broad_comparison"
        if all(m["predicted"]["mean"] > m["uniform"]["mean"] for m in measured)
        and sum(m["predicted"]["parent_improvers"] for m in measured)
        > sum(m["uniform"]["parent_improvers"] for m in measured)
        else "do_not_promote_frozen_local_model"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--laws", type=Path, nargs=2, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--lookahead", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    if args.output.resolve().is_relative_to(ROOT):
        parser.error("raw experiment output must be outside source")
    origin = json.loads(args.origin.read_text())
    manifest = unseal(args.manifest)
    selection = unseal(args.selection / "result.json")
    previous = unseal(args.selection / "configuration.json")
    model_lock = unseal(args.selection / "query_lock.json")
    history = unseal(args.lookahead / "prepared.json")
    lookahead = unseal(args.lookahead / "result.json")
    if any(r["status"] != "complete_development" for r in (origin, selection, lookahead)):
        raise ValueError("cross-parent comparison requires completed input runs")
    verify_file(args.selection / "query_lock.json", selection["query_lock_sha256"])
    verify_file(args.selection / "configuration.json", selection["configuration_sha256"])
    verify_file(args.lookahead / "prepared.json", lookahead["prepared_sha256"])
    known = history["known"].copy()
    for r in lookahead["oracle_rows"]:
        if r["smiles"] in known or r["status"] != "complete":
            raise ValueError("incomplete or duplicated historical query")
        known[r["smiles"]] = r["score"]
    parents = [manifest["tasks"][k]["parent"] for k in WORKERS]
    # Verify the declared rank/cache/slot selection against the completed run.
    measured_work = {w["worker_id"]: w["law_work"] for w in origin["workers"]}
    eligible = []
    for key, task in manifest["tasks"].items():
        work = measured_work[key]
        if any(work.get(k, 0) for k in ("fresh_laws", "memory_hits", "old_hits", "saved_hits")):
            eligible.append((key, task))
    chosen = {}
    for key, task in sorted(eligible, key=lambda v: (v[1]["slot"], v[0])):
        chosen.setdefault(task["parent"]["smiles"], (key, task))
    ranked = sorted(
        chosen.values(), key=lambda v: (-v[1]["parent"]["score"], v[1]["parent"]["smiles"])
    )
    if tuple(k for k, _ in ranked[:2]) != WORKERS:
        raise ValueError("declared parent ranking differs from saved worker evidence")
    for p, law in zip(parents, args.laws, strict=True):
        graph = p["node"]["graph"]
        if (
            unseal(law)["source"] != graph
            or canonical_state_key(decode_state(graph)) != p["smiles"]
        ):
            raise ValueError("cached law and exact parent graph disagree")
        if known[p["smiles"]] != p["score"]:
            raise ValueError("parent score differs from paid historical label")
    for p, h in origin["image_revision"]["serialized_sources"].items():
        if p.startswith(("src/compose_v4/chem/", "src/compose_v4/rewrite/")):
            verify_file(ROOT / p, h)
    software = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        **{k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch", "PyTDC")},
    }
    tdc = Path(importlib.util.find_spec("tdc").origin).parent
    oracle_files = {str(p.relative_to(tdc)): sha256_file(p) for p in sorted(tdc.rglob("*.py"))}
    if software != previous["software"] or oracle_files != previous["oracle_implementation_sha256"]:
        raise ValueError("cross-parent run requires the qualified local oracle/runtime")
    input_paths = (
        [args.origin, args.manifest, *args.laws]
        + [args.selection / p for p in ("configuration.json", "query_lock.json", "result.json")]
        + [args.lookahead / p for p in ("prepared.json", "result.json")]
    )
    implementation = set(previous["implementation"]) | {
        "tools/pmo_cross_parent_selection.py",
        "docs/PMO_CROSS_PARENT_SELECTION.md",
    }
    store = Store(args.output, lambda: None)
    config = {
        "schema_version": "pmo_cross_parent_selection_v1",
        "task": "perindopril_mpo",
        "new_call_limit": 64,
        "prepare_time_limit_seconds": 180,
        "selection_seed": 20261007,
        "inputs": {str(p): sha256_file(p) for p in input_paths},
        "implementation": {p: sha256_file(ROOT / p) for p in sorted(implementation)},
        "model_sha256": model_lock["model"]["snapshot_sha256"],
        "workers": list(WORKERS),
        "software": software,
        "oracle_implementation_sha256": oracle_files,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "hardware": {
            "device": "cpu",
            "precision": "float64",
            "threads": 1,
            "machine": platform.machine(),
        },
        "historical_prescreen_calls": history["historical_prescreen_calls"],
        "historical_development_physical_calls": history["historical_development_physical_calls"]
        + lookahead["new_calls"],
        "role": "exposed warm cross-neighborhood component; not source-held-out or benchmark AUC",
        "parent_in_model_training": [p["smiles"] in model_lock["training_smiles"] for p in parents],
    }
    old = store.read("configuration")
    if old:
        config["code_revision"] = old["code_revision"]
    _frozen_save(store, "configuration", config)
    snapshot = args.output / "source_snapshot.tar.gz"
    if not snapshot.exists():
        with tarfile.open(snapshot, "w:gz") as archive:
            for p in config["implementation"]:
                archive.add(ROOT / p, arcname=p, recursive=False)
            for p in input_paths:
                archive.add(p, arcname=f"inputs/{sha256_file(p)}.json", recursive=False)
    generations = []
    for i, law in enumerate(args.laws):
        generation = store.read(f"generation/{i}")
        if generation is None:
            before = perf_counter()
            generation = census(unseal(law))
            generation["generation_seconds"] = perf_counter() - before
            generation["input_law_sha256"] = sha256_file(law)
            store.save(f"generation/{i}", generation, durable=True)
        generations.append(generation)
        print(
            json.dumps(
                {
                    "parent": i,
                    "products": len(generation["products"]),
                    "generation_seconds": generation["generation_seconds"],
                }
            ),
            flush=True,
        )
    lock = store.read("query_lock")
    if lock is None:
        batches = []
        for i, (parent, generation) in enumerate(zip(parents, generations, strict=True)):
            pool = sorted({p["smiles"] for p in generation["products"]} - set(known))
            if len(pool) < 16:
                store.save(
                    "abstention",
                    {"status": "inconclusive_pool_shortfall", "parent": i, "unqueried": len(pool)},
                    durable=True,
                )
                return
            before = perf_counter()
            predicted = predict_frozen(model_lock, pool)
            batches.append(
                {
                    "parent": parent,
                    "pool": pool,
                    "predictions": dict(zip(pool, map(float, predicted), strict=True)),
                    "arms": transfer_arms(pool, predicted, i),
                    "prediction_seconds": perf_counter() - before,
                }
            )
            print(
                json.dumps(
                    {
                        "parent": i,
                        "unqueried": len(pool),
                        "prediction_seconds": batches[-1]["prediction_seconds"],
                    }
                ),
                flush=True,
            )
        lock = {
            "configuration_sha256": sha256_file(args.output / "configuration.json"),
            "generation_sha256": [
                sha256_file(args.output / f"generation/{i}.json") for i in range(2)
            ],
            "batches": batches,
            "new": sorted({s for b in batches for arm in b["arms"].values() for s in arm}),
            "prepared_at": _stamp(),
            "prepare_seconds": perf_counter() - started,
        }
        _frozen_save(store, "query_lock", lock)
    if not args.score:
        print(
            json.dumps(
                {"locked_queries": len(lock["new"]), "prepare_seconds": lock["prepare_seconds"]}
            ),
            flush=True,
        )
        return
    if store.read("result"):
        print(json.dumps(store.read("result")["summary"]), flush=True)
        return
    if len(lock["new"]) > 64 or set(lock["new"]) & set(known):
        raise ValueError("query lock repeats historical scores or exceeds physical budget")
    system = editing_v2_semantic_rewrite_system()
    for generation, batch in zip(generations, lock["batches"], strict=True):
        source = decode_state(generation["source"])
        products = {p["smiles"]: p for p in generation["products"]}
        for s in set().union(*map(set, batch["arms"].values())):
            row, action = products[s], products[s]["witnesses"][0]
            codec = action_codec_v4 if action["schema_version"] == 4 else action_codec
            product = system.apply(source, *codec.decode_action(action))
            if encode_state(product) != row["state"] or canonical_state_key(product) != s:
                raise ValueError("candidate failed exact primitive replay")
    if max(lock["prepare_seconds"], perf_counter() - started) > 180:
        raise RuntimeError("preparation exceeded bound; no new oracle calls authorized")
    oracle = DurableScores(args.output, make_oracle(config["task"], ROOT, {}), 64, lambda: None, {})
    oracle.context = {
        "role": "cross_parent_locked_query_comparison",
        "lock_path": str(args.output / "query_lock.json"),
    }
    values = [r["desirability"] for r in oracle.score_many(lock["new"])]
    scores = dict(zip(lock["new"], values, strict=True))
    measured = [
        {
            a: metrics(members, scores, batch["parent"]["score"])
            for a, members in batch["arms"].items()
        }
        for batch in lock["batches"]
    ]
    summary = {
        "per_parent": measured,
        "decision": decision(measured),
        "prior_champion": max(known.values()),
        "best": max(max(known.values()), max(scores.values())),
        "new_calls": len(oracle.rows),
        "prepare_seconds": lock["prepare_seconds"],
        "replay_and_scoring_seconds": perf_counter() - started,
        "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    store.save(
        "result",
        {
            "schema_version": "pmo_cross_parent_result_v1",
            "status": "complete_development",
            "configuration_sha256": sha256_file(args.output / "configuration.json"),
            "query_lock_sha256": sha256_file(args.output / "query_lock.json"),
            "oracle_rows": oracle.rows,
            "summary": summary,
            "completed_at": _stamp(),
        },
        durable=True,
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
