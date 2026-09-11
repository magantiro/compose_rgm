"""Score a locked, already-computed production neighborhood without new neural work.

This is an exposed development coverage diagnostic, not an optimizer comparison.
Positive: an existing one-edit successor exceeds the current observed champion.
Negative: the queried subset cannot do so; do not repeat its enumeration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
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
from compose_v4.experiments.pmo_macro_probe import DurableScores, make_oracle
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_repair_neighbors import enumerate_products
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state

ROOT = Path(__file__).resolve().parents[1]


def census(payload):
    graph = decode_state(payload["source"])
    pairs = [
        (action_codec_v4 if a["schema_version"] == 4 else action_codec).decode_action(a)
        for a in payload["marks"]
    ]

    def saved_law(state):
        if encode_state(state) != payload["source"]:
            raise ValueError("cached neighborhood requested for a different exact state")
        return tuple(f for f, _ in pairs), tuple(a for _, a in pairs), payload["probabilities"]

    return enumerate_products(graph, saved_law, editing_v2_semantic_rewrite_system())


def query_subset(products, known, limit=1024, seed=20261004):
    """Uniform budgeted oracle allocation, not a molecular-support restriction."""
    novel = sorted({r["smiles"] for r in products} - set(known))
    if len(novel) <= limit:
        return novel
    indices = np.random.default_rng(seed).choice(len(novel), size=limit, replace=False)
    return sorted(novel[int(i)] for i in indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--law", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--generation", type=Path)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        parser.error("raw receipts must remain outside the source tree")
    start = perf_counter()
    origin = json.loads(args.origin.read_text())
    if origin["status"] != "complete_development":
        raise ValueError("cached law origin is not a completed experiment")
    source_files = origin["image_revision"]["serialized_sources"]
    for p, h in source_files.items():
        if p.startswith(tuple(f"src/compose_v4/{d}/" for d in ("chem", "rewrite"))):
            verify_file(ROOT / p, h)
    payload = unseal(args.law)
    if payload["source"] != origin["initial_parents"][0]["node"]["graph"]:
        raise ValueError("diagnostic must use the declared initial incumbent's exact state")
    qualified = unseal(args.qualification)
    versions = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        **{k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch", "PyTDC")},
    }
    tdc_root = Path(importlib.util.find_spec("tdc").origin).parent
    oracle_files = {
        str(p.relative_to(tdc_root)): sha256_file(p) for p in sorted(tdc_root.rglob("*.py"))
    }
    if (
        versions != qualified["software"]
        or oracle_files != qualified["oracle_implementation_sha256"]
    ):
        raise ValueError("cached-neighborhood evaluation requires the qualified local oracle")
    config = {
        "schema_version": "pmo_cached_neighborhood_v1",
        "task": "perindopril_mpo",
        "new_query_limit": 1024,
        "time_limit_seconds": 120,
        "source_exact_id": identity(payload["source"]),
        "inputs": {str(p): sha256_file(p) for p in (args.law, args.origin, args.qualification)},
        "implementation": {
            p: sha256_file(ROOT / p)
            for p in sorted(set(source_files) | {"tools/pmo_cached_neighborhood.py"})
        },
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "software": versions,
        "oracle_implementation_sha256": oracle_files,
        "hardware": {"device": "cpu", "machine": platform.machine(), "threads": 1},
        "ordering": "canonical SMILES, no stochastic selection",
        "query_allocation": {
            "rule": "all novel or uniform subset without replacement",
            "seed": 20261004,
            "negative_scope": "queried subset only",
        },
        "role": "exposed warm development; all positive-mass primitive successors; not a Doob or benchmark result",
    }
    if args.generation is not None:
        config["reused_generation"] = {
            "path": str(args.generation),
            "sha256": sha256_file(args.generation),
            "configuration_sha256": sha256_file(args.generation.with_name("configuration.json")),
        }
    store = Store(args.output, lambda: None)
    existing_config = store.read("configuration")
    if existing_config is not None:
        # Allow operational metadata to change, not inputs or source bytes.
        for key in ("code_revision", "worktree_status"):
            config[key] = existing_config[key]
    _frozen_save(store, "configuration", config)
    snapshot_path = args.output / "source_snapshot.tar.gz"
    if not snapshot_path.exists():
        with tarfile.open(snapshot_path, "w:gz") as archive:
            for path in config["implementation"]:
                archive.add(ROOT / path, arcname=path, recursive=False)
    generated = store.read("generation_lock")
    if generated is None:
        if args.generation is None:
            generated = census(payload)
            generated["generation_seconds"] = perf_counter() - start
        else:
            old_config = unseal(args.generation.with_name("configuration.json"))
            for p in config["inputs"]:
                if config["inputs"][p] != old_config["inputs"][p]:
                    raise ValueError("reused generation has different inputs")
            for p, h in old_config["implementation"].items():
                if p.startswith("src/compose_v4/"):
                    verify_file(ROOT / p, h)
            generated = unseal(args.generation)
            if generated["source"] != payload["source"]:
                raise ValueError("reused generation has a different exact source")
        store.save("generation_lock", generated, durable=True)
    print(
        json.dumps(
            {
                "products": len(generated["products"]),
                "counts": generated["counts"],
                "generation_seconds": generated["generation_seconds"],
                "new_neural_calls": 0,
            }
        ),
        flush=True,
    )
    if args.history is None:
        return
    history = json.loads(args.history.read_text())
    if history["status"] != "complete_development":
        raise ValueError("wait for the independent comparison to finish before charging new labels")
    snapshot = args.history.with_name("source_snapshot.tar.gz")
    with tarfile.open(snapshot) as archive:
        raw = archive.extractfile(history["configuration"]["prepared"]["path"]).read()
    if hashlib.sha256(raw).hexdigest() != history["configuration"]["prepared"]["sha256"]:
        raise ValueError("historical observed labels differ from the executed snapshot")
    prepared = json.loads(raw)
    known = dict(prepared["observed"])
    for q in history["oracle_rows"]:
        if q["status"] != "complete" or q["smiles"] in known:
            raise ValueError("historical query is repeated or incomplete")
        known[q["smiles"]] = q["score"]
    new = query_subset(
        generated["products"], known, config["new_query_limit"], config["query_allocation"]["seed"]
    )
    if (
        len(new) > config["new_query_limit"]
        or perf_counter() - start > config["time_limit_seconds"]
    ):
        raise RuntimeError("complete cached neighborhood exceeds its declared diagnostic bound")
    _frozen_save(
        store,
        "query_lock",
        {
            "history_sha256": sha256_file(args.history),
            "snapshot_sha256": sha256_file(snapshot),
            "generation_sha256": sha256_file(args.output / "generation_lock.json"),
            "new": new,
        },
    )
    if store.read("result") is not None:
        print(json.dumps(store.read("result")["summary"]))
        return
    scores = DurableScores(
        args.output,
        make_oracle(config["task"], ROOT, {}),
        config["new_query_limit"],
        lambda: None,
        {},
    )
    scores.context = {
        "role": "all_locked_primitive_neighbors",
        "lock_path": str(args.output / "query_lock.json"),
    }
    values = [r["desirability"] for r in scores.score_many(new)]
    known.update(zip(new, values, strict=True))
    rows = [
        {**r, "score": known[r["smiles"]]} for r in generated["products"] if r["smiles"] in known
    ]
    rows.sort(key=lambda r: (-r["score"], r["smiles"]))
    threshold = max(a["best"] for r in (origin, history) for a in r["arms"].values())
    summary = {
        "initial_score": known[generated["source_smiles"]],
        "current_champion": threshold,
        "best_neighbor": rows[0]["score"] if rows else None,
        "better_than_current": sum(r["score"] > threshold for r in rows),
        "new_calls": len(scores.rows),
        "support_count": len(generated["products"]),
        "scored_count": len(rows),
        "unscored_count": len(generated["products"]) - len(rows),
        "negative_scope": "queried subset only, not the full neighborhood",
        "seconds": perf_counter() - start,
    }
    store.save(
        "result",
        {
            "schema_version": "pmo_cached_neighborhood_result_v1",
            "status": "complete_development",
            "configuration_sha256": sha256_file(args.output / "configuration.json"),
            "query_lock_sha256": sha256_file(args.output / "query_lock.json"),
            "summary": summary,
            "rows": rows,
            "oracle_rows": scores.rows,
            "historical_prescreen_calls": prepared["historical_prescreen_calls"],
            "historical_development_physical_calls": prepared[
                "historical_development_physical_calls"
            ]
            + history["new_oracle_calls"],
        },
        durable=True,
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
