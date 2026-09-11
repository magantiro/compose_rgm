"""Fit contextual edit replay from paid traces and compare fresh proposal pools."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import tarfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.donor_program import compile_transplant
from compose_v4.control.edit_replay import RECIPE, EditReplay, fit_replay
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_branch_policy import _frozen_save
from compose_v4.experiments.pmo_donor_comparison import donor_candidate
from compose_v4.experiments.pmo_macro_probe import DurableScores, make_oracle
from compose_v4.experiments.t4_matched_pilot import _stamp
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = (
    Path(
        "/private/tmp/compose-pmo-donor-comparison-runs/478421560799673857cd5aa0d2fc36df5b2ee4582828936ca6eb428699bd02fd/result.json"
    ),
    Path(
        "/private/tmp/compose-pmo-donor-replication-runs/88906fe9b25c5b283d4d68571dd8273aabbb16ba10929f182dc6f19f6df6ea5c/result.json"
    ),
)


def training_rows(probe: Path, comparisons: tuple[Path, ...]):
    batch = json.loads((probe / "batch.json").read_text())
    first = json.loads((probe / "result.json").read_text())
    inputs = {
        str(probe / name): sha256_file(probe / name) for name in ("batch.json", "result.json")
    }
    labels = {r["parent"]: r["score"] for r in first["parity"]}
    donors = [r["state"] for r in batch["parents"]]
    rows = []
    for row in first["attempts"]:
        if row["status"] == "compiled":
            parent = batch["parents"][row["parent"]]
            rows.append(
                {
                    "parent_smiles": parent["smiles"],
                    "parent_score": labels[row["parent"]],
                    "smiles": row["smiles"],
                    "score": row["score"],
                    "source": parent["state"],
                    "donor": donors[row["donor"]],
                    "donor_index": row["donor"],
                    "source_cut": row["parent_cut"],
                    "donor_cut": row["donor_cut"],
                    "origin": [str(probe / "result.json"), row["index"]],
                }
            )
    for path in comparisons:
        r = json.loads(path.read_text())
        if r["status"] != "complete_development":
            raise ValueError("unfinished replay training source")
        snapshot = path.with_name("source_snapshot.tar.gz")
        inputs[str(path)], inputs[str(snapshot)] = sha256_file(path), sha256_file(snapshot)
        with tarfile.open(snapshot, "r:gz") as archive:
            raw = archive.extractfile(r["configuration"]["prepared"]["path"]).read()
        if hashlib.sha256(raw).hexdigest() != r["configuration"]["prepared"]["sha256"]:
            raise ValueError("training donors differ from executed preparation")
        if json.loads(raw)["donors"] != donors:
            raise ValueError("replay comparisons use different donor identities")
        for arm in r["configuration"]["arms"]:
            parents = r["initial_parents"]
            for rd in r["rounds"]:
                a = rd["arms"][arm]
                for parent, child in zip(parents, a["proposals"], strict=True):
                    if child is None or child["bundle"]["option"] != "donor_transplant":
                        continue
                    if (
                        parent is None
                        or child["chain"] != parent["chain"] + [child["id"]]
                        or child["parent_smiles"] != parent["smiles"]
                        or child["parent_score"] != parent["score"]
                    ):
                        raise ValueError("replay edge has no matching exact parent ancestry")
                    b = child["bundle"]
                    rows.append(
                        {
                            "parent_smiles": parent["smiles"],
                            "parent_score": parent["score"],
                            "smiles": child["smiles"],
                            "score": child["score"],
                            "source": parent["node"]["graph"],
                            "donor": donors[b["donor_index"]],
                            "donor_index": b["donor_index"],
                            "source_cut": b["source_cut"],
                            "donor_cut": b["donor_cut"],
                            "origin": [str(path), arm, rd["boundary"], child["id"]],
                        }
                    )
                parents = [a["proposals"][i] for i in a["indices"]]
    return rows, donors, inputs


def current_data(choice: Path, audit: Path):
    source = Store(choice, lambda: None)
    old = source.read("prepared")
    nodes = {r["smiles"]: r for r in old["parents"]}
    pools = source.read("pools")
    result = source.read("result")
    if (
        sha256_file(choice / "selection.json") != result["selection_sha256"]
        or sha256_file(choice / "pools.json") != source.read("selection")["pools_sha256"]
    ):
        raise ValueError("previous candidate pool was changed")
    prior = Store(audit, lambda: None).read("result")
    known = old["known"].copy()
    for r in result["oracle_rows"] + prior["oracle_rows"]:
        if r["smiles"] in known or r["status"] != "complete":
            raise ValueError("inconsistent historical query receipt")
        known[r["smiles"]] = r["score"]
    for pool in pools["pools"]:
        for node in pool["candidates"]:
            nodes.setdefault(node["smiles"], {**node, "score": known[node["smiles"]]})
    parents = sorted(nodes.values(), key=lambda r: (-r["score"], r["smiles"]))[:16]
    if len(parents) != 16 or parents[0]["score"] != 0.6030226891555273:
        raise ValueError("replay probe must start from the current champion")
    for parent in parents:
        if (
            canonical_state_key(decode_search_state(parent["node"]).graph) != parent["smiles"]
            or known[parent["smiles"]] != parent["score"]
        ):
            raise ValueError("current exact parent does not match its oracle label")
    return {
        **old,
        "parents": parents,
        "known": known,
        "historical_development_physical_calls": old["historical_development_physical_calls"]
        + result["new_oracle_calls"]
        + prior["new_oracle_calls"],
    }, result["configuration"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        p.error("raw receipts must be outside the source tree")
    started = perf_counter()
    choice = Path("/private/tmp/compose-pmo-program-choice-20260911a")
    audit = Path("/private/tmp/compose-pmo-program-pool-audit-20260911a")
    data, previous_config = current_data(choice, audit)
    rows, donors, inputs = training_rows(
        Path("/private/tmp/compose-pmo-donor-probe-20260911a"), COMPARISONS
    )
    if data["donors"] != donors:
        raise ValueError("training and proposal donor universes differ")
    versions = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        **{k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch", "PyTDC")},
    }
    if versions != previous_config["software"]:
        raise ValueError("edit replay requires the previously qualified local environment")
    tdc_root = Path(importlib.util.find_spec("tdc").origin).parent
    oracle_hashes = {
        str(path.relative_to(tdc_root)): sha256_file(path)
        for path in sorted(tdc_root.rglob("*.py"))
    }
    if oracle_hashes != previous_config["oracle_implementation_sha256"]:
        raise ValueError("edit replay oracle implementation differs from the qualified probe")
    inputs.update(
        {
            str(path): sha256_file(path)
            for path in (
                choice / "prepared.json",
                choice / "pools.json",
                choice / "result.json",
                audit / "result.json",
            )
        }
    )
    dependencies = set(previous_config["implementation_sha256"]) | {
        "src/compose_v4/control/edit_replay.py",
        "tools/pmo_edit_replay.py",
        "docs/PMO_CONTEXTUAL_EDIT_REPLAY.md",
    }
    config = {
        "schema_version": "edit_replay_probe_v1",
        "task": "perindopril_mpo",
        "arms": ["uniform", "replay"],
        "seed": 20261001,
        "parents": 16,
        "draws_per_parent_arm": 4,
        "new_query_limit": 128,
        "time_limit_seconds": 300,
        "recipe": RECIPE,
        "compiler": previous_config["compiler"],
        "inputs_sha256": inputs,
        "implementation_sha256": {name: sha256_file(ROOT / name) for name in sorted(dependencies)},
        "oracle_implementation_sha256": previous_config["oracle_implementation_sha256"],
        "software": versions,
        "hardware": previous_config["hardware"],
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "role": "warm prescreened development; proposal-channel comparison, not full-controller benchmark",
        "training_roles": "isolated donor probe plus first two matched donor runs only; subsequent labels excluded from fitting",
    }
    store = Store(args.output, lambda: None)
    _frozen_save(store, "configuration", config)
    _frozen_save(store, "prepared", data)
    _frozen_save(
        store,
        "training",
        {"rows": rows, "inputs_sha256": inputs, "split": config["training_roles"]},
    )
    if store.read("result") is not None:
        print(json.dumps(store.read("result")))
        return
    bank = store.read("bank")
    fit_start = perf_counter()
    if bank is None:
        bank = fit_replay(rows)
        store.save("bank", bank, durable=True)
    policy = EditReplay(bank)
    fit_seconds = perf_counter() - fit_start
    print(
        f"[replay] rows={bank['raw_rows']} unique={bank['unique_edges']} positive={len(bank['entries'])} parents={bank['positive_parents']} fit={fit_seconds:.2f}s",
        flush=True,
    )
    donor_graphs = [decode_state(s) for s in donors]
    attempted, candidates, feature_seconds = [], {a: [] for a in config["arms"]}, 0.0
    for slot, parent in enumerate(data["parents"]):
        origin = decode_search_state(parent["node"])
        clock = perf_counter()
        cached = policy.distribution(origin.graph)
        feature_seconds += perf_counter() - clock
        _frozen_save(
            store,
            f"context/{slot:02}",
            {
                "cuts": [asdict(c) for c in cached[0]],
                "probabilities": cached[1].tolist(),
                "bank_sha256": bank["bank_sha256"],
            },
        )
        for arm in config["arms"]:
            for draw in range(config["draws_per_parent_arm"]):
                if perf_counter() - started > config["time_limit_seconds"]:
                    raise TimeoutError("edit-replay deadline; completed draws remain reusable")
                name = f"draws/{arm}_{slot:02}_{draw:02}"
                saved = store.read(name)
                if saved is None:
                    clock = perf_counter()
                    rng = np.random.default_rng(
                        np.random.SeedSequence([config["seed"], slot, draw, 776])
                    )
                    proposed = policy.draw(origin.graph, donor_graphs, rng, mode=arm, cached=cached)
                    candidate = None
                    if proposed is None:
                        result = {"status": "no_pendant_cut"}
                    else:
                        i, a, b = (
                            proposed["donor_index"],
                            proposed["source_cut"],
                            proposed["donor_cut"],
                        )
                        result = compile_transplant(origin.graph, donor_graphs[i], a, b)
                        candidate = donor_candidate(parent, result, name, i, a, b)
                        proposed = {**proposed, "source_cut": asdict(a), "donor_cut": asdict(b)}
                    saved = {
                        "arm": arm,
                        "slot": slot,
                        "draw": draw,
                        "proposal": proposed,
                        "result": result,
                        "candidate": candidate,
                        "seconds": perf_counter() - clock,
                    }
                    store.save(name, saved, durable=True)
                attempted.append(
                    {k: saved[k] for k in ("arm", "slot", "draw", "proposal", "seconds")}
                    | {"status": saved["result"]["status"]}
                )
                if saved["candidate"] is not None:
                    candidates[arm].append(saved["candidate"])
        print(
            f"[replay] parents={slot + 1}/16 attempts={len(attempted)}/128 elapsed={perf_counter() - started:.1f}s",
            flush=True,
        )
    _frozen_save(
        store,
        "candidate_lock",
        {"candidates": candidates, "attempts": attempted, "bank_sha256": bank["bank_sha256"]},
    )
    known = data["known"].copy()
    new = sorted({r["smiles"] for arm in candidates.values() for r in arm} - set(known))
    if (
        len(new) > config["new_query_limit"]
        or perf_counter() - started > config["time_limit_seconds"]
    ):
        raise RuntimeError("replay query batch exceeds declared bounds")
    scores = DurableScores(
        args.output,
        make_oracle(config["task"], ROOT, {}),
        config["new_query_limit"],
        lambda: None,
        {},
    )
    scores.context = {
        "role": "fixed_replay_comparison",
        "lock_path": str(args.output / "candidate_lock.json"),
    }
    known.update(zip(new, [r["desirability"] for r in scores.score_many(new)], strict=True))
    arms = {}
    for arm in config["arms"]:
        selected = [
            {**r, "score": known[r["smiles"]], "gain": known[r["smiles"]] - r["parent_score"]}
            for r in candidates[arm]
        ]
        novel = {r["smiles"]: r["score"] for r in selected if r["smiles"] not in data["known"]}
        archive = {p["smiles"]: p["score"] for p in data["parents"]} | {
            r["smiles"]: r["score"] for r in selected
        }
        arms[arm] = {
            "candidates": selected,
            "complete": len(selected),
            "improved_draws": sum(r["gain"] > 1e-12 for r in selected),
            "unique": len({r["smiles"] for r in selected}),
            "novel_unique": len(novel),
            "best": max(archive.values()),
            "best_new": max(novel.values(), default=None),
            "top10_mean": float(np.mean(sorted(archive.values(), reverse=True)[:10])),
            "mean_gain": float(np.mean([r["gain"] for r in selected])) if selected else None,
            "statuses": dict(Counter(r["status"] for r in attempted if r["arm"] == arm)),
            "proposal_seconds": sum(r["seconds"] for r in attempted if r["arm"] == arm),
        }
    result = {
        "schema_version": "edit_replay_probe_result_v1",
        "status": "complete_development",
        "configuration": config,
        "configuration_sha256": sha256_file(args.output / "configuration.json"),
        "training_sha256": sha256_file(args.output / "training.json"),
        "bank_sha256": bank["bank_sha256"],
        "candidate_lock_sha256": sha256_file(args.output / "candidate_lock.json"),
        "initial_best": data["parents"][0]["score"],
        "arms": arms,
        "new_oracle_calls": len(scores.rows),
        "oracle_rows": scores.rows,
        "historical_prescreen_calls": data["historical_prescreen_calls"],
        "historical_development_physical_calls": data["historical_development_physical_calls"],
        "fit_seconds": fit_seconds,
        "context_seconds": feature_seconds,
        "seconds": perf_counter() - started,
        "finished_at": _stamp(),
    }
    store.save("result", result, durable=True)
    publish_json(args.output / "report.json", result)
    print(
        json.dumps(
            {
                "new_calls": len(scores.rows),
                "seconds": result["seconds"],
                "context_seconds": feature_seconds,
                "arms": {
                    arm: {k: v for k, v in r.items() if k != "candidates"}
                    for arm, r in arms.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
