"""Fresh, at-most-32-query comparison of donor-program preference and random choice."""

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

from compose_v4.control.branch_policy import BranchPolicy
from compose_v4.control.donor_program import compile_transplant, pendant_cuts
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.control.program_selection import choose_pools
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_branch_policy import _frozen_save, candidate_pool
from compose_v4.experiments.pmo_donor_comparison import donor_candidate
from compose_v4.experiments.pmo_macro_probe import DurableScores, make_oracle
from compose_v4.experiments.t4_matched_pilot import _stamp
from compose_v4.experiments.winner_paths import PathConfig
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH = "46cec002fbd450f1a1fb65f9bf1a0b2278b95b613398c8345e78d185d2b3bc09"
MODEL_HASH = "2069958078422b42a35cac0428904eb8eda8f65b4e3647c8efbe1669ebaf9535"


def prepare(path: Path):
    verify_file(path, SOURCE_HASH)
    prior = json.loads(path.read_text())
    snapshot = path.with_name("source_snapshot.tar.gz")
    with tarfile.open(snapshot, "r:gz") as archive:
        content = archive.extractfile(prior["configuration"]["prepared"]["path"]).read()
    if hashlib.sha256(content).hexdigest() != prior["configuration"]["prepared"]["sha256"]:
        raise ValueError("original donor/history preparation differs from the source receipt")
    data = json.loads(content)
    known = data["observed"].copy()
    for row in prior["oracle_rows"]:
        if row["smiles"] in known or row["status"] != "complete":
            raise ValueError("invalid previous physical query receipt")
        known[row["smiles"]] = row["score"]
    nodes = {r["smiles"]: r for r in prior["initial_parents"]}
    for rd in prior["rounds"]:
        for arm in sorted(rd["arms"]):
            for row in rd["arms"][arm]["proposals"]:
                if row is not None:
                    nodes.setdefault(row["smiles"], row)
    parents = sorted(nodes.values(), key=lambda r: (-r["score"], r["smiles"]))[:16]
    for row in parents:
        if (
            canonical_state_key(decode_search_state(row["node"]).graph) != row["smiles"]
            or known[row["smiles"]] != row["score"]
        ):
            raise ValueError("parent exact-state or score mismatch")
    if len(parents) != 16 or parents[0]["score"] != 0.6030226891555273:
        raise ValueError("unexpected current-champion parent panel")
    return {
        "parents": parents,
        "donors": data["donors"],
        "known": known,
        "historical_prescreen_calls": data["historical_prescreen_calls"],
        "historical_development_physical_calls": data["historical_probe_physical_calls"]
        + data["historical_comparison_physical_calls"]
        + prior["new_oracle_calls"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        parser.error("raw receipts must be outside the serialized source tree")
    if rdBase.rdkitVersion != "2024.03.5" or importlib.metadata.version("PyTDC") != "0.3.6":
        raise ValueError("requires the qualified RDKit 2024.03.5/PyTDC 0.3.6 environment")
    model_path = ROOT / "diagnostics/pmo_program_ranking/model.json"
    verify_file(model_path, MODEL_HASH)
    data = prepare(args.prior)
    source_paths = [
        "tools/pmo_program_choice.py",
        "docs/PMO_PROGRAM_CHOICE_PROBE.md",
        "src/compose_v4/control/branch_policy.py",
        "src/compose_v4/control/program_selection.py",
        "src/compose_v4/control/donor_program.py",
        "src/compose_v4/control/docking_value.py",
        "src/compose_v4/control/continuation.py",
        "src/compose_v4/control/molecular_search_codec.py",
        "src/compose_v4/experiments/pmo_chronological.py",
        "src/compose_v4/experiments/pmo_donor_comparison.py",
        "src/compose_v4/experiments/pmo_macro_probe.py",
        "src/compose_v4/experiments/winner_paths.py",
        "src/compose_v4/experiments/pmo_branch_policy.py",
        "src/compose_v4/experiments/pmo_archive_pilot.py",
    ]
    source_paths += [
        str(p.relative_to(ROOT))
        for folder in ("chem", "rewrite", "data")
        for p in sorted((ROOT / f"src/compose_v4/{folder}").glob("*.py"))
    ]
    tdc_root = Path(importlib.util.find_spec("tdc").origin).parent
    config = {
        "schema_version": "program_choice_probe_v1",
        "task": "perindopril_mpo",
        "seed": 20260930,
        "parents": 16,
        "draws_per_parent": 8,
        "physical_query_limit": 32,
        "time_limit_seconds": 300,
        "compiler": asdict(PathConfig()),
        "model_file_sha256": MODEL_HASH,
        "inputs_sha256": {
            str(args.prior.resolve()): SOURCE_HASH,
            str(args.prior.with_name("source_snapshot.tar.gz").resolve()): sha256_file(
                args.prior.with_name("source_snapshot.tar.gz")
            ),
        },
        "implementation_sha256": {p: sha256_file(ROOT / p) for p in sorted(source_paths)},
        "oracle_implementation_sha256": {
            str(p.relative_to(tdc_root)): sha256_file(p) for p in sorted(tdc_root.rglob("*.py"))
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            **{k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch", "PyTDC")},
        },
        "hardware": {
            "platform": platform.platform(),
            "workers": 1,
            "threads": 1,
            "precision": "float64/integer molecular arrays",
            "device": "cpu",
        },
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "role": "fresh component selection within exposed prescreened development, not a whole-controller or PMO AUC comparison",
    }
    store = Store(args.output, lambda: None)
    _frozen_save(store, "configuration", config)
    _frozen_save(store, "prepared", data)
    if store.read("result") is not None:
        print(json.dumps(store.read("result")))
        return
    started = perf_counter()
    donors = [decode_state(d) for d in data["donors"]]
    cuts = [pendant_cuts(d) for d in donors]
    pools, attempts = [], []
    for slot, parent in enumerate(data["parents"]):
        origin = decode_search_state(parent["node"])
        left = pendant_cuts(origin.graph)
        candidates = []
        for draw in range(config["draws_per_parent"]):
            if perf_counter() - started > config["time_limit_seconds"]:
                raise TimeoutError(
                    "program-choice proposal deadline, completed draws remain reusable"
                )
            name = f"draws/{slot:02}_{draw:02}"
            saved = store.read(name)
            if saved is None:
                clock = perf_counter()
                rng = np.random.default_rng(
                    np.random.SeedSequence([config["seed"], slot, draw, 776])
                )
                donor_index = int(rng.integers(len(donors)))
                right = cuts[donor_index]
                candidate = None
                if left and right:
                    a, b = left[int(rng.integers(len(left)))], right[int(rng.integers(len(right)))]
                    result = compile_transplant(origin.graph, donors[donor_index], a, b)
                    candidate = donor_candidate(parent, result, name, donor_index, a, b)
                else:
                    result = {"status": "no_pendant_cut"}
                saved = {
                    "slot": slot,
                    "draw": draw,
                    "result": result,
                    "candidate": candidate,
                    "seconds": perf_counter() - clock,
                }
                store.save(name, saved, durable=True)
            attempts.append(
                {
                    "slot": slot,
                    "draw": draw,
                    "status": saved["result"]["status"],
                    "seconds": saved["seconds"],
                }
            )
            if saved["candidate"] is not None:
                candidates.append(saved["candidate"])
        pool, reference, exclusions = candidate_pool(candidates, data["known"])
        pools.append(
            {"parent": parent, "candidates": pool, "reference": reference, "exclusions": exclusions}
        )
        print(
            f"[programs] parents={slot + 1}/16 draws={len(attempts)}/128 novel_pool={len(pool)} elapsed={perf_counter() - started:.1f}s",
            flush=True,
        )
    _frozen_save(store, "pools", {"pools": pools, "attempts": attempts})
    locked = store.read("selection")
    if locked is None:
        clock = perf_counter()
        choices = choose_pools(
            pools, BranchPolicy(json.loads(model_path.read_text())), seed=config["seed"]
        )
        locked = {
            "choices": choices,
            "model_seconds": perf_counter() - clock,
            "pools_sha256": sha256_file(args.output / "pools.json"),
        }
        store.save("selection", locked, durable=True)
    if locked["pools_sha256"] != sha256_file(args.output / "pools.json"):
        raise ValueError("candidate pool changed after selection")
    progress = {}
    score = DurableScores(
        args.output, make_oracle(config["task"], ROOT, {}), 32, lambda: None, progress
    )
    score.context = {
        "role": "fresh_program_choice",
        "lock_path": str(args.output / "selection.json"),
    }
    requested = sorted(
        {
            pools[row["slot"]]["candidates"][index]["smiles"]
            for row in locked["choices"]
            if row["status"] == "selected"
            for index in row["indices"].values()
        }
    )
    if any(s in data["known"] for s in requested):
        raise ValueError("selection introduced an already-paid query")
    if perf_counter() - started > config["time_limit_seconds"]:
        raise TimeoutError("program-choice deadline before oracle calls")
    scored = dict(
        zip(requested, [r["desirability"] for r in score.score_many(requested)], strict=True)
    )
    arms = {}
    for arm in ("baseline", "learned"):
        selected = []
        for row in locked["choices"]:
            if row["status"] == "selected":
                pool = pools[row["slot"]]
                child = pool["candidates"][row["indices"][arm]]
                selected.append(
                    {
                        **child,
                        "score": scored[child["smiles"]],
                        "gain": scored[child["smiles"]] - pool["parent"]["score"],
                    }
                )
        arms[arm] = {
            "selected": selected,
            "best": max([data["parents"][0]["score"], *[r["score"] for r in selected]]),
            "mean_selected_score": float(np.mean([r["score"] for r in selected]))
            if selected
            else None,
            "mean_gain": float(np.mean([r["gain"] for r in selected])) if selected else None,
            "improved": sum(r["gain"] > 1e-12 for r in selected),
            "requests": len(selected),
            "unique_selected": len({r["smiles"] for r in selected}),
        }
    result = {
        "schema_version": "program_choice_result_v1",
        "status": "complete_development",
        "configuration": config,
        "configuration_sha256": sha256_file(args.output / "configuration.json"),
        "prepared_sha256": sha256_file(args.output / "prepared.json"),
        "selection_sha256": sha256_file(args.output / "selection.json"),
        "historical_prescreen_calls": data["historical_prescreen_calls"],
        "historical_development_physical_calls": data["historical_development_physical_calls"],
        "initial_best": data["parents"][0]["score"],
        "arms": arms,
        "new_oracle_calls": len(score.rows),
        "oracle_rows": score.rows,
        "attempt_statuses": dict(Counter(r["status"] for r in attempts)),
        "proposal_seconds": sum(r["seconds"] for r in attempts),
        "model_seconds": locked["model_seconds"],
        "seconds": perf_counter() - started,
        "finished_at": _stamp(),
    }
    store.save("result", result, durable=True)
    publish_json(args.output / "report.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("configuration", "arms", "oracle_rows")},
            indent=2,
        )
    )
    print(
        json.dumps(
            {arm: {k: v for k, v in row.items() if k != "selected"} for arm, row in arms.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
