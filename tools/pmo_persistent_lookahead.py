"""Retain witnessed two-option sequences, with matched immediate continuation."""

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
from compose_v4.control.donor_memory import build_memory
from compose_v4.control.donor_program import compile_transplant, pendant_cuts
from compose_v4.control.molecular_search_codec import decode_search_state, encode_search_state
from compose_v4.control.molecular_task_search import MolecularSearchState
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_branch_policy import _frozen_save
from compose_v4.experiments.pmo_donor_comparison import donor_candidate
from compose_v4.experiments.pmo_macro_probe import DurableScores, make_oracle
from compose_v4.experiments.t4_matched_pilot import _stamp, unseal
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]


def best_first(root, proposals):
    candidates = [(-1, root)] + [(i, p) for i, p in enumerate(proposals) if p is not None]
    return min(candidates, key=lambda item: (-item[1]["score"], item[1]["smiles"], item[0]))


def continuation_tasks(root_index, first, selected):
    slot, parent = selected
    return {
        "immediate": [(root_index, slot, draw, parent) for draw in range(8)],
        "lookahead": [(root_index, i, draw, p) for i, p in enumerate(first) for draw in range(2)],
    }


def load_data(history_path, local_path, selection_path):
    history = json.loads(history_path.read_text())
    snapshot = history_path.with_name("source_snapshot.tar.gz")
    with tarfile.open(snapshot) as archive:
        raw = archive.extractfile(history["configuration"]["prepared"]["path"]).read()
    if hashlib.sha256(raw).hexdigest() != history["configuration"]["prepared"]["sha256"]:
        raise ValueError("historical starting archive differs from executed snapshot")
    data = json.loads(raw)
    local = unseal(local_path / "result.json")
    selection = unseal(selection_path / "result.json")
    if any(r["status"] != "complete_development" for r in (history, local, selection)):
        raise ValueError("parallel lookahead requires completed historical observations")
    known = data["observed"].copy()
    for result in (history, local, selection):
        for row in result["oracle_rows"]:
            if row["status"] != "complete" or row["smiles"] in known:
                raise ValueError("historical oracle calls repeated or unresolved")
            known[row["smiles"]] = row["score"]
    for directory, result in ((local_path, local), (selection_path, selection)):
        verify_file(directory / "configuration.json", result["configuration_sha256"])
        verify_file(directory / "query_lock.json", result["query_lock_sha256"])
    initial = history["initial_parents"][0]
    local_best = max(local["rows"], key=lambda r: (r["score"], r["smiles"]))
    origin = decode_search_state(initial["node"])
    a = local_best["witnesses"][0]
    codec = action_codec_v4 if a["schema_version"] == 4 else action_codec
    lineage = origin.lineage.observe(*codec.decode_action(a))
    node = MolecularSearchState(decode_state(local_best["state"]), lineage, 64, origin.root_id)
    local_id = "cached_neighborhood/" + identity(local_best["state"])
    first_root = {
        "id": local_id,
        "node": encode_search_state(node),
        "smiles": local_best["smiles"],
        "score": local_best["score"],
        "chain": initial["chain"] + [local_id],
        "primitives": initial["primitives"] + 1,
    }
    broad = [p for rd in history["rounds"] for p in rd["arms"]["fixed"]["proposals"] if p]
    second_root = min(broad, key=lambda p: (-p["score"], p["smiles"]))
    roots = [first_root, second_root]
    if [r["score"] for r in roots] != [0.6747477697966318, 0.672021505032247]:
        raise ValueError("parallel test changed its frozen two starting states")
    for root in roots:
        if (
            canonical_state_key(decode_search_state(root["node"]).graph) != root["smiles"]
            or known[root["smiles"]] != root["score"]
        ):
            raise ValueError("root exact state and paid label disagree")
    return {
        "roots": roots,
        "known": known,
        "donor_memory": build_memory(data["initial_donor_memory"], {}, mode="fixed"),
        "historical_prescreen_calls": local["historical_prescreen_calls"],
        "historical_development_physical_calls": local["historical_development_physical_calls"]
        + local["summary"]["new_calls"]
        + selection["new_calls"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    if args.output.resolve().is_relative_to(ROOT):
        parser.error("raw numerical artifacts must remain outside serialized source")
    data = load_data(args.history, args.local, args.selection)
    previous = unseal(args.selection / "configuration.json")
    software = {
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        **{k: importlib.metadata.version(k) for k in ("numpy", "scipy", "torch", "PyTDC")},
    }
    tdc = Path(importlib.util.find_spec("tdc").origin).parent
    oracle_files = {str(p.relative_to(tdc)): sha256_file(p) for p in sorted(tdc.rglob("*.py"))}
    if software != previous["software"] or oracle_files != previous["oracle_implementation_sha256"]:
        raise ValueError("persistent planner requires previously qualified oracle/runtime")
    for p, h in previous["implementation"].items():
        if p.startswith(("src/compose_v4/chem/", "src/compose_v4/rewrite/")):
            verify_file(ROOT / p, h)
    config = {
        "schema_version": "pmo_persistent_lookahead_v1",
        "task": "perindopril_mpo",
        "seed": 20261006,
        "first_draws": 4,
        "continuation_slots_per_arm_root": 8,
        "new_call_limit": 40,
        "time_limit_seconds": 180,
        "compiler": {"max_steps": 64, "max_expansions": 128},
        "input_sha256": {
            str(p): sha256_file(p)
            for p in (
                args.history,
                args.history.with_name("source_snapshot.tar.gz"),
                args.local / "configuration.json",
                args.local / "result.json",
                args.selection / "configuration.json",
                args.selection / "result.json",
            )
        },
        "implementation": {
            p: sha256_file(ROOT / p)
            for p in sorted(
                set(previous["implementation"])
                | {"tools/pmo_persistent_lookahead.py", "docs/PMO_PERSISTENT_LOOKAHEAD.md"}
            )
        },
        "software": software,
        "oracle_implementation_sha256": oracle_files,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "hardware": {"device": "cpu", "threads": 1, "machine": platform.machine()},
        "role": "two-option proposal-channel development; not a Doob transform or full-controller benchmark",
    }
    store = Store(args.output, lambda: None)
    old_config = store.read("configuration")
    if old_config:
        config["code_revision"] = old_config["code_revision"]
    _frozen_save(store, "configuration", config)
    _frozen_save(store, "prepared", data)
    snapshot_path = args.output / "source_snapshot.tar.gz"
    if not snapshot_path.exists():
        with tarfile.open(snapshot_path, "w:gz") as archive:
            for p in config["implementation"]:
                archive.add(ROOT / p, arcname=p, recursive=False)
    if store.read("result") is not None:
        print(json.dumps(store.read("result")["arms"]))
        return
    memory = data["donor_memory"]
    donors = [decode_state(r["state"]) for r in memory["rows"]]
    known = data["known"].copy()
    oracle = DurableScores(args.output, make_oracle(config["task"], ROOT, {}), 40, lambda: None, {})
    attempts = {}

    def draw(root, phase, slot, offset, parent):
        if parent is None:
            return None
        task = {"root": root, "phase": phase, "slot": slot, "draw": offset, "parent": parent}
        key = identity(task)
        name = f"draws/{key}"
        saved = store.read(name)
        if saved is None:
            if perf_counter() - started > config["time_limit_seconds"]:
                raise TimeoutError("planner time bound; reuse finished draws, do not repeat")
            clock = perf_counter()
            rng = np.random.default_rng(
                np.random.SeedSequence([config["seed"], root, phase, slot + 1, offset])
            )
            i = int(rng.choice(len(donors), p=memory["probabilities"]))
            graph = decode_search_state(parent["node"]).graph
            left, right = pendant_cuts(graph), pendant_cuts(donors[i])
            child = None
            if left and right:
                a, b = left[int(rng.integers(len(left)))], right[int(rng.integers(len(right)))]
                result = compile_transplant(graph, donors[i], a, b)
                child = donor_candidate(parent, result, name, i, a, b)
            else:
                result = {"status": "no_pendant_cut"}
            saved = {
                "task": task,
                "candidate": child,
                "program": result,
                "seconds": perf_counter() - clock,
            }
            store.save(name, saved, durable=True)
        attempts[key] = saved
        return saved["candidate"]

    def score_phase(name, rows):
        new = sorted({p["smiles"] for p in rows if p is not None} - set(known))
        _frozen_save(store, name, {"candidates": rows, "new": new})
        if (
            len(set(new) | {q["smiles"] for q in oracle.rows}) > 40
            or perf_counter() - started > 180
        ):
            raise RuntimeError("planning phase exceeds the declared query/time cap")
        oracle.context = {"role": name, "lock_path": str(args.output / f"{name}.json")}
        known.update(zip(new, [r["desirability"] for r in oracle.score_many(new)], strict=True))
        return [None if p is None else {**p, "score": known[p["smiles"]]} for p in rows]

    first = [
        [draw(i, 0, slot, 0, root) for slot in range(4)] for i, root in enumerate(data["roots"])
    ]
    flat = score_phase("phase1_lock", [p for group in first for p in group])
    first = [flat[i * 4 : (i + 1) * 4] for i in range(2)]
    selected = [best_first(root, rows) for root, rows in zip(data["roots"], first, strict=True)]
    plans = [continuation_tasks(i, rows, selected[i]) for i, rows in enumerate(first)]
    _frozen_save(store, "phase2_parents", {"plans": plans, "selected": selected})
    second = {arm: [] for arm in ("immediate", "lookahead")}
    for plan in plans:
        for arm, tasks in plan.items():
            second[arm].extend(
                draw(i, 1, slot, offset, parent) for i, slot, offset, parent in tasks
            )
    ordered = [p for arm in second.values() for p in arm]
    scored = score_phase("phase2_lock", ordered)
    second = {"immediate": scored[:16], "lookahead": scored[16:]}
    arms = {}
    for arm, rows in second.items():
        archive = {p["smiles"]: p for p in data["roots"] + flat + rows if p is not None}
        winner = min(archive.values(), key=lambda p: (-p["score"], p["smiles"]))
        complete = [p for p in rows if p is not None]
        rescues = []
        for child in complete:
            first_parent = next(
                (p for p in flat if p is not None and p["id"] == child["chain"][-2]), None
            )
            if (
                first_parent
                and first_parent["score"] < first_parent["parent_score"]
                and child["score"] > first_parent["parent_score"]
            ):
                rescues.append(child["id"])
        arms[arm] = {
            "best": winner["score"],
            "winner": winner,
            "complete_continuations": len(complete),
            "continuation_slots": len(rows),
            "distinct_archive": len(archive),
            "top10_mean": float(
                np.mean(sorted((p["score"] for p in archive.values()), reverse=True)[:10])
            ),
            "continuation_mean": float(np.mean([p["score"] for p in complete]))
            if complete
            else None,
            "temporary_loss_rescues": rescues,
        }
    passes = arms["lookahead"]["best"] > max(arms["immediate"]["best"], data["roots"][0]["score"])
    result = {
        "schema_version": "persistent_lookahead_result_v1",
        "status": "complete_development",
        "configuration_sha256": sha256_file(args.output / "configuration.json"),
        "prepared_sha256": sha256_file(args.output / "prepared.json"),
        "phase_lock_sha256": {
            k: sha256_file(args.output / f"{k}.json")
            for k in ("phase1_lock", "phase2_parents", "phase2_lock")
        },
        "arms": arms,
        "first": first,
        "second": second,
        "attempts": {
            k: {
                "status": v["program"]["status"],
                "seconds": v["seconds"],
                "primitive_steps": v["program"].get("primitive_steps", 0),
            }
            for k, v in attempts.items()
        },
        "new_calls": len(oracle.rows),
        "oracle_rows": oracle.rows,
        "seconds": perf_counter() - started,
        "proposal_seconds": sum(v["seconds"] for v in attempts.values()),
        "oracle_seconds": sum(q["oracle_seconds"] for q in oracle.rows),
        "decision": "earned_full_controller_integration_test"
        if passes
        else "stop_unchanged_two_option_recipe",
        "completed_at": _stamp(),
    }
    store.save("result", result, durable=True)
    print(
        json.dumps(
            {
                "arms": {a: {k: v for k, v in r.items() if k != "winner"} for a, r in arms.items()},
                **{k: result[k] for k in ("new_calls", "seconds", "decision")},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
