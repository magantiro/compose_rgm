"""Short completed-macro lookahead; frozen chemistry, no winner input or Q(o) fit."""

from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from time import perf_counter

import numpy as np

from compose_v4.control.continuation import _tilted
from compose_v4.control.docking_value import DockingValue, identity, validate_archive
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.experiments.continuation_profile import verify_file
from compose_v4.experiments.t4_macro_beam import BeamConfig, exact_archive_graph, run_search
from compose_v4.experiments.t4_macro_feedback import (
    diverse_one,
    extend_candidate,
    run_episode,
    scorer,
    unique_pool,
)
from compose_v4.experiments.t4_matched_pilot import unseal

KIND = "t4_macro_lookahead"
CONTRACT_PATH = f"configs/{KIND}.json"


def contract_at(repo_root):
    contract = json.loads((repo_root / CONTRACT_PATH).read_text())
    if (
        identity({k: v for k, v in contract.items() if k != "contract_sha256"})
        != contract["contract_sha256"]
    ):
        raise ValueError("lookahead contract hash mismatch")
    if (
        contract["search"]
        != {
            "rounds": 4,
            "lineages": 8,
            "branches": 2,
            "replicas": 2,
            "primitive_budget": 110,
            "seed": 20260911,
        }
        or contract["compute"]["oracle_call_limit"] != 40
        or contract["compute"]["per_round"] != 10
    ):
        raise ValueError("lookahead exceeds the declared 4x16x6 / 40-call recipe")
    return contract


def source_state(contract, artifact_root):
    source = contract["previous_episode"]
    path = artifact_root / source["path"]
    verify_file(path, source["sha256"])
    saved = unseal(path)
    state = saved["state"]
    archive = state["archive"]
    validate_archive(archive)
    if len(archive) != 95 or saved["summary"]["new_oracle_attempts"] != 27:
        raise ValueError("lookahead requires the complete 94-call source ledger")
    for row in archive:
        exact_archive_graph(row)
    for row in state["pool"]:
        node = decode_search_state(row["node"])
        if node.stage != "where" or row["node"]["budget"] > row["root_node"]["budget"]:
            raise ValueError("source pool has an unfinished or reset trajectory")
        exact_archive_graph({"state": row["node"]["graph"], "smiles": row["smiles"]})
    return state


def initial_archive(contract, artifact_root):
    return source_state(contract, artifact_root)["archive"]


def planning_scorer(model, seed_smiles):
    ordinary = scorer(model, seed_smiles)
    labels = {r["smiles"]: r["ds"] for r in model.payload["training_rows"]}

    @lru_cache(maxsize=4096)
    def score(smiles):
        row = ordinary(smiles)
        value = labels.get(smiles, row["predicted_docking"])
        terminal = math.exp(-max(0.0, value - model.payload["best"]) / model.payload["scale"])
        return {
            **row,
            "planning_docking": value,
            "planning_label": "observed_prior" if smiles in labels else "surrogate",
            "terminal_value": terminal if row["oracle_eligible"] else 0.0,
            "repair_value": math.exp(-min(700.0, row["v"] / 0.1)),
        }

    return score


def select_branch(branches, score, rng):
    """Back up sampled completed returns before allocating additional execution."""
    live = [unique_pool(rows) for rows in branches if rows]
    if not live:
        return None, {"status": "no_complete_branch"}
    values = np.asarray([max(score(c["smiles"])["terminal_value"] for c in rows) for rows in live])
    field = "terminal_value" if values.any() else "repair_value"
    if field == "repair_value":
        values = np.asarray([max(score(c["smiles"])[field] for c in rows) for rows in live])
    p = np.full(len(live), 1 / len(live))
    q, eta, kl = _tilted(p, values, kappa=1.0, exploration=0.1)
    index = int(rng.choice(len(live), p=q))
    exploratory = bool(rng.random() < 0.1)
    node = (
        live[index][int(rng.integers(len(live[index])))]
        if exploratory
        else max(
            live[index], key=lambda c: (score(c["smiles"])[field], c["node"]["budget"], c["smiles"])
        )
    )
    return node, {
        "status": "selected",
        "branches": [[c["attempt_id"] for c in rows] for rows in live],
        "score_field": field,
        "values": values.tolist(),
        "reference": p.tolist(),
        "probabilities": q.tolist(),
        "eta": eta,
        "kl": kl,
        "selected": index,
        "exploratory_state": exploratory,
        "selected_attempt": node["attempt_id"],
    }


def run_local_search(
    parent, hierarchy, *, seed, save, read, meter, progress, prefix, task, episode, contract
):
    """Two first options, one unpruned continuation each, two guided follow-ups."""
    number = task["round"]
    value_path = episode / f"rounds/{number:02}/prior/value.json"
    verify_file(value_path, task["prior_value_sha256"])
    model = DockingValue.from_payload(unseal(value_path))
    archive_path = episode / f"rounds/{number:02}/prior/archive.json"
    verify_file(archive_path, model.payload["source_sha256"])
    archive = unseal(archive_path)["archive"]
    before = unseal(episode / f"rounds/{number:02}/before.json")
    if (
        archive != before["archive"]
        or model.payload["before_round"] != max(r["round"] for r in archive) + 1
    ):
        raise ValueError("worker task guide is not the locked strictly prior archive")
    score = planning_scorer(model, archive[0]["smiles"])
    start = perf_counter()
    all_candidates, all_attempts = [], []

    def execute(origin, name, count, offset):
        def no_primitive_score(_smiles):
            raise AssertionError("task value applies at completed options, never primitive edits")

        result = run_search(
            decode_search_state(origin["node"]),
            hierarchy,
            config=BeamConfig(
                arm="post_hoc",
                depth=1,
                width=3,
                branches=count,
                primitive_budget=110,
                seed=int(np.random.SeedSequence([seed, offset]).generate_state(1)[0]),
            ),
            score=no_primitive_score,
            save=lambda k, v: save(f"{name}/{k}", v),
            read=lambda k: read(f"{name}/{k}"),
            meter=meter,
            progress=progress,
        )
        rows = [
            extend_candidate(origin, a["candidate"], f"{prefix}/{name}")
            for a in result["attempts"]
            if a["status"] == "complete"
        ]
        all_candidates.extend(rows)
        all_attempts.extend(result["attempts"])
        return rows

    first = execute(parent, "first", 2, 0)
    branches = [
        [row, *execute(row, f"lookahead/{i:02}", 1, 100 + i)] for i, row in enumerate(first)
    ]
    selected, decision = select_branch(
        branches, score, np.random.default_rng(np.random.SeedSequence([seed, 701]))
    )
    decision["value_snapshot_sha256"] = model.payload["snapshot_sha256"]
    old = read("task_branch_decision")
    if old is not None and old != decision:
        raise ValueError("resumed lookahead changed the pre-execution branch decision")
    if old is None:
        save("task_branch_decision", decision)
    if selected is not None:
        execute(selected, "guided_followup", 2, 1000)
    return {
        "candidates": all_candidates,
        "attempts": len(all_attempts),
        "options": dict(Counter(a["bundle"]["option"] for a in all_attempts if a["bundle"])),
        "outcomes": dict(Counter(a["status"] for a in all_attempts)),
        "proposal_seconds": perf_counter() - start,
        "branch_decision": decision,
        "value_snapshot_sha256": model.payload["snapshot_sha256"],
    }


def parent_selection(pool, archive, seed, score, rng, *, expansion_counts=None):
    counts = expansion_counts or {}
    remaining = [r for r in unique_pool(pool) if r["node"]["budget"] > 0]
    chosen, audit = [], []
    labels = {r["smiles"]: r["ds"] for r in archive if r["ds"] is not None}

    def take(row, role):
        chosen.append(row)
        audit.append(
            {
                "role": role,
                "attempt_id": row["attempt_id"],
                "smiles": row["smiles"],
                "previous_expansions": counts.get(row["smiles"], 0),
                **score(row["smiles"]),
            }
        )
        remaining[:] = [r for r in remaining if r["smiles"] != row["smiles"]]

    take(seed, "original_seed")
    for role in (
        "observed_elite",
        "quality",
        "quality",
        "diversity",
        "diversity",
        "repair",
        "uniform",
    ):
        if not remaining:
            take(seed, "seed_fill")
            continue
        eligible = [r for r in remaining if score(r["smiles"])["oracle_eligible"]]
        if role == "observed_elite" and (
            observed := [r for r in eligible if r["smiles"] in labels]
        ):
            row = min(observed, key=lambda r: (labels[r["smiles"]], r["smiles"]))
        elif role == "quality" and eligible:
            row = min(eligible, key=lambda r: (score(r["smiles"])["planning_docking"], r["smiles"]))
        elif role == "diversity" and eligible:
            least = min(counts.get(r["smiles"], 0) for r in eligible)
            row = diverse_one([r for r in eligible if counts.get(r["smiles"], 0) == least], chosen)
        elif role == "repair" and (
            repair := [r for r in remaining if not score(r["smiles"])["oracle_eligible"]]
        ):
            row = min(
                repair,
                key=lambda r: (counts.get(r["smiles"], 0), score(r["smiles"])["v"], r["smiles"]),
            )
        else:
            row = remaining[int(rng.integers(len(remaining)))]
            role += "_uniform_fallback" if role != "uniform" else ""
        take(row, role)
    return chosen, audit


def oracle_selection(pool, archive, score, rng, limit=10):
    old = {r["smiles"] for r in archive}
    remaining = [{**r, **score(r["smiles"])} for r in unique_pool(pool) if r["smiles"] not in old]
    remaining = [r for r in remaining if r["oracle_eligible"]]
    chosen = []
    for index in range(limit):
        if not remaining:
            break
        role = (
            "uniform" if index % 5 == 4 else "predicted_quality" if index % 2 == 0 else "diversity"
        )
        row = (
            remaining[int(rng.integers(len(remaining)))]
            if role == "uniform"
            else min(remaining, key=lambda r: (r["predicted_docking"], r["smiles"]))
            if role == "predicted_quality"
            else diverse_one(remaining, chosen)
        )
        remaining.remove(row)
        chosen.append({**row, "oracle_selection_role": role})
    return chosen


def episode_runner(contract, task, archive, output, propose, execute, *, commit, progress):
    state = source_state(contract, output.parents[1])
    seed = next(r for r in state["pool"] if r["smiles"] == archive[0]["smiles"])
    model = DockingValue.fit(
        archive,
        before_round=max(r["round"] for r in archive) + 1,
        source_sha256=contract["previous_episode"]["sha256"],
    )
    state["parents"], initial_selection = parent_selection(
        state["pool"],
        archive,
        seed,
        planning_scorer(model, seed["smiles"]),
        np.random.default_rng(contract["search"]["seed"]),
    )

    def select_parents(pool, rows, source, score, rng):
        counts = Counter()
        for number in range(contract["search"]["rounds"]):
            path = output / f"rounds/{number:02}/before.json"
            if path.exists():
                counts.update(r["smiles"] for r in unseal(path)["parents"])
        return parent_selection(pool, rows, source, score, rng, expansion_counts=counts)

    result = run_episode(
        contract,
        task,
        archive,
        output,
        propose,
        execute,
        commit=commit,
        progress=progress,
        score_factory=planning_scorer,
        initial_state=state,
        select_parents=select_parents,
        select_oracle=oracle_selection,
    )
    return {
        **result,
        "schema_version": "t4_macro_lookahead_result_v1",
        "initial_parent_selection": initial_selection,
        "interpretation": "warm-start inspected-cell macro lookahead development, not an exact Doob law or matched causal comparison",
    }
