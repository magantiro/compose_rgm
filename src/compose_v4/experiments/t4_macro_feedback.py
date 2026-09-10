"""Bounded whole-option search with round-synchronous, observed docking feedback.

This is finite-pool adaptive search, not an exact Doob sampler or learned Q(o).
Only endpoint oracle eligibility is hard; search parents may be infeasible.
"""

from __future__ import annotations

import json
from functools import lru_cache
from time import perf_counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control.docking_value import DockingValue, identity, validate_archive
from compose_v4.control.graph_geometry import structural_displacement, topology
from compose_v4.control.molecular_search_codec import decode_search_state, encode_search_state
from compose_v4.control.molecular_task_search import MolecularSearchState
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint, calculate_properties
from compose_v4.experiments.t4_macro_beam import (
    BeamConfig,
    canonical_smiles,
    distance,
    exact_archive_graph,
    no_similarity_desirability,
    retain,
)
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal

KIND = "t4_macro_feedback"
CONTRACT_PATH = f"configs/{KIND}.json"


def storage(output, commit):
    def read(name):
        path = output / f"{name}.json"
        return unseal(path) if path.exists() else None

    def save(name, value):
        path = output / f"{name}.json"
        if path.exists():
            if unseal(path) != value:
                raise ValueError(f"immutable feedback artifact changed: {path}")
        else:
            seal(path, value)
            commit()
        return sha256_file(path)

    return read, save


def initial_archive(contract, artifact_root):
    for key in ("archive", "saved_docking", "saved_lock"):
        verify_file(artifact_root / contract[key]["path"], contract[key]["sha256"])
    old = unseal(artifact_root / contract["archive"]["path"])
    lock = unseal(artifact_root / contract["saved_lock"]["path"])
    recent = json.loads((artifact_root / contract["saved_docking"]["path"]).read_text())
    if (
        old["oracle_attempts"] != 51
        or recent["status"] != "complete"
        or recent["new_oracle_attempts"] != 16
        or recent["candidate_lock_sha256"] != contract["saved_lock"]["sha256"]
        or len(recent["docked"]) != 16
        or len(lock["take"]) != 16
    ):
        raise ValueError("feedback requires the complete 51+16 source-call ledger")
    rows = list(old["archive"])
    for index, (candidate, result) in enumerate(zip(lock["take"], recent["docked"], strict=True)):
        if (
            result["index"] != index
            or result["candidate_lock_sha256"] != contract["saved_lock"]["sha256"]
            or any(result.get(k) != v for k, v in candidate.items())
        ):
            raise ValueError("saved docking differs from its locked exact candidate")
        rows.append({**result, "round": 5})
    rows = [{**r, "smiles": canonical_smiles(r["smiles"])} for r in rows]
    if len(rows) != 68 or max(r["round"] for r in rows) != 5:
        raise ValueError("feedback source ledger must contain seed plus all 67 attempts")
    validate_archive(rows)
    for row in rows:
        exact_archive_graph(row)
    return rows


def root_candidate(row, budget):
    node = MolecularSearchState.start(
        exact_archive_graph(row), budget=budget, root_id=identity({"smiles": row["smiles"]})
    )
    encoded = encode_search_state(node)
    return {
        "attempt_id": f"initial/{node.root_id}",
        "smiles": row["smiles"],
        "node": encoded,
        "root_node": encoded,
        "chain": [],
        "topology": topology(node.graph),
    }


def extend_candidate(parent, candidate, prefix):
    """Carry exact slots, lineage, remaining horizon and global ancestry forward."""
    origin, child = decode_search_state(parent["node"]), decode_search_state(candidate["node"])
    if (
        child.stage != "where"
        or not 0 <= child.budget < origin.budget
        or child.root_id != origin.root_id
        or canonical_smiles(candidate["smiles"]) != candidate["smiles"]
    ):
        raise ValueError("completed feedback option changed root, horizon or molecular identity")
    exact_archive_graph({"state": candidate["node"]["graph"], "smiles": candidate["smiles"]})
    root = decode_search_state(parent["root_node"])
    name = f"{prefix}/{candidate['attempt_id']}"
    return {
        **candidate,
        "attempt_id": name,
        "root_node": parent["root_node"],
        "chain": [*parent["chain"], name],
        "cumulative_change": structural_displacement(
            root.graph, child.graph, root.lineage, child.lineage
        ),
        "primitive_depth": root.budget - child.budget,
        "option_depth": len(parent["chain"]) + 1,
        "parent_smiles": parent["smiles"],
    }


def unique_pool(rows):
    # Keep a reproducible representative without creating extra canonical slots.
    unique = {}
    for row in sorted(rows, key=lambda r: (-r["node"]["budget"], r["attempt_id"])):
        unique.setdefault(row["smiles"], row)
    return [unique[key] for key in sorted(unique)]


def signature(row):
    t = row["topology"]
    return (t["cycle_rank"], t["n_ring_systems"], t["n_ring_atoms"], t["n_heavy"] // 4)


def diverse_one(pool, chosen):
    represented = {signature(row) for row in chosen}
    return max(
        pool,
        key=lambda r: (
            signature(r) not in represented,
            min((distance(r["smiles"], c["smiles"]) for c in chosen), default=1.0),
            r["smiles"],
        ),
    )


def oracle_selection(pool, archive, score, rng, limit=4):
    attempted = {r["smiles"] for r in archive}
    remaining = [
        {**r, **score(r["smiles"])} for r in unique_pool(pool) if r["smiles"] not in attempted
    ]
    remaining = [r for r in remaining if r["oracle_eligible"]]
    chosen = []
    for role in ("predicted_quality", "diversity", "diversity", "uniform")[:limit]:
        if not remaining:
            break
        if role == "predicted_quality":
            row = min(remaining, key=lambda r: (r["predicted_docking"], r["smiles"]))
        elif role == "diversity":
            row = diverse_one(remaining, chosen)
        else:
            row = remaining[int(rng.integers(len(remaining)))]
        remaining.remove(row)
        chosen.append({**row, "oracle_selection_role": role})
    return chosen


def parent_selection(pool, archive, seed, score, rng):
    remaining = [r for r in unique_pool(pool) if r["node"]["budget"] > 0]
    chosen, audit = [], []

    def take(row, role):
        chosen.append(row)
        audit.append({"role": role, "attempt_id": row["attempt_id"], "smiles": row["smiles"]})
        remaining[:] = [r for r in remaining if r["smiles"] != row["smiles"]]

    take(seed, "original_seed")
    labels = {r["smiles"]: r["ds"] for r in archive if r["ds"] is not None}
    observed = [r for r in remaining if r["smiles"] in labels]
    if observed:
        take(min(observed, key=lambda r: (labels[r["smiles"]], r["smiles"])), "observed_elite")
    for role in ("guided", "guided", "diversity", "diversity", "diversity", "uniform"):
        if not remaining:
            break
        if role == "guided":
            selected, decision = retain(remaining, BeamConfig(width=1), rng, score)
            take(selected[0], role)
            audit[-1]["decision"] = decision
        elif role == "diversity":
            take(diverse_one(remaining, chosen), role)
        else:
            take(remaining[int(rng.integers(len(remaining)))], role)
    # Small pool: keep declared worker census; clones do not create extra oracle slots.
    while len(chosen) < 8:
        chosen.append(seed)
        audit.append({"role": "seed_fill", "smiles": seed["smiles"]})
    return chosen, audit


def scorer(model, seed_smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(seed_smiles))

    @lru_cache(maxsize=4096)
    def score(smiles):
        props = calculate_properties(
            Chem.MolFromSmiles(smiles),
            seed_fp=seed_fp,
            generator=generator,
            sa_scorer=sascorer.calculateScore,
            delta=0.4,
            qed_min=0.6,
            sa_max=4.0,
        )
        prediction = float(model.predict([smiles])[0])
        return {
            **props,
            "oracle_eligible": acceptable_endpoint({"smiles": smiles, **props}),
            "predicted_docking": prediction,
            "desirability": no_similarity_desirability(
                props, prediction, model.payload, {"qed_min": 0.6, "sa_max": 4.0, "scale": 0.1}
            ),
        }

    return score


def run_episode(
    contract, task, archive, output, propose, execute, *, commit, progress, score_factory=scorer
):
    """Lock proposals, finish docking, then update value/parents. Resume complete units."""
    read, save = storage(output, commit)
    search = contract["search"]
    seed = root_candidate(archive[0], search["primitive_budget"])
    observed = sorted(
        (r for r in archive if r["ds"] is not None and acceptable_endpoint(r)),
        key=lambda r: (r["ds"], r["smiles"]),
    )[:4]
    warm = [root_candidate(r, search["primitive_budget"]) for r in observed]
    if len(warm) != 4 or search["lineages"] != 8:
        raise ValueError("feedback requires original seed and four distinct observed roots")
    state = {"archive": archive, "pool": [seed, *warm], "parents": [seed] * 4 + warm}
    save("initial", state)
    initial_calls, summaries = len(archive) - 1, []
    first_round = max(r["round"] for r in archive) + 1

    def fit(rows, name, before_round):
        digest = save(f"{name}/archive", {"archive": rows})
        old = read(f"{name}/value")
        model = (
            DockingValue.from_payload(old)
            if old is not None
            else DockingValue.fit(rows, before_round=before_round, source_sha256=digest)
        )
        if (
            model.payload["source_sha256"] != digest
            or model.payload["before_round"] != before_round
        ):
            raise ValueError("round value is not fitted on its exact prior-only archive")
        save(f"{name}/value", model.payload)
        return model, score_factory(model, seed["smiles"])

    for number in range(search["rounds"]):
        folder, label_round = f"rounds/{number:02}", first_round + number
        done = read(f"{folder}/after")
        if done is not None:
            state = done["state"]
            summaries.append(done["summary"])
            continue
        started = perf_counter()
        calls = len(state["archive"]) - 1 - initial_calls
        progress.update(
            phase="proposals",
            round=number + 1,
            rounds=search["rounds"],
            new_oracle_attempts=calls,
            best_score=min(r["ds"] for r in state["archive"] if r["ds"] is not None),
        )
        before_hash = save(f"{folder}/before", state)
        model, score = fit(state["archive"], f"{folder}/prior", label_round)
        generated = read(f"{folder}/generated")
        if generated is None:
            generated = propose(number, state["parents"], before_hash)
            save(f"{folder}/generated", generated)
        pool = unique_pool([*state["pool"], *generated["candidates"]])
        round_id = identity({"run_id": task["cell"], "round": number})
        dock_folder = f"docking/{round_id}"
        lock = read(f"{dock_folder}/candidate_lock")
        if lock is None:
            rng = np.random.default_rng(np.random.SeedSequence([search["seed"], number, 701]))
            take = oracle_selection(
                pool,
                state["archive"],
                score,
                rng,
                min(
                    contract["compute"]["per_round"],
                    contract["compute"]["oracle_call_limit"] - calls,
                ),
            )
            lock = {
                "schema_version": "t4_macro_feedback_candidate_lock_v1",
                "task": task,
                "required_rdkit": contract["required_rdkit"],
                "take": take,
                "round": number,
                "before_sha256": before_hash,
                "value_snapshot_sha256": model.payload["snapshot_sha256"],
                "locked_at_utc": _stamp(),
            }
        if (
            lock["before_sha256"] != before_hash
            or lock["value_snapshot_sha256"] != model.payload["snapshot_sha256"]
        ):
            raise ValueError("round lock changed prior states or value snapshot")
        digest = save(f"{dock_folder}/candidate_lock", lock)
        barrier = output / dock_folder / "docking_started.json"
        if barrier.exists():
            if json.loads(barrier.read_text())["candidate_lock_sha256"] != digest:
                raise ValueError("round oracle barrier does not match lock")
        else:
            publish_json(
                barrier,
                {
                    "candidate_lock_sha256": digest,
                    "attempts": len(lock["take"]),
                    "started_at_utc": _stamp(),
                },
            )
            commit()
        docked = []
        for index, candidate in enumerate(lock["take"]):
            progress.update(phase="docking", batch_size=len(lock["take"]), batch_completed=index)
            result = execute(round_id, index, digest)
            if (
                result["index"] != index
                or result["smiles"] != candidate["smiles"]
                or result["candidate_lock_sha256"] != digest
            ):
                raise ValueError("oracle result does not match locked candidate")
            docked.append(
                {**candidate, **result, "round": label_round, "state": candidate["node"]["graph"]}
            )
            values = [r["ds"] for r in [*state["archive"], *docked] if r["ds"] is not None]
            progress.update(
                batch_completed=index + 1,
                new_oracle_attempts=calls + len(docked),
                best_score=min(values),
                latest_score=result["ds"],
            )
            publish_json(output / "progress.json", {**progress, "updated_at_utc": _stamp()})
            commit()
            print(
                f"round {number + 1}: docked {index + 1}/{len(lock['take'])}; score={result['ds']} best={min(values)}",
                flush=True,
            )
        rows = [*state["archive"], *docked]
        validate_archive(rows)
        if len(rows) - 1 - initial_calls > contract["compute"]["oracle_call_limit"]:
            raise ValueError("feedback exceeded authorized oracle budget")
        after_model, next_score = fit(rows, f"{folder}/posterior", label_round + 1)
        parents, decision = parent_selection(
            pool,
            rows,
            seed,
            next_score,
            np.random.default_rng(np.random.SeedSequence([search["seed"], number, 702])),
        )
        state = {"archive": rows, "pool": pool, "parents": parents}
        summary = {
            "round": number + 1,
            "attempts": generated["attempts"],
            "completed": len(generated["candidates"]),
            "options": generated["options"],
            "unique_pool": len(pool),
            "new_unique_eligible_pool": sum(
                score(r["smiles"])["oracle_eligible"]
                and r["smiles"] not in {a["smiles"] for a in rows}
                for r in pool
            ),
            "new_oracle_attempts": len(rows) - 1 - initial_calls,
            "best_score": min(r["ds"] for r in rows if r["ds"] is not None),
            "docked": docked,
            "parent_selection": decision,
            "max_option_depth": max((len(r["chain"]) for r in pool), default=0),
            "proposal_seconds": generated["proposal_seconds"],
            "round_seconds_this_invocation": perf_counter() - started,
            "prior_value_sha256": model.payload["snapshot_sha256"],
            "next_value_sha256": after_model.payload["snapshot_sha256"],
        }
        save(f"{folder}/after", {"state": state, "summary": summary})
        summaries.append(summary)
        progress.update(
            phase="round_complete",
            **{
                k: summary[k]
                for k in (
                    "round",
                    "new_oracle_attempts",
                    "best_score",
                    "completed",
                    "max_option_depth",
                )
            },
        )
        publish_json(output / "progress.json", {**progress, "updated_at_utc": _stamp()})
        commit()
        print(
            f"round {number + 1} complete: {summary['completed']} products; {summary['new_oracle_attempts']} new calls; best={summary['best_score']}; max chain={summary['max_option_depth']}",
            flush=True,
        )
    successful = [r for r in state["archive"] if r["ds"] is not None and acceptable_endpoint(r)]
    return {
        "schema_version": "t4_macro_feedback_result_v1",
        "status": "complete",
        "rounds": summaries,
        "initial_oracle_attempts": initial_calls,
        "new_oracle_attempts": len(state["archive"]) - 1 - initial_calls,
        "best": min(successful, key=lambda r: (r["ds"], r["smiles"])),
        "winner_used": False,
        "automatic_continuation": False,
        "interpretation": "inspected-cell development with adaptive finite-pool macro search, not an exact Doob sampler or held-out benchmark",
        "software": {"numpy": np.__version__, "rdkit": rdBase.rdkitVersion},
    }
