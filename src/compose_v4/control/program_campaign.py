"""Reusable bounded score/update/propose cycles with a charged query ledger.

No bundled oracle or implicit launch authority. The caller supplies the identified
oracle and frozen run inputs. Local fixture runs use explicitly synthetic labels.
"""

import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem

from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer
from compose_v4.control.docking_value import identity
from compose_v4.control.parent_edit_search import prepare_query_batch
from compose_v4.control.program_task import archive_top_k, pmo_top_ten_auc
from compose_v4.control.program_transfer import initial_program_batch
from compose_v4.experiments.continuation_profile import publish_json
from compose_v4.rewrite.trace_shard import decode_state


class ProgramQueryLedger:
    """Reserve before evaluation; cached repeats are free, failures stay charged."""

    def __init__(self, output, task, evaluate, *, budget, flush=None):
        if type(budget) is not int or budget < 1:
            raise ValueError("positive explicitly authorized query budget required")
        self.output, self.task, self.evaluate, self.budget = Path(output), task, evaluate, budget
        self.rows, self.cache = [], {}
        self.flush = flush or (lambda: None)
        self.blocked = False
        manifest = {
            "task": asdict(task),
            "budget": budget,
            "schema_version": "program_query_ledger_v1",
        }
        path = self.output / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("query ledger task or budget changed during resume")
        publish_json(path, manifest)
        for started in sorted(self.output.glob("query_*/started.json")):
            finished = started.with_name("result.json")
            if not finished.exists():
                raise RuntimeError(
                    f"ambiguous charged oracle attempt; no automatic retry: {started}"
                )
            row = json.loads(finished.read_text())
            if identity({k: v for k, v in row.items() if k != "receipt_id"}) != row.get(
                "receipt_id"
            ):
                raise ValueError(f"corrupt query receipt: {finished}")
            if row["status"] != "complete":
                raise RuntimeError(
                    f"failed query remains charged; explicit recovery required: {finished}"
                )
            if row["oracle_protocol"] != task.oracle_protocol or row["index"] != len(self.rows):
                raise ValueError("query ledger order or protocol mismatch")
            self.rows.append(row)
            self.cache[row["endpoint"]] = row
        if len(self.rows) > budget:
            raise ValueError("restored query ledger already exceeds its budget")

    @property
    def remaining(self):
        return self.budget - len(self.rows)

    def query(self, endpoint, *, lock_id, role):
        if self.blocked:
            raise RuntimeError("failed oracle attempt blocks further campaign queries")
        molecule = Chem.MolFromSmiles(endpoint)
        if molecule is None:
            raise ValueError("query endpoint is not a valid molecule")
        endpoint = Chem.MolToSmiles(molecule)
        if endpoint in self.cache:
            return self.cache[endpoint]
        if self.remaining <= 0 or not lock_id:
            raise ValueError("query lacks budget or a frozen candidate lock")
        folder = self.output / f"query_{len(self.rows):06d}"
        if folder.exists():
            raise RuntimeError("unresolved oracle reservation; no automatic retry")
        started = {
            "index": len(self.rows),
            "endpoint": endpoint,
            "role": role,
            "lock_id": lock_id,
            "oracle_protocol": self.task.oracle_protocol,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        publish_json(folder / "started.json", started)
        # Append the reservation first: exceptions do not refund the query.
        self.rows.append({**started, "status": "started"})
        self.flush()  # Remote reservation must survive worker loss before evaluation.
        begin = perf_counter()
        try:
            score = float(self.evaluate(endpoint))
            utility = self.task.utility(score)
        except Exception as error:
            self.blocked = True
            body = {**started, "status": "failed", "error": repr(error)}
            publish_json(folder / "result.json", {**body, "receipt_id": identity(body)})
            self.flush()
            raise
        body = {
            **started,
            "status": "complete",
            "score": score,
            "utility": utility,
            "seconds": perf_counter() - begin,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        row = {**body, "receipt_id": identity(body)}
        publish_json(folder / "result.json", row)
        self.rows[-1] = row
        self.cache[endpoint] = row
        self.flush()
        return row


def run_program_campaign(
    *,
    output,
    task,
    config,
    initialization,
    library,
    ledger,
    rounds,
    queries_per_round,
    hierarchy=None,
    fit_model=None,
    fit_model_id=None,
    stagnation_rounds=3,
    warm_start=None,
    bootstrap_rounds=1,
    progress=None,
    max_seconds=None,
    initialization_mode="legacy_bootstrap",
    initial_parent_fraction=0.2,
):
    """Initial states are charged, then complete programs bootstrap the archive.

    Completed-round snapshots resume directly. A crash during a query leaves its
    durable receipt; unresolved attempts block, never silently retry. Pending-round
    recovery is deliberately manual in this first runner, not a changed RNG stream.
    """
    if rounds < 1 or queries_per_round < 1 or ledger.task != task:
        raise ValueError("invalid campaign rounds, query count or task ledger")
    if type(bootstrap_rounds) is not int or bootstrap_rounds < 1:
        raise ValueError("positive bootstrap round count required")
    if initialization_mode not in ("legacy_bootstrap", "all_scored_pool"):
        raise ValueError("unknown initialization parent mode")
    if not 0 < initial_parent_fraction <= 1:
        raise ValueError("initial-parent exploration fraction must be in (0, 1]")
    if initialization_mode == "all_scored_pool" and (task.kind != "pmo" or warm_start is not None):
        raise ValueError("all-scored initialization is an explicit cold-start PMO recipe")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("positive campaign time limit required")
    progress = progress or (lambda row: None)
    began = perf_counter()
    if fit_model is not None and (not isinstance(fit_model_id, str) or len(fit_model_id) != 64):
        raise ValueError("learning cycles require an immutable update-rule identity")
    if stagnation_rounds is not None and (
        type(stagnation_rounds) is not int or stagnation_rounds < 2
    ):
        raise ValueError("stagnation window must be at least two completed rounds or None")
    if (
        identity({k: v for k, v in initialization.items() if k != "lock_sha256"})
        != initialization["lock_sha256"]
    ):
        raise ValueError("initialization lock changed")
    output = Path(output)
    recipe = {
        "task": asdict(task),
        "config": asdict(config),
        "initialization": initialization["lock_sha256"],
        "library": [
            {"program_id": p.program.program_id, "source_groups": p.source_groups} for p in library
        ],
        "rounds": rounds,
        "queries_per_round": queries_per_round,
        "learning": fit_model is not None,
        "selection_policy": "fit_only_for_excess_pool_v1",
        "fit_model_id": fit_model_id,
        "stagnation_rounds": stagnation_rounds,
        "warm_start": None if warm_start is None else warm_start["snapshot_id"],
        "bootstrap_rounds": bootstrap_rounds,
        "max_seconds": max_seconds,
        "initialization_mode": initialization_mode,
        "initial_parent_fraction": initial_parent_fraction,
    }
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(recipe)):
        raise ValueError("campaign recipe changed during resume")
    publish_json(manifest, recipe)
    publish_json(output / "initialization.json", initialization)
    for pending in output.glob("round_*/pending.json"):
        if not pending.with_name("complete.json").exists():
            raise RuntimeError(f"pending round needs explicit receipt-based recovery: {pending}")
    starts = initialization["candidates"]
    if not starts and warm_start is None:
        raise ValueError("campaign needs charged initial states or a declared warm archive")
    if sum(r["endpoint"] not in ledger.cache for r in starts) > ledger.remaining:
        raise ValueError("initialization exceeds the declared remaining query budget")
    source_group = initialization["lock_sha256"]
    for row in starts:
        ledger.query(row["endpoint"], lock_id=initialization["lock_sha256"], role="initialization")
    search = ProgramOptimizer(
        config, source_group=source_group, oracle_protocol=task.oracle_protocol, hierarchy=hierarchy
    )
    if warm_start is not None:
        search = ProgramOptimizer.restore(warm_start, hierarchy=hierarchy)
        if search.config != config or search.oracle_protocol != task.oracle_protocol:
            raise ValueError("warm archive configuration or oracle domain changed")
        if search.pending is not None or not search.entries:
            raise ValueError("warm archive must contain resolved measured programs")
    warm_observations = [
        (r["endpoint"], task.utility(r["score"])) for r in search.observations.values()
    ]
    history = []
    for round_index in range(rounds):
        folder = output / f"round_{round_index:04d}"
        complete = folder / "complete.json"
        if complete.exists():
            saved = json.loads(complete.read_text())
            search = ProgramOptimizer.restore(saved["snapshot"], hierarchy=hierarchy)
            history.append(saved["summary"])
            continue
        if not ledger.remaining:
            break
        if max_seconds is not None and perf_counter() - began >= max_seconds:
            break
        if stagnation_rounds is not None and len(history) >= stagnation_rounds:
            recent = [r["top_k_utility"] for r in history[-stagnation_rounds:]]
            if max(recent[1:]) <= recent[0]:
                break
        count = min(queries_per_round, ledger.remaining)
        progress({"phase": "proposing", "round": round_index, "oracle_calls": len(ledger.rows)})
        initial_choice = None
        bootstrap = not search.entries or (warm_start is None and round_index < bootstrap_rounds)
        if initialization_mode == "all_scored_pool":
            # A separate round-addressed stream preserves the proposal RNG. All
            # charged exact initialization states stay available after bootstrap.
            parent_rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, round_index, 31])
            )
            bootstrap = bootstrap or parent_rng.random() < initial_parent_fraction
            if bootstrap:
                initial_choice = int(parent_rng.integers(len(starts)))
        if bootstrap:
            at = round_index % len(starts) if initial_choice is None else initial_choice
            source = decode_state(starts[at]["state"])
            broad = None
            if hierarchy is not None:
                from compose_v4.control.molecular_task_search import MolecularSearchState

                def broad(rng, source=source):
                    return hierarchy.complete_reference_program(
                        MolecularSearchState.start(
                            source, budget=config.max_primitives, root_id=source_group
                        ),
                        rng,
                        max_options=config.max_blocks,
                    )

            batch = initial_program_batch(
                source,
                library,
                replace(config, seed=config.seed + round_index, candidates_per_batch=count),
                source_group=source_group,
                oracle_protocol=task.oracle_protocol,
                eligibility=task.endpoint_evaluator(),
                broad_sampler=broad,
            )
            # Metadata outside the candidate lock does not change the completed
            # transformations or invent a zero-edit construction.
            initialization_parent = {
                "endpoint": starts[at]["endpoint"],
                "index": at,
                "available_scored_parents": len(starts),
                "selection": "uniform_all_scored"
                if initial_choice is not None
                else "legacy_round_index",
                "observation_receipt": ledger.cache[starts[at]["endpoint"]]["receipt_id"],
            }
            bootstrap = True
        else:

            def lazy_model(search=search, folder=folder):
                model = fit_model(search, task)
                if model is not None:
                    publish_json(folder / "model.json", model.payload)
                return model

            batch = prepare_query_batch(
                search,
                task,
                count=count,
                seed=config.seed + round_index,
                model_factory=lazy_model if fit_model is not None else None,
                diagnostic=True,
            )
            bootstrap = False
        publish_json(folder / "pending.json", {"batch": batch, "recipe_sha256": identity(recipe)})
        ledger.flush()  # Publish the candidate lock before any charged query.
        outcomes = []
        for candidate in batch["candidates"]:
            row = ledger.query(candidate["endpoint"], lock_id=batch["batch_id"], role="candidate")
            if bootstrap:
                search.add_measured_program(
                    candidate, receipt_id=row["receipt_id"], score=row["score"]
                )
            else:
                outcomes.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "receipt_id": row["receipt_id"],
                        "score": row["score"],
                        "oracle_protocol": task.oracle_protocol,
                    }
                )
        if not bootstrap:
            search.observe_batch(batch["batch_id"], outcomes)
        summary = {
            "round": round_index,
            "queried_candidates": len(batch["candidates"]),
            "oracle_calls_including_initialization": len(ledger.rows),
            "top_k_utility": archive_top_k(
                warm_observations + [(r["endpoint"], r["utility"]) for r in ledger.rows],
                k=task.top_k,
            ),
            "top_k": task.top_k,
            "model_updated": batch.get("selection", {}).get("model_sha256") is not None,
            "proposal_seconds": batch.get("proposal_seconds", 0.0),
            "proposal_attempts": len(batch["attempts"]),
            "pool_size": len(batch.get("proposal_pool", batch)["candidates"]),
            "seconds": perf_counter() - began,
            "best_new_utility": max((r["utility"] for r in ledger.rows), default=None),
            "initialization_parent": initialization_parent if bootstrap else None,
            "work_cache": batch.get("work_cache"),
        }
        if task.kind == "pmo":
            summary["pmo_top10_auc_so_far"] = pmo_top_ten_auc(
                [r["score"] for r in ledger.rows], budget=ledger.budget
            )
        publish_json(
            complete,
            {"snapshot": search.snapshot(), "summary": summary, "recipe_sha256": identity(recipe)},
        )
        history.append(summary)
        progress({"phase": "scored", **summary})
        ledger.flush()
        if not batch["candidates"]:
            break  # Explicit proposal-yield stop, no unbounded retry-until-novel.
    return {
        "history": history,
        "snapshot": search.snapshot(),
        "oracle_calls": len(ledger.rows),
        "remaining_queries": ledger.remaining,
        "benchmark_claim": False,
        "pmo_top10_auc_with_flat_tail": pmo_top_ten_auc(
            [r["score"] for r in ledger.rows], budget=ledger.budget, finish=True
        )
        if task.kind == "pmo"
        else None,
    }
