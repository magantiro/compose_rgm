"""Thin, unlaunched adapters for the scoped Dynamic-v2.2 quick pilot."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v22 import (
    DynamicV22PolicyConfig,
    DynamicV22ProgramOptimizer,
    advance_t4_feasibility_frontier,
    initial_dynamic_program_batch_v22,
    initial_frontier_state,
)
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask, pmo_top_ten_auc
from compose_v4.experiments.continuation_profile import publish_json
from compose_v4.experiments.t4_frozen_program_benchmark import strict_endpoint_scorer
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "dynamic_v22_quick_pilot_contract_v1"
QUERY_BUDGET = 256
T4_CELLS = ("fa7_0", "braf_1", "jak2_1")
PMO_TASKS = ("gsk3b", "isomers_c7h8n2o2")


@dataclass(frozen=True)
class DynamicV22PilotAdapter:
    kind: str
    oracle_protocol: str
    source_group: str

    def __post_init__(self):
        if self.kind not in ("t4", "pmo") or not self.oracle_protocol or not self.source_group:
            raise ValueError("v2.2 pilot adapter needs kind, protocol and source group")

    @property
    def policy(self):
        return (
            DynamicV22PolicyConfig.for_t4()
            if self.kind == "t4"
            else DynamicV22PolicyConfig.for_pmo()
        )

    def optimizer(self, search_config, *, hierarchy=None):
        return DynamicV22ProgramOptimizer(
            search_config,
            source_group=self.source_group,
            oracle_protocol=self.oracle_protocol,
            hierarchy=hierarchy,
            policy_config=self.policy,
        )

    def initial_frontier(self, *, seed):
        if self.kind != "t4":
            raise ValueError("PMO does not inherit the T4 feasibility frontier")
        return initial_frontier_state(seed=seed)

    def advance_frontier(
        self,
        source,
        *,
        state,
        eligibility,
        delta,
        attempts_per_channel,
    ):
        if self.kind != "t4":
            raise ValueError("PMO does not inherit T4 eligibility-slack acquisition")
        return advance_t4_feasibility_frontier(
            source,
            config=self.policy,
            state=state,
            source_group=self.source_group,
            oracle_protocol=self.oracle_protocol,
            eligibility=eligibility,
            delta=delta,
            attempts_per_channel=attempts_per_channel,
        )


def quick_pilot_contract() -> dict:
    payload = {
        "schema_version": SCHEMA,
        "status": "implementation_ready_not_launched",
        "scientific_problem": (
            "allocate charged evaluations to productive parent-edit combinations while "
            "retaining a persistent proposal-only feasibility frontier"
        ),
        "primary_output": "complete exact-executed molecular endpoint",
        "claim_status": "prospective development hypothesis only",
        "controller_core": {
            "shallow_spine": 0.25,
            "structural_diversity": 0.25,
            "predicted_archive_gain": 0.50,
            "pre_model": "0.50 preserved shallow pool order / 0.50 score-blind diversity",
            "model": "existing ParentEditFeatures + ParentEditModel ridge ensemble",
            "model_readiness_unique_endpoints": 24,
            "refit_interval_unique_endpoints": 16,
            "route_archive_at_start": [],
            "runtime_comparator_or_winner_inputs": [],
            "support": {"modules": 3, "primitives": 32, "blocks": 8},
        },
        "task_adapters": {
            "t4": {
                "cells": list(T4_CELLS),
                "archive_k": 1,
                "score_direction": "minimize",
                "persistent_feasibility_frontier": True,
            },
            "pmo": {
                "tasks": list(PMO_TASKS),
                "archive_k": 10,
                "score_direction": "maximize",
                "t4_filters": False,
            },
        },
        "charged_query_ceiling_per_unit": QUERY_BUDGET,
        "total_new_call_ceiling": QUERY_BUDGET * (len(T4_CELLS) + len(PMO_TASKS)),
        "launch_authorized": False,
        "automatic_retry": False,
        "required_logging": [
            "proposal attempts and persistent attempt keys",
            "frontier states and non-oracle eligibility margins",
            "complete eligible pools and candidate descriptors",
            "allocation roles and model identities",
            "exact query ledger and score curves",
            "optimizer snapshots including RNG and model state",
        ],
        "comparators": "offline reporting only; absent from runtime inputs",
    }
    return {"payload": payload, "contract_sha256": identity(payload)}


def adapter_receipt(adapter: DynamicV22PilotAdapter) -> dict:
    body = {
        "schema_version": "dynamic_v22_pilot_adapter_v1",
        "adapter": asdict(adapter),
        "policy": asdict(adapter.policy),
        "query_budget": QUERY_BUDGET,
    }
    return {**body, "adapter_sha256": identity(body)}


def _score_curve(rows: list[dict], *, kind: str) -> list[dict]:
    values, curve = [], []
    for index, row in enumerate(rows, start=1):
        if row.get("status") != "complete":
            raise ValueError("v2.2 score curve cannot impute a failed query")
        values.append(float(row["score"]))
        oriented = [-value for value in values] if kind == "t4" else values
        top_k = 1 if kind == "t4" else 10
        curve.append(
            {
                "charged_queries": index,
                "best_score": min(values) if kind == "t4" else max(values),
                "top_k_utility": float(
                    sum(sorted(oriented, reverse=True)[:top_k]) / min(top_k, len(oriented))
                ),
            }
        )
    return curve


def run_t4_pilot_campaign(
    *,
    output,
    source_state,
    original_seed,
    target,
    oracle_protocol,
    config,
    evaluate,
    delta=0.4,
    rounds=64,
    max_rounds_this_invocation=None,
    flush=None,
    progress=None,
):
    """Durable no-plateau T4 runner with proposal-only frontier resume."""
    output = Path(output)
    flush = flush or (lambda: None)
    progress = progress or (lambda row: None)
    adapter = DynamicV22PilotAdapter("t4", oracle_protocol, identity(source_state))
    task = ProgramTask(target, oracle_protocol, "t4", original_seed, delta)
    eligibility = strict_endpoint_scorer(original_seed, delta=delta)
    ledger = ProgramQueryLedger(output / "oracle", task, evaluate, budget=QUERY_BUDGET, flush=flush)
    recipe = json.loads(
        json.dumps(
            {
                "schema_version": "dynamic_v22_t4_pilot_campaign_v1",
                "task": asdict(task),
                "source_state_sha256": identity(source_state),
                "configuration": asdict(config),
                "policy": asdict(adapter.policy),
                "query_budget": QUERY_BUDGET,
                "rounds": rounds,
                "plateau_stopping": False,
                "runtime_comparator_inputs": [],
            }
        )
    )
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != recipe:
        raise ValueError("v2.2 T4 campaign recipe changed during resume")
    publish_json(manifest, recipe)
    if max_rounds_this_invocation is not None and (
        type(max_rounds_this_invocation) is not int or max_rounds_this_invocation < 1
    ):
        raise ValueError("positive per-invocation round cap required")
    for pending in output.glob("round_*/pending.json"):
        if not pending.with_name("complete.json").exists():
            raise RuntimeError(f"pending v2.2 T4 round requires explicit recovery: {pending}")
    search, frontier, history = None, adapter.initial_frontier(seed=config.seed), []
    for complete in sorted(output.glob("round_*/complete.json")):
        saved = json.loads(complete.read_text())
        search = (
            None
            if saved["snapshot"] is None
            else DynamicV22ProgramOptimizer.restore(saved["snapshot"], hierarchy=None)
        )
        frontier = saved["frontier_state"]
        history.append(saved["summary"])
    source = decode_state(source_state)
    stop_round = (
        rounds
        if max_rounds_this_invocation is None
        else min(rounds, len(history) + max_rounds_this_invocation)
    )
    for round_index in range(len(history), stop_round):
        if not ledger.remaining:
            break
        folder = output / f"round_{round_index:04d}"
        count = min(config.candidates_per_batch, ledger.remaining)
        bootstrap = search is None or not search.entries
        if bootstrap:
            batch = initial_dynamic_program_batch_v22(
                source,
                (),
                config,
                source_group=adapter.source_group,
                oracle_protocol=oracle_protocol,
                eligibility=eligibility,
                policy_config=adapter.policy,
                frontier_state=frontier,
                delta=delta,
            )
            frontier = batch["frontier_state"]
            if len(batch["candidates"]) > count:
                batch["candidates"] = batch["candidates"][:count]
                body = {key: value for key, value in batch.items() if key != "batch_id"}
                batch = {**body, "batch_id": identity(body)}
        else:
            batch = search.propose_batch(eligibility)
            if len(batch["candidates"]) > count:
                selected = [row["candidate_id"] for row in batch["candidates"][:count]]
                batch = search.lock_query_subset(
                    batch["batch_id"],
                    selected,
                    {
                        "policy": "remaining_budget_prefix_v1",
                        "selected_ids": selected,
                        "remaining_budget": ledger.remaining,
                    },
                )
        publish_json(folder / "pending.json", {"batch": batch, "recipe_sha256": identity(recipe)})
        flush()
        outcomes = []
        for candidate in batch["candidates"]:
            row = ledger.query(candidate["endpoint"], lock_id=batch["batch_id"], role="candidate")
            outcomes.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "receipt_id": row["receipt_id"],
                    "score": row["score"],
                    "oracle_protocol": oracle_protocol,
                }
            )
        if bootstrap and outcomes:
            if search is None:
                search = adapter.optimizer(config, hierarchy=None)
            for candidate, outcome in zip(batch["candidates"], outcomes, strict=True):
                search.add_measured_program(
                    candidate,
                    receipt_id=outcome["receipt_id"],
                    score=outcome["score"],
                )
        elif not bootstrap:
            search.observe_batch(batch["batch_id"], outcomes)
        curve = _score_curve(ledger.rows, kind="t4")
        summary = {
            "round": round_index,
            "bootstrap": bootstrap,
            "queries_this_round": len(batch["candidates"]),
            "queries_total": len(ledger.rows),
            "best_score": None if not curve else curve[-1]["best_score"],
            "proposal_attempts": len(batch["attempts"]),
            "frontier": batch.get("frontier_summary"),
            "allocation": batch.get("allocation"),
        }
        publish_json(
            folder / "complete.json",
            {
                "summary": summary,
                "snapshot": None if search is None else search.snapshot(),
                "frontier_state": frontier,
                "recipe_sha256": identity(recipe),
            },
        )
        history.append(summary)
        progress(summary)
        flush()
    curve = _score_curve(ledger.rows, kind="t4")
    result = {
        "schema_version": "dynamic_v22_t4_pilot_result_v1",
        "status": "complete" if not ledger.remaining else "proposal_or_round_shortfall",
        "charged_oracle_calls": len(ledger.rows),
        "remaining_queries": ledger.remaining,
        "best_score": None if not curve else curve[-1]["best_score"],
        "score_curve": curve,
        "history": history,
        "frontier_state": frontier,
        "snapshot": None if search is None else search.snapshot(),
        "benchmark_claim": False,
    }
    publish_json(output / "result.json", result)
    flush()
    return result


@contextmanager
def _pmo_campaign_namespace():
    from compose_v4.control import program_campaign
    from compose_v4.experiments.pmo_dynamic_v21_recovery import (
        _merge_bootstrap_proposal_state,
    )

    original_optimizer = program_campaign.ProgramOptimizer
    original_initial = program_campaign.initial_program_batch

    class PMODynamicV22Optimizer(DynamicV22ProgramOptimizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, policy_config=DynamicV22PolicyConfig.for_pmo(), **kwargs)
            self._v22_pmo_generation_pool_ids = []

        def _adapt_bootstrap(self, record):
            v21 = record.get("provenance", {}).get("dynamic_v21_bootstrap")
            v22 = record.get("provenance", {}).get("dynamic_v22_bootstrap")
            if v21 is None or v22 is None:
                return record
            generation_id = v21.get("generation_pool_id", v21["pool_id"])
            if self._bootstrap_pool_id is None:
                if generation_id not in self._v22_pmo_generation_pool_ids:
                    self._v22_pmo_generation_pool_ids.append(generation_id)
                return record
            if generation_id not in self._v22_pmo_generation_pool_ids:
                _merge_bootstrap_proposal_state(self.allocator_state, v21["allocator_state"])
                self.shallow_rng.bit_generator.state = v21["shallow_rng"]
                self.structured_rng.bit_generator.state = v21["structured_rng"]
                self.rng.bit_generator.state = v21["arbitration_rng"]
                self.parent_rng.bit_generator.state = v22["parent_rng"]
                self._v22_pmo_generation_pool_ids.append(generation_id)
            adapted = json.loads(json.dumps(record))
            adapted["provenance"]["dynamic_v21_bootstrap"]["generation_pool_id"] = generation_id
            adapted["provenance"]["dynamic_v21_bootstrap"]["pool_id"] = self._bootstrap_pool_id
            adapted["provenance"]["dynamic_v22_bootstrap"]["frontier_state_sha256"] = (
                self._v22_bootstrap_frontier_id
            )
            return adapted

        def add_measured_program(self, record, *, receipt_id, score, static_score=None):
            return super().add_measured_program(
                self._adapt_bootstrap(record),
                receipt_id=receipt_id,
                score=score,
                static_score=static_score,
            )

        def snapshot(self, *, include_history=True):
            snapshot = super().snapshot(include_history=include_history)
            body = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
            body["dynamic_v22_pmo_continuity"] = {
                "schema_version": "dynamic_v22_pmo_continuity_v1",
                "generation_pool_ids": list(self._v22_pmo_generation_pool_ids),
            }
            return {**body, "snapshot_id": identity(body)}

        @classmethod
        def restore(cls, snapshot, *, hierarchy=None):
            result = super().restore(snapshot, hierarchy=hierarchy)
            state = snapshot.get("dynamic_v22_pmo_continuity")
            if state is None:
                result._v22_pmo_generation_pool_ids = (
                    [] if result._bootstrap_pool_id is None else [result._bootstrap_pool_id]
                )
            elif state.get("schema_version") != "dynamic_v22_pmo_continuity_v1":
                raise ValueError("invalid v2.2 PMO bootstrap-continuity state")
            else:
                values = list(state["generation_pool_ids"])
                if len(values) != len(set(values)):
                    raise ValueError("duplicate v2.2 PMO generation pool identity")
                result._v22_pmo_generation_pool_ids = values
            return result

    def initial(source, entries, config, **kwargs):
        return initial_dynamic_program_batch_v22(
            source,
            entries,
            config,
            policy_config=DynamicV22PolicyConfig.for_pmo(),
            **kwargs,
        )

    program_campaign.ProgramOptimizer = PMODynamicV22Optimizer
    program_campaign.initial_program_batch = initial
    try:
        yield
    finally:
        program_campaign.ProgramOptimizer = original_optimizer
        program_campaign.initial_program_batch = original_initial


def run_pmo_pilot_campaign(
    *,
    output,
    task_name,
    oracle_protocol,
    config,
    initialization,
    evaluate,
    rounds=16,
    flush=None,
    progress=None,
):
    """Thin PMO adapter: same core, native reward, all initialization charged."""
    if task_name not in PMO_TASKS:
        raise ValueError("task is outside the declared v2.2 PMO pilot")
    task = ProgramTask(task_name, oracle_protocol, "pmo")
    output = Path(output)
    ledger = ProgramQueryLedger(output / "oracle", task, evaluate, budget=QUERY_BUDGET, flush=flush)
    with _pmo_campaign_namespace():
        campaign = run_program_campaign(
            output=output / "campaign",
            task=task,
            config=config,
            initialization=initialization,
            library=(),
            ledger=ledger,
            rounds=rounds,
            queries_per_round=config.candidates_per_batch,
            hierarchy=None,
            fit_model=None,
            stagnation_rounds=None,
            bootstrap_rounds=1,
            initialization_mode="all_scored_pool",
            initial_parent_fraction=0.2,
            max_seconds=None,
            progress=progress,
        )
    curve = _score_curve(ledger.rows, kind="pmo")
    result = {
        "schema_version": "dynamic_v22_pmo_pilot_result_v1",
        "status": "complete" if not ledger.remaining else "proposal_or_round_shortfall",
        "task": task_name,
        "charged_oracle_calls": len(ledger.rows),
        "remaining_queries": ledger.remaining,
        "best_score": curve[-1]["best_score"],
        "final_top10": curve[-1]["top_k_utility"],
        "score_curve": curve,
        "auc_top10_256": pmo_top_ten_auc(
            [row["score"] for row in ledger.rows],
            budget=QUERY_BUDGET,
            frequency=100,
            finish=True,
        ),
        "campaign": campaign,
        "benchmark_claim": False,
    }
    publish_json(output / "result.json", result)
    if flush is not None:
        flush()
    return result
