"""Capped production continuation profile, with durable per-row evidence.

This diagnostic does not modify the T4 optimizer. Its single binding work budget
is attempted executor applications. A platform timeout is explicitly incomplete.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.continuation import (
    ContinuationBudgetExceeded,
    FiniteHorizonContinuation,
    continuation_decision,
)
from compose_v4.control.graph_geometry import topology
from compose_v4.control.option_continuation import OptionContinuationKernel, OptionState
from compose_v4.control.region import enumerate_regions
from compose_v4.control.region_rewrite import Lineage, context_from_region
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import InvalidRewrite, RewriteSystem
from compose_v4.rewrite.trace_shard import decode_state, encode_state


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def encode_action(rule, action):
    """Explicit current/legacy codec dispatch, without changing action semantics."""
    codec = action_codec_v4 if rule in action_codec_v4.ACTIVE8_EXECUTOR_RULES else action_codec
    return codec.encode_action(rule, action)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"input identity mismatch: {path}: expected {expected}, got {actual}")
    return actual


def publish_json(path: Path, value: object) -> str:
    """Atomic publication; content-addressed immutable artifacts use new paths."""
    payload = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def state_payload(node: OptionState) -> dict:
    def graph_payload(graph):
        encoded = encode_state(graph)
        decoded = decode_state(encoded)
        for name in ("atom_types", "bonds", "formal_charges", "implicit_h_counts"):
            if not np.array_equal(getattr(graph, name), getattr(decoded, name)):
                raise ValueError(f"exact state codec failed round-trip for {name}")
        return encoded

    context = asdict(node.context)
    context["frozen"] = sorted(context["frozen"])
    context["locus"] = sorted(context["locus"])
    return {
        "graph": graph_payload(node.graph),
        "origin": graph_payload(node.origin),
        "context": context,
        "lineage": {
            "slot_of": sorted(node.lineage.slot_of.items()),
            "id_of": sorted(node.lineage.id_of.items()),
            "next_id": node.lineage.next_id,
        },
        "option": node.option,
        "step": node.step,
        "horizon": node.horizon,
        "bundle_id": node.bundle_id,
    }


def initial_state(contract: dict, seed_manifest: list) -> OptionState:
    from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    source = seed_manifest[contract["source_index"]]["smiles"]
    # Fresh source initialization, NOT recovery of an intermediate from SMILES.
    graph = pad_molecular_graph(smiles_to_molecular_graph(source), contract["persistent_slots"])
    regions = [
        region
        for region in enumerate_regions(source)
        if sorted(region.atoms) == contract["region_atoms"]
    ]
    if len(regions) != 1:
        raise ValueError("profile requires exactly one frozen region, no replacement draw")
    region = regions[0]
    return OptionState(
        graph,
        graph,
        context_from_region(region),
        Lineage.initial(np.flatnonzero(is_element(graph.atom_types))),
        contract["option"],
        0,
        contract["horizon"],
        contract["prior_bundle_id"],
    )


class ExecutorMeter:
    """Scoped diagnostic hook, including public executor calls inside enumeration."""

    def __init__(self, limit):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("executor limit must be a nonnegative integer")
        self.limit = limit
        self.calls = 0
        self.seconds = 0.0
        self.attempts = []
        self.phase = "option_validation"

    @contextmanager
    def instrument(self):
        from unittest.mock import patch

        original = RewriteSystem.apply

        def measured(system, state, rule_name, action):
            if self.calls >= self.limit:
                raise ContinuationBudgetExceeded("total public-executor budget exhausted")
            self.calls += 1
            receipt = {
                "phase": self.phase,
                "source": encode_state(state),
                "mark": encode_action(rule_name, action),
            }
            started = perf_counter()
            try:
                result = original(system, state, rule_name, action)
            except InvalidRewrite as error:
                self.seconds += perf_counter() - started
                self.attempts.append(
                    {**receipt, "status": "invalid_rewrite", "message": str(error)}
                )
                raise
            else:
                self.seconds += perf_counter() - started
                self.attempts.append(
                    {**receipt, "status": "executed", "product": encode_state(result)}
                )
                return result

        with patch.object(RewriteSystem, "apply", measured):
            yield self


def run_profile(
    node: OptionState,
    enumerate_law: Callable,
    system,
    contract: dict,
    output: Path,
    *,
    snapshot_id: str,
    commit_volume: Callable[[], None],
    progress: dict,
) -> dict:
    with ExecutorMeter(contract["max_executor_applications"]).instrument() as meter:
        return _run_profile(
            node,
            enumerate_law,
            system,
            contract,
            output,
            snapshot_id=snapshot_id,
            commit_volume=commit_volume,
            progress=progress,
            meter=meter,
        )


def _run_profile(
    node, enumerate_law, system, contract, output, *, snapshot_id, commit_volume, progress, meter
):
    """Return a complete decision or honest abstention; retain every completed row.

    Raw law records and complete successor rows are restart units for later
    validated reuse. This profile never silently relaunches or re-enumerates a
    completed run; the launcher binds an immutable output identity.
    """
    started = perf_counter()
    timings = {"enumeration": 0.0, "rows": 0.0, "persistence": 0.0, "terminal": 0.0}
    row_receipts, terminal_records = [], []
    law_count = 0

    def persist(relative, payload):
        start = perf_counter()
        digest = publish_json(output / relative, payload)
        commit_volume()
        timings["persistence"] += perf_counter() - start
        return digest

    def timed_law(graph):
        nonlocal law_count
        progress.update(phase="law_enumeration", law_index=law_count)
        start = perf_counter()
        meter.phase = "law_enumeration"
        try:
            families, actions, probabilities = enumerate_law(graph)
        finally:
            timings["enumeration"] += perf_counter() - start
            meter.phase = "option_validation"
        payload = {
            "schema_version": "continuation_profile_law_v1",
            "snapshot_id": snapshot_id,
            "state": encode_state(graph),
            "marks": [encode_action(f, a) for f, a in zip(families, actions)],
            "probabilities": [float(p) for p in probabilities],
        }
        digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
        persist(f"laws/{digest}.json", payload)
        law_count += 1
        return families, actions, probabilities

    kernel = OptionContinuationKernel(
        timed_law,
        system,
        max_executor_applications=contract["max_executor_applications"],
    )
    row_keys = set()

    def row(state):
        progress.update(phase="reference_row", step=state.step)
        start = perf_counter()
        attempts_start = len(meter.attempts)
        try:
            result = kernel.row(state)
        finally:
            timings["rows"] += perf_counter() - start
            progress["kernel_work"] = asdict(kernel.work)
            progress["total_public_executor_calls"] = meter.calls
            if len(meter.attempts) > attempts_start:
                persist(
                    f"attempts/{attempts_start:06d}.json",
                    {
                        "snapshot_id": snapshot_id,
                        "source": state_payload(state),
                        "attempts": meter.attempts[attempts_start:],
                        "partial_row": state.key() not in kernel._rows,
                    },
                )
        if state.key() not in row_keys:
            payload = {
                "schema_version": "continuation_profile_row_v1",
                "snapshot_id": snapshot_id,
                "source": state_payload(state),
                "successors": [state_payload(s) for s in result.successors],
                "probabilities": result.probabilities,
                "marks": [encode_action(f, a) for f, a in kernel.marks(state)],
            }
            digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
            relative = f"rows/{digest}.json"
            physical = persist(relative, payload)
            row_receipts.append(
                {
                    "path": relative,
                    "sha256": physical,
                    "step": state.step,
                    "successors": len(result.successors),
                }
            )
            row_keys.add(state.key())
            progress["completed_rows"] = len(row_receipts)
        return result

    origin_topology = topology(node.origin)

    def terminal(state):
        start = perf_counter()
        final = topology(state.graph)
        value = float(
            state.remaining == 0
            and final["cycle_rank"] > origin_topology["cycle_rank"]
            and final["n_ring_systems"] > origin_topology["n_ring_systems"]
        )
        terminal_records.append({"state": state_payload(state), "value": value})
        timings["terminal"] += perf_counter() - start
        return value

    continuation = FiniteHorizonContinuation(
        row,
        terminal,
        OptionState.key,
        snapshot_id=snapshot_id,
        max_expansions=contract["max_expansions"],
        max_terminal_evaluations=contract["max_terminal_evaluations"],
    )
    decision, cache_parity, root_count = None, None, 0
    try:
        root = row(node)
        root_count = len(root.successors)
        work_before = asdict(kernel.work)
        cached = row(node)
        cache_parity = (
            root.probabilities == cached.probabilities
            and tuple(s.key() for s in root.successors) == tuple(s.key() for s in cached.successors)
            and kernel.work.executor_applications == work_before["executor_applications"]
            and kernel.work.law_enumerations == work_before["law_enumerations"]
        )
        if not cache_parity:
            raise RuntimeError("cached row changed exact successor law or repeated molecular work")
        progress["phase"] = "continuation_backup"
        decision = continuation_decision(
            root,
            node.remaining,
            continuation,
            fallback_values=np.ones(root_count),
        )
        status = decision.status
        if status == "budget_abstention" and not np.allclose(
            decision.probabilities, root.probabilities, rtol=1e-12, atol=1e-12
        ):
            raise RuntimeError("budget abstention changed the declared reference fallback")
    except ContinuationBudgetExceeded:
        status = "root_executor_budget_exhausted"
    persist("terminal_evaluations.json", terminal_records)
    work = asdict(kernel.work)
    if meter.calls > contract["max_executor_applications"]:
        raise RuntimeError("executor ceiling exceeded")
    return {
        "schema_version": "continuation_profile_result_v1",
        "status": status,
        "root_successors": root_count,
        "root_cache_parity": cache_parity,
        "decision": asdict(decision) if decision else None,
        "initial_state": state_payload(node),
        "origin_topology": origin_topology,
        "kernel_work": work,
        "total_public_executor_calls": meter.calls,
        "executor_calls_by_phase": {
            phase: sum(a["phase"] == phase for a in meter.attempts)
            for phase in ("law_enumeration", "option_validation")
        },
        "continuation_work": asdict(continuation.work),
        "rows": row_receipts,
        "timings_seconds": {
            **timings,
            "executor": meter.seconds,
            "total": perf_counter() - started,
        },
        "timing_note": "row time includes enumeration, execution and law persistence; categories are nested, not additive",
        "oracle_calls": 0,
        "chemistry_comparison_authorized": status == "guided",
        "limitations": [
            "One conditional bundle; no outer-controller coverage or performance estimate.",
            "Structural terminal event is not a qualified T4 task-value signal.",
            "Even a guided decision is not evidence of better sampled chemistry or docking.",
        ],
    }
