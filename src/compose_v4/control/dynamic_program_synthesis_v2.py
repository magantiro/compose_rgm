"""Selective shallow/structured routing for route-free Dynamic COMPOSE.

Both proposal channels compile to the existing complete program representation
and exact executor.  This module contains no task, seed, winner or endpoint
identity.  Its adaptive state is learned only from outcomes observed within the
current run.
"""

from __future__ import annotations

import json
import math
from time import perf_counter

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import (
    CAPACITY_AWARE_THRESHOLD,
    CAPACITY_BOOTSTRAP_PAIR_ATTEMPTS,
    CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS,
    FRESH_SYNTHESIS_PROBABILITY,
    DynamicProgramOptimizer,
    synthesize_dynamic_program,
    synthesize_named_module_sequence,
)
from compose_v4.control.dynamic_program_synthesis_v1 import (
    synthesize_dynamic_program_v1,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "dynamic_program_synthesis_v2"
SHALLOW_CHANNEL = "shallow_program_channel"
STRUCTURED_CHANNEL = "structured_program_channel"
# The third generic lane T4 samples and PMO never imported.  See pmo_channels for the
# measured yield that makes it a restoration rather than a new mechanism.
ANCHORED_CHANNEL = "anchored_replacement_channel"
CHANNELS = (SHALLOW_CHANNEL, STRUCTURED_CHANNEL)

STRUCTURED_PRIOR = 0.25
STRUCTURED_FLOOR = 0.10
STRUCTURED_CEILING = 0.50
GLOBAL_STAGNATION_FIRST = 25
GLOBAL_STAGNATION_SECOND = 75
MIN_CHANNEL_EVIDENCE = 20


def _blank_counts() -> dict[str, int]:
    return {
        "proposals": 0,
        "exact_executions": 0,
        "eligible": 0,
        "duplicates": 0,
        "ineligible": 0,
        "execution_rejections": 0,
        "charged_outcomes": 0,
        "scored_outcomes": 0,
        "parent_improvements": 0,
        "global_best_improvements": 0,
    }


def initial_gate_state(*, score_direction: str) -> dict:
    if score_direction not in ("minimize", "maximize"):
        raise ValueError("v2 score direction must be minimize or maximize")
    return {
        "schema_version": "dynamic_program_gate_state_v1",
        "score_direction": score_direction,
        "channels": {channel: _blank_counts() for channel in CHANNELS},
        "parents": {},
        "charged_since_global_best": 0,
        "global_best": None,
        "gate_decisions": 0,
    }


def molecular_gate_features(source: MolecularGraph) -> dict:
    """Compute generic opportunity features without task or oracle access."""
    slots = np.flatnonzero(is_element(source.atom_types))
    slot_set = set(map(int, slots))
    edges = sum(
        1 for i in slot_set for j in slot_set if i < j and int(source.bonds[i, j]) > 0
    )
    cycle_rank = max(0, edges - len(slot_set) + int(bool(slot_set)))
    attachment_sites = sum(
        1 for slot in slots if int(source.implicit_h_counts[int(slot)]) > 0
    )
    return {
        "heavy_atoms": int(source.n_real_atoms),
        "atom_count_slack": int(40 - source.n_real_atoms),
        "cycle_rank": int(cycle_rank),
        "editable_ring_system_present": bool(cycle_rank > 0),
        "attachment_sites": int(attachment_sites),
        "multiple_attachment_regions": bool(attachment_sites >= 4),
    }


def structured_probability(
    source: MolecularGraph,
    state: dict,
    *,
    parent_id: str,
) -> tuple[float, dict]:
    """Return one bounded interpretable parent-specific planner probability."""
    if state.get("schema_version") != "dynamic_program_gate_state_v1":
        raise ValueError("invalid Dynamic-v2 gate state")
    structural = molecular_gate_features(source)
    parent = state["parents"].get(parent_id, {})
    shallow = parent.get(SHALLOW_CHANNEL, _blank_counts())
    probability = STRUCTURED_PRIOR
    adjustments = {}

    if structural["editable_ring_system_present"]:
        adjustments["ring_opportunity"] = 0.04
    if structural["multiple_attachment_regions"]:
        adjustments["multiple_attachment_regions"] = 0.02
    if structural["atom_count_slack"] >= 4:
        adjustments["capacity_slack"] = 0.02

    shallow_attempts = shallow["proposals"]
    shallow_failures = (
        shallow["duplicates"] + shallow["ineligible"] + shallow["execution_rejections"]
    )
    failure_rate = shallow_failures / max(1, shallow_attempts)
    if shallow_attempts >= 8 and failure_rate > 0.5:
        adjustments["parent_shallow_failure_rate"] = min(
            0.10, 0.20 * (failure_rate - 0.5)
        )
    if shallow["scored_outcomes"] >= 4 and shallow["parent_improvements"] == 0:
        adjustments["parent_shallow_no_improvement"] = 0.05

    stagnation = state["charged_since_global_best"]
    if stagnation >= GLOBAL_STAGNATION_FIRST:
        adjustments["global_stagnation"] = 0.025
    if stagnation >= GLOBAL_STAGNATION_SECOND:
        adjustments["global_stagnation"] = 0.05

    global_shallow = state["channels"][SHALLOW_CHANNEL]
    global_structured = state["channels"][STRUCTURED_CHANNEL]
    if (
        global_shallow["scored_outcomes"] >= MIN_CHANNEL_EVIDENCE
        and global_structured["scored_outcomes"] >= MIN_CHANNEL_EVIDENCE
    ):
        shallow_yield = global_shallow["parent_improvements"] / max(
            1, global_shallow["scored_outcomes"]
        )
        structured_yield = global_structured["parent_improvements"] / max(
            1, global_structured["scored_outcomes"]
        )
        adjustments["within_run_channel_productivity"] = float(
            np.clip(0.10 * (structured_yield - shallow_yield), -0.05, 0.05)
        )

    probability += sum(adjustments.values())
    probability = float(np.clip(probability, STRUCTURED_FLOOR, STRUCTURED_CEILING))
    features = {
        **structural,
        "parent_shallow_proposals": shallow_attempts,
        "parent_shallow_failure_rate": failure_rate,
        "parent_shallow_scored": shallow["scored_outcomes"],
        "parent_shallow_improvements": shallow["parent_improvements"],
        "charged_since_global_best": stagnation,
        "adjustments": adjustments,
    }
    return probability, features


def _gate_record(
    *,
    parent_id: str,
    probability: float,
    features: dict,
    channel: str,
    decision_index: int,
) -> dict:
    if channel not in CHANNELS:
        raise ValueError("unknown Dynamic-v2 planner channel")
    return {
        "schema_version": "dynamic_program_gate_decision_v1",
        "decision_index": decision_index,
        "parent_id": parent_id,
        "probabilities": {
            SHALLOW_CHANNEL: 1.0 - probability,
            STRUCTURED_CHANNEL: probability,
        },
        "features": features,
        "selected_channel": channel,
    }


def _is_improvement(value: float, reference: float, direction: str) -> bool:
    return value < reference if direction == "minimize" else value > reference


class _ForceStructuredRng:
    """Consume, but override, only v1's existing top-level capability draw."""

    def __init__(self, rng):
        self._rng = rng
        self._first_random = True

    def random(self, *args, **kwargs):
        value = self._rng.random(*args, **kwargs)
        if self._first_random:
            if args or kwargs:
                raise RuntimeError("v1 capability draw unexpectedly became non-scalar")
            self._first_random = False
            return 0.0
        return value

    def __getattr__(self, name):
        return getattr(self._rng, name)


def synthesize_structured_program(
    source,
    rng,
    *,
    max_modules=3,
    max_primitives=32,
    max_blocks=8,
    panel_cache=None,
):
    """Invoke the unchanged v1 structured branch as an explicit v2 channel."""
    result = synthesize_dynamic_program_v1(
        source,
        _ForceStructuredRng(rng),
        max_modules=max_modules,
        max_primitives=max_primitives,
        max_blocks=max_blocks,
        panel_cache=panel_cache,
    )
    if result[4].get("v1_selection") == "preserved_dynamic_v0_channel":
        raise RuntimeError("explicit structured planner entered the shallow branch")
    return result


def _add_attempt_credit(state: dict, gate: dict, status: str) -> None:
    channel = gate["selected_channel"]
    global_counts = state["channels"][channel]
    parent = state["parents"].setdefault(
        gate["parent_id"], {name: _blank_counts() for name in CHANNELS}
    )[channel]
    for counts in (global_counts, parent):
        counts["proposals"] += 1
        if status != "execution_rejected":
            counts["exact_executions"] += 1
        key = {
            "eligible": "eligible",
            "duplicate": "duplicates",
            "ineligible": "ineligible",
            "execution_rejected": "execution_rejections",
        }[status]
        counts[key] += 1


def _add_outcome_credit(
    state: dict,
    gate: dict,
    *,
    score: float | None,
    parent_score: float | None,
) -> None:
    channel = gate["selected_channel"]
    parent = state["parents"].setdefault(
        gate["parent_id"], {name: _blank_counts() for name in CHANNELS}
    )[channel]
    global_counts = state["channels"][channel]
    for counts in (global_counts, parent):
        counts["charged_outcomes"] += 1
    state["charged_since_global_best"] += 1
    if score is None:
        return
    for counts in (global_counts, parent):
        counts["scored_outcomes"] += 1
        if parent_score is not None and _is_improvement(
            score, parent_score, state["score_direction"]
        ):
            counts["parent_improvements"] += 1
    best = state["global_best"]
    if best is None or _is_improvement(score, best, state["score_direction"]):
        state["global_best"] = float(score)
        state["charged_since_global_best"] = 0
        for counts in (global_counts, parent):
            counts["global_best_improvements"] += 1


def _initial_candidate(
    source,
    program,
    binding,
    trace,
    metadata,
    *,
    source_group,
    oracle_protocol,
    attempt,
    properties,
):
    candidate = {
        "source_group": source_group,
        "oracle_protocol": oracle_protocol,
        "source_state": encode_state(source),
        "program": program.payload(),
        "assignment": list(binding),
        "trace": trace,
        "endpoint": trace["endpoint"],
        "provenance": {
            "attempt": attempt,
            "channel": "dynamic_v2_gated_composition",
            "planner_channel": metadata["dynamic_v2_planner"]["selected_channel"],
            "planner_gate": metadata["dynamic_v2_planner"],
            "endpoint": trace["endpoint"],
            "status": "eligible",
            "properties": properties,
            "metadata": metadata,
            "actual_changes": trace["actual_changes"],
        },
        "score": None,
    }
    return {**candidate, "candidate_id": identity(candidate)}


def initial_dynamic_program_batch_v2(
    source,
    entries,
    config,
    *,
    source_group,
    oracle_protocol,
    eligibility,
    broad_sampler=None,
):
    """Build one route-empty cold-start batch through the generic v2 gate."""
    if entries:
        raise ValueError("Dynamic-v2 initialization requires an empty route archive")
    if broad_sampler is not None:
        raise ValueError("Dynamic-v2 initialization does not use reference inference")
    began = perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 79]))
    state = initial_gate_state(score_direction=config.score_direction)
    attempts, candidates = [], []
    seen = {canonical_state_key(source)}
    panel_cache = {}
    attempt_limit = (
        CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS + CAPACITY_BOOTSTRAP_PAIR_ATTEMPTS
        if source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD
        else config.attempts_per_batch
    )
    root_id = identity(encode_state(source))
    for index in range(attempt_limit):
        if (
            len(candidates) >= config.candidates_per_batch
            or perf_counter() - began >= config.wall_seconds
        ):
            break
        probability, features = structured_probability(source, state, parent_id=root_id)
        state["gate_decisions"] += 1
        if source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD:
            channel = SHALLOW_CHANNEL
        else:
            channel = (
                STRUCTURED_CHANNEL if rng.random() < probability else SHALLOW_CHANNEL
            )
        gate = _gate_record(
            parent_id=root_id,
            probability=probability,
            features=features,
            channel=channel,
            decision_index=state["gate_decisions"],
        )
        try:
            if source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD:
                families = (
                    ("substituent_delete",)
                    if index < CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS
                    else ("substituent_delete", "substituent_delete")
                )
                _, program, binding, trace, detail = synthesize_named_module_sequence(
                    source,
                    rng,
                    families,
                    max_primitives=config.max_primitives,
                    max_blocks=config.max_blocks,
                )
                detail["v2_selection"] = "preserved_dynamic_v0_capacity_bootstrap"
            elif channel == SHALLOW_CHANNEL:
                _, program, binding, trace, detail = synthesize_dynamic_program(
                    source,
                    rng,
                    max_modules=3,
                    max_primitives=config.max_primitives,
                    max_blocks=config.max_blocks,
                )
            else:
                _, program, binding, trace, detail = synthesize_structured_program(
                    source,
                    rng,
                    max_modules=3,
                    max_primitives=config.max_primitives,
                    max_blocks=config.max_blocks,
                    panel_cache=panel_cache,
                )
        except ValueError as error:
            _add_attempt_credit(state, gate, "execution_rejected")
            attempts.append(
                {
                    "attempt": index,
                    "planner_channel": channel,
                    "planner_gate": gate,
                    "status": "execution_rejected",
                    "reason": str(error),
                }
            )
            continue
        endpoint = trace["endpoint"]
        properties = eligibility({"smiles": endpoint})
        status = (
            "duplicate"
            if endpoint in seen
            else (
                "eligible"
                if properties.get("oracle_eligible") is True
                else "ineligible"
            )
        )
        _add_attempt_credit(state, gate, status)
        metadata = {
            "dynamic_v2_planner": gate,
            "dynamic_v2_program": detail,
            "intermediate_task_evaluations": 0,
        }
        attempts.append(
            {
                "attempt": index,
                "planner_channel": channel,
                "planner_gate": gate,
                "endpoint": endpoint,
                "status": status,
                "properties": properties,
                "metadata": metadata,
                "actual_changes": trace["actual_changes"],
            }
        )
        if status != "eligible":
            continue
        seen.add(endpoint)
        candidates.append(
            _initial_candidate(
                source,
                program,
                binding,
                trace,
                metadata,
                source_group=source_group,
                oracle_protocol=oracle_protocol,
                attempt=index,
                properties=properties,
            )
        )
    body = {
        "schema_version": "initial_dynamic_program_batch_v2",
        "source_group": source_group,
        "oracle_protocol": oracle_protocol,
        "initial_route_archive": [],
        "source_library_rows_loaded": 0,
        "candidates": candidates,
        "attempts": attempts,
        "gate_state_after_preparation": state,
        "rng_state_after_preparation": rng.bit_generator.state,
    }
    return {
        **body,
        "batch_id": identity(body),
        "proposal_seconds": perf_counter() - began,
        "new_oracle_calls": 0,
    }


class DynamicV2ProgramOptimizer(DynamicProgramOptimizer):
    """Route-free optimizer with a bounded parent-specific planner gate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_v2_state = initial_gate_state(
            score_direction=self.config.score_direction
        )
        self._v2_panel_cache = {}
        self._v2_execution_events = {}
        self._v2_error_events = {}

    def _new_gate(self, entry, source, *, force_channel=None):
        parent_id = entry["entry_id"]
        probability, features = structured_probability(
            source, self.dynamic_v2_state, parent_id=parent_id
        )
        self.dynamic_v2_state["gate_decisions"] += 1
        channel = force_channel or (
            STRUCTURED_CHANNEL if self.rng.random() < probability else SHALLOW_CHANNEL
        )
        return _gate_record(
            parent_id=parent_id,
            probability=probability,
            features=features,
            channel=channel,
            decision_index=self.dynamic_v2_state["gate_decisions"],
        )

    def add_measured_program(
        self,
        candidate,
        *,
        receipt_id,
        score,
        static_score=None,
    ):
        """Credit charged cold-start observations before normal archiving."""
        provenance = candidate.get("provenance", {})
        gate = provenance.get("planner_gate")
        if provenance.get("channel") == "dynamic_v2_gated_composition" and gate:
            _add_attempt_credit(self.dynamic_v2_state, gate, "eligible")
            _add_outcome_credit(
                self.dynamic_v2_state,
                gate,
                score=float(score),
                parent_score=None,
            )
        return super().add_measured_program(
            candidate,
            receipt_id=receipt_id,
            score=score,
            static_score=static_score,
        )

    def _remember_execution(self, source, program, binding, gate):
        key = identity(
            {
                "source": encode_state(source),
                "program": program.payload(),
                "binding": binding,
                "max_primitives": self.config.max_primitives,
                "max_blocks": self.config.max_blocks,
            }
        )
        self._v2_execution_events[key] = gate

    def _mutate(self, entry):
        current = decode_state(entry["trace"]["states"][-1])
        if self.rng.random() >= FRESH_SYNTHESIS_PROBABILITY:
            gate = self._new_gate(entry, current, force_channel=SHALLOW_CHANNEL)
            try:
                source, program, binding, metadata = ProgramOptimizer._mutate(
                    self, entry
                )
            except ValueError as error:
                token = identity({"gate": gate, "error": str(error)})
                self._v2_error_events[token] = gate
                raise ValueError(f"dynamic_v2_gate={token}:{error}") from error
            metadata = {
                **metadata,
                "dynamic_v2_planner": gate,
                "dynamic_v2_program": {"selection": "preserved_v0_route_mutation"},
            }
            self._remember_execution(source, program, binding, gate)
            return source, program, binding, metadata

        gate = self._new_gate(entry, current)
        try:
            if gate["selected_channel"] == SHALLOW_CHANNEL:
                source, program, binding, _, detail = synthesize_dynamic_program(
                    current,
                    self.rng,
                    max_modules=3,
                    max_primitives=self.config.max_primitives,
                    max_blocks=self.config.max_blocks,
                    successor_prior=getattr(self, "construction_prior", None),
                )
            else:
                source, program, binding, _, detail = synthesize_structured_program(
                    current,
                    self.rng,
                    max_modules=3,
                    max_primitives=self.config.max_primitives,
                    max_blocks=self.config.max_blocks,
                    panel_cache=self._v2_panel_cache,
                )
        except ValueError as error:
            token = identity({"gate": gate, "error": str(error)})
            self._v2_error_events[token] = gate
            raise ValueError(f"dynamic_v2_gate={token}:{error}") from error
        metadata = {
            "dynamic_v2_planner": gate,
            "dynamic_v2_program": detail,
            **self._continuation_lineage(entry),
        }
        self._remember_execution(source, program, binding, gate)
        return source, program, binding, metadata

    def _recombine(self, entry):
        current = decode_state(entry["trace"]["states"][-1])
        gate = self._new_gate(entry, current, force_channel=SHALLOW_CHANNEL)
        try:
            source, program, binding, metadata = ProgramOptimizer._recombine(
                self, entry
            )
        except ValueError as error:
            token = identity({"gate": gate, "error": str(error)})
            self._v2_error_events[token] = gate
            raise ValueError(f"dynamic_v2_gate={token}:{error}") from error
        metadata = {
            **metadata,
            "dynamic_v2_planner": gate,
            "dynamic_v2_program": {"selection": "preserved_v0_recombination"},
        }
        self._remember_execution(source, program, binding, gate)
        return source, program, binding, metadata

    def _attempt_gate(self, attempt):
        metadata = attempt.get("metadata", {})
        gate = metadata.get("dynamic_v2_planner")
        if gate is not None:
            return gate
        execution_key = attempt.get("execution_key")
        if execution_key in self._v2_execution_events:
            return self._v2_execution_events[execution_key]
        reason = attempt.get("reason", "")
        prefix = "dynamic_v2_gate="
        if reason.startswith(prefix) and ":" in reason:
            token = reason[len(prefix) :].split(":", 1)[0]
            return self._v2_error_events.get(token)
        return None

    def propose_batch(self, eligibility):
        self._v2_execution_events = {}
        self._v2_error_events = {}
        batch = super().propose_batch(eligibility)
        body = {key: value for key, value in self.pending.items() if key != "batch_id"}
        for attempt in body["attempts"]:
            gate = self._attempt_gate(attempt)
            if gate is None:
                continue
            attempt["planner_channel"] = gate["selected_channel"]
            attempt["planner_gate"] = gate
            _add_attempt_credit(self.dynamic_v2_state, gate, attempt["status"])
        for candidate in body["candidates"]:
            provenance = candidate["provenance"]
            gate = provenance.get("metadata", {}).get("dynamic_v2_planner")
            if gate is not None:
                provenance["planner_channel"] = gate["selected_channel"]
                provenance["planner_gate"] = gate
                candidate.pop("candidate_id", None)
                candidate["candidate_id"] = identity(candidate)
        body["dynamic_v2_gate_state_after_proposal"] = json.loads(
            json.dumps(self.dynamic_v2_state)
        )
        self.pending = {**body, "batch_id": identity(body)}
        return json.loads(
            json.dumps(
                {
                    **self.pending,
                    "proposal_seconds": batch["proposal_seconds"],
                    "work_cache": batch["work_cache"],
                    "new_oracle_calls": 0,
                }
            )
        )

    def observe_batch(self, batch_id, outcomes):
        if self.pending is None or self.pending.get("batch_id") != batch_id:
            raise ValueError("Dynamic-v2 outcome has no matching pending batch")
        candidates = {row["candidate_id"]: row for row in self.pending["candidates"]}
        for outcome in outcomes:
            candidate = candidates.get(outcome.get("candidate_id"))
            if candidate is None:
                continue
            provenance = candidate["provenance"]
            gate = provenance.get("planner_gate")
            if gate is None:
                continue
            _add_outcome_credit(
                self.dynamic_v2_state,
                gate,
                score=outcome.get("score"),
                parent_score=float(provenance["parent_measured_score"]),
            )
        super().observe_batch(batch_id, outcomes)

    def snapshot(self, *, include_history=True):
        validate_gate_state(self.dynamic_v2_state)
        snapshot = super().snapshot(include_history=include_history)
        body = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
        body["dynamic_v2_state"] = json.loads(json.dumps(self.dynamic_v2_state))
        return json.loads(json.dumps({**body, "snapshot_id": identity(body)}))

    @classmethod
    def restore(cls, snapshot, *, hierarchy=None):
        result = super().restore(snapshot, hierarchy=hierarchy)
        state = snapshot.get("dynamic_v2_state")
        if (
            state is None
            or state.get("schema_version") != "dynamic_program_gate_state_v1"
        ):
            raise ValueError("Dynamic-v2 snapshot lacks its planner gate state")
        validate_gate_state(state)
        result.dynamic_v2_state = json.loads(json.dumps(state))
        result._v2_panel_cache = {}
        result._v2_execution_events = {}
        result._v2_error_events = {}
        return result


def gate_state_summary(state: dict) -> dict:
    """Return stable channel-level instrumentation for reports and tests."""
    return {
        "schema_version": state["schema_version"],
        "structured_bounds": [STRUCTURED_FLOOR, STRUCTURED_CEILING],
        "charged_since_global_best": state["charged_since_global_best"],
        "global_best": state["global_best"],
        "gate_decisions": state["gate_decisions"],
        "channels": json.loads(json.dumps(state["channels"])),
        "parent_count": len(state["parents"]),
    }


def validate_gate_state(state: dict) -> None:
    """Fail closed on malformed persisted adaptive state."""
    if state.get("schema_version") != "dynamic_program_gate_state_v1":
        raise ValueError("invalid Dynamic-v2 gate-state schema")
    if state.get("score_direction") not in ("minimize", "maximize"):
        raise ValueError("invalid Dynamic-v2 gate-state score direction")
    if set(state.get("channels", {})) != set(CHANNELS):
        raise ValueError("Dynamic-v2 gate-state channels changed")
    if not isinstance(state.get("parents"), dict):
        raise TypeError("Dynamic-v2 parent state must be a mapping")
    if not isinstance(state.get("charged_since_global_best"), int):
        raise TypeError("Dynamic-v2 stagnation counter must be integral")
    for group in [
        *state["channels"].values(),
        *(counts for parent in state["parents"].values() for counts in parent.values()),
    ]:
        if set(group) != set(_blank_counts()) or any(
            type(value) is not int or value < 0 for value in group.values()
        ):
            raise ValueError("Dynamic-v2 channel counts are malformed")
    if state["global_best"] is not None and not math.isfinite(state["global_best"]):
        raise ValueError("Dynamic-v2 global best must be finite")
