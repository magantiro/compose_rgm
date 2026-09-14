"""Post-filter arbitration over independent route-free proposal channels.

Dynamic-v2.1 preserves the Dynamic-v0 shallow proposer and Dynamic-v1
structured proposer as independently seeded sources of complete programs.
Both pools are exact-executed and filtered before a bounded, run-local channel
allocator assigns expensive endpoint evaluations.  No target identity or
offline comparator result enters this module.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import asdict
from time import perf_counter

import numpy as np

from compose_v4.control.adaptive_program_optimizer import (
    ProgramOptimizer,
    dispatch_complete_proposal,
)
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
from compose_v4.control.dynamic_program_synthesis_v2 import (
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
    synthesize_structured_program,
)
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
    program_size_profile,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "dynamic_program_synthesis_v21"
CHANNELS = (SHALLOW_CHANNEL, STRUCTURED_CHANNEL)
CHANNEL_CANDIDATE_LIMIT = 16
UCB_EXPLORATION = 1.0
EXPLORATION_FLOOR = 1


def _blank_counts() -> dict[str, int | float]:
    return {
        "proposals": 0,
        "exact_executions": 0,
        "eligible_novel": 0,
        "duplicates": 0,
        "cross_channel_duplicates": 0,
        "ineligible": 0,
        "execution_rejections": 0,
        "selected_for_oracle": 0,
        "charged_outcomes": 0,
        "scored_outcomes": 0,
        "parent_improvements": 0,
        "global_best_improvements": 0,
        "positive_improvement_sum": 0.0,
    }


def initial_allocator_state(*, score_direction: str) -> dict:
    if score_direction not in ("minimize", "maximize"):
        raise ValueError("v2.1 score direction must be minimize or maximize")
    return {
        "schema_version": "dynamic_program_allocator_state_v1",
        "score_direction": score_direction,
        "channels": {channel: _blank_counts() for channel in CHANNELS},
        "contexts": {},
        "global_best": None,
        "allocation_decisions": 0,
    }


def _is_improvement(value: float, reference: float, direction: str) -> bool:
    return value < reference if direction == "minimize" else value > reference


def _improvement(value: float, reference: float, direction: str) -> float:
    return reference - value if direction == "minimize" else value - reference


def _context_key(source) -> str:
    slots = np.flatnonzero(source.atom_types > 0)
    slot_set = set(map(int, slots))
    edges = sum(
        1 for i in slot_set for j in slot_set if i < j and int(source.bonds[i, j]) > 0
    )
    cycles = max(0, edges - len(slot_set) + int(bool(slot_set)))
    attachments = sum(int(source.implicit_h_counts[int(slot)]) > 0 for slot in slots)
    return ":".join(
        (
            "ring" if cycles else "acyclic",
            "capacity" if 40 - source.n_real_atoms >= 4 else "tight",
            "multi_attachment" if attachments >= 4 else "few_attachment",
        )
    )


def _channel_score(state: dict, channel: str, *, virtual: int = 0) -> float:
    counts = state["channels"][channel]
    charged = int(counts["charged_outcomes"]) + virtual
    total = sum(int(state["channels"][name]["charged_outcomes"]) for name in CHANNELS)
    reward = float(counts["positive_improvement_sum"]) / max(
        1, int(counts["scored_outcomes"])
    )
    yield_term = int(counts["parent_improvements"]) / max(
        1, int(counts["scored_outcomes"])
    )
    uncertainty = UCB_EXPLORATION * math.sqrt(math.log(total + 2) / (charged + 1))
    return reward + yield_term + uncertainty


def arbitrate_candidates(
    candidates: list[dict],
    *,
    limit: int,
    state: dict,
    rng,
) -> tuple[list[dict], dict]:
    """Allocate a finite endpoint-query batch after strict cheap filtering."""
    if type(limit) is not int or limit < 1:
        raise ValueError("v2.1 arbitration limit must be positive")
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("v2.1 arbitration received duplicate candidate identities")
    available = {
        channel: [
            row for row in candidates if row["provenance"]["planner_channel"] == channel
        ]
        for channel in CHANNELS
    }
    state["allocation_decisions"] += 1
    decision = state["allocation_decisions"]
    if len(candidates) <= limit:
        selected = list(candidates)
        policy = "all_eligible_fit"
        scores = {channel: _channel_score(state, channel) for channel in CHANNELS}
    else:
        policy = "bounded_channel_ucb_v1"
        selected, positions = [], {channel: 0 for channel in CHANNELS}
        scores = {channel: _channel_score(state, channel) for channel in CHANNELS}
        if limit >= len(CHANNELS) * EXPLORATION_FLOOR and all(available.values()):
            for channel in CHANNELS:
                selected.append(available[channel][0])
                positions[channel] = 1
        virtual = {channel: positions[channel] for channel in CHANNELS}
        while len(selected) < limit:
            active = [
                channel
                for channel in CHANNELS
                if positions[channel] < len(available[channel])
            ]
            if not active:
                break
            values = {
                channel: _channel_score(state, channel, virtual=virtual[channel])
                for channel in active
            }
            best = max(values.values())
            tied = sorted(
                channel
                for channel, value in values.items()
                if math.isclose(value, best)
            )
            channel = tied[int(rng.integers(len(tied)))]
            selected.append(available[channel][positions[channel]])
            positions[channel] += 1
            virtual[channel] += 1
    selected_ids = [row["candidate_id"] for row in selected]
    return selected, {
        "schema_version": "dynamic_v21_allocation_v1",
        "decision_index": decision,
        "policy": policy,
        "available_by_channel": {
            channel: len(available[channel]) for channel in CHANNELS
        },
        "selected_by_channel": {
            channel: sum(
                row["provenance"]["planner_channel"] == channel for row in selected
            )
            for channel in CHANNELS
        },
        "channel_scores_before_selection": scores,
        "selected_ids": selected_ids,
        "candidate_limit": limit,
        "runtime_information": "current_run_charged_outcomes_only",
    }


def _attempt_status(state: dict, channel: str, status: str) -> None:
    counts = state["channels"][channel]
    counts["proposals"] += 1
    if status != "execution_rejected":
        counts["exact_executions"] += 1
    key = {
        "eligible": "eligible_novel",
        "duplicate": "duplicates",
        "ineligible": "ineligible",
        "execution_rejected": "execution_rejections",
    }[status]
    counts[key] += 1


def _credit_outcome(
    state: dict,
    candidate: dict,
    *,
    score: float | None,
) -> None:
    provenance = candidate["provenance"]
    channel = provenance["planner_channel"]
    counts = state["channels"][channel]
    counts["charged_outcomes"] += 1
    context = provenance["planner_context"]
    context_counts = state["contexts"].setdefault(
        context, {name: _blank_counts() for name in CHANNELS}
    )[channel]
    context_counts["charged_outcomes"] += 1
    if score is None:
        return
    counts["scored_outcomes"] += 1
    context_counts["scored_outcomes"] += 1
    parent_score = provenance.get("parent_measured_score")
    if parent_score is not None:
        gain = _improvement(float(score), float(parent_score), state["score_direction"])
        if gain > 0:
            counts["parent_improvements"] += 1
            context_counts["parent_improvements"] += 1
            counts["positive_improvement_sum"] += gain
            context_counts["positive_improvement_sum"] += gain
    best = state["global_best"]
    if best is None or _is_improvement(
        float(score), float(best), state["score_direction"]
    ):
        state["global_best"] = float(score)
        counts["global_best_improvements"] += 1
        context_counts["global_best_improvements"] += 1


@contextmanager
def _using_rng(optimizer, rng):
    original = optimizer.rng
    optimizer.rng = rng
    try:
        yield
    finally:
        optimizer.rng = original


def _candidate(
    *,
    optimizer,
    entry,
    parent,
    channel,
    attempt,
    source,
    program,
    binding,
    trace,
    metadata,
    size,
    properties,
) -> dict:
    record = {
        "attempt": attempt,
        **parent,
        "channel": "dynamic_v21_pooled_channels",
        "planner_channel": channel,
        "planner_context": _context_key(decode_state(entry["trace"]["states"][-1])),
        "endpoint": trace["endpoint"],
        "status": "eligible",
        "metadata": metadata,
        "program_size": size,
        "properties": properties,
        "actual_changes": trace["actual_changes"],
    }
    candidate = {
        "source_group": optimizer.source_group,
        "oracle_protocol": optimizer.oracle_protocol,
        "source_state": trace["states"][0],
        "program": program.payload(),
        "assignment": list(binding),
        "endpoint": trace["endpoint"],
        "trace": trace,
        "provenance": record,
        "inherited_static_score": entry["static_score"],
    }
    if "construction_ancestry" in metadata:
        for key in (
            "construction_ancestry",
            "original_seed_state",
            "ancestral_primitive_edits",
        ):
            candidate[key] = metadata[key]
    elif "construction_ancestry" in entry:
        for key in (
            "construction_ancestry",
            "original_seed_state",
            "ancestral_primitive_edits",
        ):
            candidate[key] = entry[key]
    candidate["candidate_id"] = identity(candidate)
    return candidate


class DynamicV21ProgramOptimizer(DynamicProgramOptimizer):
    """Generate both cheap pools, then allocate oracle calls over their union."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, 211, 0])
        )
        self.shallow_rng = np.random.default_rng(self.config.seed)
        self.structured_rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, 211, 1])
        )
        self.allocator_state = initial_allocator_state(
            score_direction=self.config.score_direction
        )
        self._v21_panel_cache = {}
        self._bootstrap_pool_id = None

    def _channel_proposal(self, channel: str, entry):
        rng = self.shallow_rng if channel == SHALLOW_CHANNEL else self.structured_rng
        chosen = {}
        with _using_rng(self, rng):

            def program_sampler(kind):
                chosen["proposal_kind"] = kind
                if kind == "recombination":
                    return ProgramOptimizer._recombine(self, entry)
                if channel == SHALLOW_CHANNEL:
                    return DynamicProgramOptimizer._mutate(self, entry)
                if self.rng.random() >= FRESH_SYNTHESIS_PROBABILITY:
                    return ProgramOptimizer._mutate(self, entry)
                current = decode_state(entry["trace"]["states"][-1])
                source, program, binding, _, detail = synthesize_structured_program(
                    current,
                    self.rng,
                    max_modules=3,
                    max_primitives=self.config.max_primitives,
                    max_blocks=self.config.max_blocks,
                    panel_cache=self._v21_panel_cache,
                )
                return (
                    source,
                    program,
                    binding,
                    {
                        "dynamic_v1_structured_program": detail,
                        **self._continuation_lineage(entry),
                    },
                )

            _, result = dispatch_complete_proposal(
                self.rng,
                program_sampler=program_sampler,
                broad_sampler=lambda: (_ for _ in ()).throw(
                    ValueError("v2.1 has no broad proposal channel")
                ),
                probabilities=self.config.channel_probabilities,
                proposal_mode=self.config.proposal_mode,
            )
        source, program, binding, metadata = result
        return source, program, binding, {**metadata, **chosen}

    def _generate_channel_pool(
        self, channel, eligibility, archive_seen, parent_schedule
    ):
        began = perf_counter()
        attempts, candidates, seen = [], [], set(archive_seen)
        for attempt, (entry, parent) in enumerate(parent_schedule):
            if (
                len(candidates) >= CHANNEL_CANDIDATE_LIMIT
                or perf_counter() - began >= self.config.wall_seconds
            ):
                break
            try:
                source, program, binding, metadata = self._channel_proposal(
                    channel, entry
                )
                graph = compile_program_graph(program)
                size = program_size_profile(graph, source.n_real_atoms)
                measured_atoms = decode_state(entry["trace"]["states"][-1]).n_real_atoms
                size["measured_parent_heavy_atoms"] = measured_atoms
                size["delta_from_measured_parent"] = (
                    size["final_heavy_atoms"] - measured_atoms
                )
                _, trace = execute_program_graph(
                    source,
                    graph,
                    binding,
                    max_primitives=self.config.max_primitives,
                    max_blocks=self.config.max_blocks,
                )
                if (
                    decode_state(trace["states"][-1]).n_real_atoms
                    != size["final_heavy_atoms"]
                    or trace["capacity_timeline"] != size["capacity_timeline"]
                ):
                    raise RuntimeError(
                        "v2.1 size accounting differs from exact execution"
                    )
            except ValueError as error:
                _attempt_status(self.allocator_state, channel, "execution_rejected")
                attempts.append(
                    {
                        "attempt": attempt,
                        "planner_channel": channel,
                        "status": "execution_rejected",
                        "reason": str(error),
                    }
                )
                continue
            endpoint = trace["endpoint"]
            properties = eligibility({"smiles": endpoint})
            if type(properties.get("oracle_eligible")) is not bool:
                raise ValueError(
                    "endpoint evaluator must return explicit boolean eligibility"
                )
            status = (
                "duplicate"
                if endpoint in seen
                else "eligible" if properties["oracle_eligible"] else "ineligible"
            )
            _attempt_status(self.allocator_state, channel, status)
            attempt_record = {
                "attempt": attempt,
                **parent,
                "planner_channel": channel,
                "planner_context": _context_key(
                    decode_state(entry["trace"]["states"][-1])
                ),
                "status": status,
                "endpoint": endpoint,
                "metadata": metadata,
                "program_size": size,
                "properties": properties,
                "actual_changes": trace["actual_changes"],
            }
            attempts.append(attempt_record)
            if status == "duplicate":
                self.duplicate_counts[entry["entry_id"]] = (
                    self.duplicate_counts.get(entry["entry_id"], 0) + 1
                )
            if status != "eligible":
                continue
            seen.add(endpoint)
            candidates.append(
                _candidate(
                    optimizer=self,
                    entry=entry,
                    parent=parent,
                    channel=channel,
                    attempt=attempt,
                    source=source,
                    program=program,
                    binding=binding,
                    trace=trace,
                    metadata=metadata,
                    size=size,
                    properties=properties,
                )
            )
        return attempts, candidates, perf_counter() - began

    def propose_batch(self, eligibility):
        if self.pending is not None:
            raise ValueError(
                "resolve the locked pending batch before generating another"
            )
        archive_seen = {
            row["endpoint"] for row in self.observations.values()
        } | self.failed_endpoints
        parent_schedule = [
            self._parent() for _ in range(self.config.attempts_per_batch)
        ]
        all_attempts, pools, seconds = [], {}, {}
        for channel in CHANNELS:
            attempts, candidates, elapsed = self._generate_channel_pool(
                channel, eligibility, archive_seen, parent_schedule
            )
            all_attempts.extend(attempts)
            pools[channel], seconds[channel] = candidates, elapsed
        merged, seen = [], set()
        for channel in CHANNELS:
            for candidate in pools[channel]:
                endpoint = candidate["endpoint"]
                if endpoint in seen:
                    self.allocator_state["channels"][channel][
                        "cross_channel_duplicates"
                    ] += 1
                    all_attempts.append(
                        {
                            "planner_channel": channel,
                            "status": "cross_channel_duplicate",
                            "endpoint": endpoint,
                            "candidate_id": candidate["candidate_id"],
                        }
                    )
                    continue
                seen.add(endpoint)
                merged.append(candidate)
        selected, allocation = arbitrate_candidates(
            merged,
            limit=self.config.candidates_per_batch,
            state=self.allocator_state,
            rng=self.rng,
        )
        for candidate in selected:
            self.allocator_state["channels"][
                candidate["provenance"]["planner_channel"]
            ]["selected_for_oracle"] += 1
        body = {
            "schema_version": "dynamic_program_batch_v21",
            "batch_index": self.batches,
            "configuration": asdict(self.config),
            "candidates": selected,
            "attempts": all_attempts,
            "source_group": self.source_group,
            "oracle_protocol": self.oracle_protocol,
            "eligible_pool": {
                "candidates": merged,
                "pool_id": identity([row["candidate_id"] for row in merged]),
            },
            "allocation": allocation,
            "allocator_state_after_proposal": json.loads(
                json.dumps(self.allocator_state)
            ),
        }
        self.pending = {**body, "batch_id": identity(body)}
        return json.loads(
            json.dumps(
                {
                    **self.pending,
                    "proposal_seconds": sum(seconds.values()),
                    "work_cache": {
                        **self.work_cache.report(),
                        "proposal_seconds_by_channel": seconds,
                    },
                    "new_oracle_calls": 0,
                }
            )
        )

    def add_measured_program(self, record, *, receipt_id, score, static_score=None):
        provenance = record.get("provenance", {})
        bootstrap = provenance.get("dynamic_v21_bootstrap")
        if bootstrap is not None:
            pool_id = bootstrap["pool_id"]
            if self._bootstrap_pool_id is None:
                self._bootstrap_pool_id = pool_id
                self.allocator_state = json.loads(
                    json.dumps(bootstrap["allocator_state"])
                )
                self.shallow_rng.bit_generator.state = bootstrap["shallow_rng"]
                self.structured_rng.bit_generator.state = bootstrap["structured_rng"]
                self.rng.bit_generator.state = bootstrap["arbitration_rng"]
            elif self._bootstrap_pool_id != pool_id:
                raise ValueError("v2.1 bootstrap candidates came from different pools")
            _credit_outcome(self.allocator_state, record, score=float(score))
        return super().add_measured_program(
            record,
            receipt_id=receipt_id,
            score=score,
            static_score=static_score,
        )

    def observe_batch(self, batch_id, outcomes):
        if self.pending is None or self.pending.get("batch_id") != batch_id:
            raise ValueError("v2.1 outcome has no matching pending batch")
        candidates = {row["candidate_id"]: row for row in self.pending["candidates"]}
        for outcome in outcomes:
            candidate = candidates.get(outcome.get("candidate_id"))
            if candidate is not None:
                _credit_outcome(
                    self.allocator_state, candidate, score=outcome.get("score")
                )
        super().observe_batch(batch_id, outcomes)

    def snapshot(self, *, include_history=True):
        validate_allocator_state(self.allocator_state)
        snapshot = super().snapshot(include_history=include_history)
        body = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
        body["dynamic_v21"] = {
            "allocator_state": self.allocator_state,
            "shallow_rng": self.shallow_rng.bit_generator.state,
            "structured_rng": self.structured_rng.bit_generator.state,
            "bootstrap_pool_id": self._bootstrap_pool_id,
        }
        return json.loads(json.dumps({**body, "snapshot_id": identity(body)}))

    @classmethod
    def restore(cls, snapshot, *, hierarchy=None):
        result = super().restore(snapshot, hierarchy=hierarchy)
        state = snapshot.get("dynamic_v21")
        if not isinstance(state, dict):
            raise TypeError("v2.1 snapshot lacks pooled-channel state")
        validate_allocator_state(state["allocator_state"])
        result.allocator_state = json.loads(json.dumps(state["allocator_state"]))
        result.shallow_rng.bit_generator.state = state["shallow_rng"]
        result.structured_rng.bit_generator.state = state["structured_rng"]
        result._bootstrap_pool_id = state["bootstrap_pool_id"]
        result._v21_panel_cache = {}
        return result


def _cold_channel_pool(
    source,
    *,
    channel,
    config,
    source_group,
    oracle_protocol,
    eligibility,
    rng,
    state,
):
    began, attempts, candidates = perf_counter(), [], []
    seen = {canonical_state_key(source)}
    panel_cache = {}
    attempt_limit = (
        CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS + CAPACITY_BOOTSTRAP_PAIR_ATTEMPTS
        if source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD
        else config.attempts_per_batch
    )
    for attempt in range(attempt_limit):
        if (
            len(candidates) >= CHANNEL_CANDIDATE_LIMIT
            or perf_counter() - began >= config.wall_seconds
        ):
            break
        try:
            if (
                channel == SHALLOW_CHANNEL
                and source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD
            ):
                families = (
                    ("substituent_delete",)
                    if attempt < CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS
                    else ("substituent_delete", "substituent_delete")
                )
                _, program, binding, trace, metadata = synthesize_named_module_sequence(
                    source,
                    rng,
                    families,
                    max_primitives=config.max_primitives,
                    max_blocks=config.max_blocks,
                )
            elif channel == SHALLOW_CHANNEL:
                _, program, binding, trace, metadata = synthesize_dynamic_program(
                    source,
                    rng,
                    max_modules=3,
                    max_primitives=config.max_primitives,
                    max_blocks=config.max_blocks,
                )
            else:
                _, program, binding, trace, metadata = synthesize_structured_program(
                    source,
                    rng,
                    max_modules=3,
                    max_primitives=config.max_primitives,
                    max_blocks=config.max_blocks,
                    panel_cache=panel_cache,
                )
        except ValueError as error:
            _attempt_status(state, channel, "execution_rejected")
            attempts.append(
                {
                    "attempt": attempt,
                    "planner_channel": channel,
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
        _attempt_status(state, channel, status)
        attempts.append(
            {
                "attempt": attempt,
                "channel": "dynamic_v21_pooled_channels",
                "planner_channel": channel,
                "planner_context": _context_key(source),
                "status": status,
                "endpoint": endpoint,
                "metadata": metadata,
                "properties": properties,
                "actual_changes": trace["actual_changes"],
            }
        )
        if status != "eligible":
            continue
        seen.add(endpoint)
        candidate = {
            "source_group": source_group,
            "oracle_protocol": oracle_protocol,
            "source_state": encode_state(source),
            "program": program.payload(),
            "assignment": list(binding),
            "trace": trace,
            "endpoint": endpoint,
            "provenance": attempts[-1],
            "score": None,
        }
        candidate["candidate_id"] = identity(candidate)
        candidates.append(candidate)
    return attempts, candidates, perf_counter() - began


def initial_dynamic_program_batch_v21(
    source,
    entries,
    config,
    *,
    source_group,
    oracle_protocol,
    eligibility,
    broad_sampler=None,
):
    """Build independent cold-start pools and arbitrate their eligible union."""
    if entries:
        raise ValueError("Dynamic-v2.1 initialization requires an empty route archive")
    if broad_sampler is not None:
        raise ValueError("Dynamic-v2.1 initialization does not use reference inference")
    began = perf_counter()
    state = initial_allocator_state(score_direction=config.score_direction)
    shallow_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 71]))
    structured_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 73]))
    arbitration_rng = np.random.default_rng(
        np.random.SeedSequence([config.seed, 211, 0])
    )
    pools, attempts, seconds = {}, [], {}
    for channel, rng in (
        (SHALLOW_CHANNEL, shallow_rng),
        (STRUCTURED_CHANNEL, structured_rng),
    ):
        channel_attempts, candidates, elapsed = _cold_channel_pool(
            source,
            channel=channel,
            config=config,
            source_group=source_group,
            oracle_protocol=oracle_protocol,
            eligibility=eligibility,
            rng=rng,
            state=state,
        )
        attempts.extend(channel_attempts)
        pools[channel], seconds[channel] = candidates, elapsed
    merged, seen = [], set()
    for channel in CHANNELS:
        for candidate in pools[channel]:
            if candidate["endpoint"] in seen:
                state["channels"][channel]["cross_channel_duplicates"] += 1
                attempts.append(
                    {
                        "planner_channel": channel,
                        "status": "cross_channel_duplicate",
                        "endpoint": candidate["endpoint"],
                    }
                )
                continue
            seen.add(candidate["endpoint"])
            merged.append(candidate)
    selected, allocation = arbitrate_candidates(
        merged,
        limit=config.candidates_per_batch,
        state=state,
        rng=arbitration_rng,
    )
    for candidate in selected:
        state["channels"][candidate["provenance"]["planner_channel"]][
            "selected_for_oracle"
        ] += 1
    bootstrap = {
        "pool_id": identity([row["candidate_id"] for row in merged]),
        "allocator_state": json.loads(json.dumps(state)),
        "shallow_rng": shallow_rng.bit_generator.state,
        "structured_rng": structured_rng.bit_generator.state,
        "arbitration_rng": arbitration_rng.bit_generator.state,
    }
    for candidate in selected:
        candidate["provenance"]["dynamic_v21_bootstrap"] = bootstrap
        candidate.pop("candidate_id")
        candidate["candidate_id"] = identity(candidate)
    allocation["selected_ids"] = [row["candidate_id"] for row in selected]
    body = {
        "schema_version": "initial_dynamic_program_batch_v21",
        "source_group": source_group,
        "oracle_protocol": oracle_protocol,
        "initial_route_archive": [],
        "source_library_rows_loaded": 0,
        "candidates": selected,
        "attempts": attempts,
        "proposal_pool": {
            "pool_id": bootstrap["pool_id"],
            "candidates": merged,
        },
        "allocation": allocation,
        "allocator_state_after_preparation": state,
    }
    return {
        **body,
        "batch_id": identity(body),
        "proposal_seconds": perf_counter() - began,
        "proposal_seconds_by_channel": seconds,
        "new_oracle_calls": 0,
    }


def validate_allocator_state(state: dict) -> None:
    if state.get("schema_version") != "dynamic_program_allocator_state_v1":
        raise ValueError("invalid Dynamic-v2.1 allocator-state schema")
    if state.get("score_direction") not in ("minimize", "maximize"):
        raise ValueError("invalid Dynamic-v2.1 score direction")
    if set(state.get("channels", {})) != set(CHANNELS):
        raise ValueError("Dynamic-v2.1 channel census changed")
    expected = set(_blank_counts())
    groups = list(state["channels"].values())
    for context in state.get("contexts", {}).values():
        if set(context) != set(CHANNELS):
            raise ValueError("Dynamic-v2.1 context channel census changed")
        groups.extend(context.values())
    for counts in groups:
        if set(counts) != expected:
            raise ValueError("Dynamic-v2.1 channel counter schema changed")
        for key, value in counts.items():
            if key == "positive_improvement_sum":
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError("Dynamic-v2.1 improvement magnitude is invalid")
            elif type(value) is not int or value < 0:
                raise ValueError("Dynamic-v2.1 channel count is invalid")
    if state.get("global_best") is not None and not math.isfinite(state["global_best"]):
        raise ValueError("Dynamic-v2.1 global best is invalid")
    if type(state.get("allocation_decisions")) is not int:
        raise ValueError("Dynamic-v2.1 allocation decision count is invalid")


def allocator_state_summary(state: dict) -> dict:
    validate_allocator_state(state)
    return {
        "schema_version": state["schema_version"],
        "global_best": state["global_best"],
        "allocation_decisions": state["allocation_decisions"],
        "channels": json.loads(json.dumps(state["channels"])),
        "context_count": len(state["contexts"]),
    }
