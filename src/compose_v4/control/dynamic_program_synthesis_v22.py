"""Candidate-conditioned allocation and a persistent feasibility frontier.

Dynamic-v2.2 deliberately reuses the route-free v2.1 shallow and structured
generators.  It changes two boundaries only:

* exact-valid, endpoint-ineligible proposal prefixes may survive in a bounded
  proposal-only frontier, without receiving a task score; and
* eligible endpoints are allocated as parent/edit candidates instead of by a
  global channel UCB statistic.

No comparator, winner, target identity, failed proposal, ineligible endpoint,
or unqueried endpoint can enter the utility model.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from compose_v4.control.docking_value import graph_kernel, identity, molecular_features
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
from compose_v4.control.dynamic_program_synthesis_v2 import (
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
    synthesize_structured_program,
)
from compose_v4.control.dynamic_program_synthesis_v21 import (
    CHANNELS,
    DynamicV21ProgramOptimizer,
    _attempt_status,
    initial_allocator_state,
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.control.parent_edit_model import (
    MUTATION_RECIPE,
    ParentEditFeatures,
    ParentEditModel,
)
from compose_v4.control.program_task import predicted_archive_gains
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "dynamic_program_synthesis_v22"


@dataclass(frozen=True)
class DynamicV22PolicyConfig:
    """Frozen shared policy settings; the task adapter supplies kind only."""

    task_kind: str
    archive_k: int
    minimum_training_endpoints: int = 24
    refit_interval: int = 16
    shallow_spine_fraction: float = 0.25
    structural_exploration_fraction: float = 0.25
    frontier_width: int = 16
    max_modules: int = 3
    max_primitives: int = 32
    max_blocks: int = 8

    @classmethod
    def for_t4(cls):
        return cls(task_kind="t4", archive_k=1)

    @classmethod
    def for_pmo(cls):
        return cls(task_kind="pmo", archive_k=10)

    def __post_init__(self):
        expected = 1 if self.task_kind == "t4" else 10 if self.task_kind == "pmo" else None
        if expected is None or self.archive_k != expected:
            raise ValueError("v2.2 task kind and archive-k contract disagree")
        if self.minimum_training_endpoints < 1 or self.refit_interval < 1:
            raise ValueError("v2.2 model readiness settings must be positive")
        if not math.isclose(
            self.shallow_spine_fraction + self.structural_exploration_fraction,
            0.5,
        ):
            raise ValueError("v2.2 reserves exactly half the post-warmup batch for learning")
        if any(
            type(value) is not int or value < 1
            for value in (
                self.frontier_width,
                self.max_modules,
                self.max_primitives,
                self.max_blocks,
            )
        ):
            raise ValueError("v2.2 work and frontier limits must be positive integers")
        if self.max_modules != 3 or self.max_primitives != 32 or self.max_blocks != 8:
            raise ValueError("v2.2 pilot must retain the declared 3/32/8 support")


@dataclass(frozen=True)
class EligibilityMargins:
    """Non-oracle T4 endpoint slack; positive values satisfy a constraint."""

    similarity: float
    qed: float
    sa: float

    @property
    def violations(self):
        return sum(value < 0 for value in asdict(self).values())

    @property
    def total_deficit(self):
        return float(sum(max(0.0, -value) for value in asdict(self).values()))

    @property
    def normalized_deficits(self):
        scales = {
            "similarity": 0.6,
            "qed": 0.6,
            "sa": 4.0,
        }
        return tuple(max(0.0, -float(value)) / scales[name] for name, value in asdict(self).items())


@dataclass(frozen=True)
class ProposalDescriptor:
    candidate_id: str
    endpoint: str
    parent_id: str | None
    channel: str
    high_level_modules: tuple[str, ...]
    primitive_count: int
    block_count: int
    parent_feature_sha256: str


@dataclass(frozen=True)
class FrontierNode:
    node_id: str
    source_state: dict
    program: dict
    assignment: tuple[int, ...]
    trace: dict
    endpoint: str
    channel: str
    modules: tuple[dict, ...]
    module_count: int
    properties: dict
    margins: EligibilityMargins
    parent_node_id: str | None
    attempt_key: str

    def payload(self):
        row = asdict(self)
        row["assignment"] = list(self.assignment)
        row["modules"] = list(self.modules)
        row["margins"] = asdict(self.margins)
        return row


def initial_frontier_state(*, seed: int) -> dict:
    if type(seed) is not int or seed < 0:
        raise ValueError("v2.2 frontier seed must be a nonnegative integer")
    body = {
        "schema_version": "dynamic_v22_frontier_state_v1",
        "seed": seed,
        "wave": 0,
        "nodes": [],
        "attempt_keys": [],
        "shallow_rng": np.random.default_rng(
            np.random.SeedSequence([seed, 71])
        ).bit_generator.state,
        "structured_rng": np.random.default_rng(
            np.random.SeedSequence([seed, 73])
        ).bit_generator.state,
        "attempts": 0,
    }
    return {**body, "state_sha256": identity(body)}


def validate_frontier_state(state: dict, *, config: DynamicV22PolicyConfig) -> None:
    body = {key: value for key, value in state.items() if key != "state_sha256"}
    if state.get("schema_version") != "dynamic_v22_frontier_state_v1" or identity(
        body
    ) != state.get("state_sha256"):
        raise ValueError("corrupt or incompatible v2.2 frontier state")
    if len(state.get("nodes", ())) > config.frontier_width:
        raise ValueError("v2.2 frontier exceeds its declared width")
    if len(state.get("attempt_keys", ())) != len(set(state.get("attempt_keys", ()))):
        raise ValueError("v2.2 frontier attempt keys are not unique")
    for row in state.get("nodes", ()):
        node = FrontierNode(
            **{
                **row,
                "assignment": tuple(row["assignment"]),
                "modules": tuple(row["modules"]),
                "margins": EligibilityMargins(**row["margins"]),
            }
        )
        if (
            node.module_count > config.max_modules
            or len(node.trace["actions"]) > config.max_primitives
            or len(node.trace["blocks"]) > config.max_blocks
        ):
            raise ValueError("v2.2 frontier node exceeds declared proposal support")


def t4_eligibility_margins(properties: dict, *, delta: float) -> EligibilityMargins:
    """Calculate proposal-only slack without docking or surrogate value."""
    if delta not in (0.4, 0.6):
        raise ValueError("T4 frontier requires a benchmark similarity threshold")
    for field in ("sim", "qed", "sa"):
        if not isinstance(properties.get(field), (int, float)) or not math.isfinite(
            properties[field]
        ):
            raise ValueError(f"T4 frontier lacks finite {field} property")
    return EligibilityMargins(
        similarity=float(properties["sim"]) - delta,
        qed=float(properties["qed"]) - 0.6,
        sa=4.0 - float(properties["sa"]),
    )


def _receipt_stages(receipt: dict, *, prefix: str) -> list[dict]:
    stages, start = [], 0
    for index, block in enumerate(receipt["blocks"]):
        stop = int(block["stop"])
        stages.append(
            {
                "name": f"{prefix}:{index}:{block['label']}",
                "actions": receipt["actions"][start:stop],
                "states": receipt["states"][start : stop + 1],
                "endpoint": canonical_state_key(decode_state(receipt["states"][stop])),
            }
        )
        start = stop
    return stages


def _frontier_order(nodes: list[FrontierNode]) -> list[FrontierNode]:
    """Nondominated normalized constraint deficits, then diversity."""
    if not nodes:
        return []
    remaining, layers = list(nodes), []
    while remaining:
        vectors = [row.margins.normalized_deficits for row in remaining]
        front = []
        for index, vector in enumerate(vectors):
            dominated = any(
                other != index
                and all(a <= b for a, b in zip(vectors[other], vector, strict=True))
                and any(a < b for a, b in zip(vectors[other], vector, strict=True))
                for other in range(len(remaining))
            )
            if not dominated:
                front.append(remaining[index])
        layers.append(front)
        front_ids = {row.node_id for row in front}
        remaining = [row for row in remaining if row.node_id not in front_ids]
    ordered, anchors = [], []
    for front in layers:
        pool = list(front)
        while pool:
            features = [molecular_features(row.endpoint)[1] for row in pool]
            if not anchors:
                position = min(
                    range(len(pool)),
                    key=lambda i: (
                        sum(pool[i].margins.normalized_deficits),
                        pool[i].node_id,
                    ),
                )
            else:
                similarity = graph_kernel(features, anchors)
                position = min(
                    range(len(pool)),
                    key=lambda i: (float(np.max(similarity[i])), pool[i].node_id),
                )
            ordered.append(pool.pop(position))
            anchors.append(features.pop(position))
    return ordered


def _frontier_node(payload: dict) -> FrontierNode:
    return FrontierNode(
        **{
            **payload,
            "assignment": tuple(payload["assignment"]),
            "modules": tuple(payload["modules"]),
            "margins": EligibilityMargins(**payload["margins"]),
        }
    )


def advance_t4_feasibility_frontier(
    source,
    *,
    config: DynamicV22PolicyConfig,
    state: dict,
    source_group: str,
    oracle_protocol: str,
    eligibility,
    delta: float,
    attempts_per_channel: int,
) -> dict:
    """Advance one zero-oracle frontier wave and return any eligible endpoints.

    A frontier node is an exact-valid complete molecule that failed only the
    endpoint screen. Continuations are compiled against that actual state and
    then re-extracted as one root-to-endpoint program. The full 3/32/8 limits
    apply to the combined route, not independently to its suffix.
    """
    if config.task_kind != "t4":
        raise ValueError("feasibility-slack frontier is a T4 adapter capability")
    if type(attempts_per_channel) is not int or attempts_per_channel < 1:
        raise ValueError("frontier attempts per channel must be positive")
    validate_frontier_state(state, config=config)
    shallow_rng, structured_rng = np.random.default_rng(), np.random.default_rng()
    shallow_rng.bit_generator.state = state["shallow_rng"]
    structured_rng.bit_generator.state = state["structured_rng"]
    prior_nodes = [_frontier_node(row) for row in state["nodes"]]
    prior_keys = set(state["attempt_keys"])
    root_key = canonical_state_key(source)
    seen = {root_key, *(row.endpoint for row in prior_nodes)}
    eligible, new_nodes, attempts = [], [], []
    panel_cache = {}
    for channel, rng in (
        (SHALLOW_CHANNEL, shallow_rng),
        (STRUCTURED_CHANNEL, structured_rng),
    ):
        channel_nodes = [row for row in prior_nodes if row.channel == channel]
        for local_attempt in range(attempts_per_channel):
            # Alternate root proposals and persistent-prefix continuations. This
            # preserves broad shallow opportunity while ensuring zero-query
            # rounds do not regenerate the same cold stream forever.
            parent = (
                channel_nodes[local_attempt % len(channel_nodes)]
                if channel_nodes and local_attempt % 2 == 1
                else None
            )
            current = source if parent is None else decode_state(parent.trace["states"][-1])
            prior_module_count = 0 if parent is None else parent.module_count
            remaining_modules = config.max_modules - prior_module_count
            if remaining_modules < 1:
                continue
            try:
                if channel == SHALLOW_CHANNEL:
                    _, suffix, suffix_binding, suffix_trace, metadata = synthesize_dynamic_program(
                        current,
                        rng,
                        max_modules=remaining_modules,
                        max_primitives=config.max_primitives,
                        max_blocks=config.max_blocks,
                    )
                else:
                    _, suffix, suffix_binding, suffix_trace, metadata = (
                        synthesize_structured_program(
                            current,
                            rng,
                            max_modules=remaining_modules,
                            max_primitives=config.max_primitives,
                            max_blocks=config.max_blocks,
                            panel_cache=panel_cache,
                        )
                    )
                suffix_modules = tuple(metadata.get("modules", ()))
                if parent is None:
                    program, binding, trace = suffix, tuple(suffix_binding), suffix_trace
                    modules = suffix_modules
                else:
                    stages = [
                        *_receipt_stages(parent.trace, prefix="frontier_prefix"),
                        *_receipt_stages(suffix_trace, prefix="frontier_suffix"),
                    ]
                    program, binding = extract_program(source, stages)
                    _, trace = execute_program_graph(
                        source,
                        compile_program_graph(program),
                        binding,
                        max_primitives=config.max_primitives,
                        max_blocks=config.max_blocks,
                    )
                    modules = (*parent.modules, *suffix_modules)
                if (
                    len(modules) > config.max_modules
                    or len(program.marks) > config.max_primitives
                    or len(program.blocks) > config.max_blocks
                ):
                    raise ValueError("continued prefix exceeds complete v2.2 work support")
            except ValueError as error:
                attempts.append(
                    {
                        "wave": state["wave"],
                        "attempt": local_attempt,
                        "planner_channel": channel,
                        "parent_node_id": None if parent is None else parent.node_id,
                        "status": "execution_rejected",
                        "reason": str(error),
                    }
                )
                continue
            endpoint = trace["endpoint"]
            attempt_key = identity(
                {
                    "schema_version": SCHEMA,
                    "channel": channel,
                    "source": root_key,
                    "parent_endpoint": None if parent is None else parent.endpoint,
                    "program": program.payload(),
                    "assignment": list(binding),
                }
            )
            if attempt_key in prior_keys:
                status, properties = "repeated_attempt_key", None
            elif endpoint in seen:
                prior_keys.add(attempt_key)
                status, properties = "duplicate_endpoint", None
            else:
                prior_keys.add(attempt_key)
                seen.add(endpoint)
                properties = eligibility({"smiles": endpoint})
                if type(properties.get("oracle_eligible")) is not bool:
                    raise ValueError("frontier evaluator requires boolean endpoint eligibility")
                status = "eligible" if properties["oracle_eligible"] else "frontier"
            record = {
                "wave": state["wave"],
                "attempt": local_attempt,
                "planner_channel": channel,
                "parent_node_id": None if parent is None else parent.node_id,
                "attempt_key": attempt_key,
                "status": status,
                "endpoint": endpoint,
                "module_count": len(modules),
                "primitive_count": len(program.marks),
                "block_count": len(program.blocks),
            }
            attempts.append(record)
            if status == "eligible":
                candidate = {
                    "source_group": source_group,
                    "oracle_protocol": oracle_protocol,
                    "source_state": encode_state(source),
                    "program": program.payload(),
                    "assignment": list(binding),
                    "endpoint": endpoint,
                    "trace": trace,
                    "score": None,
                    "provenance": {
                        **record,
                        "channel": "dynamic_v22_persistent_frontier",
                        "planner_context": "proposal_only_frontier",
                        "metadata": {
                            "schema_version": "dynamic_v22_frontier_route_v1",
                            "modules": list(modules),
                            "initial_stored_complete_routes": 0,
                            "source_library_rows_loaded": 0,
                            "intermediate_task_evaluations": 0,
                        },
                        "properties": properties,
                        "actual_changes": trace["actual_changes"],
                    },
                }
                candidate["candidate_id"] = identity(candidate)
                eligible.append(candidate)
            elif status == "frontier" and len(modules) < config.max_modules:
                margins = t4_eligibility_margins(properties, delta=delta)
                node_body = {
                    "source_state": encode_state(source),
                    "program": program.payload(),
                    "assignment": tuple(binding),
                    "trace": trace,
                    "endpoint": endpoint,
                    "channel": channel,
                    "modules": modules,
                    "module_count": len(modules),
                    "properties": properties,
                    "margins": margins,
                    "parent_node_id": None if parent is None else parent.node_id,
                    "attempt_key": attempt_key,
                }
                node_id = identity(
                    {**node_body, "assignment": list(binding), "margins": asdict(margins)}
                )
                new_nodes.append(FrontierNode(node_id=node_id, **node_body))
    frontier = _frontier_order([*prior_nodes, *new_nodes])[: config.frontier_width]
    body = {
        "schema_version": "dynamic_v22_frontier_state_v1",
        "seed": state["seed"],
        "wave": state["wave"] + 1,
        "nodes": [row.payload() for row in frontier],
        "attempt_keys": sorted(prior_keys),
        "shallow_rng": shallow_rng.bit_generator.state,
        "structured_rng": structured_rng.bit_generator.state,
        "attempts": state["attempts"] + len(attempts),
    }
    next_state = {**body, "state_sha256": identity(body)}
    validate_frontier_state(next_state, config=config)
    return {
        "schema_version": "dynamic_v22_frontier_wave_v1",
        "eligible_candidates": eligible,
        "attempts": attempts,
        "frontier_state": next_state,
        "frontier_summary": {
            "wave": next_state["wave"],
            "retained_nodes": len(frontier),
            "eligible_candidates": len(eligible),
            "new_attempt_keys": len(prior_keys) - len(state["attempt_keys"]),
            "new_oracle_calls": 0,
        },
        "new_oracle_calls": 0,
    }


def _module_names(candidate: dict) -> tuple[str, ...]:
    detail = candidate.get("provenance", {}).get("metadata", {})
    nested = detail.get("dynamic_v1_structured_program", detail)
    modules = nested.get("modules", ())
    names = []
    for row in modules:
        name = row.get("family", row.get("name"))
        if name:
            names.append(str(name))
    return tuple(names)


def _oriented(score: float, direction: str) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("v2.2 utility requires a finite measured score")
    return -float(score) if direction == "minimize" else float(score)


def _endpoint_utilities(search) -> tuple[list[str], list[float]]:
    grouped = {}
    for observation in search.observations.values():
        if observation.get("score") is None:
            continue
        grouped.setdefault(observation["endpoint"], []).append(
            _oriented(observation["score"], search.config.score_direction)
        )
    endpoints = sorted(grouped)
    return endpoints, [float(np.mean(grouped[endpoint])) for endpoint in endpoints]


def _candidate_features(
    search, candidates: list[dict]
) -> tuple[list[list[float]], list[ProposalDescriptor]]:
    feature_builder = ParentEditFeatures()
    rows, descriptors = [], []
    for candidate in candidates:
        parent_id = candidate.get("provenance", {}).get("entry_id")
        parent = search.entries.get(parent_id) if parent_id is not None else None
        parent_state = (
            parent["trace"]["states"][-1] if parent is not None else candidate["source_state"]
        )
        parent_scores = []
        if parent is not None:
            parent_scores = [
                _oriented(record["score"], search.config.score_direction)
                for record in search.observations.values()
                if record["endpoint"] == parent["endpoint"] and record.get("score") is not None
            ]
        features = feature_builder.with_mutation_context(
            candidate,
            parent_state=parent_state,
            parent_scores=parent_scores,
            parent_record=parent,
        )
        program = candidate["program"]
        rows.append(features)
        descriptors.append(
            ProposalDescriptor(
                candidate_id=candidate["candidate_id"],
                endpoint=candidate["endpoint"],
                parent_id=parent_id,
                channel=candidate["provenance"]["planner_channel"],
                high_level_modules=_module_names(candidate),
                primitive_count=len(program["marks"]),
                block_count=len(program["blocks"]),
                parent_feature_sha256=identity({"features": features}),
            )
        )
    return rows, descriptors


def measured_parent_edit_rows(search) -> list[dict]:
    """Chronological measured rows only; missing outcomes never become labels."""
    feature_builder = ParentEditFeatures()
    rows = []
    for entry_id, entry in sorted(search.entries.items()):
        parent_id = entry.get("provenance", {}).get("entry_id")
        parent = search.entries.get(parent_id) if parent_id is not None else None
        parent_state = parent["trace"]["states"][-1] if parent else entry["source_state"]
        parent_scores = []
        if parent is not None:
            parent_scores = [
                _oriented(record["score"], search.config.score_direction)
                for record in search.observations.values()
                if record["endpoint"] == parent["endpoint"] and record.get("score") is not None
            ]
        features = feature_builder.with_mutation_context(
            entry,
            parent_state=parent_state,
            parent_scores=parent_scores,
            parent_record=parent,
        )
        for receipt_id, observation in sorted(search.observations.items()):
            if observation["endpoint"] != entry["endpoint"] or observation.get("score") is None:
                continue
            rows.append(
                {
                    "features": features,
                    "utility": _oriented(observation["score"], search.config.score_direction),
                    "endpoint": entry["endpoint"],
                    "receipt_id": receipt_id,
                    "oracle_protocol": search.oracle_protocol,
                    "entry_id": entry_id,
                }
            )
    return rows


def fit_parent_edit_model_if_ready(
    search,
    *,
    config: DynamicV22PolicyConfig,
    previous: ParentEditModel | None,
    previous_endpoint_count: int,
) -> tuple[ParentEditModel | None, int, dict]:
    rows = measured_parent_edit_rows(search)
    endpoint_count = len({row["endpoint"] for row in rows})
    ready = endpoint_count >= config.minimum_training_endpoints
    due = previous is None or endpoint_count - previous_endpoint_count >= config.refit_interval
    if not ready or not due:
        return (
            previous,
            previous_endpoint_count,
            {
                "status": "not_ready" if not ready else "reuse_frozen_round_model",
                "measured_endpoint_count": endpoint_count,
                "minimum_training_endpoints": config.minimum_training_endpoints,
                "previous_fit_endpoint_count": previous_endpoint_count,
            },
        )
    input_body = {
        "oracle_protocol": search.oracle_protocol,
        "task_kind": config.task_kind,
        "archive_k": config.archive_k,
        "receipts": sorted(row["receipt_id"] for row in rows),
        "entry_features": sorted(
            {identity({"entry_id": row["entry_id"], "features": row["features"]}) for row in rows}
        ),
    }
    model = ParentEditModel.fit(
        rows,
        oracle_protocol=search.oracle_protocol,
        input_sha256=identity(input_body),
        recipe=MUTATION_RECIPE,
    )
    return (
        model,
        endpoint_count,
        {
            "status": "fitted_chronological_prefix",
            "measured_endpoint_count": endpoint_count,
            "model_sha256": model.payload["model_sha256"],
            "training_receipts": model.payload["training_receipts"],
            "uncertainty_used": False,
        },
    )


def _take_shallow_spine(candidates: list[dict], count: int, excluded: set[str]) -> list[dict]:
    return [
        row
        for row in candidates
        if row["candidate_id"] not in excluded
        and row["provenance"]["planner_channel"] == SHALLOW_CHANNEL
    ][:count]


def _take_diverse(
    candidates: list[dict], count: int, excluded: set[str], archive_endpoints: list[str]
) -> list[dict]:
    remaining = [row for row in candidates if row["candidate_id"] not in excluded]
    if not remaining or count < 1:
        return []
    candidate_features = [molecular_features(row["endpoint"])[1] for row in remaining]
    anchor_features = [molecular_features(endpoint)[1] for endpoint in archive_endpoints]
    selected = []
    while remaining and len(selected) < count:
        if not anchor_features:
            position = min(range(len(remaining)), key=lambda i: remaining[i]["candidate_id"])
        else:
            similarities = graph_kernel(candidate_features, anchor_features)
            position = max(
                range(len(remaining)),
                key=lambda i: (1.0 - float(np.max(similarities[i])), remaining[i]["candidate_id"]),
            )
        chosen = remaining.pop(position)
        chosen_feature = candidate_features.pop(position)
        selected.append(chosen)
        anchor_features.append(chosen_feature)
    return selected


def allocate_parent_edits(
    search,
    candidates: list[dict],
    *,
    limit: int,
    config: DynamicV22PolicyConfig,
    model: ParentEditModel | None,
) -> tuple[list[dict], dict]:
    """Lock a 25/25/50 spine/diversity/archive-gain allocation."""
    if type(limit) is not int or limit < 1:
        raise ValueError("v2.2 allocation limit must be positive")
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("v2.2 allocation requires unique candidates")
    if len(candidates) <= limit:
        return list(candidates), {
            "schema_version": "dynamic_v22_parent_edit_allocation_v1",
            "policy": "all_eligible_fit",
            "selected_ids": [row["candidate_id"] for row in candidates],
            "candidate_limit": limit,
            "model_sha256": None if model is None else model.payload["model_sha256"],
            "runtime_information": "current_run_measured_outcomes_only",
        }
    archive_endpoints, archive_utilities = _endpoint_utilities(search)
    selected: list[dict] = []
    roles: dict[str, str] = {}
    if model is None:
        shallow_quota = math.ceil(limit / 2)
        exploration_quota = limit - shallow_quota
    else:
        shallow_quota = math.ceil(limit * config.shallow_spine_fraction)
        exploration_quota = math.ceil(limit * config.structural_exploration_fraction)
    shallow = _take_shallow_spine(candidates, shallow_quota, set())
    selected.extend(shallow)
    roles.update({row["candidate_id"]: "preserved_shallow_pool_order" for row in shallow})
    diverse = _take_diverse(
        candidates,
        exploration_quota,
        {row["candidate_id"] for row in selected},
        archive_endpoints,
    )
    selected.extend(diverse)
    roles.update({row["candidate_id"]: "score_blind_structural_diversity" for row in diverse})
    prediction_log = []
    if model is not None:
        remaining = [row for row in candidates if row["candidate_id"] not in roles]
        if remaining and len(selected) < limit:
            features, descriptors = _candidate_features(search, remaining)
            predictions, disagreement = model.predict(
                features, oracle_protocol=search.oracle_protocol
            )
            virtual = list(archive_utilities)
            available = list(range(len(remaining)))
            while available and len(selected) < limit:
                gains = predicted_archive_gains(predictions, virtual, k=config.archive_k)
                position = max(
                    available,
                    key=lambda i: (gains[i], predictions[i], remaining[i]["candidate_id"]),
                )
                row = remaining[position]
                selected.append(row)
                roles[row["candidate_id"]] = "predicted_archive_gain"
                virtual.append(float(predictions[position]))
                available.remove(position)
            prediction_log = [
                {
                    **asdict(descriptor),
                    "predicted_utility": float(predictions[index]),
                    "bootstrap_disagreement": float(disagreement[index]),
                    "initial_predicted_archive_gain": float(
                        predicted_archive_gains(predictions, archive_utilities, k=config.archive_k)[
                            index
                        ]
                    ),
                    "uncertainty_used_for_selection": False,
                }
                for index, descriptor in enumerate(descriptors)
            ]
    if len(selected) < limit:
        fallback = _take_diverse(
            candidates,
            limit - len(selected),
            {row["candidate_id"] for row in selected},
            archive_endpoints,
        )
        selected.extend(fallback)
        roles.update({row["candidate_id"]: "quota_release_diversity" for row in fallback})
    selected = selected[:limit]
    return selected, {
        "schema_version": "dynamic_v22_parent_edit_allocation_v1",
        "policy": "pre_model_50_50_shallow_diverse"
        if model is None
        else "25_25_50_shallow_diverse_predicted_archive_gain",
        "task_kind": config.task_kind,
        "archive_k": config.archive_k,
        "available_by_channel": {
            channel: sum(row["provenance"]["planner_channel"] == channel for row in candidates)
            for channel in CHANNELS
        },
        "selected_by_role": {
            role: sum(value == role for value in roles.values())
            for role in sorted(set(roles.values()))
        },
        "selected_by_channel": {
            channel: sum(row["provenance"]["planner_channel"] == channel for row in selected)
            for channel in CHANNELS
        },
        "selected_ids": [row["candidate_id"] for row in selected],
        "candidate_roles": roles,
        "prediction_records": prediction_log,
        "model_sha256": None if model is None else model.payload["model_sha256"],
        "candidate_limit": limit,
        "runtime_information": "current_run_measured_outcomes_only",
        "uncertainty_used": False,
    }


class DynamicV22ProgramOptimizer(DynamicV21ProgramOptimizer):
    """v2.1 generation with chronological parent/edit query allocation."""

    def __init__(self, *args, policy_config: DynamicV22PolicyConfig | None = None, **kwargs):
        if policy_config is None:
            policy_config = getattr(type(self), "_restore_policy_config", None)
        if policy_config is None:
            raise ValueError("v2.2 optimizer requires its explicit task policy")
        super().__init__(*args, **kwargs)
        expected_direction = "minimize" if policy_config.task_kind == "t4" else "maximize"
        if self.config.score_direction != expected_direction:
            raise ValueError("v2.2 task kind and score direction disagree")
        self.policy_config = policy_config
        self.parent_edit_model = None
        self.parent_edit_fit_endpoint_count = 0
        self.parent_edit_fit_history = []
        # Parent choice is stable and independent of both proposal experts and
        # allocation. It is not claimed to reproduce v0's historically
        # interleaved single-RNG trajectory.
        self.parent_rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, 221, 0]))
        self.policy_decisions = 0
        self._v22_bootstrap_frontier_id = None

    def add_measured_program(self, record, *, receipt_id, score, static_score=None):
        bootstrap = record.get("provenance", {}).get("dynamic_v22_bootstrap")
        if bootstrap is not None:
            frontier_id = bootstrap["frontier_state_sha256"]
            if self._v22_bootstrap_frontier_id is None:
                self._v22_bootstrap_frontier_id = frontier_id
                self.parent_rng.bit_generator.state = bootstrap["parent_rng"]
            elif self._v22_bootstrap_frontier_id != frontier_id:
                raise ValueError("v2.2 bootstrap candidates came from different frontier states")
        return super().add_measured_program(
            record,
            receipt_id=receipt_id,
            score=score,
            static_score=static_score,
        )

    def propose_batch(self, eligibility):
        if self.pending is not None:
            raise ValueError("resolve the locked pending batch before generating another")
        self.parent_edit_model, self.parent_edit_fit_endpoint_count, fit = (
            fit_parent_edit_model_if_ready(
                self,
                config=self.policy_config,
                previous=self.parent_edit_model,
                previous_endpoint_count=self.parent_edit_fit_endpoint_count,
            )
        )
        self.parent_edit_fit_history.append(fit)
        archive_seen = {
            row["endpoint"] for row in self.observations.values()
        } | self.failed_endpoints
        original_rng = self.rng
        self.rng = self.parent_rng
        try:
            parent_schedule = [self._parent() for _ in range(self.config.attempts_per_batch)]
        finally:
            self.rng = original_rng
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
                    self.allocator_state["channels"][channel]["cross_channel_duplicates"] += 1
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
        selected, allocation = allocate_parent_edits(
            self,
            merged,
            limit=self.config.candidates_per_batch,
            config=self.policy_config,
            model=self.parent_edit_model,
        )
        for candidate in selected:
            channel = candidate["provenance"]["planner_channel"]
            self.allocator_state["channels"][channel]["selected_for_oracle"] += 1
        self.allocator_state["allocation_decisions"] += 1
        self.policy_decisions += 1
        body = {
            "schema_version": "dynamic_program_batch_v22",
            "batch_index": self.batches,
            "configuration": asdict(self.config),
            "policy_configuration": asdict(self.policy_config),
            "candidates": selected,
            "attempts": all_attempts,
            "source_group": self.source_group,
            "oracle_protocol": self.oracle_protocol,
            "eligible_pool": {
                "candidates": merged,
                "pool_id": identity([row["candidate_id"] for row in merged]),
            },
            "allocation": allocation,
            "parent_edit_model_update": fit,
            "parent_schedule": {
                "entry_ids": [entry["entry_id"] for entry, _ in parent_schedule],
                "rng_stream": "independent_dynamic_v22_parent_rng",
                "exact_v0_trajectory_claimed": False,
            },
            "allocator_state_after_proposal": json.loads(json.dumps(self.allocator_state)),
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

    def snapshot(self, *, include_history=True):
        snapshot = super().snapshot(include_history=include_history)
        body = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
        body["dynamic_v22"] = {
            "schema_version": "dynamic_v22_policy_state_v1",
            "policy_config": asdict(self.policy_config),
            "model": None if self.parent_edit_model is None else self.parent_edit_model.payload,
            "fit_endpoint_count": self.parent_edit_fit_endpoint_count,
            "fit_history": self.parent_edit_fit_history,
            "parent_rng": self.parent_rng.bit_generator.state,
            "policy_decisions": self.policy_decisions,
            "bootstrap_frontier_id": self._v22_bootstrap_frontier_id,
        }
        return json.loads(json.dumps({**body, "snapshot_id": identity(body)}))

    @classmethod
    def restore(cls, snapshot, *, hierarchy=None):
        state = snapshot.get("dynamic_v22")
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != "dynamic_v22_policy_state_v1"
        ):
            raise ValueError("v2.2 snapshot lacks candidate-policy state")
        policy_config = DynamicV22PolicyConfig(**state["policy_config"])
        # DynamicV21.restore instantiates ``cls``. Expose the required policy only
        # for that exact call without altering the shared v2.1 constructor.
        cls._restore_policy_config = policy_config
        try:
            result = super().restore(snapshot, hierarchy=hierarchy)
        finally:
            del cls._restore_policy_config
        result.parent_edit_model = (
            None if state["model"] is None else ParentEditModel(state["model"])
        )
        result.parent_edit_fit_endpoint_count = int(state["fit_endpoint_count"])
        result.parent_edit_fit_history = json.loads(json.dumps(state["fit_history"]))
        result.parent_rng.bit_generator.state = state["parent_rng"]
        result.policy_decisions = int(state["policy_decisions"])
        result._v22_bootstrap_frontier_id = state["bootstrap_frontier_id"]
        return result


def _cold_pre_model_allocation(candidates: list[dict], *, limit: int) -> tuple[list[dict], dict]:
    selected = _take_shallow_spine(candidates, math.ceil(limit / 2), set())
    diverse = _take_diverse(
        candidates,
        limit - len(selected),
        {row["candidate_id"] for row in selected},
        [],
    )
    selected.extend(diverse)
    if len(selected) < min(limit, len(candidates)):
        selected.extend(
            _take_diverse(
                candidates,
                min(limit, len(candidates)) - len(selected),
                {row["candidate_id"] for row in selected},
                [],
            )
        )
    return selected, {
        "schema_version": "dynamic_v22_parent_edit_allocation_v1",
        "policy": "cold_pre_model_50_50_shallow_diverse",
        "selected_ids": [row["candidate_id"] for row in selected],
        "candidate_limit": limit,
        "model_sha256": None,
        "runtime_information": "zero_task_outcomes_available",
    }


def initial_dynamic_program_batch_v22(
    source,
    entries,
    config,
    *,
    source_group,
    oracle_protocol,
    eligibility,
    policy_config: DynamicV22PolicyConfig,
    frontier_state=None,
    delta=None,
    broad_sampler=None,
):
    """Build one persistent-frontier T4 wave or pooled PMO cold batch."""
    if entries or broad_sampler is not None:
        raise ValueError("v2.2 initialization requires an empty route archive and no reference")
    if config.max_primitives != 32 or config.max_blocks != 8:
        raise ValueError("v2.2 initialization changed complete-program work support")
    if policy_config.task_kind == "pmo":
        if frontier_state is not None or delta is not None:
            raise ValueError("PMO must not inherit the T4 feasibility frontier")
        v21 = initial_dynamic_program_batch_v21(
            source,
            (),
            config,
            source_group=source_group,
            oracle_protocol=oracle_protocol,
            eligibility=eligibility,
        )
        merged = v21["proposal_pool"]["candidates"]
        if not merged:
            return {
                **v21,
                "schema_version": "initial_dynamic_program_batch_v22",
                "policy_configuration": asdict(policy_config),
            }
        generation_candidate_ids = [row["candidate_id"] for row in merged]
        old_bootstrap = v21["candidates"][0]["provenance"]["dynamic_v21_bootstrap"]
        selected, allocation = _cold_pre_model_allocation(merged, limit=config.candidates_per_batch)
        bootstrap = json.loads(json.dumps(old_bootstrap))
        for candidate in v21["candidates"]:
            channel = candidate["provenance"]["planner_channel"]
            bootstrap["allocator_state"]["channels"][channel]["selected_for_oracle"] -= 1
        for candidate in selected:
            channel = candidate["provenance"]["planner_channel"]
            bootstrap["allocator_state"]["channels"][channel]["selected_for_oracle"] += 1
            candidate["provenance"]["dynamic_v21_bootstrap"] = bootstrap
            candidate["provenance"]["dynamic_v22_bootstrap"] = {
                "frontier_state_sha256": identity({"pmo_pool": bootstrap["pool_id"]}),
                "parent_rng": np.random.default_rng(
                    np.random.SeedSequence([config.seed, 221, 0])
                ).bit_generator.state,
            }
            candidate.pop("candidate_id", None)
            candidate["candidate_id"] = identity(candidate)
        allocation["selected_ids"] = [row["candidate_id"] for row in selected]
        body = {
            **{
                key: value
                for key, value in v21.items()
                if key not in ("batch_id", "candidates", "allocation")
            },
            "schema_version": "initial_dynamic_program_batch_v22",
            "policy_configuration": asdict(policy_config),
            "candidates": selected,
            "allocation": allocation,
            "proposal_pool": {
                **v21["proposal_pool"],
                "generation_candidate_ids": generation_candidate_ids,
            },
        }
        return {**body, "batch_id": identity(body)}
    if delta not in (0.4, 0.6):
        raise ValueError("T4 v2.2 initialization requires delta")
    state = frontier_state or initial_frontier_state(seed=config.seed)
    wave = advance_t4_feasibility_frontier(
        source,
        config=policy_config,
        state=state,
        source_group=source_group,
        oracle_protocol=oracle_protocol,
        eligibility=eligibility,
        delta=delta,
        attempts_per_channel=config.attempts_per_batch,
    )
    merged = wave["eligible_candidates"]
    selected, allocation = _cold_pre_model_allocation(merged, limit=config.candidates_per_batch)
    allocator = initial_allocator_state(score_direction=config.score_direction)
    for attempt in wave["attempts"]:
        status = attempt["status"]
        if status == "eligible":
            _attempt_status(allocator, attempt["planner_channel"], "eligible")
        elif status == "frontier":
            _attempt_status(allocator, attempt["planner_channel"], "ineligible")
        elif status in ("duplicate_endpoint", "repeated_attempt_key"):
            _attempt_status(allocator, attempt["planner_channel"], "duplicate")
        elif status == "execution_rejected":
            _attempt_status(allocator, attempt["planner_channel"], status)
    for candidate in selected:
        allocator["channels"][candidate["provenance"]["planner_channel"]][
            "selected_for_oracle"
        ] += 1
    generation_candidate_ids = [row["candidate_id"] for row in merged]
    pool_id = identity(generation_candidate_ids)
    v21_bootstrap = {
        "pool_id": pool_id,
        "allocator_state": allocator,
        "shallow_rng": wave["frontier_state"]["shallow_rng"],
        "structured_rng": wave["frontier_state"]["structured_rng"],
        "arbitration_rng": np.random.default_rng(
            np.random.SeedSequence([config.seed, 211, 0])
        ).bit_generator.state,
    }
    v22_bootstrap = {
        "frontier_state_sha256": wave["frontier_state"]["state_sha256"],
        "parent_rng": np.random.default_rng(
            np.random.SeedSequence([config.seed, 221, 0])
        ).bit_generator.state,
    }
    for candidate in selected:
        candidate["provenance"]["dynamic_v21_bootstrap"] = v21_bootstrap
        candidate["provenance"]["dynamic_v22_bootstrap"] = v22_bootstrap
        candidate.pop("candidate_id", None)
        candidate["candidate_id"] = identity(candidate)
    allocation["selected_ids"] = [row["candidate_id"] for row in selected]
    body = {
        "schema_version": "initial_dynamic_program_batch_v22",
        "source_group": source_group,
        "oracle_protocol": oracle_protocol,
        "initial_route_archive": [],
        "source_library_rows_loaded": 0,
        "policy_configuration": asdict(policy_config),
        "candidates": selected,
        "attempts": wave["attempts"],
        "proposal_pool": {
            "pool_id": pool_id,
            "generation_candidate_ids": generation_candidate_ids,
            "candidates": merged,
        },
        "allocation": allocation,
        "frontier_state": wave["frontier_state"],
        "frontier_summary": wave["frontier_summary"],
        "new_oracle_calls": 0,
    }
    return {**body, "batch_id": identity(body)}
