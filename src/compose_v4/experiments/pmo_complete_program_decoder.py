"""Teacher-free PMO complete-program beam decoding and locked evaluation.

Generation consumes only a source graph, its frozen held-family fold, and
numeric fold checkpoints.  Teacher routes enter only through
``evaluate_candidate_lock`` after the candidate payload has been sealed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_dependency_region_policy_comparison import (
    CHECKPOINT_SCHEMA as DEPENDENCY_CHECKPOINT_SCHEMA,
)
from compose_v4.experiments.pmo_dependency_region_policy_comparison import (
    HEAD_SUPPORT,
    POLICIES,
)
from compose_v4.experiments.pmo_dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
)
from compose_v4.experiments.pmo_legal_action_policy import (
    CHECKPOINT_SCHEMA as LEGAL_CHECKPOINT_SCHEMA,
)
from compose_v4.experiments.pmo_legal_action_policy import (
    RULES,
    LegalSuccessor,
    action_features,
    enumerate_rule_successors,
    score_features,
)
from compose_v4.experiments.route_proposal_quality import transformation_equivalent
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "pmo_complete_program_decoder_v1"
SOURCE_MANIFEST_SCHEMA = "pmo_complete_program_decoder_sources_v1"
CANDIDATE_LOCK_SCHEMA = "pmo_complete_program_decoder_candidate_lock_v1"
REPORT_SCHEMA = "pmo_complete_program_decoder_report_v1"
UNIFORM = "uniform_where_how"
LEARNED = "learned_where_how"
DECODERS = (UNIFORM, LEARNED)


@dataclass(frozen=True)
class DecoderConfig:
    beam_widths: tuple[int, ...] = (1, 8)
    snapshot_depths: tuple[int, ...] = (8, 16, 24, 32)
    output_cutoffs: tuple[int, ...] = (1, 5, 10, 32)
    successors_per_rule: int = 32
    maximum_primitives: int = 32
    maximum_components: int = 8
    maximum_active_atoms: int = 40

    def __post_init__(self) -> None:
        for name in (
            "beam_widths",
            "snapshot_depths",
            "output_cutoffs",
        ):
            values = getattr(self, name)
            if not values or any(
                type(value) is not int or value < 1 for value in values
            ):
                raise ValueError(f"{name} must contain positive integers")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        for name in (
            "successors_per_rule",
            "maximum_primitives",
            "maximum_components",
            "maximum_active_atoms",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.snapshot_depths[-1] != self.maximum_primitives:
            raise ValueError("the final snapshot must equal the primitive horizon")
        if self.output_cutoffs[-1] > 32:
            raise ValueError("the declared candidate lock supports at most 32 outputs")


@dataclass(frozen=True)
class _Prefix:
    graph: object
    states: tuple[dict, ...]
    actions: tuple[dict, ...]
    score: float


def _fold_rows(runtime: dict) -> dict[int, dict]:
    if runtime.get("new_oracle_calls") != 0:
        raise ValueError("runtime checkpoint unexpectedly records oracle calls")
    rows = {}
    for item in runtime.get("folds", ()):
        row = item.get("checkpoint", item)
        fold = int(item.get("fold", row.get("fold_index", -1)))
        if fold in rows:
            raise ValueError(f"duplicate runtime checkpoint fold: {fold}")
        rows[fold] = row
    if sorted(rows) != [0, 1, 2]:
        raise ValueError("runtime checkpoint must contain folds 0, 1 and 2")
    return rows


def validate_fold_checkpoints(dependency_runtime: dict, legal_runtime: dict) -> None:
    if dependency_runtime.get("schema_version") != DEPENDENCY_CHECKPOINT_SCHEMA:
        raise ValueError("dependency-region runtime checkpoint schema mismatch")
    if legal_runtime.get("schema_version") != "pmo_legal_action_fold_checkpoints_v1":
        raise ValueError("legal-action runtime checkpoint schema mismatch")
    dependency = _fold_rows(dependency_runtime)
    legal = _fold_rows(legal_runtime)
    for fold in range(3):
        sequence = dependency[fold]
        if sequence.get("fold_index") != fold:
            raise ValueError("dependency checkpoint fold identity mismatch")
        policy = sequence.get("policies", {}).get(POLICIES[0])
        if not isinstance(policy, dict) or policy.get("conditioned") is not False:
            raise ValueError("balanced generic marginal checkpoint is absent")
        heads = policy.get("heads", {})
        for head in ("primitive_rule", "program_control"):
            model = heads.get(head)
            if not isinstance(model, dict):
                raise TypeError(f"balanced marginal lacks {head}")
            if tuple(model.get("classes", ())) != tuple(HEAD_SUPPORT[head]):
                raise ValueError(f"balanced marginal {head} support changed")
            probabilities = np.asarray(model.get("marginal_probabilities"), dtype=float)
            if (
                probabilities.shape != (len(HEAD_SUPPORT[head]),)
                or np.any(probabilities <= 0)
                or not np.isclose(probabilities.sum(), 1.0)
            ):
                raise ValueError(f"invalid balanced marginal {head} probabilities")
        if legal[fold].get("schema_version") != LEGAL_CHECKPOINT_SCHEMA:
            raise ValueError("legal-action fold checkpoint schema mismatch")


def source_manifest_from_panels(panel_corpus: dict, *, split_identity: str) -> dict:
    """Strip teachers and task labels, retaining one source/fold decode case."""

    split = panel_corpus.get("split", {})
    if split.get("split_identity") != split_identity:
        raise ValueError("source-panel split identity changed")
    panels = panel_corpus.get("source_conditioned_panels")
    if not isinstance(panels, list) or len(panels) != 18:
        raise ValueError("the PMO decoder requires the frozen 18 source panels")
    cases = {}
    for panel in panels:
        fold = int(panel["test_fold"])
        state = panel["source_state"]
        state_identity = identity(state)
        key = (fold, state_identity)
        cases.setdefault(
            key,
            {
                "source_case_id": identity(
                    {
                        "schema_version": "pmo_decoder_source_case_v1",
                        "fold": fold,
                        "state_identity": state_identity,
                    }
                ),
                "fold": fold,
                "source_state": state,
                "source_state_identity": state_identity,
            },
        )
    rows = sorted(cases.values(), key=lambda row: (row["fold"], row["source_case_id"]))
    if len(rows) != 9:
        raise ValueError("the 18 panels must collapse to nine source/fold decode cases")
    payload = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "split_identity": split_identity,
        "source_cases": rows,
        "panel_count": len(panels),
        "source_case_count": len(rows),
        "teacher_fields_present": False,
        "task_identity_present": False,
        "new_oracle_calls": 0,
    }
    forbidden = ("task_family", "teacher_endpoint_states", "teacher_trace_identities")
    serialized = str(payload)
    if any(value in serialized for value in forbidden):
        raise RuntimeError("source-only manifest retained a forbidden field")
    return payload


def _probability_head(sequence_checkpoint: dict, head: str) -> dict[str, float]:
    model = sequence_checkpoint["policies"][POLICIES[0]]["heads"][head]
    return dict(
        zip(
            model["classes"],
            (float(value) for value in model["marginal_probabilities"]),
            strict=True,
        )
    )


def _log_softmax(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("legal-action scores must be finite and nonempty")
    maximum = float(array.max())
    normalizer = maximum + math.log(float(np.exp(array - maximum).sum()))
    return array - normalizer


def _trace_identity(prefix: _Prefix) -> str:
    return identity(
        {
            "schema_version": "pmo_decoder_canonical_route_v1",
            "rules": [action["executor_rule"] for action in prefix.actions],
            "states": [
                canonical_state_key(decode_state(state)) for state in prefix.states
            ],
        }
    )


def _prefix_order(prefix: _Prefix) -> tuple:
    return (-prefix.score, canonical_state_key(prefix.graph), _trace_identity(prefix))


def _candidate_row(prefix: _Prefix, component_count: int) -> dict:
    endpoint_state = prefix.states[-1]
    endpoint_key = canonical_state_key(prefix.graph)
    return {
        "candidate_id": identity(
            {
                "schema_version": "pmo_decoder_candidate_v1",
                "route_identity": _trace_identity(prefix),
                "score": prefix.score,
            }
        ),
        "route_identity": _trace_identity(prefix),
        "score": prefix.score,
        "primitive_count": len(prefix.actions),
        "component_count": component_count,
        "endpoint_key": endpoint_key,
        "endpoint_state": endpoint_state,
        "trace": {"states": list(prefix.states), "actions": list(prefix.actions)},
    }


def _ranked_candidates(rows: list[dict], maximum: int) -> dict:
    routes = {}
    endpoints = {}
    for row in sorted(rows, key=lambda value: (-value["score"], value["candidate_id"])):
        routes.setdefault(row["route_identity"], row)
        endpoints.setdefault(row["endpoint_key"], row)
    return {
        "ranked_routes": list(routes.values())[:maximum],
        "ranked_endpoints": list(endpoints.values())[:maximum],
    }


def decode_source(
    source,
    sequence_checkpoint: dict,
    legal_checkpoint: dict,
    *,
    decoder: str,
    beam_width: int,
    config: DecoderConfig,
    successor_enumerator: Callable[[object, str], tuple[LegalSuccessor, ...]] = (
        enumerate_rule_successors
    ),
    learned_scorer: Callable[[object, LegalSuccessor, dict], float] | None = None,
) -> dict:
    """Decode one autonomous complete-program beam without teacher information."""

    if decoder not in DECODERS:
        raise ValueError(f"unknown decoder: {decoder}")
    if beam_width not in config.beam_widths:
        raise ValueError("beam width is outside the declared panel")
    if sequence_checkpoint.get("policies", {}).get(POLICIES[0]) is None:
        raise ValueError("decoder requires the balanced generic marginal")
    if legal_checkpoint.get("schema_version") != LEGAL_CHECKPOINT_SCHEMA:
        raise ValueError("decoder requires a legal-action fold checkpoint")
    rule_probabilities = _probability_head(sequence_checkpoint, "primitive_rule")
    stop_probabilities = _probability_head(sequence_checkpoint, "program_control")
    if learned_scorer is None:
        learned_scorer = lambda graph, candidate, checkpoint: score_features(
            checkpoint, action_features(graph, candidate)
        )
    source_key = canonical_state_key(source)
    live = (_Prefix(source, (encode_state(source),), (), 0.0),)
    completed: list[dict] = []
    snapshots = {}
    telemetry = {
        "prefixes_expanded": 0,
        "rule_fibers_enumerated": 0,
        "legal_successors_enumerated": 0,
        "legal_successors_retained": 0,
        "prefix_state_duplicates": 0,
        "component_budget_abstentions": 0,
        "active_atom_abstentions": 0,
        "self_endpoint_abstentions": 0,
        "completed_programs": 0,
        "exact_execution_failures": 0,
    }
    component_config = DependencyRegionConfig(
        runtime_maximum_primitives=config.maximum_primitives,
        runtime_maximum_components=config.maximum_components,
    )
    for depth in range(1, config.maximum_primitives + 1):
        expanded = []
        for prefix in live:
            telemetry["prefixes_expanded"] += 1
            fibers = {}
            for rule in RULES:
                candidates = tuple(successor_enumerator(prefix.graph, rule))
                telemetry["rule_fibers_enumerated"] += 1
                telemetry["legal_successors_enumerated"] += len(candidates)
                if candidates:
                    fibers[rule] = candidates
            available_mass = sum(rule_probabilities[rule] for rule in fibers)
            if not fibers or available_mass <= 0:
                continue
            for rule in RULES:
                candidates = fibers.get(rule)
                if not candidates:
                    continue
                if decoder == UNIFORM:
                    action_logs = np.full(len(candidates), -math.log(len(candidates)))
                else:
                    action_logs = _log_softmax(
                        learned_scorer(prefix.graph, candidate, legal_checkpoint)
                        for candidate in candidates
                    )
                ordered = sorted(
                    range(len(candidates)),
                    key=lambda index: (
                        -float(action_logs[index]),
                        candidates[index].successor_key,
                    ),
                )[: config.successors_per_rule]
                telemetry["legal_successors_retained"] += len(ordered)
                rule_log = math.log(rule_probabilities[rule] / available_mass)
                for index in ordered:
                    candidate = candidates[index]
                    if candidate.successor.n_real_atoms > config.maximum_active_atoms:
                        telemetry["active_atom_abstentions"] += 1
                        continue
                    expanded.append(
                        _Prefix(
                            candidate.successor,
                            (*prefix.states, encode_state(candidate.successor)),
                            (*prefix.actions, candidate.action_record),
                            prefix.score + rule_log + float(action_logs[index]),
                        )
                    )
        best_by_state = {}
        for prefix in sorted(expanded, key=_prefix_order):
            key = canonical_state_key(prefix.graph)
            if key in best_by_state:
                telemetry["prefix_state_duplicates"] += 1
                continue
            best_by_state[key] = prefix
        next_live = []
        for prefix in best_by_state.values():
            if canonical_state_key(prefix.graph) == source_key:
                telemetry["self_endpoint_abstentions"] += 1
                continue
            try:
                program = dependency_region_program(
                    prefix.states,
                    prefix.actions,
                    config=component_config,
                )
            except ValueError:
                telemetry["exact_execution_failures"] += 1
                continue
            if not program["exact_replay"]:
                telemetry["exact_execution_failures"] += 1
                continue
            if not program["component_budget_supported"]:
                telemetry["component_budget_abstentions"] += 1
            else:
                stopped = _Prefix(
                    prefix.graph,
                    prefix.states,
                    prefix.actions,
                    prefix.score + math.log(stop_probabilities["stop"]),
                )
                completed.append(_candidate_row(stopped, program["component_count"]))
                telemetry["completed_programs"] += 1
            if depth < config.maximum_primitives:
                next_live.append(
                    _Prefix(
                        prefix.graph,
                        prefix.states,
                        prefix.actions,
                        prefix.score + math.log(stop_probabilities["continue"]),
                    )
                )
        live = tuple(sorted(next_live, key=_prefix_order)[:beam_width])
        if depth in config.snapshot_depths:
            snapshots[str(depth)] = _ranked_candidates(
                completed, config.output_cutoffs[-1]
            )
        if not live and depth < config.maximum_primitives:
            for future in config.snapshot_depths:
                if future > depth:
                    snapshots[str(future)] = _ranked_candidates(
                        completed, config.output_cutoffs[-1]
                    )
            break
    if set(snapshots) != {str(value) for value in config.snapshot_depths}:
        raise RuntimeError("decoder failed to produce every declared depth snapshot")
    exact_successes = (
        telemetry["completed_programs"] + telemetry["component_budget_abstentions"]
    )
    exact_attempts = exact_successes + telemetry["exact_execution_failures"]
    telemetry["exact_execution_precision"] = (
        exact_successes / exact_attempts if exact_attempts else None
    )
    return {
        "decoder": decoder,
        "beam_width": beam_width,
        "snapshots": snapshots,
        "telemetry": telemetry,
    }


def build_candidate_lock(
    source_manifest: dict,
    dependency_runtime: dict,
    legal_runtime: dict,
    config: DecoderConfig,
) -> dict:
    """Generate both arms from a teacher-free source manifest."""

    if source_manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("source manifest schema mismatch")
    if source_manifest.get("teacher_fields_present") is not False:
        raise ValueError("source manifest is not teacher-free")
    if source_manifest.get("task_identity_present") is not False:
        raise ValueError("source manifest contains task identity")
    validate_fold_checkpoints(dependency_runtime, legal_runtime)
    dependency = _fold_rows(dependency_runtime)
    legal = _fold_rows(legal_runtime)
    cases = []
    for source_case in source_manifest["source_cases"]:
        fold = int(source_case["fold"])
        source = decode_state(source_case["source_state"])
        decoders = []
        for decoder in DECODERS:
            for width in config.beam_widths:
                decoders.append(
                    decode_source(
                        source,
                        dependency[fold],
                        legal[fold],
                        decoder=decoder,
                        beam_width=width,
                        config=config,
                    )
                )
        cases.append(
            {
                "source_case_id": source_case["source_case_id"],
                "fold": fold,
                "source_state": source_case["source_state"],
                "source_state_identity": source_case["source_state_identity"],
                "decoders": decoders,
            }
        )
    payload = {
        "schema_version": CANDIDATE_LOCK_SCHEMA,
        "split_identity": source_manifest["split_identity"],
        "configuration": {
            "beam_widths": list(config.beam_widths),
            "snapshot_depths": list(config.snapshot_depths),
            "output_cutoffs": list(config.output_cutoffs),
            "successors_per_rule": config.successors_per_rule,
            "maximum_primitives": config.maximum_primitives,
            "maximum_components": config.maximum_components,
            "maximum_active_atoms": config.maximum_active_atoms,
        },
        "sequence_policy": "balanced_generic_marginal_rule_and_program_control",
        "decoder_arms": list(DECODERS),
        "source_cases": cases,
        "teacher_fields_present": False,
        "task_identity_present": False,
        "new_oracle_calls": 0,
    }
    return payload


def seal_candidate_lock(payload: dict) -> dict:
    if payload.get("schema_version") != CANDIDATE_LOCK_SCHEMA:
        raise ValueError("candidate payload schema mismatch")
    return {"payload": payload, "payload_sha256": identity(payload)}


def validate_candidate_lock(sealed_lock: dict) -> dict:
    """Return the candidate payload only when its deterministic seal is intact."""

    payload = sealed_lock.get("payload")
    if not isinstance(payload, dict) or sealed_lock.get("payload_sha256") != identity(
        payload
    ):
        raise ValueError("candidate lock is not sealed or was modified")
    if payload.get("schema_version") != CANDIDATE_LOCK_SCHEMA:
        raise ValueError("candidate lock schema mismatch")
    if (
        payload.get("teacher_fields_present") is not False
        or payload.get("task_identity_present") is not False
    ):
        raise ValueError("candidate generation was not teacher/task blind")
    if payload.get("new_oracle_calls") != 0:
        raise ValueError("candidate lock contains oracle activity")
    return payload


def _canonical_route(trace: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(action["executor_rule"] for action in trace["actions"]),
        tuple(canonical_state_key(decode_state(state)) for state in trace["states"]),
    )


def _first_rank(rows: list[dict], predicate: Callable[[dict], bool]) -> int | None:
    return next((index for index, row in enumerate(rows, 1) if predicate(row)), None)


def _panel_outcome(
    panel: dict,
    routes: list[dict],
    snapshot: dict,
    cutoffs: tuple[int, ...],
) -> dict:
    source = decode_state(panel["source_state"])
    route_targets = {
        _canonical_route({"states": row["states"], "actions": row["actions"]})
        for row in routes
    }
    exact_endpoint_graphs = [decode_state(row["states"][-1]) for row in routes]
    endpoint_graphs = []
    for graph in exact_endpoint_graphs:
        if not any(
            transformation_equivalent(source, graph, representative)
            for representative in endpoint_graphs
        ):
            endpoint_graphs.append(graph)
    endpoint_keys = {canonical_state_key(graph) for graph in exact_endpoint_graphs}
    route_rows = snapshot["ranked_routes"]
    endpoint_rows = snapshot["ranked_endpoints"]
    exact_route_ranks = [
        _first_rank(
            route_rows,
            lambda candidate, target=target: _canonical_route(candidate["trace"])
            == target,
        )
        for target in sorted(route_targets)
    ]
    exact_endpoint_ranks = [
        _first_rank(
            endpoint_rows,
            lambda candidate, target=target: candidate["endpoint_key"] == target,
        )
        for target in sorted(endpoint_keys)
    ]
    transformation_ranks = [
        _first_rank(
            endpoint_rows,
            lambda candidate, target=target: transformation_equivalent(
                source, decode_state(candidate["endpoint_state"]), target
            ),
        )
        for target in endpoint_graphs
    ]

    def recall(ranks: list[int | None], cutoff: int) -> float:
        return sum(rank is not None and rank <= cutoff for rank in ranks) / max(
            1, len(ranks)
        )

    def mrr(ranks: list[int | None], cutoff: int) -> float:
        return sum(
            1 / rank for rank in ranks if rank is not None and rank <= cutoff
        ) / max(1, len(ranks))

    result = {
        "task_family": panel["task_family"],
        "lineage_identity": panel["lineage_identity"],
        "fold": int(panel["test_fold"]),
        "teacher_routes": len(routes),
        "runtime_supported_routes": sum(
            row["dependency_region_program"]["complete_representation_supported"]
            for row in routes
        ),
        "emitted_routes": len(route_rows),
        "emitted_endpoints": len(endpoint_rows),
        "cutoffs": {},
    }
    for cutoff in cutoffs:
        emitted_routes = route_rows[:cutoff]
        emitted = endpoint_rows[:cutoff]
        matching_routes = sum(
            _canonical_route(row["trace"]) in route_targets for row in emitted_routes
        )
        matching_exact = sum(row["endpoint_key"] in endpoint_keys for row in emitted)
        matching_transform = sum(
            any(
                transformation_equivalent(
                    source, decode_state(candidate["endpoint_state"]), target
                )
                for target in endpoint_graphs
            )
            for candidate in emitted
        )
        result["cutoffs"][str(cutoff)] = {
            "exact_route_recall": recall(exact_route_ranks, cutoff),
            "exact_route_mrr": mrr(exact_route_ranks, cutoff),
            "exact_endpoint_recall": recall(exact_endpoint_ranks, cutoff),
            "radius2_transformation_recall": recall(transformation_ranks, cutoff),
            "exact_endpoint_mrr": mrr(exact_endpoint_ranks, cutoff),
            "radius2_transformation_mrr": mrr(transformation_ranks, cutoff),
            "exact_endpoint_any": float(
                any(
                    rank is not None and rank <= cutoff for rank in exact_endpoint_ranks
                )
            ),
            "radius2_transformation_any": float(
                any(
                    rank is not None and rank <= cutoff for rank in transformation_ranks
                )
            ),
            "exact_route_any": float(
                any(rank is not None and rank <= cutoff for rank in exact_route_ranks)
            ),
            "exact_route_precision_emitted": matching_routes
            / max(1, len(emitted_routes)),
            "exact_endpoint_precision_emitted": matching_exact / max(1, len(emitted)),
            "radius2_transformation_precision_emitted": matching_transform
            / max(1, len(emitted)),
        }
    return result


def _aggregate_panel_outcomes(rows: list[dict], cutoffs: tuple[int, ...]) -> dict:
    if not rows:
        raise ValueError("decoder evaluation requires panel outcomes")
    families = sorted({row["task_family"] for row in rows})
    per_family = {
        family: [row for row in rows if row["task_family"] == family]
        for family in families
    }

    def balanced(values: Callable[[dict], float]) -> float:
        return sum(
            sum(values(row) for row in per_family[family]) / len(per_family[family])
            for family in families
        ) / len(families)

    return {
        "panels": len(rows),
        "task_families": len(families),
        "source_coverage": sum(row["emitted_endpoints"] > 0 for row in rows)
        / len(rows),
        "unique_endpoint_yield_sum": sum(row["emitted_endpoints"] for row in rows),
        "mean_unique_endpoints_per_lineage": sum(
            row["emitted_endpoints"] for row in rows
        )
        / len(rows),
        "cutoffs": {
            str(cutoff): {
                metric: balanced(
                    lambda row, name=metric, active_cutoff=cutoff: row["cutoffs"][
                        str(active_cutoff)
                    ][name]
                )
                for metric in next(iter(rows))["cutoffs"][str(cutoff)]
            }
            for cutoff in cutoffs
        },
    }


def evaluate_candidate_lock(
    sealed_lock: dict,
    panel_corpus: dict,
    dependency_corpus: dict,
    config: DecoderConfig,
) -> dict:
    """Evaluate a previously sealed autonomous lock against held teachers."""

    payload = validate_candidate_lock(sealed_lock)
    expected_configuration = {
        "beam_widths": list(config.beam_widths),
        "snapshot_depths": list(config.snapshot_depths),
        "output_cutoffs": list(config.output_cutoffs),
        "successors_per_rule": config.successors_per_rule,
        "maximum_primitives": config.maximum_primitives,
        "maximum_components": config.maximum_components,
        "maximum_active_atoms": config.maximum_active_atoms,
    }
    if payload.get("configuration") != expected_configuration:
        raise ValueError("candidate lock decoder configuration mismatch")
    split_identity = payload["split_identity"]
    if panel_corpus.get("split", {}).get("split_identity") != split_identity:
        raise ValueError("panel split differs from candidate lock")
    if dependency_corpus.get("split", {}).get("split_identity") != split_identity:
        raise ValueError("dependency corpus split differs from candidate lock")
    cases = {
        (row["fold"], row["source_state_identity"]): row
        for row in payload["source_cases"]
    }
    routes = dependency_corpus["routes"]
    supported_routes = [
        route
        for route in routes
        if route["dependency_region_program"]["complete_representation_supported"]
    ]
    if len(routes) != 184 or len(supported_routes) != 106:
        raise ValueError("teacher route census differs from the declared 184/106 scope")
    routes_by_lineage = {}
    for route in routes:
        routes_by_lineage.setdefault(route["lineage_identity"], []).append(route)
    all_outcome_groups = {}
    runtime_outcome_groups = {}
    for panel in panel_corpus["source_conditioned_panels"]:
        key = (int(panel["test_fold"]), identity(panel["source_state"]))
        case = cases.get(key)
        if case is None:
            raise ValueError("candidate lock is missing a held source/fold case")
        teachers = routes_by_lineage.get(panel["lineage_identity"], [])
        if not teachers:
            raise ValueError("held panel has no teacher routes")
        runtime_teachers = [
            route
            for route in teachers
            if route["dependency_region_program"]["complete_representation_supported"]
        ]
        for decoded in case["decoders"]:
            for depth, snapshot in decoded["snapshots"].items():
                group = (decoded["decoder"], int(decoded["beam_width"]), int(depth))
                all_outcome_groups.setdefault(group, []).append(
                    _panel_outcome(panel, teachers, snapshot, config.output_cutoffs)
                )
                if runtime_teachers:
                    runtime_outcome_groups.setdefault(group, []).append(
                        _panel_outcome(
                            panel,
                            runtime_teachers,
                            snapshot,
                            config.output_cutoffs,
                        )
                    )
    evaluations = []
    for (decoder, width, depth), rows in sorted(all_outcome_groups.items()):
        runtime_rows = runtime_outcome_groups[(decoder, width, depth)]
        runtime_folds = {
            str(fold): _aggregate_panel_outcomes(
                [row for row in runtime_rows if row["fold"] == fold],
                config.output_cutoffs,
            )
            for fold in range(3)
        }
        evaluations.append(
            {
                "decoder": decoder,
                "beam_width": width,
                "depth": depth,
                "all_18_lineages": _aggregate_panel_outcomes(
                    rows, config.output_cutoffs
                ),
                "runtime_supported_8_lineages": _aggregate_panel_outcomes(
                    runtime_rows, config.output_cutoffs
                ),
                "runtime_supported_folds": runtime_folds,
                "per_lineage_all_routes": rows,
                "per_lineage_runtime_supported_routes": runtime_rows,
            }
        )
    primary = {
        row["decoder"]: row
        for row in evaluations
        if row["beam_width"] == 8 and row["depth"] == 32
    }
    if set(primary) != set(DECODERS):
        raise RuntimeError("primary decoder comparison is incomplete")
    cutoff = str(config.output_cutoffs[-1])
    uniform = primary[UNIFORM]["runtime_supported_8_lineages"]["cutoffs"][cutoff]
    learned = primary[LEARNED]["runtime_supported_8_lineages"]["cutoffs"][cutoff]
    improved_folds = sum(
        primary[LEARNED]["runtime_supported_folds"][str(fold)]["cutoffs"][cutoff][
            "radius2_transformation_recall"
        ]
        > primary[UNIFORM]["runtime_supported_folds"][str(fold)]["cutoffs"][cutoff][
            "radius2_transformation_recall"
        ]
        for fold in range(3)
    )
    scientific = {
        "nonzero_autonomous_radius2_recovery": learned["radius2_transformation_recall"]
        > 0,
        "radius2_recall_strictly_above_uniform": learned[
            "radius2_transformation_recall"
        ]
        > uniform["radius2_transformation_recall"],
        "radius2_mrr_strictly_above_uniform": learned["radius2_transformation_mrr"]
        > uniform["radius2_transformation_mrr"],
        "exact_endpoint_recall_not_lower": learned["exact_endpoint_recall"]
        >= uniform["exact_endpoint_recall"],
        "improvement_in_at_least_two_folds": improved_folds >= 2,
    }
    source_cases = payload["source_cases"]
    locked_candidates = [
        candidate
        for source_case in source_cases
        for decoded in source_case["decoders"]
        for snapshot in decoded["snapshots"].values()
        for ranking in ("ranked_routes", "ranked_endpoints")
        for candidate in snapshot[ranking]
    ]
    generation_summaries = []
    for decoder in DECODERS:
        for width in config.beam_widths:
            decoded_rows = [
                next(
                    row
                    for row in source_case["decoders"]
                    if row["decoder"] == decoder and row["beam_width"] == width
                )
                for source_case in source_cases
            ]
            generation_summaries.append(
                {
                    "decoder": decoder,
                    "beam_width": width,
                    "source_cases": len(decoded_rows),
                    "snapshots": {
                        str(depth): {
                            "source_coverage": sum(
                                bool(row["snapshots"][str(depth)]["ranked_endpoints"])
                                for row in decoded_rows
                            )
                            / len(decoded_rows),
                            "unique_endpoint_yield_sum": sum(
                                len(row["snapshots"][str(depth)]["ranked_endpoints"])
                                for row in decoded_rows
                            ),
                        }
                        for depth in config.snapshot_depths
                    },
                    "telemetry_sum": {
                        name: sum(row["telemetry"][name] for row in decoded_rows)
                        for name in (
                            "prefixes_expanded",
                            "rule_fibers_enumerated",
                            "legal_successors_enumerated",
                            "legal_successors_retained",
                            "prefix_state_duplicates",
                            "component_budget_abstentions",
                            "active_atom_abstentions",
                            "self_endpoint_abstentions",
                            "completed_programs",
                            "exact_execution_failures",
                        )
                    },
                }
            )
    engineering = {
        "zero_oracle_calls": payload["new_oracle_calls"] == 0,
        "teacher_and_task_blind_generation": not payload["teacher_fields_present"]
        and not payload["task_identity_present"],
        "all_source_cases_present": len(source_cases) == 9,
        "all_held_lineages_reported": all(
            row["all_18_lineages"]["panels"] == 18 for row in evaluations
        ),
        "runtime_supported_scope_reported": all(
            row["runtime_supported_8_lineages"]["panels"] == 8 for row in evaluations
        ),
        "declared_support_enforced": all(
            candidate["primitive_count"] <= config.maximum_primitives
            and candidate["component_count"] <= config.maximum_components
            and decode_state(candidate["endpoint_state"]).n_real_atoms
            <= config.maximum_active_atoms
            for candidate in locked_candidates
        ),
        "source_coverage_one": all(
            row["all_18_lineages"]["source_coverage"] == 1.0 for row in evaluations
        ),
        "exact_execution_precision_one": all(
            decoded["telemetry"]["exact_execution_precision"] == 1.0
            for case in source_cases
            for decoded in case["decoders"]
        ),
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "candidate_lock_payload_sha256": sealed_lock["payload_sha256"],
        "split_identity": split_identity,
        "teacher_census": {
            "lineages": len(routes_by_lineage),
            "routes": len(routes),
            "runtime_supported_lineages": sum(
                any(
                    route["dependency_region_program"][
                        "complete_representation_supported"
                    ]
                    for route in lineage_routes
                )
                for lineage_routes in routes_by_lineage.values()
            ),
            "runtime_supported_routes": len(supported_routes),
        },
        "evaluations": evaluations,
        "generation_summaries": generation_summaries,
        "engineering_gate": {**engineering, "passed": all(engineering.values())},
        "scientific_where_how_gate": {
            **scientific,
            "passed": all(scientific.values()),
        },
        "exact_route_recovery_claim": learned["exact_route_recall"] > 0,
        "scored_pmo_pilot_authorized": False,
        "new_oracle_calls": 0,
    }


__all__ = [
    "CANDIDATE_LOCK_SCHEMA",
    "DECODERS",
    "LEARNED",
    "REPORT_SCHEMA",
    "SCHEMA",
    "SOURCE_MANIFEST_SCHEMA",
    "UNIFORM",
    "DecoderConfig",
    "build_candidate_lock",
    "decode_source",
    "evaluate_candidate_lock",
    "seal_candidate_lock",
    "source_manifest_from_panels",
    "validate_candidate_lock",
    "validate_fold_checkpoints",
]
