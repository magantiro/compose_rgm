"""Joint, local structural-patch generation without a whole-patch vocabulary.

The learned object contains only weighted numeric densities.  At runtime the
unchanged Active8 executor supplies the generic legal construction-event domain.
The first event selects an address-free source region ``R``; a second event may
extend the same partial patch through a source or created role.  The completed
event sequence is converted to ``(H, alpha, D)`` by the existing structural
extractor and compiled by the existing exact structural realizer.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    MolecularGraph,
    is_element,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.generic_legal_action_policy import (
    RULES,
    LegalSuccessor,
    _candidate_actions,
    _record_parts,
    action_features,
    enumerate_legal_successors,
)
from compose_v4.control.structural_subgoal import (
    StructuralGoal,
    extract_structural_goal,
)
from compose_v4.control.structural_subgoal_policy import minimize_subgoal
from compose_v4.control.structural_subgoal_realizer import (
    RealizerConfig,
    realize_structural_goal,
)
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    SemanticAtomRestate,
    resolve_semantic_atom_restate_action,
)
from compose_v4.rewrite.semantic_atom_restate import (
    prepare_semantic_atom_restate_context,
)
from compose_v4.rewrite.trace_shard import decode_state, encode_state

CHECKPOINT_SCHEMA = "compositional_structural_subgoal_generator_v1"
DENSITY_SCHEMA = "weighted_diagonal_density_v1"
PATCH_SCHEMA = "generated_compositional_structural_patch_v1"
POLICY_UNIFORM = "uniform_joint_grammar"
POLICY_LEARNED = "balanced_joint_autoregressive"


@dataclass(frozen=True)
class WeightedDiagonalDensity:
    """A deterministic density fitted jointly over one complete feature vector."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    effective_weight: float

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        if (
            mean.ndim != 1
            or not len(mean)
            or scale.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
            or not math.isfinite(self.effective_weight)
            or self.effective_weight <= 0
        ):
            raise ValueError("invalid weighted diagonal density")

    def score(self, features: np.ndarray) -> float:
        values = np.asarray(features, dtype=float)
        if values.shape != (len(self.mean),) or not np.isfinite(values).all():
            raise ValueError("density feature dimension mismatch")
        standardized = (values - np.asarray(self.mean)) / np.asarray(self.scale)
        return -0.5 * float(np.mean(standardized * standardized))

    def payload(self) -> dict:
        return {
            "schema_version": DENSITY_SCHEMA,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "effective_weight": self.effective_weight,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> WeightedDiagonalDensity:
        if payload.get("schema_version") != DENSITY_SCHEMA:
            raise ValueError("weighted-density schema mismatch")
        return cls(
            tuple(map(float, payload["mean"])),
            tuple(map(float, payload["scale"])),
            float(payload["effective_weight"]),
        )


def fit_weighted_density(
    vectors: list[np.ndarray], weights: list[float], *, variance_floor: float
) -> WeightedDiagonalDensity:
    if not vectors or len(vectors) != len(weights) or variance_floor <= 0:
        raise ValueError("weighted-density fit inputs are invalid")
    matrix = np.stack(vectors).astype(float)
    weight = np.asarray(weights, dtype=float)
    if (
        matrix.ndim != 2
        or not np.isfinite(matrix).all()
        or not np.isfinite(weight).all()
        or np.any(weight <= 0)
    ):
        raise ValueError("weighted-density fit contains invalid values")
    weight /= weight.sum()
    mean = np.sum(matrix * weight[:, None], axis=0)
    variance = np.sum((matrix - mean) ** 2 * weight[:, None], axis=0)
    scale = np.maximum(np.sqrt(variance), variance_floor)
    return WeightedDiagonalDensity(
        tuple(map(float, mean)),
        tuple(map(float, scale)),
        float(1 / np.sum(weight * weight)),
    )


@dataclass(frozen=True)
class PatchTrainingEvent:
    """One split-first event used only while fitting numeric densities."""

    domain: str
    lineage: str
    route_id: str
    component_id: str
    component_events: int
    depth: int
    component_source: MolecularGraph
    current: MolecularGraph
    candidate: LegalSuccessor
    region_slots: tuple[int, ...]
    mutable_slots: tuple[int, ...]
    previous_rule: str | None

    def __post_init__(self) -> None:
        if (
            not all((self.domain, self.lineage, self.route_id, self.component_id))
            or self.component_events < 1
            or not 0 <= self.depth < self.component_events
            or not self.region_slots
            or len(set(self.region_slots)) != len(self.region_slots)
            or len(set(self.mutable_slots)) != len(self.mutable_slots)
            or self.previous_rule not in (*RULES, None)
        ):
            raise ValueError("invalid compositional patch training event")


def balanced_event_weights(events: list[PatchTrainingEvent]) -> list[float]:
    """Equalize domain, lineage, route, component and event mass."""

    if not events:
        raise ValueError("balanced weights require training events")
    domains = sorted({row.domain for row in events})
    lineages = {
        domain: sorted({row.lineage for row in events if row.domain == domain})
        for domain in domains
    }
    routes = defaultdict(set)
    components = defaultdict(set)
    decisions = Counter()
    for row in events:
        routes[(row.domain, row.lineage)].add(row.route_id)
        components[(row.domain, row.lineage, row.route_id)].add(row.component_id)
        decisions[(row.domain, row.lineage, row.route_id, row.component_id)] += 1
    result = []
    for row in events:
        route_key = (row.domain, row.lineage)
        component_key = (*route_key, row.route_id)
        event_key = (*component_key, row.component_id)
        result.append(
            1
            / (
                len(domains)
                * len(lineages[row.domain])
                * len(routes[route_key])
                * len(components[component_key])
                * decisions[event_key]
            )
        )
    total = sum(result)
    return [value / total for value in result]


def _cycle_slots(graph: MolecularGraph) -> set[int]:
    slots = tuple(int(value) for value in np.flatnonzero(is_element(graph.atom_types)))
    network = nx.Graph()
    network.add_nodes_from(slots)
    network.add_edges_from(
        (left, right)
        for left in slots
        for right in slots
        if left < right and int(graph.bonds[left, right])
    )
    return {slot for cycle in nx.cycle_basis(network) for slot in cycle}


def region_features(graph: MolecularGraph, slots: tuple[int, ...]) -> np.ndarray:
    """Address-free features for the selected source-role set ``R``."""

    unique = tuple(sorted(set(map(int, slots))))
    if not unique or any(not bool(is_element(graph.atom_types[slot])) for slot in unique):
        raise ValueError("region roles must be live source atoms")
    elements = np.zeros(10, dtype=float)
    degrees = np.zeros(5, dtype=float)
    hydrogens = np.zeros(5, dtype=float)
    charges = np.zeros(5, dtype=float)
    bond_orders = np.zeros(4, dtype=float)
    cyclic = _cycle_slots(graph)
    for slot in unique:
        element = int(graph.atom_types[slot])
        if 1 <= element <= 10:
            elements[element - 1] += 1
        degrees[min(int(np.count_nonzero(graph.bonds[slot])), 4)] += 1
        hydrogens[min(int(graph.implicit_h_counts[slot]), 4)] += 1
        charges[min(max(int(graph.formal_charges[slot]), -2), 2) + 2] += 1
    internal_edges = 0
    for index, left in enumerate(unique):
        for right in unique[index + 1 :]:
            order = int(graph.bonds[left, right])
            if order:
                internal_edges += 1
                bond_orders[min(order, 4) - 1] += 1
    denominator = len(unique)
    result = np.concatenate(
        (
            molecule_features(canonical_state_key(graph)).astype(float),
            elements / denominator,
            degrees / denominator,
            hydrogens / denominator,
            charges / denominator,
            bond_orders / max(1, internal_edges),
            np.asarray(
                [
                    denominator / 40,
                    internal_edges / max(1, denominator),
                    sum(slot in cyclic for slot in unique) / denominator,
                ],
                dtype=float,
            ),
        )
    )
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite structural region features")
    return result


def joint_event_features(
    *,
    component_source: MolecularGraph,
    current: MolecularGraph,
    candidate: LegalSuccessor,
    region_slots: tuple[int, ...],
    mutable_slots: tuple[int, ...],
    depth: int,
    previous_rule: str | None,
) -> np.ndarray:
    """One coupled feature vector for topology, attributes, attachment and D."""

    footprint, references, created, deleted = _record_parts(candidate.action_record, current)
    region = set(region_slots)
    mutable = set(mutable_slots)
    previous = np.zeros(len(RULES) + 1, dtype=float)
    previous[len(RULES) if previous_rule is None else RULES.index(previous_rule)] = 1
    relation = np.asarray(
        [
            depth / 2,
            len(region) / 40,
            len(mutable) / 40,
            len(references & region) / max(1, len(references)),
            len(references & mutable) / max(1, len(references)),
            len(footprint - mutable) / max(1, len(footprint)),
            float(created is not None),
            float(deleted is not None),
        ],
        dtype=float,
    )
    result = np.concatenate(
        (
            action_features(current, candidate),
            region_features(component_source, region_slots),
            previous,
            relation,
        )
    )
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite joint patch-event features")
    return result


@dataclass(frozen=True)
class CompositionalPatchGenerator:
    region_density: WeightedDiagonalDensity
    joint_event_density: WeightedDiagonalDensity
    stop_after_first: float
    training_identity: str
    training_summary: dict

    def __post_init__(self) -> None:
        if (
            not 0 < self.stop_after_first < 1
            or not self.training_identity
            or not isinstance(self.training_summary, dict)
        ):
            raise ValueError("invalid compositional patch generator")

    def checkpoint(self) -> dict:
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "region_density": self.region_density.payload(),
            "joint_event_density": self.joint_event_density.payload(),
            "stop_after_first": self.stop_after_first,
            "training_identity": self.training_identity,
            "training_summary": self.training_summary,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict) -> CompositionalPatchGenerator:
        if payload.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("compositional patch checkpoint schema mismatch")
        return cls(
            WeightedDiagonalDensity.from_payload(payload["region_density"]),
            WeightedDiagonalDensity.from_payload(payload["joint_event_density"]),
            float(payload["stop_after_first"]),
            str(payload["training_identity"]),
            dict(payload["training_summary"]),
        )


def fit_compositional_patch_generator(
    events: list[PatchTrainingEvent], *, variance_floor: float
) -> CompositionalPatchGenerator:
    """Fit numeric joint densities after the caller has frozen its split."""

    weights = balanced_event_weights(events)
    first = [index for index, row in enumerate(events) if row.depth == 0]
    region_density = fit_weighted_density(
        [
            region_features(events[index].component_source, events[index].region_slots)
            for index in first
        ],
        [weights[index] for index in first],
        variance_floor=variance_floor,
    )
    vectors = [
        joint_event_features(
            component_source=row.component_source,
            current=row.current,
            candidate=row.candidate,
            region_slots=row.region_slots,
            mutable_slots=row.mutable_slots,
            depth=row.depth,
            previous_rule=row.previous_rule,
        )
        for row in events
    ]
    joint_density = fit_weighted_density(vectors, weights, variance_floor=variance_floor)
    first_weight = sum(weights[index] for index in first)
    stop_weight = sum(weights[index] for index in first if events[index].component_events == 1)
    stop_after_first = (stop_weight + 0.5 * first_weight) / (2 * first_weight)
    domains = Counter(row.domain for row in events)
    summary = {
        "domain_count": len(domains),
        "events_per_domain_sorted": sorted(domains.values()),
        "events": len(events),
        "components": len({(r.domain, r.lineage, r.route_id, r.component_id) for r in events}),
        "routes": len({(r.domain, r.lineage, r.route_id) for r in events}),
        "lineages": len({(r.domain, r.lineage) for r in events}),
        "maximum_observed_component_events": max(row.component_events for row in events),
        "weight_sum": sum(weights),
    }
    training_identity = identity(
        {
            "schema_version": "compositional_patch_numeric_fit_identity_v1",
            "region_density": region_density.payload(),
            "joint_event_density": joint_density.payload(),
            "stop_after_first": stop_after_first,
            "summary": summary,
            "variance_floor": variance_floor,
        }
    )
    return CompositionalPatchGenerator(
        region_density,
        joint_density,
        stop_after_first,
        training_identity,
        summary,
    )


@dataclass(frozen=True)
class GeneratedPatch:
    goal: StructuralGoal
    bindings: tuple[tuple[int, ...], ...]
    endpoint_state: dict
    actions: tuple[dict, ...]
    construction_dependencies: tuple[tuple[int, int, str], ...]
    score: float
    policy_id: str
    patch_ids: tuple[str, ...]
    realization: dict

    @property
    def endpoint_key(self) -> str:
        return canonical_state_key(decode_state(self.endpoint_state))

    def payload(self) -> dict:
        return {
            "schema_version": PATCH_SCHEMA,
            "goal": self.goal.payload(),
            "bindings": [list(row) for row in self.bindings],
            "endpoint_state": self.endpoint_state,
            "construction_events": len(self.actions),
            "actions": list(self.actions),
            "construction_dependencies": [list(row) for row in self.construction_dependencies],
            "score": self.score,
            "policy_id": self.policy_id,
            "patch_ids": list(self.patch_ids),
            "realization": self.realization,
        }


@dataclass(frozen=True)
class _Prefix:
    states: tuple[dict, ...]
    actions: tuple[dict, ...]
    component_source: MolecularGraph
    current: MolecularGraph
    region_slots: tuple[int, ...]
    mutable_slots: tuple[int, ...]
    event_scores: tuple[float, ...]
    score: float


def _event_region(graph: MolecularGraph, candidate: LegalSuccessor) -> tuple[int, ...]:
    _, references, _, _ = _record_parts(candidate.action_record, graph)
    rows = tuple(sorted(slot for slot in references if bool(is_element(graph.atom_types[slot]))))
    if not rows:
        raise ValueError("construction event has no live source region")
    return rows


def _mutable_after(
    source: MolecularGraph,
    current: MolecularGraph,
    region_slots: tuple[int, ...],
) -> tuple[int, ...]:
    changed = set(region_slots)
    for slot in range(len(source.atom_types)):
        if (
            int(source.atom_types[slot]) != int(current.atom_types[slot])
            or int(source.formal_charges[slot]) != int(current.formal_charges[slot])
            or int(source.implicit_h_counts[slot]) != int(current.implicit_h_counts[slot])
            or not np.array_equal(source.bonds[slot], current.bonds[slot])
        ):
            changed.add(slot)
    return tuple(sorted(changed))


def _is_local(prefix: _Prefix, candidate: LegalSuccessor) -> bool:
    _, references, _, _ = _record_parts(candidate.action_record, prefix.current)
    return bool(references) and references <= set(prefix.mutable_slots)


def enumerate_local_legal_successors(
    graph: MolecularGraph,
    allowed_slots: tuple[int, ...],
) -> tuple[LegalSuccessor, ...]:
    """Enumerate the exact Active8 fiber whose references are local.

    This is decision-equivalent to filtering :func:`enumerate_legal_successors`
    after complete enumeration.  It filters address roles before exact execution
    and therefore avoids paying for molecule-wide nonlocal candidates.  Semantic
    atom restatement retains its authoritative resolver and shared prepared
    context.
    """

    allowed = {int(slot) for slot in allowed_slots}
    if not allowed:
        return ()
    source_key = canonical_state_key(graph)
    system = editing_v2_semantic_rewrite_system()
    successors: dict[str, LegalSuccessor] = {}
    for rule in RULES:
        resolved = []
        if rule == "atom_restate_semantic":
            context = prepare_semantic_atom_restate_context(graph)
            for slot in sorted(allowed):
                if (
                    slot < 0
                    or slot >= len(graph.atom_types)
                    or not bool(is_element(graph.atom_types[slot]))
                ):
                    continue
                for target_class in range(len(ORGANIC_VOCABULARY)):
                    action = SemanticAtomRestate(slot, target_class)
                    resolution = resolve_semantic_atom_restate_action(
                        graph,
                        action,
                        context=context,
                    )
                    if (
                        resolution is not None
                        and resolution.admitted
                        and resolution.successor is not None
                    ):
                        resolved.append((action, resolution.successor))
        else:
            for action in _candidate_actions(graph, rule):
                action_record = encode_action(rule, action)
                _, references, _, _ = _record_parts(action_record, graph)
                if not references or not references <= allowed:
                    continue
                try:
                    successor = system.apply(graph, rule, action)
                except (InvalidRewrite, ValueError):
                    continue
                resolved.append((action, successor))
        for action, successor in resolved:
            key = canonical_state_key(successor)
            if key == source_key or key in successors:
                continue
            successors[key] = LegalSuccessor(
                rule=rule,
                action_record=encode_action(rule, action),
                successor=successor,
                successor_key=key,
            )
    return tuple(successors[key] for key in sorted(successors))


def _dependencies(actions: tuple[dict, ...]) -> tuple[tuple[int, int, str], ...]:
    if len(actions) < 2:
        return ()
    first = actions[0]["payload"]
    second = actions[1]["payload"]
    created = int(first["slot"]) if actions[0]["executor_rule"] == "atom_insert" else None
    if created is None:
        return ((0, 1, "shared_source_role"),)
    encoded = json.dumps(second, sort_keys=True, separators=(",", ":"))
    return ((0, 1, "created_role") if str(created) in encoded else (0, 1, "shared_source_role"),)


def _compile_prefix(source: MolecularGraph, prefix: _Prefix, *, policy_id: str) -> GeneratedPatch:
    goal, bindings, _ = extract_structural_goal(prefix.states, prefix.actions)
    result = realize_structural_goal(source, goal, bindings, config=RealizerConfig())
    if result["status"] != "realized" or not result["endpoint_matches_bound_target"]:
        raise ValueError(f"exact structural realization abstained: {result['status']}")
    endpoint = decode_state(result["states"][-1])
    if canonical_state_key(endpoint) != canonical_state_key(prefix.current):
        raise RuntimeError("structural realizer changed the generated patch endpoint")
    patch_ids = tuple(minimize_subgoal(row)[0].template_id for row in goal.subgoals)
    return GeneratedPatch(
        goal,
        bindings,
        encode_state(endpoint),
        prefix.actions,
        _dependencies(prefix.actions),
        prefix.score,
        policy_id,
        patch_ids,
        {
            "status": result["status"],
            "primitive_count": len(result["actions"]),
            "expanded": int(result["expanded"]),
            "attempted": int(result["attempted"]),
            "compiler_strategy": result["compiler_strategy"],
            "endpoint_matches_bound_target": bool(result["endpoint_matches_bound_target"]),
            "primitive_teacher_actions_used": int(result["primitive_teacher_actions_used"]),
        },
    )


def _candidate_score(
    model: CompositionalPatchGenerator | None,
    *,
    component_source: MolecularGraph,
    current: MolecularGraph,
    candidate: LegalSuccessor,
    region_slots: tuple[int, ...],
    mutable_slots: tuple[int, ...],
    depth: int,
    previous_rule: str | None,
) -> float:
    if model is None:
        return 0.0
    score = model.joint_event_density.score(
        joint_event_features(
            component_source=component_source,
            current=current,
            candidate=candidate,
            region_slots=region_slots,
            mutable_slots=mutable_slots,
            depth=depth,
            previous_rule=previous_rule,
        )
    )
    if depth == 0:
        score += model.region_density.score(region_features(component_source, region_slots))
    return score


def generate_compositional_patches(
    source: MolecularGraph,
    model: CompositionalPatchGenerator | None,
    *,
    pool_size: int = 128,
    first_event_beam: int = 64,
    second_event_beam: int = 32,
    second_event_expansion_per_prefix: int = 16,
    per_rule_first_event_cap: int = 32,
) -> tuple[list[GeneratedPatch], dict]:
    """Generate and exactly realize autonomous one- or two-event local patches."""

    if (
        min(
            pool_size,
            first_event_beam,
            second_event_beam,
            second_event_expansion_per_prefix,
            per_rule_first_event_cap,
        )
        < 1
    ):
        raise ValueError("compositional patch generation limits must be positive")
    policy_id = POLICY_UNIFORM if model is None else POLICY_LEARNED
    source_state = encode_state(source)
    first = enumerate_legal_successors(source)
    telemetry = Counter(first_legal_successors=len(first))
    scored_by_rule = defaultdict(list)
    for candidate in first:
        region = _event_region(source, candidate)
        mutable = _mutable_after(source, candidate.successor, region)
        event_score = _candidate_score(
            model,
            component_source=source,
            current=source,
            candidate=candidate,
            region_slots=region,
            mutable_slots=region,
            depth=0,
            previous_rule=None,
        )
        stop = 0.5 if model is None else model.stop_after_first
        prefix = _Prefix(
            (source_state, encode_state(candidate.successor)),
            (candidate.action_record,),
            source,
            candidate.successor,
            region,
            mutable,
            (event_score,),
            event_score + math.log(stop),
        )
        scored_by_rule[candidate.rule].append((event_score, candidate.successor_key, prefix))
    retained_first = []
    for rule in RULES:
        rows = sorted(scored_by_rule[rule], key=lambda row: (-row[0], row[1]))
        retained_first.extend(row[2] for row in rows[:per_rule_first_event_cap])
    retained_first.sort(key=lambda row: (-row.score, canonical_state_key(row.current)))
    retained_first = retained_first[:first_event_beam]
    telemetry["retained_first_prefixes"] = len(retained_first)
    prefixes = list(retained_first)
    continue_prefixes = retained_first[:second_event_beam]
    for prefix in continue_prefixes:
        legal = enumerate_local_legal_successors(prefix.current, prefix.mutable_slots)
        telemetry["second_legal_successors"] += len(legal)
        candidates = []
        for candidate in legal:
            if not _is_local(prefix, candidate):
                telemetry["second_nonlocal_rejections"] += 1
                continue
            score = _candidate_score(
                model,
                component_source=prefix.component_source,
                current=prefix.current,
                candidate=candidate,
                region_slots=prefix.region_slots,
                mutable_slots=prefix.mutable_slots,
                depth=1,
                previous_rule=prefix.actions[-1]["executor_rule"],
            )
            candidates.append((score, candidate.successor_key, candidate))
        candidates.sort(key=lambda row: (-row[0], row[1]))
        for score, _, candidate in candidates[:second_event_expansion_per_prefix]:
            prefixes.append(
                _Prefix(
                    (*prefix.states, encode_state(candidate.successor)),
                    (*prefix.actions, candidate.action_record),
                    prefix.component_source,
                    candidate.successor,
                    prefix.region_slots,
                    _mutable_after(source, candidate.successor, prefix.region_slots),
                    (*prefix.event_scores, score),
                    sum((*prefix.event_scores, score))
                    + math.log(1 - (0.5 if model is None else model.stop_after_first)),
                )
            )
    telemetry["constructed_prefixes"] = len(prefixes)
    prefixes.sort(
        key=lambda row: (
            -row.score,
            canonical_state_key(row.current),
            identity(list(row.actions)),
        )
    )
    generated = []
    seen = set()
    for prefix in prefixes:
        key = canonical_state_key(prefix.current)
        if key in seen:
            telemetry["endpoint_aliases"] += 1
            continue
        telemetry["compile_attempts"] += 1
        try:
            patch = _compile_prefix(source, prefix, policy_id=policy_id)
        except (RuntimeError, TypeError, ValueError) as error:
            telemetry[f"compile_abstention:{type(error).__name__}"] += 1
            continue
        seen.add(key)
        generated.append(patch)
        telemetry["exactly_realized"] += 1
        if len(generated) >= pool_size:
            break
    telemetry["candidate_shortfall"] = max(0, pool_size - len(generated))
    telemetry["unique_endpoints"] = len(generated)
    telemetry["legal_patch_compile_coverage_numerator"] = len(generated)
    telemetry["legal_patch_compile_coverage_denominator"] = telemetry["compile_attempts"]
    telemetry["exact_realization_precision_numerator"] = len(generated)
    telemetry["exact_realization_precision_denominator"] = len(generated)
    return generated, dict(sorted(telemetry.items()))


__all__ = [
    "CHECKPOINT_SCHEMA",
    "PATCH_SCHEMA",
    "POLICY_LEARNED",
    "POLICY_UNIFORM",
    "CompositionalPatchGenerator",
    "GeneratedPatch",
    "PatchTrainingEvent",
    "WeightedDiagonalDensity",
    "balanced_event_weights",
    "enumerate_local_legal_successors",
    "fit_compositional_patch_generator",
    "fit_weighted_density",
    "generate_compositional_patches",
    "joint_event_features",
    "region_features",
]
