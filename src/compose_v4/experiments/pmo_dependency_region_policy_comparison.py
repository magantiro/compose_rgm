"""Goal-free PMO primitive and dependency-region proposal rankers."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.experiments.pmo_dependency_region_program import (
    dependency_region_program,
)
from compose_v4.experiments.pmo_exact_program_segmentation import _record_parts
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "pmo_dependency_region_policy_comparison_v1"
CHECKPOINT_SCHEMA = "pmo_dependency_region_policy_fold_checkpoints_v1"
POLICIES = (
    "balanced_generic_marginal",
    "graph_conditioned_flat_autoregressive",
    "graph_region_dependency_hierarchical",
)
COMMON_HEADS = (
    "primitive_rule",
    "region_origin_count",
    "region_degree",
    "created_dependency",
    "creates_output",
    "program_control",
)
HIERARCHY_HEADS = ("component_control", "component_close")
RULES = (
    "atom_delete",
    "atom_insert",
    "atom_restate_semantic",
    "bond_reorder",
    "bond_reroute",
    "cycle_close",
    "cycle_open",
    "ring_system_restate",
)
HEAD_SUPPORT = {
    "primitive_rule": RULES,
    "region_origin_count": (
        "none",
        *(
            f"{origin}_{count}"
            for origin in ("source", "created", "mixed")
            for count in ("one", "two", "many")
        ),
    ),
    "region_degree": ("none", "degree_0", "degree_1", "degree_2", "degree_3_plus"),
    "created_dependency": (
        "none",
        "single_lag_1",
        "single_lag_2_to_4",
        "single_lag_5_plus",
        "multiple_created_references",
    ),
    "creates_output": ("no_created_output", "creates_relative_output"),
    "program_control": ("continue", "stop"),
    "component_control": (
        "open",
        *(f"continue_component_backref_{index}" for index in range(8)),
    ),
    "component_close": ("keep_open", "close"),
}


@dataclass(frozen=True)
class PolicyConfig:
    maximum_selected_features_per_head: int = 32
    uniform_probability_floor: float = 0.05
    centroid_distance_strength: float = 0.5
    minimum_scale: float = 1e-6

    def __post_init__(self):
        if self.maximum_selected_features_per_head < 1:
            raise ValueError("maximum selected features must be positive")
        if not 0 < self.uniform_probability_floor < 1:
            raise ValueError("uniform probability floor must be in (0,1)")
        if self.centroid_distance_strength < 0 or self.minimum_scale <= 0:
            raise ValueError("invalid centroid policy configuration")


@dataclass(frozen=True)
class PolicyEvent:
    event_id: str
    trace_identity: str
    lineage_identity: str
    task_family: str
    test_fold: int
    runtime_supported: bool
    weight: float
    flat_features: np.ndarray
    hierarchical_features: np.ndarray
    targets: dict[str, str]


def _count_bucket(count: int) -> str:
    if count == 1:
        return "one"
    if count == 2:
        return "two"
    return "many"


def _degree_bucket(degrees: list[int]) -> str:
    if not degrees:
        return "none"
    rounded = round(sum(degrees) / len(degrees))
    return f"degree_{rounded}" if rounded <= 2 else "degree_3_plus"


def _rule_vector(rule: str | None) -> np.ndarray:
    labels = ("start", *RULES)
    value = "start" if rule is None else rule
    if value not in labels:
        raise ValueError(f"unsupported primitive rule in prefix: {value!r}")
    result = np.zeros(len(labels), dtype=float)
    result[labels.index(value)] = 1.0
    return result


def _event_targets(
    graph,
    action: dict,
    emission: dict,
    active_created_slots: set[int],
    component_control: str,
) -> dict[str, str]:
    _, references, _, _ = _record_parts(action, graph)
    created_count = sum(slot in active_created_slots for slot in references)
    source_count = len(references) - created_count
    if not references:
        region = "none"
    else:
        origin = (
            "mixed"
            if created_count and source_count
            else "created" if created_count else "source"
        )
        region = f"{origin}_{_count_bucket(len(references))}"
    degrees = [int(np.count_nonzero(graph.bonds[slot])) for slot in references]
    created_refs = emission["created_handle_references"]
    if not created_refs:
        dependency = "none"
    elif len(created_refs) > 1:
        dependency = "multiple_created_references"
    else:
        lag = int(created_refs[0]["producer_primitive_lag"])
        dependency = (
            "single_lag_1"
            if lag == 1
            else "single_lag_2_to_4" if lag <= 4 else "single_lag_5_plus"
        )
    return {
        "primitive_rule": str(action["executor_rule"]),
        "region_origin_count": region,
        "region_degree": _degree_bucket(degrees),
        "created_dependency": dependency,
        "creates_output": (
            "creates_relative_output"
            if emission["creates_output"]
            else "no_created_output"
        ),
        "component_control": component_control,
        "component_close": (
            "close" if emission["control_after"] == "close" else "keep_open"
        ),
    }


def route_events(route: dict, *, weight: float) -> tuple[PolicyEvent, ...]:
    """Build causal teacher-forced events from one exact represented route."""

    states, actions = tuple(route["states"]), tuple(route["actions"])
    program = route.get("dependency_region_program")
    if program is None:
        program = dependency_region_program(states, actions)
    emissions = program["emissions"]
    if len(emissions) != len(actions):
        raise ValueError("dependency-region emissions do not align with actions")
    graphs = [decode_state(state) for state in states]
    component_state: dict[int, dict] = {}
    active_created_slots: set[int] = set()
    opened_components: set[int] = set()
    previous_rule = None
    result = []
    for step, (graph, action, emission) in enumerate(
        zip(graphs[:-1], actions, emissions, strict=True)
    ):
        component = int(emission["component_index"])
        if emission["control_before"] == "open":
            if component in opened_components:
                raise ValueError("component reopened after its first emission")
            component_control = "open"
            state = {
                "emissions": 0,
                "created": 0,
                "first_step": step,
                "last_step": step,
                "previous_rule": None,
            }
            component_state[component] = state
            opened_components.add(component)
        else:
            if component not in opened_components:
                raise ValueError("component continued before opening")
            component_control = (
                f"continue_component_backref_{max(opened_components) - component}"
            )
            state = component_state[component]
        flat = np.concatenate(
            (
                np.asarray(molecule_features(canonical_state_key(graph)), dtype=float),
                _rule_vector(previous_rule),
                np.asarray(
                    [
                        step / 32,
                        len(program["created_handles"][:0])
                        + len(
                            [
                                row
                                for row in program["created_handles"]
                                if row["producer"] < step
                            ]
                        )
                        / 32,
                    ],
                    dtype=float,
                ),
            )
        )
        hierarchy = np.concatenate(
            (
                flat,
                np.asarray(
                    [
                        len(opened_components) / 8,
                        state["emissions"] / 32,
                        state["created"] / 32,
                        (step - state["first_step"]) / 32,
                        (step - state["last_step"]) / 32,
                    ],
                    dtype=float,
                ),
                _rule_vector(state["previous_rule"]),
            )
        )
        targets = _event_targets(
            graph,
            action,
            emission,
            active_created_slots,
            component_control,
        )
        targets["program_control"] = "stop" if step + 1 == len(actions) else "continue"
        if any(target not in HEAD_SUPPORT[head] for head, target in targets.items()):
            raise ValueError(
                f"event target falls outside fixed generic support: {targets}"
            )
        result.append(
            PolicyEvent(
                event_id=f"{route.get('trace_identity', 'candidate')}:{step}",
                trace_identity=str(route.get("trace_identity", "candidate")),
                lineage_identity=str(route.get("lineage_identity", "candidate")),
                task_family=str(route.get("task_family", "candidate")),
                test_fold=int(route.get("test_fold", -1)),
                runtime_supported=bool(program["runtime_length_supported"]),
                weight=weight,
                flat_features=flat,
                hierarchical_features=hierarchy,
                targets=targets,
            )
        )
        rule, payload = action["executor_rule"], action["payload"]
        if rule == "atom_delete":
            active_created_slots.discard(int(payload["v"]))
        elif rule == "atom_insert":
            active_created_slots.add(int(payload["slot"]))
            state["created"] += 1
        state["emissions"] += 1
        state["last_step"] = step
        state["previous_rule"] = str(rule)
        previous_rule = str(rule)
    return tuple(result)


def weighted_teacher_events(routes: list[dict]) -> tuple[PolicyEvent, ...]:
    families = sorted({str(route["task_family"]) for route in routes})
    lineage_family = {}
    lineage_decisions = Counter()
    for route in routes:
        lineage = str(route["lineage_identity"])
        family = str(route["task_family"])
        if lineage in lineage_family and lineage_family[lineage] != family:
            raise ValueError("one lineage crosses task families")
        lineage_family[lineage] = family
        lineage_decisions[lineage] += len(route["actions"])
    family_lineages = {
        family: sum(observed == family for observed in lineage_family.values())
        for family in families
    }
    result = []
    for route in routes:
        family, lineage = str(route["task_family"]), str(route["lineage_identity"])
        weight = 1 / (
            len(families) * family_lineages[family] * lineage_decisions[lineage]
        )
        result.extend(route_events(route, weight=weight))
    if not np.isclose(sum(event.weight for event in result), 1.0):
        raise RuntimeError("family/lineage/decision weights are not normalized")
    return tuple(result)


def _fit_head(
    events: list[PolicyEvent],
    head: str,
    feature_name: str,
    config: PolicyConfig,
) -> dict:
    if not events or head not in HEAD_SUPPORT:
        raise ValueError("cannot fit an empty or unknown categorical head")
    classes = tuple(HEAD_SUPPORT[head])
    x = np.stack([getattr(event, feature_name) for event in events])
    weights = np.asarray([event.weight for event in events], dtype=float)
    weights /= weights.sum()
    targets = [event.targets[head] for event in events]
    mass = np.asarray(
        [
            sum(
                weight
                for weight, target in zip(weights, targets, strict=True)
                if target == label
            )
            for label in classes
        ],
        dtype=float,
    )
    probabilities = (1 - config.uniform_probability_floor) * mass
    probabilities += config.uniform_probability_floor / len(classes)
    mean = np.sum(x * weights[:, None], axis=0)
    variance = np.sum((x - mean) ** 2 * weights[:, None], axis=0)
    centroids, between = [], np.zeros(x.shape[1], dtype=float)
    for label, class_mass in zip(classes, mass, strict=True):
        mask = np.asarray([target == label for target in targets])
        if class_mass:
            class_weights = weights[mask] / weights[mask].sum()
            centroid = np.sum(x[mask] * class_weights[:, None], axis=0)
            between += class_mass * (centroid - mean) ** 2
        else:
            centroid = mean.copy()
        centroids.append(centroid)
    informative = np.flatnonzero(variance >= config.minimum_scale**2)
    ranked = informative[
        np.argsort(
            -between[informative]
            / np.maximum(variance[informative], config.minimum_scale**2),
            kind="stable",
        )
    ]
    selected = ranked[: config.maximum_selected_features_per_head]
    return {
        "classes": list(classes),
        "marginal_probabilities": probabilities.tolist(),
        "selected_feature_indices": selected.tolist(),
        "feature_mean": mean[selected].tolist(),
        "feature_scale": np.maximum(
            np.sqrt(variance[selected]), config.minimum_scale
        ).tolist(),
        "class_centroids": [centroid[selected].tolist() for centroid in centroids],
        "centroid_distance_strength": config.centroid_distance_strength,
        "training_examples": len(events),
    }


def _marginal_model(model: dict) -> dict:
    return {
        "classes": model["classes"],
        "marginal_probabilities": model["marginal_probabilities"],
        "training_examples": model["training_examples"],
    }


def fit_fold_checkpoint(
    events: tuple[PolicyEvent, ...], test_fold: int, config: PolicyConfig
) -> dict:
    train = [event for event in events if event.test_fold != test_fold]
    test = [event for event in events if event.test_fold == test_fold]
    if not train or not test:
        raise ValueError("one frozen fold has empty train or test events")
    flat = {
        head: _fit_head(train, head, "flat_features", config) for head in COMMON_HEADS
    }
    hierarchical = {
        head: _fit_head(train, head, "hierarchical_features", config)
        for head in (*COMMON_HEADS, *HIERARCHY_HEADS)
    }
    marginal = {head: _marginal_model(flat[head]) for head in COMMON_HEADS}
    return {
        "schema_version": "pmo_dependency_region_policy_fold_v1",
        "fold_index": test_fold,
        "flat_feature_dim": len(train[0].flat_features),
        "hierarchical_feature_dim": len(train[0].hierarchical_features),
        "fixed_generic_support": {
            head: list(labels) for head, labels in HEAD_SUPPORT.items()
        },
        "policies": {
            POLICIES[0]: {"heads": marginal, "conditioned": False},
            POLICIES[1]: {"heads": flat, "conditioned": True},
            POLICIES[2]: {"heads": hierarchical, "conditioned": True},
        },
        "training_identity": identity(
            {
                "schema_version": "pmo_dependency_region_policy_training_v1",
                "fold_index": test_fold,
                "training_events": len(train),
                "configuration": config.__dict__,
                "support": HEAD_SUPPORT,
            }
        ),
    }


def _probabilities(model: dict, features: np.ndarray, conditioned: bool) -> np.ndarray:
    marginal = np.asarray(model["marginal_probabilities"], dtype=float)
    if not conditioned or "selected_feature_indices" not in model:
        return marginal
    indices = np.asarray(model["selected_feature_indices"], dtype=int)
    if not len(indices):
        return marginal
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    centroids = np.asarray(model["class_centroids"], dtype=float)
    standardized = (features[indices] - mean) / scale
    standardized_centroids = (centroids - mean[None, :]) / scale[None, :]
    distances = np.mean((standardized_centroids - standardized[None, :]) ** 2, axis=1)
    scores = np.log(marginal) - float(model["centroid_distance_strength"]) * distances
    scores -= np.max(scores)
    result = np.exp(scores)
    return result / result.sum()


def _policy_heads(policy: str) -> tuple[str, ...]:
    if policy not in POLICIES:
        raise ValueError(f"unknown PMO proposal policy: {policy}")
    return (*COMMON_HEADS, *HIERARCHY_HEADS) if policy == POLICIES[2] else COMMON_HEADS


def event_outcomes(
    checkpoint: dict,
    events: tuple[PolicyEvent, ...],
    policy: str,
) -> tuple[dict, ...]:
    model = checkpoint["policies"][policy]
    outcomes = []
    for event in events:
        if event.test_fold != checkpoint["fold_index"]:
            continue
        for head in _policy_heads(policy):
            head_model = model["heads"][head]
            features = (
                event.hierarchical_features
                if policy == POLICIES[2]
                else event.flat_features
            )
            probabilities = _probabilities(
                head_model, features, bool(model["conditioned"])
            )
            target = head_model["classes"].index(event.targets[head])
            order = np.argsort(-probabilities, kind="stable")
            rank = int(np.flatnonzero(order == target)[0]) + 1
            outcomes.append(
                {
                    "event_id": event.event_id,
                    "head": head,
                    "weight": event.weight,
                    "runtime_supported": event.runtime_supported,
                    "nll": -math.log(max(float(probabilities[target]), 1e-300)),
                    "rank": rank,
                    "top1": rank == 1,
                }
            )
    return tuple(outcomes)


def _weighted_mean(rows: list[dict], field: str) -> float:
    denominator = sum(row["weight"] for row in rows)
    return sum(row["weight"] * row[field] for row in rows) / denominator


def teacher_forced_metrics(outcomes: tuple[dict, ...]) -> dict:
    if not outcomes:
        raise ValueError("teacher-forced metrics require outcomes")
    heads = {}
    for head in sorted({row["head"] for row in outcomes}):
        rows = [row for row in outcomes if row["head"] == head]
        heads[head] = {
            "events": len(rows),
            "coverage": 1.0,
            "weighted_negative_log_likelihood": _weighted_mean(rows, "nll"),
            "weighted_mean_rank": _weighted_mean(rows, "rank"),
            "weighted_top1_precision": _weighted_mean(rows, "top1"),
        }
    event_rows = defaultdict(list)
    for row in outcomes:
        event_rows[row["event_id"]].append(row)
    common = []
    full = []
    for rows in event_rows.values():
        common_rows = [row for row in rows if row["head"] in COMMON_HEADS]
        weight = common_rows[0]["weight"]
        common.append(
            {
                "weight": weight,
                "nll": sum(row["nll"] for row in common_rows),
                "top1": all(row["top1"] for row in common_rows),
            }
        )
        full.append(
            {
                "weight": weight,
                "nll": sum(row["nll"] for row in rows),
                "top1": all(row["top1"] for row in rows),
            }
        )
    return {
        "heads": heads,
        "common_factorization": {
            "events": len(common),
            "coverage": 1.0,
            "weighted_negative_log_likelihood": _weighted_mean(common, "nll"),
            "weighted_top1_precision": _weighted_mean(common, "top1"),
        },
        "full_policy_factorization": {
            "events": len(full),
            "coverage": 1.0,
            "weighted_negative_log_likelihood": _weighted_mean(full, "nll"),
            "weighted_top1_precision": _weighted_mean(full, "top1"),
        },
    }


def score_candidate(checkpoint: dict, route: dict, policy: str) -> float:
    events = route_events(route, weight=1.0)
    model = checkpoint["policies"][policy]
    nll = 0.0
    factors = 0
    for event in events:
        for head in _policy_heads(policy):
            head_model = model["heads"][head]
            features = (
                event.hierarchical_features
                if policy == POLICIES[2]
                else event.flat_features
            )
            probabilities = _probabilities(
                head_model, features, bool(model["conditioned"])
            )
            target = head_model["classes"].index(event.targets[head])
            nll -= math.log(max(float(probabilities[target]), 1e-300))
            factors += 1
    if not factors:
        raise ValueError("cannot score an empty complete candidate")
    return -nll / factors


def runtime_checkpoint_leakage(checkpoint: dict, teacher_values: set[str]) -> dict:
    forbidden_key_terms = {
        "task",
        "family",
        "lineage",
        "route",
        "endpoint",
        "smiles",
        "source_graph",
        "slot",
        "assignment",
        "executable_program",
        "teacher",
    }
    key_hits, strings = [], set()

    def walk(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = str(key).lower()
                if any(term in lowered for term in forbidden_key_terms):
                    key_hits.append(f"{path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str):
            strings.add(value)

    walk(checkpoint)
    return {
        "forbidden_key_hits": sorted(key_hits),
        "forbidden_teacher_value_hits": sorted(strings & teacher_values),
    }


def checkpoint_envelope(folds: list[dict]) -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "policy_role": "goal_free_common_support_complete_program_rankers",
        "folds": folds,
        "new_oracle_calls": 0,
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "COMMON_HEADS",
    "HEAD_SUPPORT",
    "HIERARCHY_HEADS",
    "POLICIES",
    "RULES",
    "SCHEMA",
    "PolicyConfig",
    "PolicyEvent",
    "checkpoint_envelope",
    "event_outcomes",
    "fit_fold_checkpoint",
    "route_events",
    "runtime_checkpoint_leakage",
    "score_candidate",
    "teacher_forced_metrics",
    "weighted_teacher_events",
]
