"""Grouped local-policy comparison for sanitized PMO route supervision."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from compose_v4.control.docking_value import identity

SCHEMA = "pmo_route_policy_quality_v1"
CHECKPOINT_SCHEMA = "pmo_hierarchical_local_action_ranker_folds_v1"
POLICIES = (
    "balanced_marginal_action_prior",
    "graph_conditioned_hierarchical_local_ranker",
)
HEADS = ("rule", "where", "how", "dependency", "create", "stop")


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
            raise ValueError("invalid centroid configuration")


@dataclass(frozen=True)
class Decision:
    row_index: int
    state_features: np.ndarray
    rule: str
    operands: tuple[dict, ...]
    parameters: dict
    dependencies: tuple[dict, ...]
    creates_output: bool
    stop: bool
    task_family: str
    lineage_identity: str
    trace_identity: str
    runtime_supported: bool
    weight: float


@dataclass(frozen=True)
class FactorExample:
    event_id: str
    decision_id: int
    head: str
    context: str
    target: str
    features: np.ndarray
    weight: float
    runtime_supported: bool


def _token(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _role_kind(role: str) -> str:
    if re.fullmatch(r"neighbor_\d+", role):
        return "neighbor"
    match = re.fullmatch(r"change_\d+_([ab])", role)
    if match:
        return f"change_endpoint_{match.group(1)}"
    if role in {"atom", "endpoint_a", "endpoint_b"}:
        return role
    raise ValueError(f"unsupported generic operand role: {role!r}")


def _region_payload(descriptor: dict) -> dict:
    expected = {
        "origin",
        "atom_type",
        "formal_charge",
        "implicit_hydrogens",
        "degree",
        "bond_class_histogram",
        "neighbor_element_histogram",
    }
    if not expected <= set(descriptor):
        raise ValueError("structural operand descriptor is incomplete")
    return {
        key: descriptor[key]
        for key in (
            "atom_type",
            "formal_charge",
            "implicit_hydrogens",
            "degree",
            "bond_class_histogram",
            "neighbor_element_histogram",
        )
    }


def _region_vector(descriptor: dict) -> np.ndarray:
    region = _region_payload(descriptor)
    return np.asarray(
        [
            float(region["atom_type"]) / 11,
            float(region["formal_charge"]) / 4,
            float(region["implicit_hydrogens"]) / 4,
            float(region["degree"]) / 4,
            *(float(value) / 4 for value in region["bond_class_histogram"]),
            *(float(value) / 4 for value in region["neighbor_element_histogram"]),
        ],
        dtype=float,
    )


def _where_summary(operands: tuple[dict, ...]) -> np.ndarray:
    if not operands:
        return np.zeros(21, dtype=float)
    regions = np.stack([_region_vector(row["descriptor"]) for row in operands])
    return np.concatenate(
        (np.asarray([min(len(operands), 16) / 16], dtype=float), regions.mean(axis=0))
    )


def _dependency_target(descriptor: dict) -> str:
    origin = descriptor["origin"]
    if origin == "preexisting":
        return "preexisting"
    if origin != "route_created":
        raise ValueError(f"unknown operand origin: {origin!r}")
    lag = descriptor.get("creation_lag")
    ordinal = descriptor.get("created_ordinal")
    if type(lag) is not int or lag < 1 or type(ordinal) is not int or ordinal < 0:
        raise ValueError("route-created operand lacks relative handle provenance")
    return f"route_created_lag_{lag}"


def decisions_from_dataset(dataset: dict) -> tuple[Decision, ...]:
    """Validate and align generic rows with separate training provenance."""

    if dataset.get("schema_version") != "pmo_route_distillation_training_dataset_v1":
        raise ValueError("PMO route-supervision schema mismatch")
    if dataset.get("actor_training_performed") is not False:
        raise ValueError("input is not the frozen training-only supervision export")
    generic = dataset.get("generic_rows", [])
    provenance = dataset.get("training_provenance", [])
    if len(generic) != len(provenance) or not generic:
        raise ValueError("generic rows and provenance are not one-to-one")
    lengths = {
        row["trace_identity"]: int(row["primitive_transitions"])
        for row in dataset.get("route_manifest", [])
    }
    if len(lengths) != len(dataset.get("route_manifest", [])):
        raise ValueError("route manifest has duplicate trace identities")
    result = []
    for index, (row, source) in enumerate(zip(generic, provenance, strict=True)):
        if row.get("row_index") != index or source.get("row_index") != index:
            raise ValueError("dataset row indices are not contiguous")
        members = source.get("members", [])
        families = {member.get("task_family") for member in members}
        if len(families) != 1:
            raise ValueError("one supervision row crosses frozen task families")
        trace = str(source["trace_identity"])
        primitive_index = int(source["primitive_index"])
        if trace not in lengths or not 0 <= primitive_index < lengths[trace]:
            raise ValueError("supervision primitive index is outside its route")
        state = np.asarray(row["state_features"], dtype=float)
        if state.shape != (517,) or not np.isfinite(state).all():
            raise ValueError("invalid PMO state feature vector")
        action = row["action_supervision"]
        operands = tuple(action["operands"])
        dependencies = tuple(action["created_handle_dependencies"])
        dependent_roles = {entry["operand"] for entry in dependencies}
        observed_roles = {
            operand["role"]
            for operand in operands
            if operand["descriptor"]["origin"] == "route_created"
        }
        if dependent_roles != observed_roles:
            raise ValueError("created-handle dependency list disagrees with operands")
        result.append(
            Decision(
                row_index=index,
                state_features=state,
                rule=str(action["executor_rule"]),
                operands=operands,
                parameters=dict(action["parameters"]),
                dependencies=dependencies,
                creates_output=action["created_output_ordinal"] is not None,
                stop=primitive_index + 1 == lengths[trace],
                task_family=str(next(iter(families))),
                lineage_identity=str(source["lineage_identity"]),
                trace_identity=trace,
                runtime_supported=bool(row["runtime_complete_route_supported"]),
                weight=float(row["balance"]["weight"]),
            )
        )
    return tuple(result)


def factor_examples(decisions: tuple[Decision, ...]) -> tuple[FactorExample, ...]:
    """Create teacher-forced generic factors without absolute action addresses."""

    result = []
    for decision in decisions:
        prefix = str(decision.row_index)
        base = decision.state_features
        where_summary = _where_summary(decision.operands)
        action_features = np.concatenate((base, where_summary))

        def add(
            head,
            suffix,
            context,
            target,
            features,
            weight,
            decision_id=decision.row_index,
            event_prefix=prefix,
            runtime_supported=decision.runtime_supported,
        ):
            result.append(
                FactorExample(
                    f"{event_prefix}:{head}:{suffix}",
                    decision_id,
                    head,
                    context,
                    target,
                    features,
                    weight,
                    runtime_supported,
                )
            )

        add("rule", "0", "all", decision.rule, base, decision.weight)
        operand_weight = decision.weight / max(1, len(decision.operands))
        for operand_index, operand in enumerate(decision.operands):
            role = _role_kind(str(operand["role"]))
            descriptor = operand["descriptor"]
            context = f"{decision.rule}|{role}"
            add(
                "where",
                str(operand_index),
                context,
                _token(_region_payload(descriptor)),
                base,
                operand_weight,
            )
            add(
                "dependency",
                str(operand_index),
                context,
                _dependency_target(descriptor),
                np.concatenate((base, _region_vector(descriptor))),
                operand_weight,
            )
        add(
            "how",
            "0",
            decision.rule,
            _token(decision.parameters),
            action_features,
            decision.weight,
        )
        add(
            "create",
            "0",
            decision.rule,
            (
                "creates_relative_handle"
                if decision.creates_output
                else "no_created_output"
            ),
            action_features,
            decision.weight,
        )
        add(
            "stop",
            "0",
            decision.rule,
            "stop" if decision.stop else "continue",
            action_features,
            decision.weight,
        )
    return tuple(result)


def predeclared_family_folds(decisions: tuple[Decision, ...], fold_rows: list[dict]):
    observed = {decision.task_family for decision in decisions}
    held_out = [set(row["held_out_task_families"]) for row in fold_rows]
    if set().union(*held_out) != observed or sum(map(len, held_out)) != len(observed):
        raise ValueError("predeclared folds do not partition task families")
    result = []
    for row, families in zip(fold_rows, held_out, strict=True):
        test = tuple(d.row_index for d in decisions if d.task_family in families)
        train = tuple(d.row_index for d in decisions if d.task_family not in families)
        if len(test) != int(row["expected_test_decisions"]) or set(test) & set(train):
            raise ValueError(f"frozen PMO split changed for fold {row['fold']}")
        test_lineages = {decisions[index].lineage_identity for index in test}
        train_lineages = {decisions[index].lineage_identity for index in train}
        if test_lineages & train_lineages:
            raise ValueError("lineage leakage across PMO train/test split")
        result.append(
            {
                "fold": int(row["fold"]),
                "held_out_task_families": sorted(families),
                "train_decisions": train,
                "test_decisions": test,
                "train_lineages": sorted(train_lineages),
                "test_lineages": sorted(test_lineages),
            }
        )
    return tuple(result)


def _fit_context(examples: list[FactorExample], config: PolicyConfig) -> dict:
    if not examples:
        raise ValueError("cannot fit an empty categorical context")
    width = {example.features.shape for example in examples}
    if len(width) != 1:
        raise ValueError("one categorical context has inconsistent feature width")
    classes = sorted({example.target for example in examples})
    x = np.stack([example.features for example in examples])
    weights = np.asarray([example.weight for example in examples], dtype=float)
    weights /= weights.sum()
    class_mass = np.asarray(
        [
            sum(
                w
                for w, row in zip(weights, examples, strict=True)
                if row.target == label
            )
            for label in classes
        ]
    )
    probabilities = (1 - config.uniform_probability_floor) * class_mass
    probabilities += config.uniform_probability_floor / len(classes)
    mean = np.sum(x * weights[:, None], axis=0)
    variance = np.sum((x - mean) ** 2 * weights[:, None], axis=0)
    scale = np.sqrt(variance)
    centroids = []
    between = np.zeros(x.shape[1], dtype=float)
    for label, mass in zip(classes, class_mass, strict=True):
        mask = np.asarray([row.target == label for row in examples])
        class_weights = weights[mask] / weights[mask].sum()
        centroid = np.sum(x[mask] * class_weights[:, None], axis=0)
        centroids.append(centroid)
        between += mass * (centroid - mean) ** 2
    informative = np.flatnonzero(scale >= config.minimum_scale)
    ranked = informative[
        np.argsort(
            -between[informative]
            / np.maximum(variance[informative], config.minimum_scale**2),
            kind="stable",
        )
    ]
    selected = ranked[: config.maximum_selected_features_per_head]
    safe_scale = np.maximum(scale[selected], config.minimum_scale)
    return {
        "classes": classes,
        "marginal_probabilities": probabilities.tolist(),
        "selected_feature_indices": selected.tolist(),
        "feature_mean": mean[selected].tolist(),
        "feature_scale": safe_scale.tolist(),
        "class_centroids": [centroid[selected].tolist() for centroid in centroids],
        "centroid_distance_strength": config.centroid_distance_strength,
        "training_examples": len(examples),
    }


def fit_fold(
    examples: tuple[FactorExample, ...],
    train_decisions: set[int],
    config: PolicyConfig,
) -> dict:
    grouped = defaultdict(list)
    for example in examples:
        if example.decision_id in train_decisions:
            grouped[(example.head, example.context)].append(example)
    heads = {head: {} for head in HEADS}
    for (head, context), rows in sorted(grouped.items()):
        heads[head][context] = _fit_context(rows, config)
    checkpoint = {
        "schema_version": "pmo_hierarchical_local_action_ranker_fold_v1",
        "state_feature_dim": 517,
        "factor_order": list(HEADS),
        "heads": heads,
        "training_identity": identity(
            {
                "schema_version": "pmo_local_policy_training_identity_v1",
                "decision_count": len(train_decisions),
                "event_count": sum(
                    example.decision_id in train_decisions for example in examples
                ),
                "configuration": config.__dict__,
                "head_context_class_counts": {
                    head: {
                        context: len(model["classes"])
                        for context, model in sorted(contexts.items())
                    }
                    for head, contexts in heads.items()
                },
            }
        ),
    }
    return checkpoint


def _probabilities(
    model: dict, features: np.ndarray, *, conditioned: bool
) -> np.ndarray:
    marginal = np.asarray(model["marginal_probabilities"], dtype=float)
    if not conditioned or len(model["classes"]) == 1:
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
    scores -= scores.max()
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum()


def evaluate_fold(
    checkpoint: dict,
    examples: tuple[FactorExample, ...],
    test_decisions: set[int],
    *,
    conditioned: bool,
) -> tuple[dict, ...]:
    outcomes = []
    for example in examples:
        if example.decision_id not in test_decisions:
            continue
        model = checkpoint["heads"].get(example.head, {}).get(example.context)
        if model is None or example.target not in model["classes"]:
            outcomes.append(
                {
                    "event_id": example.event_id,
                    "decision_id": example.decision_id,
                    "head": example.head,
                    "weight": example.weight,
                    "runtime_supported": example.runtime_supported,
                    "covered": False,
                    "correct": False,
                    "nll": None,
                    "rank": None,
                }
            )
            continue
        probabilities = _probabilities(model, example.features, conditioned=conditioned)
        target = model["classes"].index(example.target)
        order = np.argsort(-probabilities, kind="stable")
        rank = int(np.flatnonzero(order == target)[0]) + 1
        outcomes.append(
            {
                "event_id": example.event_id,
                "decision_id": example.decision_id,
                "head": example.head,
                "weight": example.weight,
                "runtime_supported": example.runtime_supported,
                "covered": True,
                "correct": rank == 1,
                "nll": -math.log(max(float(probabilities[target]), 1e-300)),
                "rank": rank,
            }
        )
    return tuple(outcomes)


def _weighted_mean(rows: list[dict], field: str) -> float | None:
    available = [row for row in rows if row[field] is not None]
    if not available:
        return None
    denominator = sum(row["weight"] for row in available)
    return sum(row["weight"] * row[field] for row in available) / denominator


def component_metrics(outcomes: tuple[dict, ...]) -> dict:
    result = {}
    for head in HEADS:
        rows = [row for row in outcomes if row["head"] == head]
        covered = [row for row in rows if row["covered"]]
        total_weight = sum(row["weight"] for row in rows)
        covered_weight = sum(row["weight"] for row in covered)
        correct_weight = sum(row["weight"] for row in covered if row["correct"])
        result[head] = {
            "events": len(rows),
            "covered_events": len(covered),
            "coverage": len(covered) / len(rows) if rows else None,
            "weighted_coverage": (
                covered_weight / total_weight if total_weight else None
            ),
            "top1_precision_when_covered": (
                correct_weight / covered_weight if covered_weight else None
            ),
            "top1_recall": correct_weight / total_weight if total_weight else None,
            "covered_negative_log_likelihood": _weighted_mean(covered, "nll"),
            "covered_mean_rank": _weighted_mean(covered, "rank"),
            "covered_mean_reciprocal_rank": _weighted_mean(
                [{**row, "reciprocal_rank": 1 / row["rank"]} for row in covered],
                "reciprocal_rank",
            ),
        }
    return result


def action_metrics(
    outcomes: tuple[dict, ...], decisions: tuple[Decision, ...], scope: str
) -> dict:
    if scope not in {"all", "runtime_supported", "local_only"}:
        raise ValueError("unknown action metric scope")
    selected = {
        decision.row_index
        for decision in decisions
        if scope == "all"
        or (scope == "runtime_supported" and decision.runtime_supported)
        or (scope == "local_only" and not decision.runtime_supported)
    }
    grouped = defaultdict(list)
    for row in outcomes:
        if row["decision_id"] in selected:
            grouped[row["decision_id"]].append(row)
    rows = []
    for decision_id in sorted(selected):
        factors = grouped[decision_id]
        covered = bool(factors) and all(row["covered"] for row in factors)
        rows.append(
            {
                "weight": decisions[decision_id].weight,
                "covered": covered,
                "correct": covered and all(row["correct"] for row in factors),
                "nll": sum(row["nll"] for row in factors) if covered else None,
            }
        )
    total_weight = sum(row["weight"] for row in rows)
    covered = [row for row in rows if row["covered"]]
    covered_weight = sum(row["weight"] for row in covered)
    correct_weight = sum(row["weight"] for row in covered if row["correct"])
    return {
        "decisions": len(rows),
        "covered_decisions": len(covered),
        "factorized_coverage": len(covered) / len(rows) if rows else None,
        "weighted_factorized_coverage": covered_weight / total_weight,
        "factorized_top1_precision_when_covered": (
            correct_weight / covered_weight if covered_weight else None
        ),
        "factorized_top1_recall": correct_weight / total_weight,
        "covered_factorized_negative_log_likelihood": _weighted_mean(covered, "nll"),
    }


def runtime_checkpoint_leakage(checkpoints: dict, forbidden_values: set[str]) -> dict:
    forbidden_keys = {
        "absolute_slot",
        "actions",
        "assignment",
        "endpoint",
        "lineage",
        "lineage_identity",
        "marks",
        "member_id",
        "program",
        "route",
        "route_id",
        "smiles",
        "source_graph",
        "source_path",
        "task",
        "task_family",
    }
    keys, strings = set(), set()

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                keys.add(str(key).lower())
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            strings.add(value)

    walk(checkpoints)
    return {
        "forbidden_key_hits": sorted(keys & forbidden_keys),
        "forbidden_teacher_value_hits": sorted(strings & forbidden_values),
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "HEADS",
    "POLICIES",
    "SCHEMA",
    "PolicyConfig",
    "action_metrics",
    "component_metrics",
    "decisions_from_dataset",
    "evaluate_fold",
    "factor_examples",
    "fit_fold",
    "predeclared_family_folds",
    "runtime_checkpoint_leakage",
]
