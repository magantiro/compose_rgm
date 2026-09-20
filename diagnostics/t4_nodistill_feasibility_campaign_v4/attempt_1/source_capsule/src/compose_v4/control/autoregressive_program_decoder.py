"""Target-free autoregressive decoding over exact legal molecular rewrites.

The decoder operates only on exact molecular states, generic numeric model
parameters and the fixed Active8 legal-action support. Teacher routes and task
identities are deliberately absent from its runtime interface.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from compose_v4.control.docking_value import identity
from compose_v4.control.generic_legal_action_policy import (
    CHECKPOINT_SCHEMA as LEGAL_CHECKPOINT_SCHEMA,
)
from compose_v4.control.generic_legal_action_policy import (
    RULES,
    LegalSuccessor,
    RankerConfig,
    action_features,
    enumerate_rule_successors,
    fit_diagonal_contrastive_ranker,
    rank_of_teacher,
    reciprocal_rank,
    score_features,
)
from compose_v4.control.route_distilled_program_policy import (
    STAGE_DESCRIPTOR_NAMES,
    stage_descriptor,
)
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "autoregressive_program_decoder_v1"
CHECKPOINT_SCHEMA = "autoregressive_program_policy_v1"
CANDIDATE_LOCK_SCHEMA = "autoregressive_program_candidate_lock_v1"
MARGINAL = "marginal_legal_decoder"
LEARNED = "learned_legal_decoder"
DECODERS = (MARGINAL, LEARNED)


@dataclass(frozen=True)
class TrainingConfig:
    hidden: int = 128
    updates: int = 1000
    batch_size: int = 128
    learning_rate: float = 0.003
    l2: float = 0.0001
    exploration_floor: float = 0.05
    variance_floor: float = 0.05
    seed: int = 20260914

    def __post_init__(self) -> None:
        for name in ("hidden", "updates", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.learning_rate <= 0 or self.l2 < 0 or self.variance_floor <= 0:
            raise ValueError("invalid autoregressive training scale")
        if not 0 <= self.exploration_floor < 1:
            raise ValueError("exploration_floor must be in [0, 1)")


@dataclass(frozen=True)
class DecoderConfig:
    maximum_primitives: int = 32
    maximum_regions: int = 8
    maximum_active_atoms: int = 40
    beam_width: int = 8
    successors_per_rule: int = 32
    maximum_outputs: int = 128
    snapshots: tuple[int, ...] = (8, 16, 24, 32)

    def __post_init__(self) -> None:
        for name in (
            "maximum_primitives",
            "maximum_regions",
            "maximum_active_atoms",
            "beam_width",
            "successors_per_rule",
            "maximum_outputs",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not self.snapshots
            or tuple(sorted(set(self.snapshots))) != self.snapshots
            or self.snapshots[-1] != self.maximum_primitives
        ):
            raise ValueError("snapshots must be sorted and end at the horizon")


class ProgramPolicyNetwork(nn.Module):
    """Small graph/prefix-conditioned rule and continuation policy."""

    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
        )
        self.rule_head = nn.Linear(self.hidden, len(RULES))
        self.control_head = nn.Linear(self.hidden, 2)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(values)
        return self.rule_head(hidden), self.control_head(hidden)


@dataclass(frozen=True)
class AutoregressivePolicy:
    network: ProgramPolicyNetwork
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    exploration_floor: float
    training_identity: str

    def __post_init__(self) -> None:
        mean = np.asarray(self.feature_mean, dtype=float)
        scale = np.asarray(self.feature_scale, dtype=float)
        if (
            mean.shape != (self.network.input_dim,)
            or scale.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
            or not 0 <= self.exploration_floor < 1
            or not self.training_identity
        ):
            raise ValueError("invalid autoregressive policy contract")

    def probabilities(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(features, dtype=np.float32)
        if values.shape != (self.network.input_dim,) or not np.isfinite(values).all():
            raise ValueError("invalid autoregressive prefix features")
        normalized = (values - np.asarray(self.feature_mean)) / np.asarray(
            self.feature_scale
        )
        with torch.inference_mode():
            rule_logits, control_logits = self.network(
                torch.from_numpy(normalized.astype(np.float32))[None, :]
            )
            rule = torch.softmax(rule_logits[0], dim=0).numpy().astype(float)
            control = torch.softmax(control_logits[0], dim=0).numpy().astype(float)
        floor = self.exploration_floor
        rule = floor / len(RULES) + (1 - floor) * rule
        control = floor / 2 + (1 - floor) * control
        return rule / rule.sum(), control / control.sum()

    def to_checkpoint(self) -> dict:
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "input_dim": self.network.input_dim,
            "hidden": self.network.hidden,
            "parameters": {
                name: tensor.detach().cpu().tolist()
                for name, tensor in sorted(self.network.state_dict().items())
            },
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "exploration_floor": self.exploration_floor,
            "training_identity": self.training_identity,
            "runtime_teacher_rows": 0,
            "runtime_task_fields": 0,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> AutoregressivePolicy:
        forbidden = {
            "target",
            "protein",
            "seed",
            "route",
            "endpoint",
            "smiles",
            "source_state",
            "actions",
            "absolute_atom",
        }
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("autoregressive checkpoint schema mismatch")
        if forbidden & {str(key).lower() for key in checkpoint}:
            raise ValueError("autoregressive checkpoint contains a forbidden field")
        if checkpoint.get("runtime_teacher_rows") != 0:
            raise ValueError("autoregressive runtime checkpoint contains teachers")
        network = ProgramPolicyNetwork(
            int(checkpoint["input_dim"]), int(checkpoint["hidden"])
        )
        expected = network.state_dict()
        raw = checkpoint.get("parameters")
        if not isinstance(raw, dict) or set(raw) != set(expected):
            raise ValueError("autoregressive checkpoint parameter names mismatch")
        parameters = {}
        for name, template in expected.items():
            value = torch.tensor(raw[name], dtype=template.dtype)
            if value.shape != template.shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid autoregressive parameter: {name}")
            parameters[name] = value
        network.load_state_dict(parameters, strict=True)
        network.eval()
        return cls(
            network,
            tuple(map(float, checkpoint["feature_mean"])),
            tuple(map(float, checkpoint["feature_scale"])),
            float(checkpoint["exploration_floor"]),
            str(checkpoint["training_identity"]),
        )


def _record_slots(record: dict) -> tuple[set[int], int | None, int | None]:
    rule, payload = record["executor_rule"], record["payload"]
    created = deleted = None
    if rule == "atom_insert":
        created = int(payload["slot"])
        references = {int(row[0]) for row in payload["neighbors"]}
    elif rule in {"atom_delete", "atom_restate_semantic"}:
        references = {int(payload["v"])}
        if rule == "atom_delete":
            deleted = int(payload["v"])
    elif rule in {"cycle_close", "cycle_open", "bond_reorder"}:
        references = {int(payload["a"]), int(payload["b"])}
    elif rule == "bond_reroute":
        references = {int(payload[name]) for name in ("a", "b", "u", "v")}
    elif rule == "ring_system_restate":
        references = {
            int(change[name]) for change in payload["changes"] for name in ("a", "b")
        }
    else:
        raise ValueError(f"unsupported primitive rule: {rule!r}")
    return references, created, deleted


def dependency_region_count(actions: tuple[dict, ...]) -> int:
    """Count action components joined by shared or created-handle references."""

    if not actions:
        return 0
    parents = list(range(len(actions)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    active_creator: dict[int, int] = {}
    footprints: list[set[int]] = []
    for index, record in enumerate(actions):
        references, created, deleted = _record_slots(record)
        footprint = set(references)
        if created is not None:
            footprint.add(created)
        for earlier, prior in enumerate(footprints):
            if footprint & prior:
                union(index, earlier)
        for slot in references:
            if slot in active_creator:
                union(index, active_creator[slot])
        if created is not None:
            active_creator[created] = index
        if deleted is not None:
            active_creator.pop(deleted, None)
        footprints.append(footprint)
    return len({find(index) for index in range(len(actions))})


def prefix_features(source, current, actions: tuple[dict, ...], maximum: int) -> np.ndarray:
    """Address-free graph and action-history features for one decoder prefix."""

    source_features = molecule_features(canonical_state_key(source)).astype(float)
    current_features = molecule_features(canonical_state_key(current)).astype(float)
    if actions:
        history = stage_descriptor(source, actions)
    else:
        history = np.zeros(len(STAGE_DESCRIPTOR_NAMES), dtype=float)
    regions = dependency_region_count(actions)
    tail = np.asarray(
        [
            len(actions) / maximum,
            (current.n_real_atoms - source.n_real_atoms) / 8,
            regions / 8,
        ],
        dtype=float,
    )
    result = np.concatenate((source_features, current_features, history, tail))
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite autoregressive prefix features")
    return result


def training_rows(traces: list[dict], maximum: int) -> list[dict]:
    """Create source/route/decision-balanced teacher-forced prefix rows."""

    if not traces:
        raise ValueError("autoregressive fitting requires teacher traces")
    by_source: dict[str, list[dict]] = {}
    for trace in traces:
        source_id = str(trace["source_group"])
        by_source.setdefault(source_id, []).append(trace)
    rows = []
    for source_id, routes in sorted(by_source.items()):
        for route_index, teacher in enumerate(routes):
            trace = teacher["trace"]
            states = tuple(trace["states"])
            actions = tuple(trace["actions"])
            if not actions or len(states) != len(actions) + 1 or len(actions) > maximum:
                raise ValueError("teacher trace is outside autoregressive support")
            source = teacher["source"]
            row_weight = 1 / (len(by_source) * len(routes) * (len(actions) + 1))
            for step in range(len(actions) + 1):
                current = decode_state(states[step])
                rows.append(
                    {
                        "source_id": source_id,
                        "route_index": route_index,
                        "step": step,
                        "features": prefix_features(
                            source, current, actions[:step], maximum
                        ),
                        "rule_label": (
                            None
                            if step == len(actions)
                            else RULES.index(actions[step]["executor_rule"])
                        ),
                        "control_label": int(step == len(actions)),
                        "weight": row_weight,
                    }
                )
    total = sum(row["weight"] for row in rows)
    if not np.isclose(total, 1.0):
        raise RuntimeError("autoregressive training weights are not normalized")
    return rows


def fit_autoregressive_policy(
    traces: list[dict], *, maximum: int, config: TrainingConfig
) -> tuple[AutoregressivePolicy, list[dict]]:
    """Fit one deterministic graph/prefix-conditioned rule and stop model."""

    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    rows = training_rows(traces, maximum)
    matrix = np.stack([row["features"] for row in rows]).astype(np.float32)
    weights = np.asarray([row["weight"] for row in rows], dtype=np.float32)
    mean = np.sum(matrix * weights[:, None], axis=0)
    variance = np.sum(np.square(matrix - mean) * weights[:, None], axis=0)
    scale = np.maximum(np.sqrt(variance), config.variance_floor)
    normalized = ((matrix - mean) / scale).astype(np.float32)
    features = torch.from_numpy(normalized)
    controls = torch.tensor(
        [row["control_label"] for row in rows], dtype=torch.long
    )
    network = ProgramPolicyNetwork(features.shape[1], config.hidden)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=config.learning_rate, weight_decay=config.l2
    )
    generator = np.random.default_rng(config.seed)
    history = []
    probabilities = weights / weights.sum()
    for update in range(1, config.updates + 1):
        indices = generator.choice(
            len(rows),
            size=min(config.batch_size, len(rows)),
            replace=True,
            p=probabilities,
        )
        batch = torch.from_numpy(indices.astype(np.int64))
        rule_logits, control_logits = network(features[batch])
        control_loss = F.cross_entropy(
            control_logits, controls[batch], reduction="none"
        ).mean()
        batch_rule_mask = torch.tensor(
            [rows[index]["rule_label"] is not None for index in indices],
            dtype=torch.bool,
        )
        if batch_rule_mask.any():
            labels = torch.tensor(
                [
                    rows[index]["rule_label"]
                    for index, keep in zip(indices, batch_rule_mask.tolist(), strict=True)
                    if keep
                ],
                dtype=torch.long,
            )
            rule_loss = F.cross_entropy(rule_logits[batch_rule_mask], labels)
        else:
            rule_loss = rule_logits.sum() * 0
        loss = rule_loss + control_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite autoregressive training loss")
        if update in {1, config.updates}:
            history.append(
                {
                    "update": update,
                    "loss": float(loss.detach()),
                    "rule_loss": float(rule_loss.detach()),
                    "control_loss": float(control_loss.detach()),
                }
            )
    training_identity = identity(
        {
            "schema_version": "autoregressive_training_identity_v1",
            "sources": sorted({row["source_id"] for row in rows}),
            "routes": len(traces),
            "rows": len(rows),
            "maximum": maximum,
            "configuration": {
                name: getattr(config, name)
                for name in (
                    "hidden",
                    "updates",
                    "batch_size",
                    "learning_rate",
                    "l2",
                    "exploration_floor",
                    "variance_floor",
                    "seed",
                )
            },
        }
    )
    return (
        AutoregressivePolicy(
            network.eval(),
            tuple(map(float, mean)),
            tuple(map(float, scale)),
            config.exploration_floor,
            training_identity,
        ),
        history,
    )


def teacher_forced_metrics(
    policy: AutoregressivePolicy, traces: list[dict], *, maximum: int
) -> dict:
    """Measure weighted teacher-forced rule/control likelihood and rank."""

    rows = training_rows(traces, maximum)
    rule_nll = control_nll = rule_top1 = rule_top5 = 0.0
    rule_weight = control_weight = 0.0
    for row in rows:
        rule, control = policy.probabilities(row["features"])
        weight = float(row["weight"])
        control_nll -= weight * math.log(control[row["control_label"]])
        control_weight += weight
        if row["rule_label"] is not None:
            label = int(row["rule_label"])
            rule_nll -= weight * math.log(rule[label])
            order = np.argsort(-rule)
            rank = int(np.flatnonzero(order == label)[0]) + 1
            rule_top1 += weight * (rank <= 1)
            rule_top5 += weight * (rank <= 5)
            rule_weight += weight
    return {
        "teacher_rows": len(rows),
        "rule_rows": sum(row["rule_label"] is not None for row in rows),
        "weighted_rule_nll": rule_nll / rule_weight,
        "weighted_rule_top1": rule_top1 / rule_weight,
        "weighted_rule_top5": rule_top5 / rule_weight,
        "weighted_control_nll": control_nll / control_weight,
    }


def marginal_rule_control_probabilities(
    traces: list[dict], *, maximum: int, exploration_floor: float
) -> tuple[tuple[float, ...], tuple[float, float]]:
    """Fit source/route/decision-balanced generic rule and stop marginals."""

    if not 0 <= exploration_floor < 1:
        raise ValueError("exploration_floor must be in [0, 1)")
    rows = training_rows(traces, maximum)
    rules = np.zeros(len(RULES), dtype=float)
    controls = np.zeros(2, dtype=float)
    for row in rows:
        weight = float(row["weight"])
        controls[int(row["control_label"])] += weight
        if row["rule_label"] is not None:
            rules[int(row["rule_label"])] += weight
    rules /= rules.sum()
    controls /= controls.sum()
    rules = exploration_floor / len(RULES) + (1 - exploration_floor) * rules
    controls = exploration_floor / 2 + (1 - exploration_floor) * controls
    return tuple(map(float, rules / rules.sum())), tuple(
        map(float, controls / controls.sum())
    )


def fit_legal_where_how_ranker(
    traces: list[dict], *, maximum: int, config: RankerConfig
) -> tuple[dict, dict]:
    """Fit a target-free same-state ranker over exact legal action fibers."""

    if not traces:
        raise ValueError("legal WHERE/HOW fitting requires teacher traces")
    by_source: dict[str, list[dict]] = {}
    for trace in traces:
        by_source.setdefault(str(trace["source_group"]), []).append(trace)
    differences = []
    supported = decisions = fibers = successors = 0
    for source_id, routes in sorted(by_source.items()):
        del source_id
        for teacher in routes:
            states = tuple(teacher["trace"]["states"])
            actions = tuple(teacher["trace"]["actions"])
            if not actions or len(actions) > maximum:
                raise ValueError("legal-ranker teacher is outside decoder support")
            decision_weight = 1 / (len(by_source) * len(routes) * len(actions))
            for step, action in enumerate(actions):
                decisions += 1
                graph = decode_state(states[step])
                teacher_key = canonical_state_key(decode_state(states[step + 1]))
                candidates = enumerate_rule_successors(graph, action["executor_rule"])
                fibers += 1
                successors += len(candidates)
                teacher_candidate = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.successor_key == teacher_key
                    ),
                    None,
                )
                if teacher_candidate is None:
                    continue
                supported += 1
                teacher_features = action_features(graph, teacher_candidate)
                negatives = [
                    candidate
                    for candidate in candidates
                    if candidate.successor_key != teacher_key
                ][: config.negatives_per_teacher]
                if not negatives:
                    continue
                pair_weight = decision_weight / len(negatives)
                differences.extend(
                    (
                        teacher_features - action_features(graph, negative),
                        pair_weight,
                    )
                    for negative in negatives
                )
    if not differences:
        raise RuntimeError("no legal WHERE/HOW contrastive pairs were available")
    checkpoint = fit_diagonal_contrastive_ranker(differences, config)
    checkpoint["training_identity"] = identity(
        {
            "schema_version": "legal_where_how_training_identity_v1",
            "source_count": len(by_source),
            "trace_count": len(traces),
            "decision_count": decisions,
            "supported_decisions": supported,
            "configuration": {
                "negatives_per_teacher": config.negatives_per_teacher,
                "variance_floor": config.variance_floor,
                "coefficient_norm": config.coefficient_norm,
            },
        }
    )
    checkpoint["runtime_teacher_rows"] = 0
    return checkpoint, {
        "teacher_decisions": decisions,
        "supported_teacher_decisions": supported,
        "teacher_successor_coverage": supported / max(1, decisions),
        "fibers_enumerated": fibers,
        "legal_successors_enumerated": successors,
        "training_pairs": len(differences),
    }


def legal_where_how_metrics(traces: list[dict], checkpoint: dict) -> dict:
    """Evaluate teacher-successor ranks without changing the legal fiber."""

    if checkpoint.get("schema_version") != LEGAL_CHECKPOINT_SCHEMA:
        raise ValueError("legal WHERE/HOW checkpoint schema mismatch")
    ranks, uniform_ranks, decisions, supported = [], [], 0, 0
    for teacher in traces:
        states = tuple(teacher["trace"]["states"])
        for step, action in enumerate(teacher["trace"]["actions"]):
            decisions += 1
            graph = decode_state(states[step])
            teacher_key = canonical_state_key(decode_state(states[step + 1]))
            candidates = enumerate_rule_successors(graph, action["executor_rule"])
            if not any(row.successor_key == teacher_key for row in candidates):
                continue
            supported += 1
            scores = {
                row.successor_key: score_features(checkpoint, action_features(graph, row))
                for row in candidates
            }
            ranks.append(rank_of_teacher(candidates, teacher_key, scores))
            uniform_ranks.append(rank_of_teacher(candidates, teacher_key))

    def metrics(values: list[int | None]) -> dict:
        return {
            "mean_reciprocal_rank": float(
                np.mean([reciprocal_rank(rank) for rank in values])
            )
            if values
            else 0.0,
            "top1": float(np.mean([rank is not None and rank <= 1 for rank in values]))
            if values
            else 0.0,
            "top5": float(np.mean([rank is not None and rank <= 5 for rank in values]))
            if values
            else 0.0,
            "top10": float(
                np.mean([rank is not None and rank <= 10 for rank in values])
            )
            if values
            else 0.0,
        }

    return {
        "teacher_decisions": decisions,
        "supported_teacher_decisions": supported,
        "teacher_successor_coverage": supported / max(1, decisions),
        "uniform": metrics(uniform_ranks),
        "learned": metrics(ranks),
    }


@dataclass(frozen=True)
class Prefix:
    graph: object
    states: tuple[dict, ...]
    actions: tuple[dict, ...]
    score: float


def _prefix_identity(prefix: Prefix) -> str:
    return identity(
        {
            "schema_version": "autoregressive_prefix_identity_v1",
            "state": canonical_state_key(prefix.graph),
            "rules": [action["executor_rule"] for action in prefix.actions],
        }
    )


def _prefix_order(prefix: Prefix) -> tuple:
    return (-prefix.score, canonical_state_key(prefix.graph), _prefix_identity(prefix))


def _log_softmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("candidate scores must be finite and nonempty")
    maximum = float(array.max())
    return array - (maximum + math.log(float(np.exp(array - maximum).sum())))


def _candidate(prefix: Prefix, regions: int) -> dict:
    endpoint_key = canonical_state_key(prefix.graph)
    route_id = identity(
        {
            "schema_version": "autoregressive_route_identity_v1",
            "states": [
                canonical_state_key(decode_state(state)) for state in prefix.states
            ],
            "rules": [action["executor_rule"] for action in prefix.actions],
        }
    )
    return {
        "candidate_id": identity(
            {
                "schema_version": "autoregressive_candidate_identity_v1",
                "route_id": route_id,
                "model_score": prefix.score,
            }
        ),
        "route_identity": route_id,
        "model_score": prefix.score,
        "primitive_count": len(prefix.actions),
        "dependency_regions": regions,
        "endpoint_key": endpoint_key,
        "endpoint_state": prefix.states[-1],
        "trace": {"states": list(prefix.states), "actions": list(prefix.actions)},
    }


def _ranked(rows: list[dict], maximum: int) -> list[dict]:
    endpoints = {}
    for row in sorted(rows, key=lambda item: (-item["model_score"], item["candidate_id"])):
        endpoints.setdefault(row["endpoint_key"], row)
    return list(endpoints.values())[:maximum]


def decode_programs(
    source,
    *,
    decoder: str,
    config: DecoderConfig,
    marginal_rule_probabilities: tuple[float, ...],
    marginal_control_probabilities: tuple[float, float],
    policy: AutoregressivePolicy | None = None,
    legal_checkpoint: dict | None = None,
    successor_enumerator: Callable[[object, str], tuple[LegalSuccessor, ...]] = (
        enumerate_rule_successors
    ),
    legal_scorer: Callable[[object, LegalSuccessor, dict], float] = (
        lambda graph, candidate, checkpoint: score_features(
            checkpoint, action_features(graph, candidate)
        )
    ),
) -> dict:
    """Generate complete programs without teacher or task information."""

    if decoder not in DECODERS:
        raise ValueError(f"unknown decoder: {decoder}")
    rule_marginal = np.asarray(marginal_rule_probabilities, dtype=float)
    control_marginal = np.asarray(marginal_control_probabilities, dtype=float)
    if (
        rule_marginal.shape != (len(RULES),)
        or control_marginal.shape != (2,)
        or np.any(rule_marginal <= 0)
        or np.any(control_marginal <= 0)
        or not np.isclose(rule_marginal.sum(), 1)
        or not np.isclose(control_marginal.sum(), 1)
    ):
        raise ValueError("invalid marginal rule/control probabilities")
    if decoder == LEARNED:
        if policy is None or legal_checkpoint is None:
            raise ValueError("learned decoding requires rule/control and legal models")
        if legal_checkpoint.get("schema_version") != LEGAL_CHECKPOINT_SCHEMA:
            raise ValueError("legal-action checkpoint schema mismatch")
    source_key = canonical_state_key(source)
    live = (Prefix(source, (encode_state(source),), (), 0.0),)
    completed: list[dict] = []
    snapshots = {}
    telemetry = {
        "prefixes_expanded": 0,
        "rule_fibers_enumerated": 0,
        "legal_successors_enumerated": 0,
        "legal_successors_retained": 0,
        "canonical_prefix_duplicates": 0,
        "active_atom_abstentions": 0,
        "dependency_region_abstentions": 0,
        "self_endpoint_abstentions": 0,
        "completed_programs": 0,
        "exact_execution_failures": 0,
    }
    for depth in range(1, config.maximum_primitives + 1):
        expanded: list[Prefix] = []
        for prefix in live:
            telemetry["prefixes_expanded"] += 1
            if decoder == LEARNED:
                features = prefix_features(
                    source, prefix.graph, prefix.actions, config.maximum_primitives
                )
                rule_probabilities, control_probabilities = policy.probabilities(features)
            else:
                rule_probabilities = rule_marginal
                control_probabilities = control_marginal
            fibers = {}
            for rule in RULES:
                candidates = tuple(successor_enumerator(prefix.graph, rule))
                telemetry["rule_fibers_enumerated"] += 1
                telemetry["legal_successors_enumerated"] += len(candidates)
                if candidates:
                    fibers[rule] = candidates
            available = sum(rule_probabilities[RULES.index(rule)] for rule in fibers)
            if available <= 0:
                continue
            for rule in RULES:
                candidates = fibers.get(rule)
                if not candidates:
                    continue
                if decoder == LEARNED:
                    action_logs = _log_softmax(
                        [
                            legal_scorer(prefix.graph, candidate, legal_checkpoint)
                            for candidate in candidates
                        ]
                    )
                else:
                    action_logs = np.full(len(candidates), -math.log(len(candidates)))
                order = sorted(
                    range(len(candidates)),
                    key=lambda index: (
                        -float(action_logs[index]),
                        candidates[index].successor_key,
                    ),
                )[: config.successors_per_rule]
                telemetry["legal_successors_retained"] += len(order)
                rule_log = math.log(
                    rule_probabilities[RULES.index(rule)] / available
                )
                continue_log = math.log(control_probabilities[0])
                for index in order:
                    candidate = candidates[index]
                    if candidate.successor.n_real_atoms > config.maximum_active_atoms:
                        telemetry["active_atom_abstentions"] += 1
                        continue
                    actions = (*prefix.actions, candidate.action_record)
                    regions = dependency_region_count(actions)
                    if regions > config.maximum_regions:
                        telemetry["dependency_region_abstentions"] += 1
                        continue
                    expanded.append(
                        Prefix(
                            candidate.successor,
                            (*prefix.states, encode_state(candidate.successor)),
                            actions,
                            prefix.score
                            + continue_log
                            + rule_log
                            + float(action_logs[index]),
                        )
                    )
        best_by_state = {}
        for prefix in sorted(expanded, key=_prefix_order):
            key = canonical_state_key(prefix.graph)
            if key in best_by_state:
                telemetry["canonical_prefix_duplicates"] += 1
                continue
            best_by_state[key] = prefix
        next_live = []
        for prefix in best_by_state.values():
            if canonical_state_key(prefix.graph) == source_key:
                telemetry["self_endpoint_abstentions"] += 1
                continue
            try:
                regions = dependency_region_count(prefix.actions)
                if regions > config.maximum_regions:
                    telemetry["dependency_region_abstentions"] += 1
                    continue
                if decoder == LEARNED:
                    features = prefix_features(
                        source, prefix.graph, prefix.actions, config.maximum_primitives
                    )
                    _, control_probabilities = policy.probabilities(features)
                else:
                    control_probabilities = control_marginal
                stopped = Prefix(
                    prefix.graph,
                    prefix.states,
                    prefix.actions,
                    prefix.score + math.log(control_probabilities[1]),
                )
                completed.append(_candidate(stopped, regions))
                telemetry["completed_programs"] += 1
                if depth < config.maximum_primitives:
                    next_live.append(prefix)
            except (RuntimeError, ValueError):
                telemetry["exact_execution_failures"] += 1
        live = tuple(sorted(next_live, key=_prefix_order)[: config.beam_width])
        if depth in config.snapshots:
            snapshots[str(depth)] = _ranked(completed, config.maximum_outputs)
        if not live and depth < config.maximum_primitives:
            for future in config.snapshots:
                if future > depth:
                    snapshots[str(future)] = _ranked(
                        completed, config.maximum_outputs
                    )
            break
    if set(snapshots) != {str(value) for value in config.snapshots}:
        raise RuntimeError("decoder did not publish every declared snapshot")
    attempts = telemetry["completed_programs"] + telemetry["exact_execution_failures"]
    telemetry["exact_execution_precision"] = (
        telemetry["completed_programs"] / attempts if attempts else None
    )
    return {
        "schema_version": SCHEMA,
        "decoder": decoder,
        "configuration": {
            "maximum_primitives": config.maximum_primitives,
            "maximum_regions": config.maximum_regions,
            "maximum_active_atoms": config.maximum_active_atoms,
            "beam_width": config.beam_width,
            "successors_per_rule": config.successors_per_rule,
            "maximum_outputs": config.maximum_outputs,
            "snapshots": list(config.snapshots),
        },
        "snapshots": snapshots,
        "telemetry": telemetry,
        "teacher_fields_present": False,
        "task_identity_present": False,
        "new_oracle_calls": 0,
    }


__all__ = [
    "CANDIDATE_LOCK_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "DECODERS",
    "LEARNED",
    "MARGINAL",
    "AutoregressivePolicy",
    "DecoderConfig",
    "ProgramPolicyNetwork",
    "TrainingConfig",
    "decode_programs",
    "dependency_region_count",
    "fit_autoregressive_policy",
    "fit_legal_where_how_ranker",
    "legal_where_how_metrics",
    "marginal_rule_control_probabilities",
    "prefix_features",
    "teacher_forced_metrics",
    "training_rows",
]
