"""Grouped, zero-oracle comparison of complete-program proposal policies.

The module owns no T4 task oracle.  Teacher endpoints are accepted only by the
offline training and shared-panel paths.  Autonomous generation receives an
exact source graph and numeric, target-free checkpoints.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v1 import (
    GENERIC_MODULES,
    _following_context,
    compile_generic_module_v1,
)
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.option_demonstrations import recognize_trace
from compose_v4.control.route_distilled_program_policy import (
    option_family_weights,
    stage_descriptor,
)
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state

SCHEMA = "t4_route_policy_comparison_v1"
MODEL_SCHEMA = "t4_contrastive_complete_candidate_ranker_v1"
POLICY_GENERIC = "generic_marginal"
POLICY_ACTOR = "context_module_prototype"
POLICY_HYBRID = "graph_contrastive_candidate_ranker"


@dataclass(frozen=True)
class ComparisonConfig:
    attempts_per_source: int = 128
    training_negative_attempts_per_source: int = 32
    max_modules: int = 3
    max_primitives: int = 32
    max_blocks: int = 8
    exploration_floor: float = 0.10
    ranker_updates: int = 400
    ranker_learning_rate: float = 0.03
    ranker_l2: float = 0.01
    seed: int = 20260914

    def __post_init__(self):
        for name in (
            "attempts_per_source",
            "training_negative_attempts_per_source",
            "max_modules",
            "max_primitives",
            "max_blocks",
            "ranker_updates",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_modules > 3:
            raise ValueError("the declared generic support has at most three modules")
        if not 0 < self.exploration_floor < 1:
            raise ValueError("exploration floor must be in (0,1)")
        if self.ranker_learning_rate <= 0 or self.ranker_l2 < 0:
            raise ValueError("invalid contrastive ranker optimization settings")


@dataclass(frozen=True)
class MarginalPolicy:
    family_probabilities: tuple[float, ...]
    module_count_probabilities: tuple[float, float, float]
    exploration_floor: float
    training_identity: str

    def __post_init__(self):
        families = np.asarray(self.family_probabilities, dtype=float)
        counts = np.asarray(self.module_count_probabilities, dtype=float)
        if (
            families.shape != (len(GENERIC_MODULES),)
            or counts.shape != (3,)
            or np.any(families <= 0)
            or np.any(counts <= 0)
            or not np.isclose(families.sum(), 1)
            or not np.isclose(counts.sum(), 1)
            or not 0 < self.exploration_floor < 1
            or not self.training_identity
        ):
            raise ValueError("invalid source-balanced marginal policy")

    def score(self, families: tuple[str, ...]) -> float:
        if (
            not families
            or len(families) > 3
            or any(family not in GENERIC_MODULES for family in families)
        ):
            return float("-inf")
        probabilities = dict(
            zip(GENERIC_MODULES, self.family_probabilities, strict=True)
        )
        return math.log(self.module_count_probabilities[len(families) - 1]) + sum(
            math.log(probabilities[family]) for family in families
        )

    def checkpoint(self) -> dict:
        return {
            "schema_version": "t4_source_balanced_marginal_v1",
            "families": list(GENERIC_MODULES),
            "family_probabilities": list(self.family_probabilities),
            "module_count_probabilities": list(self.module_count_probabilities),
            "exploration_floor": self.exploration_floor,
            "training_identity": self.training_identity,
        }


@dataclass(frozen=True)
class Candidate:
    attempt_id: str
    status: str
    endpoint_state: dict | None
    actions: tuple[dict, ...]
    families: tuple[str, ...]
    blocks: int
    generation_index: int
    failure: str | None = None

    def __post_init__(self):
        if self.status == "complete":
            if self.endpoint_state is None or not self.actions or not self.families:
                raise ValueError("complete candidate lacks its executed representation")
        elif self.endpoint_state is not None or self.actions or self.families:
            raise ValueError("rejected candidate cannot carry fabricated chemistry")


def require_nonself_endpoint(source, endpoint) -> None:
    """Reject executed programs whose canonical endpoint equals their source."""

    if canonical_state_key(endpoint) == canonical_state_key(source):
        raise ValueError("canonical self event")


def predeclared_source_folds(source_metadata: dict[str, dict]) -> tuple[dict, ...]:
    """Hold out one source seed per protein in each of three fixed folds."""

    if len(source_metadata) != 15:
        raise ValueError("the predeclared T4 split requires exactly 15 sources")
    folds = []
    for fold in range(3):
        test = sorted(
            source
            for source, row in source_metadata.items()
            if int(row["source_idx"]) == fold
        )
        train = sorted(set(source_metadata) - set(test))
        proteins = {source_metadata[source]["target"] for source in test}
        if len(test) != 5 or len(proteins) != 5 or set(test) & set(train):
            raise ValueError("each grouped fold must hold one seed from five proteins")
        folds.append(
            {
                "fold": fold,
                "train_sources": train,
                "calibration_sources": [],
                "test_sources": test,
            }
        )
    if Counter(source for row in folds for source in row["test_sources"]) != Counter(
        source_metadata.keys()
    ):
        raise ValueError("each source must be held out exactly once")
    return tuple(folds)


def predeclared_target_folds(source_metadata: dict[str, dict]) -> tuple[dict, ...]:
    """Hold out every source from one target in each fixed target fold."""

    targets = sorted({str(row["target"]) for row in source_metadata.values()})
    if len(source_metadata) != 15 or len(targets) != 5:
        raise ValueError("the predeclared T4 target split requires 15 sources and 5 targets")
    folds = []
    for fold, target in enumerate(targets):
        test = sorted(
            source
            for source, row in source_metadata.items()
            if str(row["target"]) == target
        )
        train = sorted(set(source_metadata) - set(test))
        if len(test) != 3 or set(test) & set(train):
            raise ValueError("each target fold must hold all three seeds for one target")
        folds.append(
            {
                "fold": fold,
                "held_out_target": target,
                "train_sources": train,
                "calibration_sources": [],
                "test_sources": test,
            }
        )
    if Counter(source for row in folds for source in row["test_sources"]) != Counter(
        source_metadata.keys()
    ):
        raise ValueError("each source must be held out exactly once in target folds")
    return tuple(folds)


def _recognized_families(trace: dict) -> tuple[dict[str, float], int]:
    segments = recognize_trace(trace["states"], trace["actions"])
    totals = Counter()
    for segment in segments:
        for family, weight in option_family_weights(segment.option).items():
            totals[family] += float(weight)
    total = sum(totals.values())
    return (
        {family: totals[family] / max(total, 1.0) for family in GENERIC_MODULES},
        len(segments),
    )


def fit_marginal_policy(
    traces: list[dict], *, exploration_floor: float
) -> MarginalPolicy:
    """Fit source/route/decision-balanced family and route-depth marginals."""

    if not traces:
        raise ValueError("marginal fitting requires teacher traces")
    by_source: dict[str, list[dict]] = defaultdict(list)
    for teacher in traces:
        by_source[teacher["source_group"]].append(teacher)
    family_counts = np.zeros(len(GENERIC_MODULES), dtype=float)
    count_counts = np.zeros(3, dtype=float)
    for source, routes in sorted(by_source.items()):
        del source
        for teacher in routes:
            family_weights, decisions = _recognized_families(teacher["trace"])
            route_weight = 1 / (len(by_source) * len(routes))
            for family, value in family_weights.items():
                family_counts[GENERIC_MODULES.index(family)] += route_weight * value
            count_counts[min(3, decisions) - 1] += route_weight
    family_empirical = family_counts / family_counts.sum()
    count_empirical = count_counts / count_counts.sum()
    family_uniform = np.full(len(GENERIC_MODULES), 1 / len(GENERIC_MODULES))
    count_uniform = np.full(3, 1 / 3)
    family = (
        exploration_floor * family_uniform + (1 - exploration_floor) * family_empirical
    )
    counts = (
        exploration_floor * count_uniform + (1 - exploration_floor) * count_empirical
    )
    training_identity = identity(
        {
            "schema_version": "t4_marginal_training_identity_v1",
            "sources": len(by_source),
            "routes": len(traces),
            "family_counts": family_counts.tolist(),
            "count_counts": count_counts.tolist(),
            "exploration_floor": exploration_floor,
        }
    )
    return MarginalPolicy(
        tuple(family), tuple(counts), exploration_floor, training_identity
    )


def _weighted_order(
    rng, names: tuple[str, ...], probabilities: np.ndarray
) -> list[str]:
    remaining, ordered = list(names), []
    weights = dict(zip(names, probabilities, strict=True))
    while remaining:
        current = np.asarray([weights[name] for name in remaining], dtype=float)
        index = int(rng.choice(len(remaining), p=current / current.sum()))
        ordered.append(remaining.pop(index))
    return ordered


def _synthesize_marginal(source, rng, policy: MarginalPolicy, config: ComparisonConfig):
    count = int(
        rng.choice(
            np.arange(1, config.max_modules + 1),
            p=np.asarray(policy.module_count_probabilities[: config.max_modules])
            / sum(policy.module_count_probabilities[: config.max_modules]),
        )
    )
    current, stages, families = source, [], []
    preferred, panel_cache = frozenset(), {}
    for _ in range(count):
        accepted = None
        for family in _weighted_order(
            rng, GENERIC_MODULES, np.asarray(policy.family_probabilities)
        ):
            try:
                product, stage = compile_generic_module_v1(
                    current,
                    rng,
                    family,
                    preferred_anchors=preferred,
                    panel_cache=panel_cache,
                )
                program, assignment = extract_program(source, [*stages, stage])
            except ValueError:
                continue
            if (
                len(program.marks) > config.max_primitives
                or len(program.blocks) > config.max_blocks
            ):
                continue
            accepted = product, stage, program, assignment, family
            break
        if accepted is None:
            if not stages:
                raise ValueError("no marginal module compiled")
            break
        current, stage, program, assignment, family = accepted
        stages.append(stage)
        families.append(family)
        preferred = _following_context(current, stage)
    program, assignment = extract_program(source, stages)
    endpoint, trace = execute_program_graph(
        source,
        compile_program_graph(program),
        assignment,
        max_primitives=config.max_primitives,
        max_blocks=config.max_blocks,
    )
    if canonical_state_key(endpoint) != canonical_state_key(current):
        raise RuntimeError("marginal synthesis changed on exact replay")
    require_nonself_endpoint(source, endpoint)
    return endpoint, trace, tuple(families), len(program.blocks)


def generate_marginal_candidates(
    source,
    policy: MarginalPolicy,
    config: ComparisonConfig,
    *,
    seed: int,
    prefix: str,
    progress_callback=None,
) -> tuple[list[Candidate], float]:
    rng = np.random.default_rng(seed)
    began, result = perf_counter(), []
    for index in range(config.attempts_per_source):
        try:
            endpoint, trace, families, blocks = _synthesize_marginal(
                source, rng, policy, config
            )
            result.append(
                Candidate(
                    f"{prefix}-{index}",
                    "complete",
                    encode_state(endpoint),
                    tuple(trace["actions"]),
                    families,
                    blocks,
                    index,
                )
            )
        except ValueError as error:
            result.append(
                Candidate(
                    f"{prefix}-{index}",
                    "rejected",
                    None,
                    (),
                    (),
                    0,
                    index,
                    str(error),
                )
            )
        if progress_callback is not None and (
            (index + 1) % 16 == 0 or index + 1 == config.attempts_per_source
        ):
            progress_callback(index + 1, config.attempts_per_source)
    return result, perf_counter() - began


def candidate_features(
    source,
    endpoint,
    actions: tuple[dict, ...],
    family_weights: dict[str, float],
    *,
    module_count: int,
    blocks: int,
    max_primitives: int = 32,
    max_blocks: int = 8,
) -> np.ndarray:
    """Address-free source, endpoint, change, module, dependency and stop features."""

    source_vector = molecule_features(canonical_state_key(source)).astype(np.float32)
    endpoint_vector = molecule_features(canonical_state_key(endpoint)).astype(
        np.float32
    )
    delta = endpoint_vector - source_vector
    stage = stage_descriptor(source, actions).astype(np.float32)
    family = np.asarray(
        [family_weights.get(name, 0.0) for name in GENERIC_MODULES], dtype=np.float32
    )
    if family.sum() <= 0:
        raise ValueError("candidate features require at least one generic family")
    family /= family.sum()
    numeric = np.asarray(
        [
            min(module_count, 3) / 3,
            len(actions) / max_primitives,
            blocks / max_blocks,
            (max_primitives - len(actions)) / max_primitives,
            (max_blocks - blocks) / max_blocks,
            source.n_real_atoms / 40,
            endpoint.n_real_atoms / 40,
        ],
        dtype=np.float32,
    )
    result = np.concatenate(
        (source_vector, endpoint_vector, delta, stage, family, numeric)
    )
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite complete-candidate features")
    return result


@dataclass(frozen=True)
class ContrastiveRanker:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    exploration_floor: float
    training_identity: str

    def __post_init__(self):
        mean = np.asarray(self.mean)
        scale = np.asarray(self.scale)
        coefficients = np.asarray(self.coefficients)
        if (
            not len(mean)
            or mean.shape != scale.shape
            or mean.shape != coefficients.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(coefficients).all()
            or np.any(scale <= 0)
            or not 0 < self.exploration_floor < 1
            or not self.training_identity
        ):
            raise ValueError("invalid contrastive complete-candidate ranker")

    def score(self, features: np.ndarray) -> float:
        value = np.asarray(features, dtype=float)
        if value.shape != (len(self.mean),):
            raise ValueError("complete-candidate feature dimension mismatch")
        return float(
            ((value - np.asarray(self.mean)) / np.asarray(self.scale))
            @ np.asarray(self.coefficients)
        )

    def checkpoint(self) -> dict:
        return {
            "schema_version": MODEL_SCHEMA,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "exploration_floor": self.exploration_floor,
            "training_identity": self.training_identity,
            "runtime_teacher_rows": 0,
            "runtime_executable_rows": 0,
        }


def fit_contrastive_ranker(
    positives: list[tuple[str, str, np.ndarray]],
    negatives: dict[str, list[np.ndarray]],
    config: ComparisonConfig,
) -> tuple[ContrastiveRanker, dict]:
    """Fit a source/route/negative-balanced linear pairwise ranker."""

    if not positives:
        raise ValueError("contrastive fitting requires positives")
    sources = sorted({source for source, _, _ in positives})
    routes_by_source = Counter(source for source, _, _ in positives)
    all_features = [feature for _, _, feature in positives]
    all_features.extend(feature for rows in negatives.values() for feature in rows)
    matrix = np.stack(all_features).astype(np.float32)
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 0.05)
    differences, pair_weights = [], []
    for source, route, positive in positives:
        source_negatives = negatives.get(source, [])
        if not source_negatives:
            continue
        weight = 1 / (len(sources) * routes_by_source[source] * len(source_negatives))
        for negative in source_negatives:
            differences.append((positive - negative) / scale)
            pair_weights.append(weight)
    if not differences:
        raise ValueError("contrastive fitting has no same-source executable negatives")
    x = torch.from_numpy(np.stack(differences).astype(np.float32))
    weights = torch.tensor(pair_weights, dtype=torch.float32)
    weights /= weights.sum()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        coefficients = torch.zeros(x.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([coefficients], lr=config.ranker_learning_rate)
    history = []
    for update in range(config.ranker_updates):
        margins = x @ coefficients
        loss = (weights * F.softplus(-margins)).sum()
        loss += config.ranker_l2 * coefficients.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite contrastive ranker loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update in (0, config.ranker_updates - 1):
            history.append({"update": update + 1, "loss": float(loss.detach())})
    training_identity = identity(
        {
            "schema_version": "t4_contrastive_training_identity_v1",
            "sources": len(sources),
            "routes": len(positives),
            "pairs": len(differences),
            "configuration": {
                "updates": config.ranker_updates,
                "learning_rate": config.ranker_learning_rate,
                "l2": config.ranker_l2,
                "exploration_floor": config.exploration_floor,
            },
            "feature_moments": identity(
                {"mean": mean.tolist(), "scale": scale.tolist()}
            ),
        }
    )
    model = ContrastiveRanker(
        tuple(map(float, mean)),
        tuple(map(float, scale)),
        tuple(map(float, coefficients.detach().numpy())),
        config.exploration_floor,
        training_identity,
    )
    return model, {"pairs": len(differences), "history": history}


def complete_attempt_rows(
    candidates: list[Candidate], scores: dict[str, float] | None = None
) -> list[dict]:
    """Rank complete candidates, then retain every failed attempt for precision."""

    complete = [candidate for candidate in candidates if candidate.status == "complete"]
    rejected = [candidate for candidate in candidates if candidate.status != "complete"]
    if scores is None:
        complete.sort(key=lambda row: row.generation_index)
    else:
        complete.sort(
            key=lambda row: (
                -scores[row.attempt_id],
                row.generation_index,
                row.attempt_id,
            )
        )
    ordered = [*complete, *rejected]
    return [
        {
            "attempt_id": candidate.attempt_id,
            "rank": rank,
            "status": candidate.status,
            "endpoint_state": candidate.endpoint_state,
        }
        for rank, candidate in enumerate(ordered, 1)
    ]


def teacher_candidate(teacher: dict, *, attempt_id: str, index: int) -> Candidate:
    families, decisions = _recognized_families(teacher["trace"])
    ordered = tuple(family for family, weight in sorted(families.items()) if weight > 0)
    # ``families`` here is only a feature summary. The literal teacher route is
    # never exposed to autonomous generation.
    return Candidate(
        attempt_id,
        "complete",
        teacher["trace"]["states"][-1],
        tuple(teacher["trace"]["actions"]),
        ordered[: max(1, min(3, decisions))],
        1,
        index,
    )


def family_feature_weights(candidate: Candidate) -> dict[str, float]:
    totals = Counter(candidate.families)
    return {family: float(totals[family]) for family in GENERIC_MODULES}
