"""Split-first utility ranking of exact T4 source/endpoint graph pairs.

This module deliberately does not generate or execute molecular programs.  It
fits a small pairwise ranker only from measured, comparable docking outcomes.
Target conditioning is implemented as regularized interactions on top of one
shared structural model, not as separately fitted target controllers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from compose_v4.control.trajectory_value import molecule_features

TARGETS = ("5ht1b", "braf", "fa7", "jak2", "parp1")
CHECKPOINT_SCHEMA = "t4_target_conditioned_utility_ranker_v1"


def stable_identity(value: object) -> str:
    """Deterministic SHA-256 identity for JSON-compatible scientific rows."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MeasuredEndpoint:
    """One physical measured request joined to exact source and endpoint graphs."""

    row_id: str
    artifact: str
    target: str
    cell: str
    source_idx: int
    delta: float
    oracle_protocol: str
    docking_seed: int
    docking_score: float
    source_smiles: str
    endpoint_smiles: str
    source_state_sha256: str
    endpoint_state_sha256: str
    scaffold: str
    lineage_ids: tuple[str, ...]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            not self.row_id
            or not self.artifact
            or self.target not in TARGETS
            or self.cell != f"{self.target}_{self.source_idx}"
            or self.source_idx not in (0, 1, 2)
            or self.delta not in (0.4, 0.6)
            or not self.oracle_protocol
            or type(self.docking_seed) is not int
            or not math.isfinite(self.docking_score)
            or not self.source_smiles
            or not self.endpoint_smiles
            or len(self.source_state_sha256) != 64
            or len(self.endpoint_state_sha256) != 64
            or not self.scaffold
            or not self.lineage_ids
        ):
            raise ValueError(f"invalid measured endpoint row: {self.row_id}")

    @property
    def stratum(self) -> tuple[str, str, float, str, int]:
        return (
            self.target,
            self.cell,
            self.delta,
            self.oracle_protocol,
            self.docking_seed,
        )

    @property
    def utility(self) -> float:
        return -self.docking_score

    def public_payload(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "artifact": self.artifact,
            "target": self.target,
            "cell": self.cell,
            "source_idx": self.source_idx,
            "delta": self.delta,
            "oracle_protocol": self.oracle_protocol,
            "docking_seed": self.docking_seed,
            "docking_score": self.docking_score,
            "source_smiles": self.source_smiles,
            "endpoint_smiles": self.endpoint_smiles,
            "source_state_sha256": self.source_state_sha256,
            "endpoint_state_sha256": self.endpoint_state_sha256,
            "scaffold": self.scaffold,
            "lineage_ids": list(self.lineage_ids),
            "provenance": dict(self.provenance),
        }


def graph_pair_features(source_smiles: str, endpoint_smiles: str) -> np.ndarray:
    """Target-blind features for one exact, potentially unscored graph pair."""

    if not source_smiles or not endpoint_smiles:
        raise ValueError("graph-pair features require source and endpoint molecules")
    source = molecule_features(source_smiles).astype(np.float32)
    endpoint = molecule_features(endpoint_smiles).astype(np.float32)
    result = np.concatenate((source, endpoint, endpoint - source)).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite graph-pair features")
    return result


def endpoint_features(row: MeasuredEndpoint) -> np.ndarray:
    """Target-blind structural features shared by both fitted utility arms."""

    return graph_pair_features(row.source_smiles, row.endpoint_smiles)


def fixed_structural_graph_pair_score(source_smiles: str, endpoint_smiles: str) -> float:
    """Fit-free structural-control score for an unscored graph pair."""

    features = graph_pair_features(source_smiles, endpoint_smiles)
    width = len(features) // 3
    return -float(np.linalg.norm(features[-width:]))


def fixed_structural_score(row: MeasuredEndpoint) -> float:
    """Fit-free target-blind control favoring smaller feature displacement."""

    return fixed_structural_graph_pair_score(row.source_smiles, row.endpoint_smiles)


def strict_pairs(rows: Sequence[MeasuredEndpoint]) -> list[tuple[str, str]]:
    """Return high-utility/low-utility row identities within comparable strata."""

    by_stratum: dict[tuple, list[MeasuredEndpoint]] = defaultdict(list)
    for row in rows:
        by_stratum[row.stratum].append(row)
    result: list[tuple[str, str]] = []
    for stratum in sorted(by_stratum):
        group = sorted(by_stratum[stratum], key=lambda item: item.row_id)
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                if left.utility == right.utility:
                    continue
                if left.utility > right.utility:
                    result.append((left.row_id, right.row_id))
                else:
                    result.append((right.row_id, left.row_id))
    return result


def target_pair_counts(rows: Sequence[MeasuredEndpoint]) -> dict[str, int]:
    lookup = {row.row_id: row for row in rows}
    counts = {target: 0 for target in TARGETS}
    for better, _ in strict_pairs(rows):
        counts[lookup[better].target] += 1
    return counts


def _pair_rows(
    rows: Sequence[MeasuredEndpoint],
) -> list[tuple[MeasuredEndpoint, MeasuredEndpoint]]:
    lookup = {row.row_id: row for row in rows}
    return [(lookup[better], lookup[worse]) for better, worse in strict_pairs(rows)]


def _pair_weights(
    pairs: Sequence[tuple[MeasuredEndpoint, MeasuredEndpoint]],
) -> np.ndarray:
    """Equalize target, cell, stratum and strict pair mass in that order."""

    if not pairs:
        raise ValueError("utility ranker requires at least one strict pair")
    by_target: dict[str, list[int]] = defaultdict(list)
    by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_stratum: dict[tuple, list[int]] = defaultdict(list)
    for index, (better, worse) in enumerate(pairs):
        if better.stratum != worse.stratum:
            raise ValueError("utility pair crosses a comparability stratum")
        by_target[better.target].append(index)
        by_cell[(better.target, better.cell)].append(index)
        by_stratum[better.stratum].append(index)
    weights = np.zeros(len(pairs), dtype=np.float64)
    targets = sorted(by_target)
    for target in targets:
        cells = sorted({pairs[index][0].cell for index in by_target[target]})
        for cell in cells:
            strata = sorted(
                {pairs[index][0].stratum for index in by_cell[(target, cell)]}
            )
            for stratum in strata:
                indices = by_stratum[stratum]
                value = 1 / (len(targets) * len(cells) * len(strata) * len(indices))
                weights[indices] = value
    weights /= weights.sum()
    return weights


@dataclass(frozen=True)
class UtilityRanker:
    """Shared pairwise linear model, optionally with target interactions."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    base_coefficients: tuple[float, ...]
    target_coefficients: tuple[tuple[float, ...], ...]
    targets: tuple[str, ...]
    conditioned: bool
    training_targets: tuple[str, ...]
    training_identity: str

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean)
        scale = np.asarray(self.scale)
        base = np.asarray(self.base_coefficients)
        target = np.asarray(self.target_coefficients)
        if (
            mean.ndim != 1
            or not len(mean)
            or mean.shape != scale.shape
            or mean.shape != base.shape
            or target.shape != (len(self.targets), len(mean))
            or self.targets != TARGETS
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(base).all()
            or not np.isfinite(target).all()
            or np.any(scale <= 0)
            or not self.training_identity
        ):
            raise ValueError("invalid utility-ranker checkpoint")
        if not self.conditioned and np.any(target != 0):
            raise ValueError("target-blind ranker has target coefficients")

    def supports(self, target: str) -> bool:
        return not self.conditioned or target in self.training_targets

    def score(self, row: MeasuredEndpoint) -> float:
        return self.score_graph_pair(
            row.source_smiles,
            row.endpoint_smiles,
            target=row.target,
        )

    def score_graph_pair(
        self, source_smiles: str, endpoint_smiles: str, *, target: str
    ) -> float:
        """Score one unmeasured pair without fabricating a measured label row."""

        if not self.supports(target):
            raise ValueError(f"target absent from ranker training pairs: {target}")
        values = (
            graph_pair_features(source_smiles, endpoint_smiles) - np.asarray(self.mean)
        ) / np.asarray(self.scale)
        score = float(values @ np.asarray(self.base_coefficients))
        if self.conditioned:
            score += float(
                values
                @ np.asarray(self.target_coefficients)[self.targets.index(target)]
            )
        return score

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "base_coefficients": list(self.base_coefficients),
            "target_coefficients": [list(row) for row in self.target_coefficients],
            "targets": list(self.targets),
            "conditioned": self.conditioned,
            "training_targets": list(self.training_targets),
            "training_identity": self.training_identity,
            "runtime_rows": 0,
            "runtime_scores": 0,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, object]) -> UtilityRanker:
        expected = {
            "schema_version",
            "mean",
            "scale",
            "base_coefficients",
            "target_coefficients",
            "targets",
            "conditioned",
            "training_targets",
            "training_identity",
            "runtime_rows",
            "runtime_scores",
        }
        if (
            payload.get("schema_version") != CHECKPOINT_SCHEMA
            or set(payload) != expected
            or payload.get("runtime_rows") != 0
            or payload.get("runtime_scores") != 0
        ):
            raise ValueError("utility-ranker checkpoint schema mismatch")
        return cls(
            tuple(map(float, payload["mean"])),
            tuple(map(float, payload["scale"])),
            tuple(map(float, payload["base_coefficients"])),
            tuple(tuple(map(float, row)) for row in payload["target_coefficients"]),
            tuple(map(str, payload["targets"])),
            bool(payload["conditioned"]),
            tuple(map(str, payload["training_targets"])),
            str(payload["training_identity"]),
        )


def fit_utility_ranker(
    rows: Sequence[MeasuredEndpoint],
    *,
    conditioned: bool,
    updates: int,
    learning_rate: float,
    base_l2: float,
    target_interaction_l2: float,
    seed: int,
) -> tuple[UtilityRanker, dict[str, object]]:
    """Fit deterministic within-stratum pairwise logistic utility ranking."""

    if (
        not rows
        or updates <= 0
        or learning_rate <= 0
        or base_l2 < 0
        or target_interaction_l2 < 0
    ):
        raise ValueError("invalid utility-ranker fit configuration")
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError("utility-ranker fit rows contain duplicate identities")
    pairs = _pair_rows(rows)
    if not pairs:
        raise ValueError("utility-ranker training fold has no strict measured pairs")
    matrix = np.stack([endpoint_features(row) for row in rows]).astype(np.float32)
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 0.05)
    differences = np.stack(
        [
            (endpoint_features(better) - endpoint_features(worse)) / scale
            for better, worse in pairs
        ]
    ).astype(np.float32)
    weights = _pair_weights(pairs).astype(np.float32)
    target_indices = np.asarray([TARGETS.index(better.target) for better, _ in pairs])

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = torch.zeros(
            differences.shape[1], dtype=torch.float32, requires_grad=True
        )
        interactions = torch.zeros(
            (len(TARGETS), differences.shape[1]),
            dtype=torch.float32,
            requires_grad=True,
        )
    parameters = [base, interactions] if conditioned else [base]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    x = torch.from_numpy(differences)
    pair_weights = torch.from_numpy(weights)
    target_tensor = torch.from_numpy(target_indices)
    history = []
    for update in range(updates):
        margin = x @ base
        if conditioned:
            margin += (x * interactions[target_tensor]).sum(dim=1)
        loss = (pair_weights * F.softplus(-margin)).sum()
        loss += base_l2 * base.square().mean()
        if conditioned:
            loss += target_interaction_l2 * interactions.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite utility-ranker loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update in (0, updates - 1):
            history.append({"update": update + 1, "loss": float(loss.detach())})
    target_values = (
        interactions.detach().numpy()
        if conditioned
        else np.zeros((len(TARGETS), differences.shape[1]), dtype=np.float32)
    )
    training_targets = tuple(sorted({better.target for better, _ in pairs}))
    training_identity = stable_identity(
        {
            "schema_version": "t4_utility_ranker_training_identity_v1",
            "rows": sorted(row.row_id for row in rows),
            "pairs": sorted((better.row_id, worse.row_id) for better, worse in pairs),
            "conditioned": conditioned,
            "updates": updates,
            "learning_rate": learning_rate,
            "base_l2": base_l2,
            "target_interaction_l2": target_interaction_l2,
            "seed": seed,
        }
    )
    ranker = UtilityRanker(
        tuple(map(float, mean)),
        tuple(map(float, scale)),
        tuple(map(float, base.detach().numpy())),
        tuple(tuple(map(float, row)) for row in target_values),
        TARGETS,
        conditioned,
        training_targets,
        training_identity,
    )
    return ranker, {
        "rows": len(rows),
        "strict_pairs": len(pairs),
        "targets_with_pairs": list(training_targets),
        "history": history,
        "pair_weight_sum": float(weights.sum()),
    }


def _ndcg(group: Sequence[MeasuredEndpoint], scores: Mapping[str, float]) -> float:
    utilities = np.asarray([row.utility for row in group], dtype=float)
    relevance = utilities - utilities.min()
    if not np.any(relevance > 0):
        return 1.0
    predicted = sorted(group, key=lambda row: (-scores[row.row_id], row.row_id))
    ideal = sorted(group, key=lambda row: (-row.utility, row.row_id))

    def dcg(order: Sequence[MeasuredEndpoint]) -> float:
        return sum(
            float(relevance[list(group).index(row)]) / math.log2(rank + 2)
            for rank, row in enumerate(order)
        )

    return dcg(predicted) / dcg(ideal)


def evaluate_scores(
    rows: Sequence[MeasuredEndpoint], scores: Mapping[str, float]
) -> dict[str, object]:
    """Evaluate one score mapping on identical comparable candidate strata."""

    if set(scores) != {row.row_id for row in rows}:
        raise ValueError("evaluation scores do not match candidate rows")
    by_stratum: dict[tuple, list[MeasuredEndpoint]] = defaultdict(list)
    for row in rows:
        by_stratum[row.stratum].append(row)
    strata = []
    correct = predicted_pairs = strict_pair_count = 0
    for key in sorted(by_stratum):
        group = sorted(by_stratum[key], key=lambda row: row.row_id)
        if len(group) < 2:
            continue
        pairs = _pair_rows(group)
        if not pairs:
            continue
        stratum_correct = stratum_predicted = 0
        for better, worse in pairs:
            difference = scores[better.row_id] - scores[worse.row_id]
            if difference == 0:
                continue
            stratum_predicted += 1
            stratum_correct += int(difference > 0)
        ordered = sorted(group, key=lambda row: (-scores[row.row_id], row.row_id))
        best_score = min(row.docking_score for row in group)
        true_best = {row.row_id for row in group if row.docking_score == best_score}
        top3 = ordered[: min(3, len(ordered))]
        strata.append(
            {
                "target": key[0],
                "cell": key[1],
                "delta": key[2],
                "oracle_protocol": key[3],
                "docking_seed": key[4],
                "candidates": len(group),
                "strict_pairs": len(pairs),
                "pair_prediction_coverage": stratum_predicted / len(pairs),
                "pairwise_precision": (
                    stratum_correct / stratum_predicted if stratum_predicted else None
                ),
                "ndcg": _ndcg(group, scores),
                "top1_regret": ordered[0].docking_score - best_score,
                "top3_regret": min(row.docking_score for row in top3) - best_score,
                "best_recall_at_1": float(ordered[0].row_id in true_best),
                "best_recall_at_3": float(any(row.row_id in true_best for row in top3)),
                "best_selection_precision_at_1": float(ordered[0].row_id in true_best),
                "best_selection_precision_at_3": sum(
                    row.row_id in true_best for row in top3
                )
                / len(top3),
                "selected_top1_row_id": ordered[0].row_id,
                "selected_top1_docking_score": ordered[0].docking_score,
                "best_docking_score": best_score,
            }
        )
        strict_pair_count += len(pairs)
        predicted_pairs += stratum_predicted
        correct += stratum_correct
    if not strata:
        raise ValueError("evaluation has no comparable candidate stratum")

    def mean(name: str) -> float:
        values = [float(row[name]) for row in strata if row[name] is not None]
        return float(np.mean(values))

    return {
        "candidate_rows": len(rows),
        "strata": len(strata),
        "strict_pairs": strict_pair_count,
        "pair_prediction_coverage": predicted_pairs / strict_pair_count,
        "pairwise_precision": correct / predicted_pairs if predicted_pairs else None,
        "source_balanced_ndcg": mean("ndcg"),
        "source_balanced_top1_regret": mean("top1_regret"),
        "source_balanced_top3_regret": mean("top3_regret"),
        "source_balanced_best_recall_at_1": mean("best_recall_at_1"),
        "source_balanced_best_recall_at_3": mean("best_recall_at_3"),
        "source_balanced_best_selection_precision_at_1": mean(
            "best_selection_precision_at_1"
        ),
        "source_balanced_best_selection_precision_at_3": mean(
            "best_selection_precision_at_3"
        ),
        "per_stratum": strata,
    }


def rows_by_fold(
    rows: Iterable[MeasuredEndpoint], fold: int
) -> tuple[list[MeasuredEndpoint], list[MeasuredEndpoint]]:
    if fold not in (0, 1, 2):
        raise ValueError(f"invalid T4 source fold: {fold}")
    train, test = [], []
    for row in rows:
        (test if row.source_idx == fold else train).append(row)
    return sorted(train, key=lambda row: row.row_id), sorted(
        test, key=lambda row: row.row_id
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "TARGETS",
    "MeasuredEndpoint",
    "UtilityRanker",
    "endpoint_features",
    "evaluate_scores",
    "fit_utility_ranker",
    "fixed_structural_graph_pair_score",
    "fixed_structural_score",
    "graph_pair_features",
    "rows_by_fold",
    "stable_identity",
    "strict_pairs",
    "target_pair_counts",
]
