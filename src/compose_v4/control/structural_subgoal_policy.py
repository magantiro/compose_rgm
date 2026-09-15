"""Split-first proposal policies over transferable structural graph deltas.

The runtime object is a training-fold vocabulary of address-free patch deltas,
not a collection of source molecules or executable teacher routes.  A patch is
rebound to the exact current graph, materialized with current atom state, and
compiled by the ordinary structural-subgoal realizer.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from compose_v4.chem.molecular_graph import ELEMENTS, MolecularGraph, is_element
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import atom_signature, environment
from compose_v4.control.structural_subgoal import (
    StructuralGoal,
    StructuralSubgoal,
    instantiate_goal,
)
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state

TEMPLATE_SCHEMA = "structural_delta_template_v1"
MARGINAL_SCHEMA = "t4_structural_subgoal_marginal_v1"
RANKER_SCHEMA = "t4_structural_subgoal_context_ranker_v1"


@dataclass(frozen=True)
class StructuralDeltaTemplate:
    """A minimal address-free before/after patch learned on training sources."""

    input_atoms: tuple[tuple[int, int, int, int], ...]
    input_bonds: tuple[tuple[int, ...], ...]
    target_atoms: tuple[tuple[int, int, int, int] | None, ...]
    output_atoms: tuple[tuple[int, int, int, int], ...]
    target_bonds: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        n_input = len(self.input_atoms)
        n_target = n_input + len(self.output_atoms)
        if not n_input or len(self.target_atoms) != n_input:
            raise ValueError("a structural delta needs aligned nonempty input roles")
        if len(self.input_bonds) != n_input or any(len(row) != n_input for row in self.input_bonds):
            raise ValueError("structural delta input bonds are malformed")
        if len(self.target_bonds) != n_target or any(
            len(row) != n_target for row in self.target_bonds
        ):
            raise ValueError("structural delta target bonds are malformed")
        for matrix in (self.input_bonds, self.target_bonds):
            if any(matrix[i][i] for i in range(len(matrix))) or any(
                matrix[i][j] != matrix[j][i] for i in range(len(matrix)) for j in range(len(matrix))
            ):
                raise ValueError("structural delta bond matrices must be symmetric")

    @property
    def template_id(self) -> str:
        return identity(self.payload())

    def payload(self) -> dict:
        return {"schema_version": TEMPLATE_SCHEMA, **asdict(self)}

    @classmethod
    def from_payload(cls, payload: dict) -> StructuralDeltaTemplate:
        expected = {
            "schema_version",
            "input_atoms",
            "input_bonds",
            "target_atoms",
            "output_atoms",
            "target_bonds",
        }
        if payload.get("schema_version") != TEMPLATE_SCHEMA or set(payload) != expected:
            raise ValueError("unexpected structural-delta template schema")
        return cls(
            input_atoms=tuple(tuple(map(int, row)) for row in payload["input_atoms"]),
            input_bonds=tuple(tuple(map(int, row)) for row in payload["input_bonds"]),
            target_atoms=tuple(
                None if row is None else tuple(map(int, row)) for row in payload["target_atoms"]
            ),
            output_atoms=tuple(tuple(map(int, row)) for row in payload["output_atoms"]),
            target_bonds=tuple(tuple(map(int, row)) for row in payload["target_bonds"]),
        )


def minimize_subgoal(subgoal: StructuralSubgoal) -> tuple[StructuralDeltaTemplate, tuple[int, ...]]:
    """Drop unchanged context roles while preserving the complete graph delta."""

    n_input = len(subgoal.input_atoms)
    retained: set[int] = {
        index
        for index, (before, after) in enumerate(
            zip(subgoal.input_atoms, subgoal.target_atoms, strict=True)
        )
        if before != after
    }
    for left in range(n_input):
        for right in range(left + 1, n_input):
            if subgoal.input_bonds[left][right] != subgoal.target_bonds[left][right]:
                retained.update((left, right))
        if any(
            subgoal.target_bonds[left][n_input + output]
            for output in range(len(subgoal.output_atoms))
        ):
            retained.add(left)
    if not retained:
        raise ValueError("structural subgoal has no transferable graph delta")
    inputs = tuple(sorted(retained))
    targets = (*inputs, *range(n_input, n_input + len(subgoal.output_atoms)))
    template = StructuralDeltaTemplate(
        input_atoms=tuple(subgoal.input_atoms[index] for index in inputs),
        input_bonds=tuple(
            tuple(subgoal.input_bonds[left][right] for right in inputs) for left in inputs
        ),
        target_atoms=tuple(subgoal.target_atoms[index] for index in inputs),
        output_atoms=subgoal.output_atoms,
        target_bonds=tuple(
            tuple(subgoal.target_bonds[left][right] for right in targets) for left in targets
        ),
    )
    return template, inputs


@dataclass(frozen=True)
class TransferBindingCensus:
    assignments: tuple[tuple[int, ...], ...]
    visits: int
    truncated: bool


def transfer_bindings(
    template: StructuralDeltaTemplate,
    graph: MolecularGraph,
    *,
    max_bindings: int = 8,
    max_visits: int = 16_384,
) -> TransferBindingCensus:
    """Bind topology and atom role, while allowing charge/H context to adapt."""

    if min(max_bindings, max_visits) < 1:
        raise ValueError("transfer binding limits must be positive")
    real = tuple(int(slot) for slot in np.flatnonzero(is_element(graph.atom_types)))
    candidates = [
        [
            slot
            for slot in real
            if int(graph.atom_types[slot]) == signature[0]
            and int(np.count_nonzero(graph.bonds[slot])) == signature[3]
        ]
        for signature in template.input_atoms
    ]
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            len(candidates[index]),
            -sum(bool(value) for value in template.input_bonds[index]),
            index,
        ),
    )
    assigned: dict[int, int] = {}
    rows: list[tuple[int, ...]] = []
    visits = 0
    truncated = False

    def visit(depth: int) -> None:
        nonlocal visits, truncated
        if len(rows) >= max_bindings or visits >= max_visits:
            truncated = True
            return
        visits += 1
        if depth == len(order):
            rows.append(tuple(assigned[index] for index in range(len(order))))
            return
        role = order[depth]
        for slot in candidates[role]:
            if slot in assigned.values():
                continue
            if any(
                int(graph.bonds[slot, other_slot]) != template.input_bonds[role][other_role]
                for other_role, other_slot in assigned.items()
            ):
                continue
            assigned[role] = slot
            visit(depth + 1)
            del assigned[role]
            if truncated:
                return

    visit(0)
    return TransferBindingCensus(tuple(rows), visits, truncated)


def materialize_template(
    template: StructuralDeltaTemplate,
    graph: MolecularGraph,
    binding: tuple[int, ...],
) -> StructuralSubgoal:
    """Turn one generic delta into a concrete address-free current-state goal."""

    if len(binding) != len(template.input_atoms) or len(set(binding)) != len(binding):
        raise ValueError("transfer binding is malformed")
    current = tuple(atom_signature(graph, slot) for slot in binding)
    targets = []
    for learned_before, learned_after, actual in zip(
        template.input_atoms, template.target_atoms, current, strict=True
    ):
        if learned_after is None:
            targets.append(None)
            continue
        element = actual[0] if learned_after[0] == learned_before[0] else learned_after[0]
        charge = actual[1] if learned_after[1] == learned_before[1] else learned_after[1]
        hydrogens = actual[2] + learned_after[2] - learned_before[2]
        degree = actual[3] + learned_after[3] - learned_before[3]
        if not (0 <= hydrogens <= 4 and 0 <= degree <= 6):
            raise ValueError("transferred atom-state delta is outside support")
        targets.append((element, charge, hydrogens, degree))
    return StructuralSubgoal(
        input_atoms=current,
        input_bonds=template.input_bonds,
        environments=tuple(environment(graph, slot) for slot in binding),
        target_atoms=tuple(targets),
        output_atoms=template.output_atoms,
        target_bonds=template.target_bonds,
    )


def template_features(template: StructuralDeltaTemplate) -> np.ndarray:
    """Fixed-size, task-free summary of a structural graph delta."""

    n_input = len(template.input_atoms)
    before_atoms = Counter(row[0] for row in template.input_atoms)
    after_atoms = Counter(row[0] for row in template.target_atoms if row is not None)
    output_atoms = Counter(row[0] for row in template.output_atoms)
    deleted = sum(row is None for row in template.target_atoms)
    restated = sum(
        after is not None and before[:3] != after[:3]
        for before, after in zip(template.input_atoms, template.target_atoms, strict=True)
    )
    input_edges = Counter(
        template.input_bonds[left][right]
        for left in range(n_input)
        for right in range(left + 1, n_input)
        if template.input_bonds[left][right]
    )
    target_edges = Counter(
        template.target_bonds[left][right]
        for left in range(len(template.target_bonds))
        for right in range(left + 1, len(template.target_bonds))
        if template.target_bonds[left][right]
    )
    numeric = [
        n_input / 40,
        len(template.output_atoms) / 40,
        deleted / 40,
        restated / 40,
        (len(template.output_atoms) - deleted) / 40,
    ]
    numeric.extend(before_atoms[index] / max(1, n_input) for index in range(len(ELEMENTS)))
    numeric.extend(after_atoms[index] / max(1, n_input) for index in range(len(ELEMENTS)))
    numeric.extend(
        output_atoms[index] / max(1, len(template.output_atoms)) for index in range(len(ELEMENTS))
    )
    numeric.extend(input_edges[index] / 40 for index in range(1, 5))
    numeric.extend(target_edges[index] / 40 for index in range(1, 5))
    result = np.asarray(numeric, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite structural-delta template features")
    return result


def proposal_features(
    source: MolecularGraph,
    endpoint: MolecularGraph,
    templates: tuple[StructuralDeltaTemplate, ...],
) -> np.ndarray:
    """Graph-conditioned complete structural-goal features."""

    if not templates or len(templates) > 4:
        raise ValueError("proposal features require one to four templates")
    source_features = molecule_features(canonical_state_key(source)).astype(np.float32)
    endpoint_features = molecule_features(canonical_state_key(endpoint)).astype(np.float32)
    template_matrix = np.stack([template_features(row) for row in templates])
    aggregate = np.concatenate(
        (
            template_matrix.mean(axis=0),
            template_matrix.max(axis=0),
            np.asarray(
                [
                    len(templates) / 4,
                    source.n_real_atoms / 40,
                    endpoint.n_real_atoms / 40,
                    (endpoint.n_real_atoms - source.n_real_atoms) / 40,
                ],
                dtype=np.float32,
            ),
        )
    )
    result = np.concatenate(
        (source_features, endpoint_features, endpoint_features - source_features, aggregate)
    )
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite structural-subgoal proposal features")
    return result


@dataclass(frozen=True)
class MarginalSubgoalPolicy:
    template_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    goal_count_probabilities: tuple[float, float, float, float]
    exploration_floor: float
    training_identity: str

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=float)
        counts = np.asarray(self.goal_count_probabilities, dtype=float)
        if (
            not self.template_ids
            or len(set(self.template_ids)) != len(self.template_ids)
            or probabilities.shape != (len(self.template_ids),)
            or np.any(probabilities <= 0)
            or not np.isclose(probabilities.sum(), 1)
            or counts.shape != (4,)
            or np.any(counts <= 0)
            or not np.isclose(counts.sum(), 1)
            or not 0 < self.exploration_floor < 1
            or not self.training_identity
        ):
            raise ValueError("invalid source-balanced subgoal marginal")

    def score(self, templates: tuple[StructuralDeltaTemplate, ...]) -> float:
        lookup = dict(zip(self.template_ids, self.probabilities, strict=True))
        if not templates or len(templates) > 4:
            return float("-inf")
        return math.log(self.goal_count_probabilities[len(templates) - 1]) + sum(
            math.log(lookup[row.template_id]) for row in templates
        )

    def checkpoint(self) -> dict:
        return {
            "schema_version": MARGINAL_SCHEMA,
            "template_ids": list(self.template_ids),
            "probabilities": list(self.probabilities),
            "goal_count_probabilities": list(self.goal_count_probabilities),
            "exploration_floor": self.exploration_floor,
            "training_identity": self.training_identity,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict) -> MarginalSubgoalPolicy:
        if payload.get("schema_version") != MARGINAL_SCHEMA:
            raise ValueError("structural-subgoal marginal schema mismatch")
        return cls(
            tuple(payload["template_ids"]),
            tuple(map(float, payload["probabilities"])),
            tuple(map(float, payload["goal_count_probabilities"])),
            float(payload["exploration_floor"]),
            str(payload["training_identity"]),
        )


def fit_marginal_subgoal_policy(
    rows: list[dict], *, exploration_floor: float
) -> MarginalSubgoalPolicy:
    """Fit source, route and region-balanced patch/count frequencies."""

    if not rows or not 0 < exploration_floor < 1:
        raise ValueError("marginal fitting requires rows and an exploration floor")
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source_group"]].append(row)
    weights: Counter[str] = Counter()
    counts = np.zeros(4, dtype=float)
    for routes in by_source.values():
        for route in routes:
            route_weight = 1 / (len(by_source) * len(routes))
            templates = route["templates"]
            counts[len(templates) - 1] += route_weight
            for template in templates:
                weights[template.template_id] += route_weight / len(templates)
    template_ids = tuple(sorted(weights))
    empirical = np.asarray([weights[name] for name in template_ids], dtype=float)
    empirical /= empirical.sum()
    probabilities = exploration_floor / len(empirical) + (1 - exploration_floor) * empirical
    counts = exploration_floor / 4 + (1 - exploration_floor) * counts / counts.sum()
    training_identity = identity(
        {
            "schema_version": "source_balanced_subgoal_marginal_fit_v1",
            "sources": len(by_source),
            "routes": len(rows),
            "template_ids": list(template_ids),
            "probabilities": probabilities.tolist(),
            "goal_count_probabilities": counts.tolist(),
            "exploration_floor": exploration_floor,
        }
    )
    return MarginalSubgoalPolicy(
        template_ids,
        tuple(map(float, probabilities)),
        tuple(map(float, counts)),
        exploration_floor,
        training_identity,
    )


@dataclass(frozen=True)
class ContextSubgoalRanker:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    training_identity: str

    def __post_init__(self) -> None:
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
            or not self.training_identity
        ):
            raise ValueError("invalid structural-subgoal context ranker")

    def score(self, features: np.ndarray) -> float:
        values = np.asarray(features, dtype=float)
        if values.shape != (len(self.mean),):
            raise ValueError("structural-subgoal feature dimension mismatch")
        return float(
            ((values - np.asarray(self.mean)) / np.asarray(self.scale))
            @ np.asarray(self.coefficients)
        )

    def checkpoint(self) -> dict:
        return {
            "schema_version": RANKER_SCHEMA,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "training_identity": self.training_identity,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict) -> ContextSubgoalRanker:
        if payload.get("schema_version") != RANKER_SCHEMA:
            raise ValueError("structural-subgoal ranker schema mismatch")
        return cls(
            tuple(map(float, payload["mean"])),
            tuple(map(float, payload["scale"])),
            tuple(map(float, payload["coefficients"])),
            str(payload["training_identity"]),
        )


def fit_context_subgoal_ranker(
    positives: list[tuple[str, np.ndarray]],
    negatives: dict[str, list[np.ndarray]],
    *,
    updates: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> tuple[ContextSubgoalRanker, dict]:
    """Fit a source-balanced pairwise ranker on train-fold candidates only."""

    if not positives or min(updates, learning_rate) <= 0 or l2 < 0:
        raise ValueError("invalid structural-subgoal ranker fit")
    sources = sorted({source for source, _ in positives})
    positives_per_source = Counter(source for source, _ in positives)
    all_features = [features for _, features in positives]
    all_features.extend(value for rows in negatives.values() for value in rows)
    matrix = np.stack(all_features).astype(np.float32)
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 0.05)
    differences, weights = [], []
    for source, positive in positives:
        pool = negatives.get(source, ())
        if not pool:
            continue
        weight = 1 / (len(sources) * positives_per_source[source] * len(pool))
        for negative in pool:
            differences.append((positive - negative) / scale)
            weights.append(weight)
    if not differences:
        raise ValueError("structural-subgoal ranker has no same-source negatives")
    x = torch.from_numpy(np.stack(differences).astype(np.float32))
    sample_weights = torch.tensor(weights, dtype=torch.float32)
    sample_weights /= sample_weights.sum()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        coefficients = torch.zeros(x.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([coefficients], lr=learning_rate)
    history = []
    for update in range(updates):
        margin = x @ coefficients
        loss = (sample_weights * F.softplus(-margin)).sum()
        loss += l2 * coefficients.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite structural-subgoal ranker loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update in (0, updates - 1):
            history.append({"update": update + 1, "loss": float(loss.detach())})
    training_identity = identity(
        {
            "schema_version": "context_subgoal_ranker_fit_v1",
            "sources": len(sources),
            "positives": len(positives),
            "pairs": len(differences),
            "feature_moments": identity({"mean": mean.tolist(), "scale": scale.tolist()}),
            "updates": updates,
            "learning_rate": learning_rate,
            "l2": l2,
            "seed": seed,
        }
    )
    model = ContextSubgoalRanker(
        tuple(map(float, mean)),
        tuple(map(float, scale)),
        tuple(map(float, coefficients.detach().numpy())),
        training_identity,
    )
    return model, {"pairs": len(differences), "history": history}


@dataclass(frozen=True)
class ProposedStructuralGoal:
    """One generated, bound, valid structural goal before exact realization."""

    goal: StructuralGoal
    bindings: tuple[tuple[int, ...], ...]
    templates: tuple[StructuralDeltaTemplate, ...]
    endpoint: MolecularGraph
    constituent_keys: tuple[str, ...]
    score: float

    @property
    def endpoint_state(self) -> dict:
        return encode_state(self.endpoint)


def _single_goal_candidates(
    source: MolecularGraph,
    templates: tuple[StructuralDeltaTemplate, ...],
    *,
    max_bindings_per_template: int,
) -> tuple[list[ProposedStructuralGoal], dict]:
    rows: list[ProposedStructuralGoal] = []
    seen: set[tuple[str, str]] = set()
    telemetry = Counter()
    for template in templates:
        census = transfer_bindings(
            template,
            source,
            max_bindings=max_bindings_per_template,
        )
        telemetry["binding_visits"] += census.visits
        telemetry["binding_truncations"] += int(census.truncated)
        telemetry["templates_with_bindings"] += int(bool(census.assignments))
        for binding in census.assignments:
            telemetry["materialization_attempts"] += 1
            try:
                subgoal = materialize_template(template, source, binding)
                goal = StructuralGoal((subgoal,))
                endpoint, _ = instantiate_goal(source, goal, (binding,))
            except ValueError:
                telemetry["materialization_rejections"] += 1
                continue
            endpoint_key = canonical_state_key(endpoint)
            key = (template.template_id, endpoint_key)
            if key in seen:
                telemetry["single_aliases"] += 1
                continue
            seen.add(key)
            constituent = identity(
                {
                    "schema_version": "bound_structural_delta_constituent_v1",
                    "template_id": template.template_id,
                    "binding": list(binding),
                    "endpoint": endpoint_key,
                }
            )
            rows.append(
                ProposedStructuralGoal(
                    goal,
                    (binding,),
                    (template,),
                    endpoint,
                    (constituent,),
                    0.0,
                )
            )
    telemetry["valid_unique_single_goals"] = len(rows)
    return rows, dict(sorted(telemetry.items()))


def propose_structural_goals(
    source: MolecularGraph,
    templates: tuple[StructuralDeltaTemplate, ...],
    marginal: MarginalSubgoalPolicy,
    *,
    ranker: ContextSubgoalRanker | None,
    pool_size: int = 128,
    beam_width: int = 64,
    expansion_width: int = 48,
    max_bindings_per_template: int = 8,
) -> tuple[list[ProposedStructuralGoal], dict]:
    """Rebind and compose training-fold deltas into a teacher-free candidate pool."""

    if min(pool_size, beam_width, expansion_width, max_bindings_per_template) < 1:
        raise ValueError("structural-subgoal proposal limits must be positive")
    expected = set(marginal.template_ids)
    if {row.template_id for row in templates} != expected:
        raise ValueError("runtime template vocabulary and marginal disagree")
    singles, telemetry = _single_goal_candidates(
        source,
        templates,
        max_bindings_per_template=max_bindings_per_template,
    )

    def score(row: ProposedStructuralGoal) -> float:
        if ranker is None:
            return marginal.score(row.templates)
        return ranker.score(proposal_features(source, row.endpoint, row.templates))

    singles = [
        ProposedStructuralGoal(
            row.goal,
            row.bindings,
            row.templates,
            row.endpoint,
            row.constituent_keys,
            score(row),
        )
        for row in singles
    ]
    singles.sort(
        key=lambda row: (-row.score, canonical_state_key(row.endpoint), row.constituent_keys)
    )
    expansion = singles[:expansion_width]
    frontier = singles[:beam_width]
    all_rows = list(singles)
    telemetry = Counter(telemetry)
    seen = {(len(row.templates), canonical_state_key(row.endpoint)): row for row in singles}
    for depth in range(2, 5):
        next_rows = []
        for prefix in frontier:
            for addition in expansion:
                telemetry["composition_attempts"] += 1
                if set(prefix.constituent_keys) & set(addition.constituent_keys):
                    telemetry["composition_repeated_constituent"] += 1
                    continue
                pairs = sorted(
                    zip(
                        (*prefix.constituent_keys, *addition.constituent_keys),
                        (*prefix.goal.subgoals, *addition.goal.subgoals),
                        (*prefix.bindings, *addition.bindings),
                        (*prefix.templates, *addition.templates),
                        strict=True,
                    ),
                    key=lambda row: row[0],
                )
                keys = tuple(row[0] for row in pairs)
                if len(set(keys)) != depth:
                    continue
                goal = StructuralGoal(tuple(row[1] for row in pairs))
                bindings = tuple(row[2] for row in pairs)
                goal_templates = tuple(row[3] for row in pairs)
                try:
                    endpoint, _ = instantiate_goal(source, goal, bindings)
                except ValueError:
                    telemetry["composition_rejections"] += 1
                    continue
                endpoint_key = canonical_state_key(endpoint)
                dedup = (depth, endpoint_key)
                if dedup in seen:
                    telemetry["composition_aliases"] += 1
                    continue
                candidate = ProposedStructuralGoal(
                    goal,
                    bindings,
                    goal_templates,
                    endpoint,
                    keys,
                    0.0,
                )
                candidate = ProposedStructuralGoal(
                    candidate.goal,
                    candidate.bindings,
                    candidate.templates,
                    candidate.endpoint,
                    candidate.constituent_keys,
                    score(candidate),
                )
                seen[dedup] = candidate
                next_rows.append(candidate)
        next_rows.sort(
            key=lambda row: (
                -row.score,
                canonical_state_key(row.endpoint),
                row.constituent_keys,
            )
        )
        frontier = next_rows[:beam_width]
        all_rows.extend(frontier)
        telemetry[f"valid_depth_{depth}"] = len(next_rows)
        if not frontier:
            break
    best_by_endpoint: dict[str, ProposedStructuralGoal] = {}
    for row in all_rows:
        key = canonical_state_key(row.endpoint)
        previous = best_by_endpoint.get(key)
        if previous is None or (row.score, tuple(reversed(row.constituent_keys))) > (
            previous.score,
            tuple(reversed(previous.constituent_keys)),
        ):
            best_by_endpoint[key] = row
    result = sorted(
        best_by_endpoint.values(),
        key=lambda row: (-row.score, canonical_state_key(row.endpoint), row.constituent_keys),
    )[:pool_size]
    telemetry["unique_ranked_endpoints"] = len(result)
    telemetry["candidate_shortfall"] = max(0, pool_size - len(result))
    return result, dict(sorted(telemetry.items()))


__all__ = [
    "ContextSubgoalRanker",
    "MarginalSubgoalPolicy",
    "ProposedStructuralGoal",
    "StructuralDeltaTemplate",
    "TransferBindingCensus",
    "fit_context_subgoal_ranker",
    "fit_marginal_subgoal_policy",
    "materialize_template",
    "minimize_subgoal",
    "proposal_features",
    "propose_structural_goals",
    "template_features",
    "transfer_bindings",
]
