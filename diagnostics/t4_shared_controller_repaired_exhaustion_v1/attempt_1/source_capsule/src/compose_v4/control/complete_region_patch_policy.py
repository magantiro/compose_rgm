"""Canonical complete-region graph-delta tokens and split-first policy support.

One decoded object is a whole :class:`StructuralSubgoal`.  The token stream is
an internal graph-construction language for a semantic target patch; it never
contains or predicts executor primitive actions.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal import StructuralSubgoal

TOKEN_SCHEMA = "complete_region_patch_token_v1"
STREAM_SCHEMA = "complete_region_patch_stream_v1"

ATOM_TYPES = tuple(sorted(set(ORGANIC_VOCABULARY.element_index)))
FORMAL_CHARGES = tuple(range(-2, 3))
HYDROGEN_COUNTS = tuple(range(5))
DEGREES = tuple(range(7))
BOND_ORDERS = tuple(range(1, 5))
MAXIMUM_ROLES = 40
POLICY_SCHEMA = "complete_region_conditional_patch_policy_v1"


@dataclass(frozen=True)
class PatchToken:
    """One typed decision in a canonical complete-patch graph stream."""

    kind: str
    value: int
    factor: str

    def payload(self) -> dict:
        return {
            "schema_version": TOKEN_SCHEMA,
            "kind": self.kind,
            "value": self.value,
            "factor": self.factor,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> PatchToken:
        if payload.get("schema_version") != TOKEN_SCHEMA or set(payload) != {
            "schema_version",
            "kind",
            "value",
            "factor",
        }:
            raise ValueError("complete-region patch token schema mismatch")
        return cls(str(payload["kind"]), int(payload["value"]), str(payload["factor"]))


@dataclass(frozen=True)
class SourceRegionContext:
    """Address-free current-state roles supplied by the WHERE decision."""

    input_atoms: tuple[tuple[int, int, int, int], ...]
    input_bonds: tuple[tuple[int, ...], ...]
    environments: tuple[str, ...]

    def __post_init__(self) -> None:
        size = len(self.input_atoms)
        if (
            not 1 <= size <= MAXIMUM_ROLES
            or len(self.environments) != size
            or len(self.input_bonds) != size
            or any(len(row) != size for row in self.input_bonds)
        ):
            raise ValueError("complete-region source context is malformed")

    @classmethod
    def from_subgoal(cls, subgoal: StructuralSubgoal) -> SourceRegionContext:
        return cls(subgoal.input_atoms, subgoal.input_bonds, subgoal.environments)


def _atom_tokens(prefix: str, atom: tuple[int, int, int, int]) -> tuple[PatchToken, ...]:
    element, charge, hydrogens, degree = map(int, atom)
    return (
        PatchToken(f"{prefix}_element", element, "atom_attributes"),
        PatchToken(f"{prefix}_formal_charge", charge, "atom_attributes"),
        PatchToken(f"{prefix}_implicit_hydrogens", hydrogens, "atom_attributes"),
        PatchToken(f"{prefix}_degree", degree, "atom_attributes"),
    )


def _dependency_parent(subgoal: StructuralSubgoal, output: int) -> int:
    role = len(subgoal.input_atoms) + output
    neighbors = [
        other
        for other, order in enumerate(subgoal.target_bonds[role])
        if int(order) and other != role
    ]
    return min(neighbors) if neighbors else -1


def encode_patch_stream(subgoal: StructuralSubgoal) -> tuple[PatchToken, ...]:
    """Losslessly encode one whole target patch without executor actions."""

    tokens: list[PatchToken] = [PatchToken("output_count", len(subgoal.output_atoms), "control")]
    live: list[bool] = []
    for target in subgoal.target_atoms:
        survives = target is not None
        tokens.append(PatchToken("input_survival", int(survives), "atom_attributes"))
        live.append(survives)
        if target is not None:
            tokens.extend(_atom_tokens("input_target", target))
    for atom in subgoal.output_atoms:
        live.append(True)
        tokens.extend(_atom_tokens("output", atom))
    for left in range(len(live)):
        if not live[left]:
            continue
        for right in range(left + 1, len(live)):
            if not live[right]:
                continue
            order = int(subgoal.target_bonds[left][right])
            attachment = left < len(subgoal.input_atoms) <= right
            factor = "attachments" if attachment else "target_topology"
            tokens.append(PatchToken("edge_presence", int(bool(order)), factor))
            if order:
                tokens.append(
                    PatchToken(
                        "bond_order",
                        order,
                        "attachments" if attachment else "bond_attributes",
                    )
                )
    for output in range(len(subgoal.output_atoms)):
        tokens.append(
            PatchToken(
                "output_dependency_parent",
                _dependency_parent(subgoal, output),
                "dependencies",
            )
        )
    tokens.append(PatchToken("patch_stop", 1, "control"))
    return tuple(tokens)


class _Reader:
    def __init__(self, tokens: Iterable[PatchToken]) -> None:
        self.tokens = tuple(tokens)
        self.index = 0

    def take(self, kind: str, factor: str, allowed: tuple[int, ...]) -> int:
        if self.index >= len(self.tokens):
            raise ValueError(f"missing complete-patch token: {kind}")
        token = self.tokens[self.index]
        self.index += 1
        if token.kind != kind or token.factor != factor or token.value not in allowed:
            raise ValueError(f"unsupported complete-patch token at {self.index - 1}: {token}")
        return token.value

    def finish(self) -> None:
        if self.index != len(self.tokens):
            raise ValueError("complete-patch stream has trailing tokens")


def _read_atom(reader: _Reader, prefix: str) -> tuple[int, int, int, int]:
    return (
        reader.take(f"{prefix}_element", "atom_attributes", ATOM_TYPES),
        reader.take(f"{prefix}_formal_charge", "atom_attributes", FORMAL_CHARGES),
        reader.take(f"{prefix}_implicit_hydrogens", "atom_attributes", HYDROGEN_COUNTS),
        reader.take(f"{prefix}_degree", "atom_attributes", DEGREES),
    )


def decode_patch_stream(
    context: SourceRegionContext, tokens: Iterable[PatchToken]
) -> StructuralSubgoal:
    """Decode one canonical graph stream into a complete structural patch."""

    reader = _Reader(tokens)
    output_count = reader.take("output_count", "control", tuple(range(MAXIMUM_ROLES + 1)))
    if len(context.input_atoms) + output_count > MAXIMUM_ROLES:
        raise ValueError("complete target patch exceeds active-role support")
    target_atoms = []
    live = []
    for _ in context.input_atoms:
        survives = bool(reader.take("input_survival", "atom_attributes", (0, 1)))
        live.append(survives)
        target_atoms.append(_read_atom(reader, "input_target") if survives else None)
    output_atoms = tuple(_read_atom(reader, "output") for _ in range(output_count))
    live.extend([True] * output_count)
    target_bonds = [[0 for _ in live] for _ in live]
    for left in range(len(live)):
        if not live[left]:
            continue
        for right in range(left + 1, len(live)):
            if not live[right]:
                continue
            attachment = left < len(context.input_atoms) <= right
            factor = "attachments" if attachment else "target_topology"
            present = reader.take("edge_presence", factor, (0, 1))
            order = (
                reader.take(
                    "bond_order",
                    "attachments" if attachment else "bond_attributes",
                    BOND_ORDERS,
                )
                if present
                else 0
            )
            target_bonds[left][right] = target_bonds[right][left] = order
    dependencies = [
        reader.take(
            "output_dependency_parent",
            "dependencies",
            tuple(range(-1, len(live))),
        )
        for _ in range(output_count)
    ]
    reader.take("patch_stop", "control", (1,))
    reader.finish()
    subgoal = StructuralSubgoal(
        input_atoms=context.input_atoms,
        input_bonds=context.input_bonds,
        environments=context.environments,
        target_atoms=tuple(target_atoms),
        output_atoms=output_atoms,
        target_bonds=tuple(tuple(row) for row in target_bonds),
    )
    expected = [_dependency_parent(subgoal, output) for output in range(output_count)]
    if dependencies != expected:
        raise ValueError("dependency-role tokens disagree with the decoded target graph")
    return subgoal


def patch_stream_support(subgoal: StructuralSubgoal) -> tuple[bool, str | None]:
    """Check declared grammar support and exact canonical round-trip."""

    try:
        context = SourceRegionContext.from_subgoal(subgoal)
        decoded = decode_patch_stream(context, encode_patch_stream(subgoal))
    except (TypeError, ValueError) as error:
        return False, f"{type(error).__name__}:{error}"
    if decoded != subgoal:
        return False, "roundtrip_mismatch"
    return True, None


@dataclass(frozen=True)
class PatchPolicyTrainingRow:
    """One complete region used only for split-first fitting."""

    source_group: str
    route_id: str
    region_index: int
    route_region_count: int
    context: SourceRegionContext
    tokens: tuple[PatchToken, ...]
    control_after: str

    def __post_init__(self) -> None:
        if (
            not self.source_group
            or not self.route_id
            or self.region_index < 0
            or not 1 <= self.route_region_count <= 4
            or self.control_after not in {"continue", "stop"}
            or not self.tokens
        ):
            raise ValueError("invalid complete-region patch training row")


def _source_summary(context: SourceRegionContext) -> str:
    elements = Counter(atom[0] for atom in context.input_atoms)
    edges = sum(
        bool(context.input_bonds[left][right])
        for left in range(len(context.input_atoms))
        for right in range(left + 1, len(context.input_atoms))
    )
    payload = {
        "n": min(len(context.input_atoms), 12),
        "n_overflow": int(len(context.input_atoms) > 12),
        "edges": min(edges, 16),
        "edge_overflow": int(edges > 16),
        "carbon": min(elements.get(2, 0), 12),
        "hetero": min(len(context.input_atoms) - elements.get(2, 0), 8),
        "max_degree": max(atom[3] for atom in context.input_atoms),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def token_domain(
    context: SourceRegionContext,
    prefix: tuple[PatchToken, ...],
    *,
    kind: str,
    factor: str,
) -> tuple[int, ...]:
    """Return the frozen platform value fiber for one graph-delta token."""

    del factor
    if kind == "output_count":
        return tuple(range(MAXIMUM_ROLES - len(context.input_atoms) + 1))
    if kind == "input_survival" or kind == "edge_presence":
        return (0, 1)
    if kind.endswith("_element"):
        return ATOM_TYPES
    if kind.endswith("_formal_charge"):
        return FORMAL_CHARGES
    if kind.endswith("_implicit_hydrogens"):
        return HYDROGEN_COUNTS
    if kind.endswith("_degree"):
        return DEGREES
    if kind == "bond_order":
        return BOND_ORDERS
    if kind == "patch_stop":
        return (1,)
    if kind == "output_dependency_parent":
        output = next((token.value for token in prefix if token.kind == "output_count"), 0)
        return tuple(range(-1, len(context.input_atoms) + output))
    raise ValueError(f"unknown complete-patch token kind: {kind}")


def _count_key(
    context: SourceRegionContext,
    prefix: tuple[PatchToken, ...],
    token: PatchToken,
    level: int,
) -> str:
    previous = [(row.kind, row.value) for row in prefix[-level:]] if level else []
    return json.dumps(
        {
            "level": level,
            "kind": token.kind,
            "factor": token.factor,
            "source": _source_summary(context),
            "previous": previous,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _marginal_key(token: PatchToken) -> str:
    return f"{token.factor}|{token.kind}"


def _freeze_counts(counts: dict[str, Counter[int]]) -> tuple:
    return tuple(
        (key, tuple((int(value), float(count)) for value, count in sorted(rows.items())))
        for key, rows in sorted(counts.items())
    )


def _thaw_counts(rows: tuple) -> dict[str, dict[int, float]]:
    return {key: {int(value): float(count) for value, count in values} for key, values in rows}


@dataclass(frozen=True)
class ConditionalPatchPolicy:
    """Smoothed autoregressive distribution over complete target-patch tokens."""

    conditional_counts: tuple
    marginal_counts: tuple
    region_size_counts: tuple[float, ...]
    program_count_counts: tuple[float, float, float, float]
    smoothing_alpha: float
    training_identity: str
    training_summary: dict

    def __post_init__(self) -> None:
        if (
            len(self.region_size_counts) != MAXIMUM_ROLES
            or len(self.program_count_counts) != 4
            or self.smoothing_alpha <= 0
            or not self.training_identity
            or not isinstance(self.training_summary, dict)
        ):
            raise ValueError("invalid complete-region conditional patch policy")

    def checkpoint(self) -> dict:
        return {
            "schema_version": POLICY_SCHEMA,
            "conditional_counts": [
                [key, [[value, count] for value, count in values]]
                for key, values in self.conditional_counts
            ],
            "marginal_counts": [
                [key, [[value, count] for value, count in values]]
                for key, values in self.marginal_counts
            ],
            "region_size_counts": list(self.region_size_counts),
            "program_count_counts": list(self.program_count_counts),
            "smoothing_alpha": self.smoothing_alpha,
            "training_identity": self.training_identity,
            "training_summary": self.training_summary,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict) -> ConditionalPatchPolicy:
        if payload.get("schema_version") != POLICY_SCHEMA:
            raise ValueError("complete-region conditional patch policy schema mismatch")
        return cls(
            tuple(
                (str(key), tuple((int(value), float(count)) for value, count in values))
                for key, values in payload["conditional_counts"]
            ),
            tuple(
                (str(key), tuple((int(value), float(count)) for value, count in values))
                for key, values in payload["marginal_counts"]
            ),
            tuple(map(float, payload["region_size_counts"])),
            tuple(map(float, payload["program_count_counts"])),
            float(payload["smoothing_alpha"]),
            str(payload["training_identity"]),
            dict(payload["training_summary"]),
        )

    def _counts(
        self,
        context: SourceRegionContext,
        prefix: tuple[PatchToken, ...],
        token: PatchToken,
        *,
        learned: bool,
    ) -> dict[int, float]:
        marginal = _thaw_counts(self.marginal_counts)
        if learned:
            conditional = _thaw_counts(self.conditional_counts)
            for level in (2, 1, 0):
                rows = conditional.get(_count_key(context, prefix, token, level))
                if rows:
                    return rows
        return marginal.get(_marginal_key(token), {})

    def token_probability(
        self,
        context: SourceRegionContext,
        prefix: tuple[PatchToken, ...],
        token: PatchToken,
        *,
        learned: bool,
    ) -> float:
        domain = token_domain(context, prefix, kind=token.kind, factor=token.factor)
        if token.value not in domain:
            return 0.0
        counts = self._counts(context, prefix, token, learned=learned)
        total = sum(counts.get(value, 0.0) for value in domain)
        return (counts.get(token.value, 0.0) + self.smoothing_alpha) / (
            total + self.smoothing_alpha * len(domain)
        )

    def score_stream(
        self,
        context: SourceRegionContext,
        tokens: tuple[PatchToken, ...],
        *,
        learned: bool,
    ) -> dict:
        prefix: tuple[PatchToken, ...] = ()
        factor_nll: Counter[str] = Counter()
        factor_count: Counter[str] = Counter()
        total = 0.0
        for token in tokens:
            probability = self.token_probability(context, prefix, token, learned=learned)
            if probability <= 0:
                return {
                    "supported": False,
                    "log_probability": float("-inf"),
                    "nll": float("inf"),
                    "factor_nll": {},
                    "factor_counts": {},
                }
            nll = -math.log(probability)
            total += nll
            factor_nll[token.factor] += nll
            factor_count[token.factor] += 1
            prefix = (*prefix, token)
        return {
            "supported": True,
            "log_probability": -total,
            "nll": total,
            "mean_nll": total / len(tokens),
            "factor_nll": dict(sorted(factor_nll.items())),
            "factor_counts": dict(sorted(factor_count.items())),
        }

    def sample_value(
        self,
        context: SourceRegionContext,
        prefix: tuple[PatchToken, ...],
        *,
        kind: str,
        factor: str,
        allowed: tuple[int, ...],
        learned: bool,
        rng: np.random.Generator,
    ) -> int:
        probe = PatchToken(kind, allowed[0], factor)
        domain = tuple(
            value
            for value in token_domain(context, prefix, kind=kind, factor=factor)
            if value in set(allowed)
        )
        if not domain:
            raise ValueError(f"empty legality mask for {factor}/{kind}")
        counts = self._counts(context, prefix, probe, learned=learned)
        weights = np.asarray(
            [counts.get(value, 0.0) + self.smoothing_alpha for value in domain],
            dtype=float,
        )
        return int(rng.choice(np.asarray(domain), p=weights / weights.sum()))


def fit_conditional_patch_policy(
    rows: list[PatchPolicyTrainingRow], *, smoothing_alpha: float
) -> ConditionalPatchPolicy:
    """Fit after the caller freezes whole-source folds."""

    if not rows or smoothing_alpha <= 0:
        raise ValueError("complete-region policy fitting requires rows and smoothing")
    by_source: dict[str, dict[str, list[PatchPolicyTrainingRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_source[row.source_group][row.route_id].append(row)
    conditional: dict[str, Counter[int]] = defaultdict(Counter)
    marginal: dict[str, Counter[int]] = defaultdict(Counter)
    region_size = np.zeros(MAXIMUM_ROLES, dtype=float)
    program_count = np.zeros(4, dtype=float)
    route_seen = set()
    for source, routes in sorted(by_source.items()):
        source_mass = 1 / len(by_source)
        for route_id, regions in sorted(routes.items()):
            route_mass = source_mass / len(routes)
            route_key = (source, route_id)
            if route_key not in route_seen:
                program_count[regions[0].route_region_count - 1] += route_mass
                route_seen.add(route_key)
            for row in regions:
                region_mass = route_mass / len(regions)
                region_size[len(row.context.input_atoms) - 1] += region_mass
                prefix: tuple[PatchToken, ...] = ()
                token_mass = region_mass / len(row.tokens)
                for token in row.tokens:
                    for level in (0, 1, 2):
                        conditional[_count_key(row.context, prefix, token, level)][token.value] += (
                            token_mass
                        )
                    marginal[_marginal_key(token)][token.value] += token_mass
                    prefix = (*prefix, token)
    training_summary = {
        "sources": len(by_source),
        "routes": sum(len(value) for value in by_source.values()),
        "regions": len(rows),
        "tokens": sum(len(row.tokens) for row in rows),
        "source_mass": {source: 1 / len(by_source) for source in sorted(by_source)},
    }
    # Source hashes are fit-report evidence only and are removed from the runtime checkpoint.
    runtime_summary = {
        key: value for key, value in training_summary.items() if key != "source_mass"
    }
    frozen_conditional = _freeze_counts(conditional)
    frozen_marginal = _freeze_counts(marginal)
    training_identity = identity(
        {
            "schema_version": "complete_region_conditional_fit_identity_v1",
            "conditional_counts": frozen_conditional,
            "marginal_counts": frozen_marginal,
            "region_size_counts": region_size.tolist(),
            "program_count_counts": program_count.tolist(),
            "smoothing_alpha": smoothing_alpha,
            "summary": runtime_summary,
        }
    )
    return ConditionalPatchPolicy(
        frozen_conditional,
        frozen_marginal,
        tuple(map(float, region_size)),
        tuple(map(float, program_count)),
        smoothing_alpha,
        training_identity,
        runtime_summary,
    )


def _sample_atom(
    policy: ConditionalPatchPolicy,
    context: SourceRegionContext,
    prefix: list[PatchToken],
    *,
    token_prefix: str,
    learned: bool,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    values = []
    for suffix, domain in (
        ("element", ATOM_TYPES),
        ("formal_charge", FORMAL_CHARGES),
        ("implicit_hydrogens", HYDROGEN_COUNTS),
        ("degree", DEGREES),
    ):
        kind = f"{token_prefix}_{suffix}"
        value = policy.sample_value(
            context,
            tuple(prefix),
            kind=kind,
            factor="atom_attributes",
            allowed=domain,
            learned=learned,
            rng=rng,
        )
        prefix.append(PatchToken(kind, value, "atom_attributes"))
        values.append(value)
    return tuple(values)  # type: ignore[return-value]


def sample_patch_stream(
    policy: ConditionalPatchPolicy,
    context: SourceRegionContext,
    *,
    learned: bool,
    rng: np.random.Generator,
) -> tuple[StructuralSubgoal, tuple[PatchToken, ...]]:
    """Sample one whole target graph with degree and valence legality masks."""

    prefix: list[PatchToken] = []
    output_count = policy.sample_value(
        context,
        (),
        kind="output_count",
        factor="control",
        allowed=tuple(range(MAXIMUM_ROLES - len(context.input_atoms) + 1)),
        learned=learned,
        rng=rng,
    )
    prefix.append(PatchToken("output_count", output_count, "control"))
    target_atoms: list[tuple[int, int, int, int] | None] = []
    live: list[bool] = []
    for input_index, _ in enumerate(context.input_atoms):
        # The last input is forced to survive if no output exists and every
        # earlier role was deleted. This is a legality mask, not teacher input.
        allowed = (
            (1,)
            if input_index == len(context.input_atoms) - 1 and output_count == 0 and not any(live)
            else (0, 1)
        )
        survives = policy.sample_value(
            context,
            tuple(prefix),
            kind="input_survival",
            factor="atom_attributes",
            allowed=allowed,
            learned=learned,
            rng=rng,
        )
        prefix.append(PatchToken("input_survival", survives, "atom_attributes"))
        live.append(bool(survives))
        target_atoms.append(
            _sample_atom(
                policy,
                context,
                prefix,
                token_prefix="input_target",
                learned=learned,
                rng=rng,
            )
            if survives
            else None
        )
    output_atoms = [
        _sample_atom(
            policy,
            context,
            prefix,
            token_prefix="output",
            learned=learned,
            rng=rng,
        )
        for _ in range(output_count)
    ]
    live.extend([True] * output_count)
    attributes = [*target_atoms, *output_atoms]
    desired_degree = [0 if value is None else int(value[3]) for value in attributes]
    live_roles = [index for index, value in enumerate(live) if value]
    if any(desired_degree[index] >= len(live_roles) for index in live_roles):
        raise ValueError("sampled target degree exceeds available target roles")
    target_bonds = [[0 for _ in live] for _ in live]
    current_degree = [0 for _ in live]
    pairs = [
        (left, right)
        for offset, left in enumerate(live_roles)
        for right in live_roles[offset + 1 :]
    ]
    for pair_index, (left, right) in enumerate(pairs):
        remaining = pairs[pair_index + 1 :]
        future_left = sum(left in row for row in remaining)
        future_right = sum(right in row for row in remaining)
        must_present = (
            current_degree[left] + future_left < desired_degree[left]
            or current_degree[right] + future_right < desired_degree[right]
        )
        can_present = (
            current_degree[left] < desired_degree[left]
            and current_degree[right] < desired_degree[right]
        )
        if must_present and not can_present:
            raise ValueError("sampled target degree sequence is not graph-realizable")
        allowed_presence = (1,) if must_present else (0, 1) if can_present else (0,)
        attachment = left < len(context.input_atoms) <= right
        factor = "attachments" if attachment else "target_topology"
        present = policy.sample_value(
            context,
            tuple(prefix),
            kind="edge_presence",
            factor=factor,
            allowed=allowed_presence,
            learned=learned,
            rng=rng,
        )
        prefix.append(PatchToken("edge_presence", present, factor))
        if not present:
            continue
        order = policy.sample_value(
            context,
            tuple(prefix),
            kind="bond_order",
            factor="attachments" if attachment else "bond_attributes",
            allowed=BOND_ORDERS,
            learned=learned,
            rng=rng,
        )
        prefix.append(
            PatchToken(
                "bond_order",
                order,
                "attachments" if attachment else "bond_attributes",
            )
        )
        target_bonds[left][right] = target_bonds[right][left] = order
        current_degree[left] += 1
        current_degree[right] += 1
    if any(current_degree[index] != desired_degree[index] for index in live_roles):
        raise ValueError("sampled target degree sequence was not satisfied")
    for index in live_roles:
        atom = attributes[index]
        assert atom is not None
        heavy_sum = sum(target_bonds[index])
        if ORGANIC_VOCABULARY.class_index(atom[0], heavy_sum, atom[2], atom[1]) is None:
            raise ValueError("sampled target atom has unsupported valence")
    patch_without_dependencies = StructuralSubgoal(
        input_atoms=context.input_atoms,
        input_bonds=context.input_bonds,
        environments=context.environments,
        target_atoms=tuple(target_atoms),
        output_atoms=tuple(output_atoms),
        target_bonds=tuple(tuple(row) for row in target_bonds),
    )
    # Dependency roles and STOP are deterministic consequences/control for the
    # sampled joint graph. Encoding them here keeps a single canonical stream.
    tokens = encode_patch_stream(patch_without_dependencies)
    if tuple(prefix) != tokens[: len(prefix)]:
        raise AssertionError("sampled patch prefix is not canonical")
    return patch_without_dependencies, tokens


__all__ = [
    "ATOM_TYPES",
    "BOND_ORDERS",
    "DEGREES",
    "FORMAL_CHARGES",
    "HYDROGEN_COUNTS",
    "MAXIMUM_ROLES",
    "ConditionalPatchPolicy",
    "PatchPolicyTrainingRow",
    "PatchToken",
    "SourceRegionContext",
    "decode_patch_stream",
    "encode_patch_stream",
    "fit_conditional_patch_policy",
    "patch_stream_support",
    "sample_patch_stream",
    "token_domain",
]
