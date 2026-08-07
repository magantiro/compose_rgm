"""Differentiable canonical-successor supervision for the factorized editor.

The production model parameterizes executable mark probabilities.  This module
precompiles the subset of legal mark coordinates that execute to one teacher
molecule and aggregates their normalized probability with ``logsumexp`` during
training.  The executor and canonicalizer are used only during compilation;
the hot gradient path indexes existing action tables and does not replay
chemistry.
"""

from __future__ import annotations

import hashlib
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.experiments.production_successor_kernel import (
    ProductionSuccessorKernelError,
    enumerate_factorized_marked_law,
)
from compose_v4.experiments.score_free_successor_support import (
    enumerate_factorized_legal_support_many,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    _CYCLE_OP_EXECUTOR_TO_FAMILY,
    MARK_RULE_TO_INDEX,
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
    _masked_family_logits,
    is_semantic_editing_v2_process,
)
from compose_v4.rewrite import action_codec as action_codec_v2
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.kernel import (
    RewriteSystem,
    canonical_state_key,
    de_novo_rewrite_system,
    editing_v2_semantic_rewrite_system,
)


class SuccessorTrainingError(RuntimeError):
    """A teacher fiber or differentiable score is inconsistent with production."""


@dataclass(frozen=True)
class SuccessorProcessRuntime:
    """Executor and action-codec coordinates selected by model semantics."""

    system: RewriteSystem
    action_codec_schema_version: int
    editing_process_semantics: str


@dataclass(frozen=True, order=True)
class TeacherSuccessorAlias:
    """One factorized table coordinate in a teacher-successor fiber."""

    family_name: str
    table_name: str
    coordinate: tuple[int, ...]


@dataclass(frozen=True)
class StateProductiveSupport:
    """Precompiled coordinates whose execution is a canonical self-event.

    The editing kernel is conditioned on taking a productive molecular jump.
    Its normalizer is therefore one minus the aggregate probability of these
    virtual aliases.  An empty tuple means every legal mark is productive.
    """

    source_key: str
    source_state_sha256: str
    virtual_aliases: tuple[TeacherSuccessorAlias, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_key:
            raise ValueError("state-support source key must be nonempty")
        if len(self.source_state_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_state_sha256
        ):
            raise ValueError("state-support source_state_sha256 must be a lowercase SHA-256")
        if len(set(self.virtual_aliases)) != len(self.virtual_aliases):
            raise ValueError("state support contains duplicate virtual coordinates")


@dataclass(frozen=True)
class TeacherSuccessorFiber:
    """All legal mark aliases that execute to one canonical teacher successor."""

    source_key: str
    target_key: str
    target_state_sha256: str
    aliases: tuple[TeacherSuccessorAlias, ...]
    state_support: StateProductiveSupport

    def __post_init__(self) -> None:
        if not self.source_key or not self.target_key:
            raise ValueError("teacher-successor keys must be nonempty")
        if self.source_key == self.target_key:
            raise ValueError("a teacher molecular jump cannot target the source state")
        if len(self.target_state_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.target_state_sha256
        ):
            raise ValueError("teacher target_state_sha256 must be a lowercase SHA-256")
        if not self.aliases:
            raise ValueError("a teacher-successor fiber must contain at least one alias")
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("a teacher-successor fiber contains duplicate coordinates")
        if self.state_support.source_key != self.source_key:
            raise ValueError("teacher fiber and state support have different sources")
        if set(self.aliases) & set(self.state_support.virtual_aliases):
            raise ValueError("a teacher successor cannot also be a virtual self-event")


@dataclass(frozen=True)
class SupportOnlyTeacherFiberCompilation:
    """Compact exact geometry needed by successor-level optimization.

    ``raw_mark_count`` is the all-family mask census. ``decoded_mark_count`` is
    the smaller invariant-compatible subset that was executed to recover the
    teacher and canonical-self fibers.  Neither value contains model scores.
    """

    teacher_fiber: TeacherSuccessorFiber
    exact_teacher_alias: TeacherSuccessorAlias
    raw_mark_count: int
    decoded_mark_count: int
    marks_by_family: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.exact_teacher_alias not in self.teacher_fiber.aliases:
            raise ValueError("support-only exact teacher alias is absent from its fiber")
        if type(self.raw_mark_count) is not int or self.raw_mark_count <= 0:
            raise ValueError("support-only raw_mark_count must be positive")
        if (
            type(self.decoded_mark_count) is not int
            or not 0 < self.decoded_mark_count <= self.raw_mark_count
        ):
            raise ValueError("support-only decoded mark count is inconsistent")
        if tuple(sorted(dict(self.marks_by_family).items())) != self.marks_by_family:
            raise ValueError("support-only family census must be sorted and unique")


@dataclass(frozen=True, order=True)
class CompiledSuccessorMark:
    """One enumerated mark tied to its alias and exact executor output."""

    alias: TeacherSuccessorAlias
    successor_state_sha256: str
    action_sha256: str

    def __post_init__(self) -> None:
        for name in ("successor_state_sha256", "action_sha256"):
            digest = getattr(self, name)
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class CanonicalSuccessorAliasGroup:
    """Static enumerated marks for one canonical molecular successor.

    Each entry associates the factorized table coordinate, full canonical
    model-selected action-codec JSON SHA-256, and exact persistent-slot
    executor output.  The group carries no model score, probability, rate,
    weight, or time value and is discarded after cache-row projection.
    """

    target_key: str
    marks: tuple[CompiledSuccessorMark, ...]

    def __post_init__(self) -> None:
        if not self.target_key:
            raise ValueError("canonical successor alias-group key must be nonempty")
        if not self.marks:
            raise ValueError("canonical successor alias group must retain an enumerated mark")
        if tuple(sorted(set(self.marks))) != self.marks:
            raise ValueError("canonical successor marks must be sorted and unique")
        aliases = tuple(mark.alias for mark in self.marks)
        if len(set(aliases)) != len(aliases):
            raise ValueError("canonical successor group repeats a factorized mark coordinate")

    @property
    def successor_state_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted({mark.successor_state_sha256 for mark in self.marks}))

    @property
    def aliases(self) -> tuple[TeacherSuccessorAlias, ...]:
        return tuple(sorted(mark.alias for mark in self.marks))

    def contains_exact_action(
        self,
        *,
        successor_state_sha256: str,
        action_sha256: str,
    ) -> bool:
        """Whether one enumerated mark has this action and exact output."""

        return any(
            mark.successor_state_sha256 == successor_state_sha256
            and mark.action_sha256 == action_sha256
            for mark in self.marks
        )


@dataclass(frozen=True)
class CompiledStateSuccessorMap:
    """Complete static successor-coordinate grouping for one exact source.

    This is deliberately a transient compilation object.  Persistent cache
    records retain only ``state_support`` and the one requested teacher fiber.
    """

    state_support: StateProductiveSupport
    successor_groups: tuple[CanonicalSuccessorAliasGroup, ...]
    virtual_marks: tuple[CompiledSuccessorMark, ...] = ()

    def __post_init__(self) -> None:
        target_keys = tuple(group.target_key for group in self.successor_groups)
        if target_keys != tuple(sorted(set(target_keys))):
            raise ValueError("compiled successor groups must have sorted unique target keys")
        if self.state_support.source_key in target_keys:
            raise ValueError("compiled productive successor groups contain a canonical self-event")
        productive_marks = [mark for group in self.successor_groups for mark in group.marks]
        all_aliases = [mark.alias for mark in productive_marks]
        if len(all_aliases) != len(set(all_aliases)):
            raise ValueError("one legal mark coordinate appears in multiple successor groups")
        if tuple(sorted(set(self.virtual_marks))) != self.virtual_marks:
            raise ValueError("compiled virtual marks must be sorted and unique")
        virtual_aliases = tuple(sorted(mark.alias for mark in self.virtual_marks))
        if len(set(virtual_aliases)) != len(virtual_aliases):
            raise ValueError("compiled virtual marks repeat a factorized mark coordinate")
        if virtual_aliases != tuple(sorted(self.state_support.virtual_aliases)):
            raise ValueError("compiled virtual mark identities disagree with state support")
        if set(all_aliases) & set(virtual_aliases):
            raise ValueError("one legal mark coordinate is both productive and virtual")
        all_exact_digests = [
            digest for group in self.successor_groups for digest in group.successor_state_sha256s
        ]
        if len(all_exact_digests) != len(set(all_exact_digests)):
            raise ValueError("one exact successor state appears in multiple canonical groups")

    @property
    def source_key(self) -> str:
        return self.state_support.source_key

    @property
    def source_state_sha256(self) -> str:
        return self.state_support.source_state_sha256


@dataclass(frozen=True)
class FactorizedSuccessorPrediction:
    """Marked and productive molecular quantities for successor supervision."""

    total_hazard: Tensor
    productive_hazard: Tensor
    selected_successor_log_probability: Tensor
    selected_productive_successor_log_probability: Tensor
    productive_log_probability: Tensor
    teacher_family_log_probability: Tensor
    selected_within_teacher_family_log_probability: Tensor
    family_log_probabilities: Tensor
    enabled_families: Tensor

    @property
    def selected_successor_rate(self) -> Tensor:
        return self.total_hazard * self.selected_successor_log_probability.exp()


@dataclass(frozen=True)
class FactorizedSuccessorPartitionPrediction:
    """Differentiable probabilities for complete precompiled successor rows.

    ``successor_log_probabilities[i]`` follows ``successor_groups[i]`` exactly
    and is conditioned on taking a productive molecular jump.  The static
    groups contain coordinates only; no cached model score enters this path.
    """

    successor_keys: tuple[tuple[str, ...], ...]
    successor_log_probabilities: tuple[Tensor, ...]
    productive_log_probability: Tensor
    family_log_probabilities: Tensor
    enabled_families: Tensor

    def __post_init__(self) -> None:
        if len(self.successor_keys) != len(self.successor_log_probabilities):
            raise ValueError("successor partition rows are not aligned")
        for keys, values in zip(
            self.successor_keys,
            self.successor_log_probabilities,
            strict=True,
        ):
            if tuple(sorted(set(keys))) != keys or values.ndim != 1:
                raise ValueError("successor partition row is not canonical")
            if len(keys) != int(values.numel()):
                raise ValueError("successor partition keys and values disagree")


def _alias(
    *,
    family_name: str,
    table_name: str,
    coordinate: tuple[int, ...],
) -> TeacherSuccessorAlias:
    return TeacherSuccessorAlias(
        family_name=family_name,
        table_name=table_name,
        coordinate=coordinate,
    )


def rewrite_action_codec_sha256(
    executor_rule_name: str,
    action: object,
    *,
    schema_version: int = action_codec_v2.SCHEMA_VERSION,
) -> str:
    """Full SHA-256 of one explicitly selected canonical action payload."""

    if schema_version == action_codec_v2.SCHEMA_VERSION:
        codec = action_codec_v2
        error_type = action_codec_v2.ActionCodecError
    elif schema_version == action_codec_v4.SCHEMA_VERSION:
        codec = action_codec_v4
        error_type = action_codec_v4.ActionCodecV4Error
    else:
        raise SuccessorTrainingError(
            f"unsupported rewrite action codec schema_version={schema_version}"
        )
    try:
        encoded = codec.encode_action(executor_rule_name, action)
    except error_type as error:
        raise SuccessorTrainingError(
            "rewrite action cannot be represented by the selected action codec"
        ) from error
    return hashlib.sha256(codec.canonical_json(encoded).encode("utf-8")).hexdigest()


def _uses_semantic_editing_process(model: FactorizedTraceletRateModel) -> bool:
    return is_semantic_editing_v2_process(model.editing_process_semantics)


def _rewrite_system_signature(system: RewriteSystem) -> tuple[object, ...]:
    """Return an in-process exact signature for a rewrite-system instance."""

    if not isinstance(system, RewriteSystem):
        raise TypeError("successor compilation system must be RewriteSystem")
    rules = tuple(
        (
            name,
            rule.action_type,
            rule.validate,
            rule.execute,
        )
        for name, rule in sorted(system.rules.items())
    )
    return rules, tuple(system.constraints)


def resolve_successor_process_runtime(
    model: FactorizedTraceletRateModel,
    *,
    system: RewriteSystem | None = None,
) -> SuccessorProcessRuntime:
    """Select executor and codec only from the model's process semantics.

    An explicit system is accepted solely when its complete rule and constraint
    signature equals the system implied by the model.  This keeps test oracles
    injectable without permitting a semantic Editing-V2 model to be compiled
    through the legacy de-novo executor.
    """

    if not isinstance(model, FactorizedTraceletRateModel):
        raise TypeError("successor compilation requires FactorizedTraceletRateModel")
    semantic_editing_process = _uses_semantic_editing_process(model)
    expected_system = (
        editing_v2_semantic_rewrite_system()
        if semantic_editing_process
        else de_novo_rewrite_system()
    )
    selected_system = expected_system if system is None else system
    if _rewrite_system_signature(selected_system) != _rewrite_system_signature(expected_system):
        raise SuccessorTrainingError(
            "explicit successor executor disagrees with model process semantics"
        )
    return SuccessorProcessRuntime(
        system=selected_system,
        action_codec_schema_version=(
            action_codec_v4.SCHEMA_VERSION
            if semantic_editing_process
            else action_codec_v2.SCHEMA_VERSION
        ),
        editing_process_semantics=model.editing_process_semantics,
    )


def compile_state_successor_map(
    model: FactorizedTraceletRateModel,
    source: MolecularGraph,
    *,
    time: float = 0.5,
    system: RewriteSystem | None = None,
) -> CompiledStateSuccessorMap:
    """Execute and group every legal mark once for one exact source state."""

    process = resolve_successor_process_runtime(model, system=system)
    runtime = process.system
    action_codec_version = process.action_codec_schema_version
    try:
        marked_law = enumerate_factorized_marked_law(model, source, float(time))
    except ProductionSuccessorKernelError as error:
        raise SuccessorTrainingError(
            "could not enumerate the production marked law for state-successor compilation"
        ) from error

    source_key = marked_law.source_key
    grouped_marks: dict[str, list[CompiledSuccessorMark]] = {}
    virtual_marks: list[CompiledSuccessorMark] = []
    for mark in marked_law.marks:
        try:
            successor = runtime.apply(
                source,
                mark.executor_rule_name,
                mark.action,
            )
        except Exception as error:
            raise SuccessorTrainingError(
                "a production-enumerated mark failed during state-successor compilation"
            ) from error
        successor_key = canonical_state_key(successor)
        compiled_alias = _alias(
            family_name=mark.family_name,
            table_name=mark.table_name,
            coordinate=mark.coordinate,
        )
        compiled_mark = CompiledSuccessorMark(
            alias=compiled_alias,
            successor_state_sha256=persistent_slot_state_sha256(successor),
            action_sha256=rewrite_action_codec_sha256(
                mark.executor_rule_name,
                mark.action,
                schema_version=action_codec_version,
            ),
        )
        if successor_key == source_key:
            virtual_marks.append(compiled_mark)
            continue
        grouped_marks.setdefault(successor_key, []).append(compiled_mark)

    state_support = StateProductiveSupport(
        source_key=source_key,
        source_state_sha256=persistent_slot_state_sha256(source),
        virtual_aliases=tuple(sorted(mark.alias for mark in virtual_marks)),
    )
    return CompiledStateSuccessorMap(
        state_support=state_support,
        successor_groups=tuple(
            CanonicalSuccessorAliasGroup(
                target_key=target_key,
                marks=tuple(sorted(marks)),
            )
            for target_key, marks in sorted(grouped_marks.items())
        ),
        virtual_marks=tuple(sorted(virtual_marks)),
    )


_SAME_CARDINALITY_EDGE_PRESERVING_FAMILIES = frozenset(
    {
        "atom_restate",
        "bond_reorder",
        "bond_reroute",
        "ring_system_restate",
    }
)
_CANONICAL_SELF_CANDIDATE_FAMILIES = frozenset(
    {
        # Insert/delete and cycle moves change active cardinality or edge count.
        # A legal semantic atom restate changes one atom-label multiset entry;
        # a legal bond reorder changes total bond-order sum. Canonical identity
        # preserves each invariant. Reroute and ring restatement preserve these
        # coarse invariants and are therefore still executed against source_key.
        # The independent exhaustive 512-state oracle additionally observed
        # zero missed self aliases across all eight families.
        "bond_reroute",
        "ring_system_restate",
    }
)


def _active_atom_and_edge_count(state: MolecularGraph) -> tuple[int, int]:
    real_slots = np.flatnonzero(is_element(np.asarray(state.atom_types)))
    active = int(real_slots.size)
    real_bonds = np.asarray(state.bonds)[np.ix_(real_slots, real_slots)]
    edges = int(np.count_nonzero(np.triu(real_bonds, k=1)))
    return active, edges


def _target_compatible_families(
    source: MolecularGraph,
    target: MolecularGraph,
) -> frozenset[str]:
    """Families that can share a canonical target by hard graph invariants."""

    source_atoms, source_edges = _active_atom_and_edge_count(source)
    target_atoms, target_edges = _active_atom_and_edge_count(target)
    atom_delta = target_atoms - source_atoms
    edge_delta = target_edges - source_edges
    if atom_delta == 1:
        return frozenset({"atom_insert"})
    if atom_delta == -1:
        return frozenset({"atom_delete"})
    if atom_delta != 0:
        return frozenset()
    if edge_delta == 1:
        return frozenset({"cycle_insert"})
    if edge_delta == -1:
        return frozenset({"cycle_attach"})
    if edge_delta == 0:
        return _SAME_CARDINALITY_EDGE_PRESERVING_FAMILIES
    return frozenset()


def compile_teacher_successor_fibers_support_only(
    model: FactorizedTraceletRateModel,
    sources: tuple[MolecularGraph, ...],
    targets: tuple[MolecularGraph, ...],
    *,
    teacher_action_sha256s: tuple[str, ...],
    teacher_families: tuple[str, ...],
    times: tuple[float, ...],
    system: RewriteSystem | None = None,
    chemistry_feature_cache: MutableMapping[Any, Any] | None = None,
    chemistry_feature_cache_limit: int = 2048,
) -> tuple[SupportOnlyTeacherFiberCompilation, ...]:
    """Compile exact teacher/self fibers without scoring irrelevant marks.

    Canonical-equivalent aliases cannot cross active-atom or real-edge-count
    deltas.  The routine therefore decodes only families compatible with the
    exact source/target transition, plus every edge-preserving family that
    could in principle create a canonical self-event.  The complete production
    mask census is still recorded.  An exhaustive bounded equivalence gate is
    the independent oracle for these invariant reductions.
    """

    count = len(sources)
    if not (
        count
        and len(targets) == count
        and len(teacher_action_sha256s) == count
        and len(teacher_families) == count
        and len(times) == count
    ):
        raise ValueError("support-only teacher compilation inputs must be nonempty and aligned")
    process = resolve_successor_process_runtime(model, system=system)
    included: list[frozenset[str]] = []
    for source, target, teacher_family in zip(sources, targets, teacher_families, strict=True):
        target_families = _target_compatible_families(source, target)
        if teacher_family not in target_families:
            raise SuccessorTrainingError(
                "teacher family disagrees with exact cardinality/edge-count transition"
            )
        included.append(target_families | _CANONICAL_SELF_CANDIDATE_FAMILIES)
    try:
        support_rows = enumerate_factorized_legal_support_many(
            model,
            sources,
            times,
            included_families=tuple(included),
            chemistry_feature_cache=chemistry_feature_cache,
            chemistry_feature_cache_limit=chemistry_feature_cache_limit,
        )
    except (ProductionSuccessorKernelError, ValueError) as error:
        raise SuccessorTrainingError(
            "could not enumerate score-free production support for teacher fibers"
        ) from error

    compiled_rows: list[SupportOnlyTeacherFiberCompilation] = []
    for source, target, teacher_action_sha256, teacher_family, support in zip(
        sources,
        targets,
        teacher_action_sha256s,
        teacher_families,
        support_rows,
        strict=True,
    ):
        source_key = canonical_state_key(source)
        target_key = canonical_state_key(target)
        target_digest = persistent_slot_state_sha256(target)
        if support.source_key != source_key or target_key == source_key:
            raise SuccessorTrainingError(
                "support-only teacher compilation received a virtual or mismatched target"
            )
        target_marks: list[CompiledSuccessorMark] = []
        virtual_marks: list[CompiledSuccessorMark] = []
        exact_teacher_aliases: list[TeacherSuccessorAlias] = []
        for mark in support.marks:
            try:
                successor = process.system.apply(
                    source,
                    mark.executor_rule_name,
                    mark.action,
                )
            except Exception as error:
                raise SuccessorTrainingError(
                    "a score-free legal mark failed under the production executor"
                ) from error
            successor_key = canonical_state_key(successor)
            if successor_key not in {source_key, target_key}:
                continue
            successor_digest = persistent_slot_state_sha256(successor)
            action_sha256 = rewrite_action_codec_sha256(
                mark.executor_rule_name,
                mark.action,
                schema_version=process.action_codec_schema_version,
            )
            compiled = CompiledSuccessorMark(
                alias=_alias(
                    family_name=mark.family_name,
                    table_name=mark.table_name,
                    coordinate=mark.coordinate,
                ),
                successor_state_sha256=successor_digest,
                action_sha256=action_sha256,
            )
            if successor_key == source_key:
                virtual_marks.append(compiled)
                continue
            target_marks.append(compiled)
            if (
                mark.family_name == teacher_family
                and successor_digest == target_digest
                and action_sha256 == teacher_action_sha256
            ):
                exact_teacher_aliases.append(compiled.alias)
        if len(exact_teacher_aliases) != 1:
            raise SuccessorTrainingError(
                "teacher action is not one exact coordinate of the support-only target fiber"
            )
        ordered_target = tuple(sorted(target_marks))
        ordered_virtual = tuple(sorted(virtual_marks))
        state_support = StateProductiveSupport(
            source_key=source_key,
            source_state_sha256=persistent_slot_state_sha256(source),
            virtual_aliases=tuple(mark.alias for mark in ordered_virtual),
        )
        fiber = TeacherSuccessorFiber(
            source_key=source_key,
            target_key=target_key,
            target_state_sha256=target_digest,
            aliases=tuple(mark.alias for mark in ordered_target),
            state_support=state_support,
        )
        compiled_rows.append(
            SupportOnlyTeacherFiberCompilation(
                teacher_fiber=fiber,
                exact_teacher_alias=exact_teacher_aliases[0],
                raw_mark_count=support.raw_mark_count,
                decoded_mark_count=len(support.marks),
                marks_by_family=support.marks_by_family,
            )
        )
    return tuple(compiled_rows)


def teacher_successor_fiber_from_compiled_state(
    compiled: CompiledStateSuccessorMap,
    target: MolecularGraph,
) -> TeacherSuccessorFiber:
    """Derive the public teacher fiber for ``target`` without source replay."""

    target_key = canonical_state_key(target)
    if compiled.source_key == target_key:
        raise SuccessorTrainingError(
            "teacher successor equals the source; virtual events are not molecular jumps"
        )
    group = next(
        (
            candidate
            for candidate in compiled.successor_groups
            if candidate.target_key == target_key
        ),
        None,
    )
    if group is None:
        raise SuccessorTrainingError(
            "teacher successor is absent from the production marked support"
        )
    return TeacherSuccessorFiber(
        source_key=compiled.source_key,
        target_key=target_key,
        target_state_sha256=persistent_slot_state_sha256(target),
        aliases=group.aliases,
        state_support=compiled.state_support,
    )


def _exact_successor_group(
    compiled: CompiledStateSuccessorMap,
    target_state_sha256: str,
) -> CanonicalSuccessorAliasGroup:
    matches = tuple(
        group
        for group in compiled.successor_groups
        if target_state_sha256 in group.successor_state_sha256s
    )
    if len(matches) != 1:
        raise SuccessorTrainingError(
            "exact teacher successor is absent from the production marked support"
            if not matches
            else "exact teacher successor belongs to multiple canonical groups"
        )
    return matches[0]


def teacher_successor_fiber_from_exact_digest(
    compiled: CompiledStateSuccessorMap,
    target_state_sha256: str,
) -> TeacherSuccessorFiber:
    """Derive a fiber only when an exact executor output matches the target.

    This stricter builder path catches a stored trace transition that is merely
    canonically equivalent to, but not the exact persistent-slot result of, a
    legal production mark.
    """

    group = _exact_successor_group(compiled, target_state_sha256)
    return TeacherSuccessorFiber(
        source_key=compiled.source_key,
        target_key=group.target_key,
        target_state_sha256=target_state_sha256,
        aliases=group.aliases,
        state_support=compiled.state_support,
    )


def require_exact_successor_action_identity(
    compiled: CompiledStateSuccessorMap,
    *,
    target_state_sha256: str,
    action_sha256: str,
) -> None:
    """Require one enumerated action identity paired with this exact output."""

    group = _exact_successor_group(compiled, target_state_sha256)
    if not group.contains_exact_action(
        successor_state_sha256=target_state_sha256,
        action_sha256=action_sha256,
    ):
        raise SuccessorTrainingError(
            "teacher action identity is absent from the exact successor marked support"
        )


def compile_teacher_successor_fiber(
    model: FactorizedTraceletRateModel,
    source: MolecularGraph,
    target: MolecularGraph,
    *,
    time: float = 0.5,
    system: RewriteSystem | None = None,
) -> TeacherSuccessorFiber:
    """Compile every production mark from ``source`` that canonicalizes to ``target``."""

    return teacher_successor_fiber_from_compiled_state(
        compile_state_successor_map(
            model,
            source,
            time=time,
            system=system,
        ),
        target,
    )


def compile_state_productive_support(
    model: FactorizedTraceletRateModel,
    source: MolecularGraph,
    *,
    time: float = 0.5,
    system: RewriteSystem | None = None,
) -> StateProductiveSupport:
    """Compile canonical self-event coordinates for a terminal/no-teacher row."""

    return compile_state_successor_map(
        model,
        source,
        time=time,
        system=system,
    ).state_support


def _log1mexp(log_probability: Tensor) -> Tensor:
    """Stable ``log(1 - exp(x))`` for log probabilities ``x <= 0``."""

    if bool((log_probability > 1e-6).any()):
        raise SuccessorTrainingError("virtual probability exceeds one")
    cutoff = -0.6931471805599453
    return torch.where(
        log_probability < cutoff,
        torch.log1p(-log_probability.exp()),
        torch.log(-torch.expm1(log_probability)),
    )


def _factorized_action_probability_tables(
    model: FactorizedTraceletRateModel,
    batch: FactorizedMarkBatch,
) -> tuple[
    FactorizedMarkBatch,
    Tensor,
    dict[str, Tensor],
    dict[str, Tensor],
    Tensor,
    Tensor,
    Tensor,
]:
    """Compute the one shared differentiable marked-law table representation."""

    if batch.atom_types.device != model.device:
        batch = batch.to(model.device)
    node, global_state, pair = model._encode_batch(batch)
    masks, logits, action_log_z = model._action_tables(
        batch,
        node,
        global_state,
        pair,
    )
    enabled = torch.isfinite(action_log_z)
    family_logits = _masked_family_logits(
        model._family_base_logits(batch, global_state),
        action_log_z,
        enabled,
        rate_factorization=model.rate_factorization,
    )
    family_log_probability = torch.log_softmax(family_logits, dim=-1)
    return (
        batch,
        global_state,
        masks,
        logits,
        action_log_z,
        enabled,
        family_log_probability,
    )


def _compiled_alias_log_probability(
    *,
    batch_index: int,
    alias: TeacherSuccessorAlias,
    masks: dict[str, Tensor],
    logits: dict[str, Tensor],
    action_log_z: Tensor,
    family_log_probability: Tensor,
) -> tuple[Tensor, Tensor]:
    family_index = MARK_RULE_TO_INDEX.get(alias.family_name)
    if family_index is None:
        raise SuccessorTrainingError(
            f"compiled successor references unknown family {alias.family_name!r}"
        )
    table_logits = logits.get(alias.table_name)
    table_mask = masks.get(alias.table_name)
    if table_logits is None or table_mask is None:
        raise SuccessorTrainingError(
            f"compiled successor references unknown table {alias.table_name!r}"
        )
    key = (batch_index, *alias.coordinate)
    try:
        legal = table_mask[key]
        raw_logit = table_logits[key]
    except IndexError as error:
        raise SuccessorTrainingError(
            "compiled successor coordinate lies outside the current action table"
        ) from error
    return (
        family_log_probability[batch_index, family_index]
        + raw_logit
        - action_log_z[batch_index, family_index],
        legal,
    )


def forward_compiled_successor_partitions(
    model: FactorizedTraceletRateModel,
    batch: FactorizedMarkBatch,
    partitions: tuple[CompiledStateSuccessorMap, ...],
    *,
    normalization_atol: float = 2e-5,
) -> FactorizedSuccessorPartitionPrediction:
    """Score complete static successor partitions without replaying chemistry.

    This is the full-partition counterpart to
    :func:`forward_teacher_successor_batch`.  It shares the production action
    tables, validates every cached coordinate against the current masks and
    proves that productive groups plus virtual aliases exhaust unit mark mass.
    """

    if len(partitions) != batch.batch_size:
        raise ValueError("successor partitions must align with the batch")
    if not 0.0 < float(normalization_atol) < 1.0:
        raise ValueError("normalization_atol must lie in (0, 1)")
    (
        batch,
        global_state,
        masks,
        logits,
        action_log_z,
        enabled,
        family_log_probability,
    ) = _factorized_action_probability_tables(model, batch)
    productive_log_probability = global_state.new_empty(batch.batch_size)
    successor_keys: list[tuple[str, ...]] = []
    successor_rows: list[Tensor] = []

    for batch_index, partition in enumerate(partitions):
        observed_source_digest = persistent_slot_state_sha256(batch.states[batch_index])
        if observed_source_digest != partition.source_state_sha256:
            raise SuccessorTrainingError(
                "compiled successor partition does not match the exact batch state"
            )
        virtual_values: list[Tensor] = []
        virtual_legalities: list[Tensor] = []
        for alias in partition.state_support.virtual_aliases:
            value, legal = _compiled_alias_log_probability(
                batch_index=batch_index,
                alias=alias,
                masks=masks,
                logits=logits,
                action_log_z=action_log_z,
                family_log_probability=family_log_probability,
            )
            virtual_values.append(value)
            virtual_legalities.append(legal)
        if virtual_legalities and not bool(torch.stack(virtual_legalities).all()):
            raise SuccessorTrainingError("compiled virtual successor coordinate is no longer legal")
        virtual_log_probability = (
            global_state.new_tensor(float("-inf"))
            if not virtual_values
            else torch.logsumexp(torch.stack(virtual_values), dim=0)
        )
        productive_log_probability[batch_index] = _log1mexp(virtual_log_probability)

        group_values: list[Tensor] = []
        for group in partition.successor_groups:
            alias_values: list[Tensor] = []
            alias_legalities: list[Tensor] = []
            for alias in group.aliases:
                value, legal = _compiled_alias_log_probability(
                    batch_index=batch_index,
                    alias=alias,
                    masks=masks,
                    logits=logits,
                    action_log_z=action_log_z,
                    family_log_probability=family_log_probability,
                )
                alias_values.append(value)
                alias_legalities.append(legal)
            if not bool(torch.stack(alias_legalities).all()):
                raise SuccessorTrainingError(
                    "compiled productive successor coordinate is no longer legal"
                )
            group_values.append(torch.logsumexp(torch.stack(alias_values), dim=0))
        if not group_values:
            raise SuccessorTrainingError("compiled state has no productive molecular successor")
        raw_group_values = torch.stack(group_values)
        observed_productive = torch.logsumexp(raw_group_values, dim=0)
        if not bool(
            torch.isclose(
                observed_productive.detach(),
                productive_log_probability[batch_index].detach(),
                rtol=0.0,
                atol=float(normalization_atol),
            )
        ):
            raise SuccessorTrainingError(
                "compiled productive and virtual coordinates do not exhaust mark mass"
            )
        successor_keys.append(tuple(group.target_key for group in partition.successor_groups))
        successor_rows.append(raw_group_values - productive_log_probability[batch_index])

    return FactorizedSuccessorPartitionPrediction(
        successor_keys=tuple(successor_keys),
        successor_log_probabilities=tuple(successor_rows),
        productive_log_probability=productive_log_probability,
        family_log_probabilities=family_log_probability,
        enabled_families=enabled,
    )


def forward_teacher_successor_batch(
    model: FactorizedTraceletRateModel,
    batch: FactorizedMarkBatch,
    fibers: tuple[TeacherSuccessorFiber | None, ...],
    *,
    state_supports: tuple[StateProductiveSupport | None, ...] | None = None,
) -> FactorizedSuccessorPrediction:
    """Score precompiled teacher-successor fibers on the differentiable action tables."""

    if len(fibers) != batch.batch_size:
        raise ValueError("teacher-successor fibers must align with the batch")
    if state_supports is not None and len(state_supports) != batch.batch_size:
        raise ValueError("state productive supports must align with the batch")
    nonterminal_flags = tuple(float(value) > 0.0 for value in batch.teacher_rates.detach().cpu())
    (
        batch,
        global_state,
        masks,
        logits,
        action_log_z,
        enabled,
        family_log_probability,
    ) = _factorized_action_probability_tables(model, batch)
    has_legal_mark = enabled.any(dim=-1)

    selected = global_state.new_zeros(batch.batch_size)
    selected_productive = global_state.new_zeros(batch.batch_size)
    productive_log_probability = global_state.new_zeros(batch.batch_size)
    teacher_family_log_probability = global_state.new_zeros(batch.batch_size)
    selected_within_teacher_family_log_probability = global_state.new_zeros(batch.batch_size)

    def alias_log_probability(
        batch_index: int,
        alias: TeacherSuccessorAlias,
    ) -> tuple[Tensor, Tensor]:
        return _compiled_alias_log_probability(
            batch_index=batch_index,
            alias=alias,
            masks=masks,
            logits=logits,
            action_log_z=action_log_z,
            family_log_probability=family_log_probability,
        )

    for batch_index, fiber in enumerate(fibers):
        is_nonterminal = nonterminal_flags[batch_index]
        if not is_nonterminal:
            if fiber is not None:
                raise SuccessorTrainingError(
                    "terminal training row carries a teacher-successor fiber"
                )
            support = None if state_supports is None else state_supports[batch_index]
            if support is None:
                raise SuccessorTrainingError(
                    "terminal training row is missing productive state support"
                )
            observed_source_digest = persistent_slot_state_sha256(batch.states[batch_index])
            if observed_source_digest != support.source_state_sha256:
                raise SuccessorTrainingError(
                    "productive state support does not match the exact terminal batch state"
                )
            virtual_aliases = support.virtual_aliases
        else:
            if fiber is None:
                raise SuccessorTrainingError(
                    "nonterminal training row is missing its teacher-successor fiber"
                )
            observed_source_digest = persistent_slot_state_sha256(batch.states[batch_index])
            if observed_source_digest != fiber.state_support.source_state_sha256:
                raise SuccessorTrainingError(
                    "teacher-successor fiber source does not match the exact batch state"
                )
            explicit_support = None if state_supports is None else state_supports[batch_index]
            if explicit_support is not None and explicit_support != fiber.state_support:
                raise SuccessorTrainingError(
                    "explicit productive support disagrees with the teacher fiber"
                )
            virtual_aliases = fiber.state_support.virtual_aliases

        virtual_log_probabilities: list[Tensor] = []
        virtual_legalities: list[Tensor] = []
        for alias in virtual_aliases:
            log_probability, legal = alias_log_probability(batch_index, alias)
            virtual_log_probabilities.append(log_probability)
            virtual_legalities.append(legal)
        if virtual_legalities and not bool(torch.stack(virtual_legalities).all()):
            raise SuccessorTrainingError(
                "precompiled virtual alias is no longer legal in the current batch"
            )
        virtual_log_probability = (
            global_state.new_tensor(float("-inf"))
            if not virtual_log_probabilities
            else torch.logsumexp(torch.stack(virtual_log_probabilities), dim=0)
        )
        productive_log_probability[batch_index] = _log1mexp(virtual_log_probability)

        if not is_nonterminal:
            continue

        alias_log_probabilities: list[Tensor] = []
        alias_legalities: list[Tensor] = []
        teacher_family_name = _CYCLE_OP_EXECUTOR_TO_FAMILY.get(
            batch.teacher_rule_names[batch_index],
            batch.teacher_rule_names[batch_index],
        )
        if teacher_family_name is None:
            alias_families = {alias.family_name for alias in fiber.aliases}
            if len(alias_families) != 1:
                raise SuccessorTrainingError(
                    "teacher family is absent and cannot be inferred uniquely "
                    "from a cross-family successor fiber"
                )
            teacher_family_name = next(iter(alias_families))
        teacher_family_index = MARK_RULE_TO_INDEX.get(teacher_family_name)
        if teacher_family_index is None:
            raise SuccessorTrainingError(
                f"teacher row references unknown family {teacher_family_name!r}"
            )
        within_teacher_family: list[Tensor] = []
        for alias in fiber.aliases:
            log_probability, legal = alias_log_probability(batch_index, alias)
            alias_legalities.append(legal)
            alias_log_probabilities.append(log_probability)
            if alias.family_name == teacher_family_name:
                within_teacher_family.append(log_probability)
        if not bool(torch.stack(alias_legalities).all()):
            raise SuccessorTrainingError(
                "precompiled teacher alias is no longer legal in the current batch"
            )
        selected[batch_index] = torch.logsumexp(
            torch.stack(alias_log_probabilities),
            dim=0,
        )
        selected_productive[batch_index] = (
            selected[batch_index] - productive_log_probability[batch_index]
        )
        if not within_teacher_family:
            raise SuccessorTrainingError(
                "teacher successor fiber has no alias in the recorded teacher family"
            )
        teacher_family_log_probability[batch_index] = family_log_probability[
            batch_index,
            teacher_family_index,
        ]
        selected_within_teacher_family_log_probability[batch_index] = (
            torch.logsumexp(torch.stack(within_teacher_family), dim=0)
            - teacher_family_log_probability[batch_index]
        )

    total_hazard = F.softplus(
        model.total_hazard_head(global_state).squeeze(-1)
    ) * has_legal_mark.to(global_state.dtype)
    productive_hazard = total_hazard * productive_log_probability.exp()
    return FactorizedSuccessorPrediction(
        total_hazard=total_hazard,
        productive_hazard=productive_hazard,
        selected_successor_log_probability=selected,
        selected_productive_successor_log_probability=selected_productive,
        productive_log_probability=productive_log_probability,
        teacher_family_log_probability=teacher_family_log_probability,
        selected_within_teacher_family_log_probability=(
            selected_within_teacher_family_log_probability
        ),
        family_log_probabilities=family_log_probability,
        enabled_families=enabled,
    )


def factorized_successor_bregman_loss(
    prediction: FactorizedSuccessorPrediction,
    batch: FactorizedMarkBatch,
) -> Tensor:
    """Poisson-KL GM loss on the canonical molecular-successor generator.

    Canonical self-events do not contribute to a state generator.  The positive
    term is therefore the productive hazard, while the selected rate remains
    ``total_hazard * raw_teacher_successor_mass``.
    """

    teacher_rates = batch.teacher_rates.to(prediction.total_hazard.device)
    weights = batch.importance_weights.to(prediction.total_hazard.device)
    log_hazard = torch.log(prediction.total_hazard.clamp_min(1e-12))
    nonterminal = teacher_rates > 0
    teacher_term = teacher_rates * (log_hazard + prediction.selected_successor_log_probability)
    per_example = prediction.productive_hazard - torch.where(
        nonterminal,
        teacher_term,
        torch.zeros_like(teacher_term),
    )
    return (per_example * weights).mean()


def factorized_successor_identity_loss(
    prediction: FactorizedSuccessorPrediction,
    batch: FactorizedMarkBatch,
) -> Tensor:
    """Editing-only NLL under the productive embedded molecular jump chain."""

    weights = batch.importance_weights.to(prediction.total_hazard.device)
    nonterminal = batch.teacher_rates.to(prediction.total_hazard.device) > 0
    if not bool(nonterminal.any()):
        raise SuccessorTrainingError("successor-identity loss requires at least one molecular jump")
    selected_weights = weights[nonterminal]
    denominator = selected_weights.sum()
    if not bool(denominator > 0):
        raise SuccessorTrainingError(
            "successor-identity loss has zero nonterminal importance weight"
        )
    return (
        -(
            prediction.selected_productive_successor_log_probability[nonterminal] * selected_weights
        ).sum()
        / denominator
    )


def factorized_hazard_bregman_loss(
    prediction: FactorizedSuccessorPrediction,
    batch: FactorizedMarkBatch,
) -> Tensor:
    """Poisson-KL total-hazard loss kept separate from successor identity."""

    teacher_rates = batch.teacher_rates.to(prediction.total_hazard.device)
    weights = batch.importance_weights.to(prediction.total_hazard.device)
    log_hazard = torch.log(prediction.total_hazard.clamp_min(1e-12))
    teacher_term = torch.where(
        teacher_rates > 0,
        teacher_rates * log_hazard,
        torch.zeros_like(teacher_rates),
    )
    return ((prediction.total_hazard - teacher_term) * weights).mean()


__all__ = [
    "CanonicalSuccessorAliasGroup",
    "CompiledStateSuccessorMap",
    "CompiledSuccessorMark",
    "FactorizedSuccessorPrediction",
    "FactorizedSuccessorPartitionPrediction",
    "StateProductiveSupport",
    "SupportOnlyTeacherFiberCompilation",
    "SuccessorProcessRuntime",
    "SuccessorTrainingError",
    "TeacherSuccessorAlias",
    "TeacherSuccessorFiber",
    "compile_state_productive_support",
    "compile_state_successor_map",
    "compile_teacher_successor_fiber",
    "compile_teacher_successor_fibers_support_only",
    "factorized_hazard_bregman_loss",
    "factorized_successor_bregman_loss",
    "factorized_successor_identity_loss",
    "forward_compiled_successor_partitions",
    "forward_teacher_successor_batch",
    "require_exact_successor_action_identity",
    "resolve_successor_process_runtime",
    "rewrite_action_codec_sha256",
    "teacher_successor_fiber_from_compiled_state",
    "teacher_successor_fiber_from_exact_digest",
]
